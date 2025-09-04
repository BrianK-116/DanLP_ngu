# -*- coding: utf-8 -*-
# app.py — Flask backend tối giản, có comment từng dòng

import bisect
from flask import Flask, render_template, Response, request, send_file, jsonify, abort, current_app, url_for, make_response, redirect  # import các hàm Flask
from pathlib import Path                                            # chuẩn hoá đường dẫn
import json                                                         # đọc file JSON
import os, traceback      
import re
from collections import OrderedDict
from functools import lru_cache
import io, csv                          # ghi CSV vào bộ nhớ
from datetime import datetime              # fallback tên file

# ====== Khởi tạo app & đường dẫn gốc ======
app = Flask(__name__, template_folder='templates')                  # tạo Flask app, chỉ ra thư mục template
BASE = Path(__file__).resolve().parent                              # thư mục chứa file app.py
MAP_DIR   = BASE / "data" / "map_keyframes"                         # chứa các CSV vcode.csv
VIDEOS_RT = BASE / "data" / "videos"                                # gốc thư mục videos
os.environ['CUDA_VISIBLE_DEVICES'] = '0'                            # chọn GPU nếu cần (không bắt buộc)

# ====== Tải mapping ID->path ảnh ======
try:                                                                # cố gắng mở image_path.json
    with open(BASE / 'image_path.json', 'r', encoding='utf-8') as f:# đọc JSON ở cùng cấp với app.py
        _map = json.load(f)                                         # parse thành dict
    ID2PATH = {int(k): v for k, v in _map.items()}                  # ép key string -> int
except Exception:                                                   # nếu lỗi đọc file
    print("[DATA] Cannot load image_path.json")                     # log lỗi
    print(traceback.format_exc())                                   # in traceback
    ID2PATH = {}                                                    # rỗng để không văng 500

N_ITEMS = len(ID2PATH)                                              # tổng số ảnh trong index

# ====== (Tuỳ chọn) FAISS helper ======
try:                                                                # import lớp tìm kiếm (nếu có)
    from utils.query_processing import Translation                  # dịch vi↔en
    from utils.faiss import Myfaiss                                 # wrapper FAISS
    # đường dẫn 2 index (cùng cấp app.py)
    VIS_BIN = str(BASE / 'faiss_normal_ViT.bin')                    # index ảnh
    OCR_BIN = str(BASE / 'faiss_ocr_ViT.bin')                       # index OCR
    # khởi tạo MyFaiss (ưu tiên CUDA, fallback CPU)
    try:
        MF = Myfaiss(VIS_BIN, OCR_BIN, ID2PATH, 'cuda',             # thử CUDA trước
                     Translation(), "ViT-B/32")                     # backbone CLIP
    except Exception:
        print("[FAISS] CUDA failed -> CPU")                         # nếu CUDA fail -> dùng CPU
        MF = Myfaiss(VIS_BIN, OCR_BIN, ID2PATH, 'cpu',              # khởi tạo CPU
                     Translation(), "ViT-B/32")                     # backbone CLIP
    FAISS_OK = True                                                 # cờ cho biết FAISS sẵn sàng
except Exception:                                                   # nếu không import được
    print("[FAISS] Not available; search routes will return empty") # thông báo
    MF = None                                                       # không có MyFaiss
    FAISS_OK = False         # không sẵn FAISS

# ====== Helpers giữ trạng thái UI ======
def to_list(x):
    if x is None:
        return []
    # numpy -> list
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return [x.item()]
    except Exception:
        pass
    # list/tuple -> list
    if isinstance(x, (list, tuple)):
        return list(x)
    # rơi vào scalar -> bọc list 1 phần tử
    return [x]

def to_float_scalar(x, default=0.0):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return float(x.item())
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return float(default)
            return float(x.flatten()[0].item())
    except Exception:
        pass
    if isinstance(x, (list, tuple)) and x:
        return to_float_scalar(x[0], default)
    try:
        return float(x)
    except Exception:
        return float(default)

def call_text_search_best_effort(mf, text, k=None, search_type=None):
    """Thử lần lượt các chữ ký phổ biến: (text,k,search_type) -> (text,k) -> (text)"""
    try:
        return mf.text_search(text, k=k, search_type=search_type)   # thử đầy đủ
    except TypeError:
        try:
            return mf.text_search(text, k=k)                        # thiếu search_type
        except TypeError:
            try:
                return mf.text_search(text)                         # chỉ có text
            except Exception:
                print("[call_text_search_best_effort] all variants failed")
                print(traceback.format_exc())
                return None

# --- helper: tách mã video từ đường dẫn keyframe ---
#   "data/keyframes/Keyframes_L21/L21_V001/020.jpg"  ->  "L21_V001"
def _video_code_from_path(p: str) -> str:
    try:
        return Path(p).parent.name  # thư mục ngay trước filename
    except Exception:
        m = re.search(r'(L\d+_V\d+)', str(p))
        return m.group(1) if m else "unknown"

# --- helper: tách số frame từ tên file ---
#   ".../020.jpg" -> 20
def _frame_no_from_path(p: str) -> int:
    try:
        name = Path(p).stem
        m = re.search(r'(\d+)$', name)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1

# --- helper: nhóm list item theo video, sắp xếp tăng dần theo frame ---
def group_by_video(items):
    """
    items: list[{'id', 'path', 'score', 'ocr'?}]
    return: list[(video_code, [items...])]
    """
    bucket = OrderedDict()
    for r in items:
        path = r.get("path", "")
        vcode = _video_code_from_path(path)
        obj = dict(r)
        obj["video"] = vcode
        obj["frame"] = _frame_no_from_path(path)
        bucket.setdefault(vcode, []).append(obj)

    for k in bucket:
        bucket[k].sort(key=lambda x: (x.get("frame", -1), x.get("id", 0)))

    # Trả list để template duyệt ổn định theo thứ tự xuất hiện
    return list(bucket.items())

def _video_bucket(vcode: str) -> str:
    # "L25_V085" -> "Videos_L25"
    try:
        return f"Videos_{vcode.split('_', 1)[0]}"
    except Exception:
        return "Videos_unknown"

@lru_cache(maxsize=512)
def load_pts_map(vcode: str):
    """
    Đọc data/map_keyframes/<vcode>.csv -> {n:int -> pts_time:float (giây)}.
    - Tìm cột 'n' (chỉ số keyframe) và 'pts_time' (thời gian giây).
    - Nếu không thấy đúng tên thì thử các alias phổ biến.
    """
    path = MAP_DIR / f"{vcode}.csv"
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    rows = []
    reader = None
    for delim in (",", ";", "\t", "|"):
        try:
            reader = csv.DictReader(raw.splitlines(), delimiter=delim)
            rows = list(reader)
            if rows and reader.fieldnames:
                break
        except Exception:
            pass
    if not rows:
        return {}

    names = [c or "" for c in (reader.fieldnames or [])]
    def pick(names, cands):
        for c in names:
            if c.strip().lower() in cands:
                return c
        return None

    N_CANDS = {"n", "idx", "index", "no", "num", "number", "frame_no", "frame", "kf", "kf_idx"}
    T_CANDS = {"pts_time", "time", "sec", "secs", "seconds", "ts"}

    ncol = pick(names, N_CANDS)
    tcol = pick(names, T_CANDS)
    if not ncol or not tcol:
        return {}

    def to_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None
    def to_float(x):
        try:
            return float(str(x).strip().replace(",", "."))
        except Exception:
            return None

    mp = {}
    for r in rows:
        n = to_int(r.get(ncol))
        t = to_float(r.get(tcol))
        if n is None or t is None:
            continue
        mp[n] = t
    return mp

def lookup_time_by_n(vcode: str, n: int):
    rows = _load_map_rows(vcode)
    mp = {r["n"]: r["pts_time"] for r in rows if r["pts_time"] is not None}
    if not mp:
        return None
    if n in mp:
        return mp[n]
    ks = sorted(mp.keys())
    pos = bisect.bisect_right(ks, n) - 1
    return mp[ks[pos]] if pos >= 0 else None

# --- Chuẩn hoá kết quả trả về từ Myfaiss về (scores, ids, paths) an toàn ---
def normalize_ret(ret):
    """ret có thể là None / tuple / dict. Trả về (scores:list, ids:list, paths:list)."""
    scores, ids, paths = [], [], []
    if ret is None:
        return scores, ids, paths
    if isinstance(ret, dict):
        scores = list(ret.get("scores") or [])
        ids    = list(ret.get("ids")    or [])
        paths  = list(ret.get("paths")  or [])
        return scores, ids, paths
    if isinstance(ret, (list, tuple)):
        if len(ret) >= 4:
            a, b, _, d = ret
            scores = list(a) if a is not None else []
            ids    = list(b) if b is not None else []
            paths  = list(d) if d is not None else []
        elif len(ret) == 3:
            a, b, c = ret
            def is_paths(x): return isinstance(x, (list, tuple)) and (len(x) == 0 or isinstance(x[0], str))
            if is_paths(a):
                paths, ids, scores = list(a), list(b or []), list(c or [])
            elif is_paths(b):
                ids, paths, scores = list(a or []), list(b), list(c or [])
            elif is_paths(c):
                ids, scores, paths = list(a or []), list(b or []), list(c)
            else:
                ids, scores, paths = list(a or []), list(b or []), list(c or [])
        elif len(ret) == 2:
            a, b = ret
            if isinstance(a, (list, tuple)) and a and isinstance(a[0], str):
                paths, ids = list(a), list(b or [])
            elif isinstance(b, (list, tuple)) and b and isinstance(b[0], str):
                ids, paths = list(a or []), list(b)
            else:
                ids, paths = list(a or []), list(b or [])
                scores = [0.0] * min(len(ids), len(paths))
    return scores, ids, paths

def get_ui(args):
    return {
        "textquery": args.get("textquery", ""),
        "search_type": args.get("search_type", "visual"),
        "topk": int(args.get("topk", 100)),
    }

def page_bounds(index, total, size):
    first = max(index, 0) * max(size, 1)
    last  = min(first + max(size, 1), total)
    return first, last

# ====== Helpers xử lý file map keyframe & đường dẫn ======
def _sanitize_filename(name: str) -> str:
    """Làm sạch tên file: loại kí tự lạ, đảm bảo .csv"""
    # Chỉ giữ chữ/số/_/-/., còn lại thay bằng _
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe

def _map_file_for(vcode: str) -> str:
    """Trả về đường dẫn CSV map cho video code."""
    return os.path.join('data', 'map_keyframes', f'{vcode}.csv')

@lru_cache(maxsize=256)
def _load_map_index(vcode: str) -> dict:
    rows = _load_map_rows(vcode)
    return {r["n"]: r["frame_idx"] for r in rows}

@lru_cache(maxsize=256)
def _load_map_rows(vcode: str):
    """
    Đọc data/map_keyframes/<vcode>.csv và trả về list các bản ghi:
    mỗi bản ghi là dict {'n': int, 'frame_idx': int, 'pts_time': float}
    - Linh hoạt tên cột: n / frame / idx / frameid ; frame_idx / frameid ; pts_time / time / seconds / timestamp
    - Nếu không có file hoặc lỗi parse -> trả [].
    """
    path = os.path.join('data', 'map_keyframes', f'{vcode}.csv')
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return rows
            lowers = [c or "" for c in (reader.fieldnames or [])]
            def pick(*cands):
                for c in lowers:
                    if c.strip().lower() in cands:
                        return reader.fieldnames[lowers.index(c)]
                return None
            n_col   = pick('n', 'frame', 'idx', 'frame_no', 'frameindex')
            fi_col  = pick('frame_idx', 'frameindex', 'frame_id', 'frameid')
            t_col   = pick('pts_time', 'time', 'sec', 'seconds', 'timestamp')
            for row in reader:
                try:
                    n = int(str(row[n_col]).strip()) if n_col else 0
                except:
                    n = 0
                try:
                    frame_idx = int(str(row[fi_col]).strip()) if fi_col else n
                except:
                    frame_idx = n
                try:
                    pts_time = float(str(row[t_col]).strip()) if t_col else None
                except:
                    pts_time = None
                rows.append({'n': n, 'frame_idx': frame_idx, 'pts_time': pts_time})
    except Exception:
        return []
    try:
        rows.sort(key=lambda r: (r['pts_time'] is None, r['pts_time']))
    except Exception:
        pass
    return rows

def _keyframe_bucket(vcode: str) -> str:
    """
    Từ L21_V001 -> 'Keyframes_L21' để ghép đường dẫn ảnh keyframe.
    """
    try:
        prefix = vcode.split('_')[0]
        return f"Keyframes_{prefix}"
    except Exception:
        return "Keyframes"

def _candidate_frame_names(n: int):
    """
    Sinh các tên file khung hình có thể có:
    - zero-pad 3, 4, 5; và dạng không pad. Kèm các đuôi phổ biến.
    """
    base_nums = {str(n), f"{n:03d}", f"{n:04d}", f"{n:05d}"}
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for bn in base_nums:
        for ext in exts:
            yield bn + ext

def _normalize_path(p: str) -> str:
    """Chuẩn hoá path để so khớp (lower + normpath)."""
    return os.path.normpath(p).replace("\\", "/").lower()

PATH2ID = None
def _ensure_PATH2ID():
    """Đảm bảo có bảng tra PATH->ID để tìm imgid theo path ảnh."""
    global PATH2ID
    if PATH2ID is not None:
        return PATH2ID
    try:
        from app import ID2PATH
    except Exception:
        try:
            ID2PATH  # noqa
        except NameError:
            return {}
    PATH2ID = {}
    for _id, p in ID2PATH.items():
        PATH2ID[_normalize_path(p)] = int(_id)
    return PATH2ID

# ====== Route: Trang chủ (lưới ảnh tĩnh) ======
@app.route("/", methods=["GET"], endpoint="home")
def home():
    idx = int(request.args.get("index", 0))
    ui = get_ui(request.args)
    size = ui["topk"]
    f, l = page_bounds(idx, N_ITEMS, size)
    results = []
    for i in range(f, l):
        rel = ID2PATH.get(i, "")
        results.append({"id": i, "path": rel, "score": 0.0, "ocr": ""})
    has_more = l < N_ITEMS
    groups = group_by_video(results)
    return render_template("home.html",
                           results=results,
                           groups=groups,
                           total=N_ITEMS,
                           elapsed=0,
                           page=idx,
                           has_more=has_more,
                           ui=ui)

# ====== Route: Tìm kiếm ảnh-ảnh ======
@app.route("/imgsearch", methods=["GET"])
def imgsearch():
    ui = get_ui(request.args)
    idx = int(request.args.get("index", 0))
    if not FAISS_OK:
        return render_template("home.html", results=[], total=0,
                               elapsed=0, page=idx, has_more=False, ui=ui)
    qid = int(request.args.get("imgid", 0))
    try:
        ret = MF.image_search(qid, k=ui["topk"])
    except Exception:
        print("[/imgsearch] failed")
        print(traceback.format_exc())
        ret = None
    scores, ids, paths = [], [], []
    if isinstance(ret, (list, tuple)) and len(ret) >= 4:
        scores, ids, paths = ret[0], ret[1], ret[3]
    elif isinstance(ret, dict):
        scores, ids, paths = ret.get("scores"), ret.get("ids"), ret.get("paths")
    scores = to_list(scores)
    ids = to_list(ids)
    paths = to_list(paths)
    total = min(len(paths), len(ids))
    size = max(int(ui["topk"]), 1)
    first = max(idx, 0) * size
    last = min(first + size, total)
    out = []
    for i in range(first, last):
        s = to_float_scalar(scores[i], 0.0) if i < len(scores) else 0.0
        _id = ids[i]
        try:
            import numpy as np
            if isinstance(_id, np.generic):
                _id = _id.item()
        except Exception:
            pass
        try:
            _id = int(_id)
        except Exception:
            pass
        out.append({
            "id": _id,
            "path": paths[i],
            "score": s,
            "ocr": ""
        })
    groups = group_by_video(out)
    has_more = last < total
    return render_template("home.html",
                           results=out, groups=groups, total=total, elapsed=0,
                           page=idx, has_more=has_more, ui=ui)

# ====== Route: Tìm kiếm text (visual/ocr/hybrid) ======
@app.route("/textsearch", methods=["GET"])
def textsearch():
    ui = get_ui(request.args)
    idx = int(request.args.get("index", 0))
    if (not FAISS_OK) or (ui["textquery"].strip() == ""):
        return render_template("home.html",
                               results=[], total=0, elapsed=0,
                               page=idx, has_more=False, ui=ui)
    ret = call_text_search_best_effort(
        MF,
        text=ui["textquery"],
        k=ui["topk"],
        search_type=ui["search_type"]
    )
    scores, ids, paths = normalize_ret(ret)
    total = min(len(paths), len(ids))
    size = max(int(ui["topk"]), 1)
    first = max(idx, 0) * size
    last = min(first + size, total)
    out = []
    for i in range(first, last):
        s = float(scores[i]) if i < len(scores) else 0.0
        _id = ids[i]
        _id = int(_id) if isinstance(_id, (int, float, str)) and str(_id).isdigit() else _id
        out.append({"id": _id, "path": paths[i], "score": s, "ocr": ""})
    has_more = last < total
    groups = group_by_video(out)
    return render_template("home.html",
                           results=out, groups=groups, total=total, elapsed=0,
                           page=idx, has_more=has_more, ui=ui)

# ====== Route: Trả file ảnh theo path tương đối ======
@app.route("/get_img", methods=["GET"])
def get_img():
    rel = (request.args.get("fpath") or "").strip()
    if not rel:
        abort(400, "Missing fpath")
    p = Path(rel)
    if not p.is_absolute():
        p = (BASE / rel).resolve()
    try:
        _ = p.relative_to(BASE)
    except Exception:
        abort(403)
    if not p.exists():
        abort(404)
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    return send_file(p, mimetype=mime, conditional=True)

@app.route("/get_video")
def get_video():
    vcode = (request.args.get("vcode") or "").strip()
    if not vcode:
        abort(400, "Missing vcode")
    bucket = _video_bucket(vcode)
    fpath = VIDEOS_RT / bucket / f"{vcode}.mp4"
    if not fpath.exists():
        abort(404, "Video not found")
    return send_file(fpath, mimetype="video/mp4", conditional=True)

@app.route("/watch")
def watch():
    vcode = (request.args.get("vcode") or "").strip()
    try:
        n = int(request.args.get("frame")) if request.args.get("frame") else None
    except Exception:
        n = None
    if not vcode:
        abort(400, "Missing vcode")
    pts = lookup_time_by_n(vcode, n) if n is not None else None
    video_src = url_for("get_video", vcode=vcode)
    fragment = f"#t={pts:.3f}" if isinstance(pts, (int, float)) else ""
    return render_template("player.html",
                           video_src=video_src + fragment,
                           pts_time=pts, vcode=vcode, frame=n)

@app.route("/frame_time")
def frame_time():
    vcode = (request.args.get("vcode") or "").strip()
    try:
        n = int(request.args.get("frame"))
    except Exception:
        abort(400, "Missing or invalid frame")
    if not vcode:
        abort(400, "Missing vcode")
    t = lookup_time_by_n(vcode, n)
    return jsonify({"vcode": vcode, "n": n, "time": t})

@app.route("/debug_map")
def debug_map():
    vcode = (request.args.get('vcode') or '').strip()
    if not vcode:
        return jsonify(error="Missing vcode"), 400
    _ = _load_map_rows.cache_clear()
    rows = _load_map_rows(vcode)
    mp_time = {r["n"]: r["pts_time"] for r in rows}
    keys = sorted(mp_time.keys())
    head = [{ "frame": k, "time": mp_time[k] } for k in keys[:10]]
    tail = [{ "frame": k, "time": mp_time[k] } for k in keys[-10:]]
    return jsonify({
        "vcode": vcode,
        "count": len(keys),
        "head": head,
        "tail": tail
    })

@app.post("/export/kis")
def export_kis_csv():
    """
    Xuất CSV cho KIS:
    - Client gửi rows = [{ vcode, frame_idx }] (frame_idx phía client chính là 'n' - stt keyframe).
    - Server sẽ đổi 'n' -> 'frame_idx' thật theo file data/map_keyframes/<vcode>.csv (cột frame_idx).
    - Trả về file CSV: <Tên file video>, <Frame Idx>.
    - Pad lên 100 dòng nếu thiếu (nhưng không pad khi rỗng).
    """
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            try:
                import json as _json
                raw = request.get_data(as_text=True) or ""
                data = _json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {}
        rows = data.get("rows") or []
        filename = data.get("filename") or f"submit_kis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filename = _sanitize_filename(filename)
        pairs_n = []
        for it in rows:
            vcode = str((it or {}).get("vcode", "")).strip()
            fidx_raw = (it or {}).get("frame_idx", None)
            try:
                n_val = int(fidx_raw)
            except Exception:
                continue
            if not vcode or n_val < 0:
                continue
            pairs_n.append((vcode, n_val))
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        if not pairs_n:
            resp = make_response(output.getvalue())
            resp.headers["Content-Type"] = "text/csv; charset=utf-8"
            resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
            return resp
        rows_idx = []
        for vcode, n_val in pairs_n:
            mapping = _load_map_index(vcode)
            frame_idx = mapping.get(n_val, n_val)
            rows_idx.append((vcode, frame_idx))
        MAX_LINES = 100
        padded = list(rows_idx)
        if padded:
            while len(padded) < MAX_LINES:
                padded.append(padded[-1])
        for vcode, frame_idx in padded[:MAX_LINES] if padded else []:
            writer.writerow([vcode, frame_idx])
        resp = make_response(output.getvalue())
        resp.headers["Content-Type"] = "text/csv; charset=utf-8"
        resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp
    except Exception as e:
        return make_response(f"Export error: {e}", 400)

@app.get("/lookup_frame")
def lookup_frame():
    """
    GET /lookup_frame?vcode=<Lxx_Vyyy>&time=<seconds.float>
    - Trả về frame gần nhất theo pts_time: { "n": int, "frame_idx": int }
    - Nếu map không có pts_time: ước lượng gần đúng bằng cách lấy theo thứ tự n (best-effort).
    """
    vcode = (request.args.get('vcode') or '').strip()
    try:
        t = float(request.args.get('time') or '0')
    except Exception:
        t = 0.0
    if not vcode:
        return jsonify({"error": "missing vcode"}), 400
    rows = _load_map_rows(vcode)
    if not rows:
        return jsonify({"n": 0, "frame_idx": 0})
    best = None
    best_diff = None
    for r in rows:
        if r['pts_time'] is None:
            continue
        d = abs(r['pts_time'] - t)
        if (best is None) or (d < best_diff):
            best, best_diff = r, d
    if best is None:
        best = rows[min(len(rows) - 1, max(0, int(len(rows) * 0.5)))]
    return jsonify({"n": int(best['n']), "frame_idx": int(best['frame_idx'])})

@app.get("/imgid_from_vcode_frame")
def imgid_from_vcode_frame():
    """
    GET /imgid_from_vcode_frame?vcode=<Lxx_Vyyy>&n=<int>
    - Tìm ảnh keyframe tương ứng trong image_path.json để suy ra imgid nội bộ.
    - Trả về: { "imgid": int|null, "fpath": "data/keyframes/.../xxx.jpg"|null }
    """
    vcode = (request.args.get('vcode') or '').strip()
    try:
        n = int(request.args.get('n') or '0')
    except Exception:
        n = 0
    if not vcode:
        return jsonify({"imgid": None, "fpath": None})
    path2id = _ensure_PATH2ID()
    if not path2id:
        return jsonify({"imgid": None, "fpath": None})
    bucket = _keyframe_bucket(vcode)
    base_dir = f"data/keyframes/{bucket}/{vcode}"
    for fname in _candidate_frame_names(n):
        candidate = _normalize_path(f"{base_dir}/{fname}")
        if candidate in path2id:
            return jsonify({"imgid": path2id[candidate], "fpath": candidate})
    return jsonify({"imgid": None, "fpath": None})

@app.get("/allframes")
def allframes():
    # (1) Lấy tham số video code; thiếu thì quay lại trang chủ
    vcode = (request.args.get("vcode") or "").strip()
    if not vcode:
        return redirect(url_for("home"))

    # (2) Ghép needle để lọc những path thuộc đúng video
    #     VD: .../L21_V001/...
    needle = f"/{vcode}/".replace("\\", "/")

    # (3) Nếu có file map_keyframes/<vcode>.csv, nạp để lấy frame_idx & pts_time
    #     _load_map_rows(vcode) đã linh hoạt tên cột; trả list dict {'n','frame_idx','pts_time'}
    rows = _load_map_rows(vcode)           # có thể là [] nếu không tồn tại
    # Dựng map nhanh: n -> (frame_idx, pts_time) để tra cứu
    n2meta = {r["n"]: (r.get("frame_idx"), r.get("pts_time")) for r in rows}

    # (4) Lọc toàn bộ ảnh thuộc video này từ ID2PATH, chuẩn hoá thành danh sách phẳng
    flat_items = []                        # list các item cho grid
    for imgid, path in ID2PATH.items():
        p = str(path).replace("\\", "/")   # chuẩn hoá dấu gạch chéo
        if needle in p:                    # chỉ lấy ảnh của video đang xem
            n = _frame_no_from_path(p)     # số thứ tự frame lấy từ tên file (vd 020.jpg -> 20)
            # Lấy meta nếu có trong CSV map
            fi, ts = n2meta.get(n, (None, None))
            flat_items.append({
                "id": imgid,               # id nội bộ (dùng cho /imgsearch?imgid=...)
                "path": p,                 # đường dẫn ảnh (dùng với /get_img?fpath=...)
                "score": 0.0,              # giữ cấu trúc đồng nhất với trang home
                "ocr": "",                 # placeholder
                "frame": n,                # n lấy từ tên file (để hiển thị/ sort)
                "frame_idx": fi,           # frame index thực (nếu có file map)
                "pts_time": ts,            # thời gian giây (nếu có file map)
                "vcode": vcode,            # để link sang /watch
            })

    # (5) Sắp xếp tăng dần theo 'frame' để grid đúng thứ tự
    flat_items.sort(key=lambda r: r.get("frame", -1))

    # (6) Render về template "home.html" với cờ 'mode=allframes' + danh sách phẳng
    #     (template sẽ biết hiển thị dạng grid xuống dòng, không group)
    return render_template(
        "home.html",
        mode="allframes",          # cờ báo chế độ allframes
        current_vcode=vcode,       # hiển thị tiêu đề video đang xem
        flat_items=flat_items,     # danh sách ảnh phẳng để render grid
        total=len(flat_items),     # tổng số frame
        elapsed=0,                 # placeholder
        page=0,                    # không phân trang
        has_more=False,            # không phân trang
        ui=None                    # không cần trạng thái form tìm kiếm ở trang này
    )
# ====== Main ======
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
