# -*- coding: utf-8 -*-
# app.py — Flask backend tối giản, có comment từng dòng

from flask import Flask, render_template, request, send_file, abort  # import các hàm Flask
from pathlib import Path                                            # chuẩn hoá đường dẫn
import json                                                         # đọc file JSON
import os, traceback                                                # tiện ích hệ thống + log lỗi

# ====== Khởi tạo app & đường dẫn gốc ======
app = Flask(__name__, template_folder='templates')                  # tạo Flask app, chỉ ra thư mục template
BASE = Path(__file__).resolve().parent                              # thư mục chứa file app.py
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
# --- ép object về list Python an toàn (hỗ trợ numpy) ---
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

# --- ép 1 phần tử về float an toàn (hỗ trợ numpy/list lồng) ---
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

# --- Chuẩn hoá kết quả trả về từ Myfaiss về (scores, ids, paths) an toàn ---
def normalize_ret(ret):
    """ret có thể là None / tuple / dict. Trả về (scores:list, ids:list, paths:list)."""
    scores, ids, paths = [], [], []                                 # mặc định rỗng
    if ret is None:                                                 # nếu None -> giữ rỗng
        return scores, ids, paths
    # Dạng dict: {'scores': [...], 'ids': [...], 'paths': [...]}
    if isinstance(ret, dict):
        scores = list(ret.get("scores") or [])
        ids    = list(ret.get("ids")    or [])
        paths  = list(ret.get("paths")  or [])
        return scores, ids, paths
    # Dạng tuple/list
    if isinstance(ret, (list, tuple)):
        if len(ret) >= 4:                                           # (scores, ids, _, paths)
            a, b, _, d = ret
            scores = list(a) if a is not None else []
            ids    = list(b) if b is not None else []
            paths  = list(d) if d is not None else []
        elif len(ret) == 3:                                         # (a,b,c) – cố đoán trường nào là paths
            a, b, c = ret
            # bên nào là list str -> paths
            def is_paths(x): return isinstance(x, (list, tuple)) and (len(x)==0 or isinstance(x[0], str))
            if is_paths(a):
                paths, ids, scores = list(a), list(b or []), list(c or [])
            elif is_paths(b):
                ids, paths, scores = list(a or []), list(b), list(c or [])
            elif is_paths(c):
                ids, scores, paths = list(a or []), list(b or []), list(c)
            else:
                ids, scores, paths = list(a or []), list(b or []), list(c or [])
        elif len(ret) == 2:                                         # (ids, paths) hoặc (paths, ids)
            a, b = ret
            if isinstance(a, (list, tuple)) and a and isinstance(a[0], str):
                paths, ids = list(a), list(b or [])
            elif isinstance(b, (list, tuple)) and b and isinstance(b[0], str):
                ids, paths = list(a or []), list(b)
            else:
                ids, paths = list(a or []), list(b or [])
                scores = [0.0] * min(len(ids), len(paths))
    return scores, ids, paths
def get_ui(args):                                                   # gom tham số UI từ query string
    return {
        "textquery": args.get("textquery", ""),                     # nội dung ô search
        "search_type": args.get("search_type", "hybrid"),           # visual|ocr|hybrid
        "topk": int(args.get("topk", 100)),                         # số kết quả / trang
        "wvis": float(args.get("wvis", 0.6)),                       # trọng số ảnh
        "wtxt": float(args.get("wtxt", 0.4)),                       # trọng số text
        "constraints": args.get("constraints", ""),                 # ràng buộc (nếu dùng)
    }

def page_bounds(index, total, size):                                # tính phạm vi phần tử cho 1 trang
    first = max(index, 0) * max(size, 1)                            # vị trí bắt đầu (an toàn)
    last  = min(first + max(size, 1), total)                        # vị trí kết thúc (exclusive)
    return first, last                                              # trả về (first, last)

# ====== Route: Trang chủ (lưới ảnh tĩnh) ======
@app.route("/", methods=["GET"], endpoint="home")                   # endpoint tên 'home' cho tiện url_for
def home():
    idx = int(request.args.get("index", 0))                         # trang hiện tại (0-based)
    ui = get_ui(request.args)                                       # lấy tham số UI
    size = ui["topk"]                                               # số ảnh mỗi trang = topk
    f, l = page_bounds(idx, N_ITEMS, size)                          # tính phạm vi ảnh
    results = []                                                    # danh sách kết quả để render
    for i in range(f, l):                                           # duyệt id trong trang
        rel = ID2PATH.get(i, "")                                    # path tương đối, ví dụ "data/keyframes/.../001.jpg"
        results.append({"id": i, "path": rel, "score": 0.0, "ocr": ""}) # object hiển thị (path quan trọng)
    has_more = l < N_ITEMS                                          # còn trang sau không?
    return render_template("home.html",                             # render template
                           results=results,                         # danh sách thẻ ảnh
                           total=N_ITEMS,                           # tổng ảnh
                           elapsed=0,                               # thời gian (placeholder)
                           page=idx,                                # số trang hiện tại
                           has_more=has_more,                       # có Next không
                           ui=ui)                                   # trạng thái UI để đổ vào form

# ====== Route: Tìm kiếm ảnh-ảnh ======
@app.route("/imgsearch", methods=["GET"])
def imgsearch():
    ui = get_ui(request.args)                                     # tham số UI để giữ trạng thái
    idx = int(request.args.get("index", 0))                       # trang hiện tại
    if not FAISS_OK:
        return render_template("home.html", results=[], total=0,
                               elapsed=0, page=idx, has_more=False, ui=ui)

    qid = int(request.args.get("imgid", 0))                       # id ảnh query

    try:
        ret = MF.image_search(qid, k=ui["topk"])                  # gọi search
    except Exception:
        print("[/imgsearch] failed"); print(traceback.format_exc())
        ret = None

    # Chuẩn hoá output về 3 list: scores, ids, paths
    scores, ids, paths = [], [], []
    if isinstance(ret, (list, tuple)) and len(ret) >= 4:          # (scores, ids, _, paths)
        scores, ids, paths = ret[0], ret[1], ret[3]
    elif isinstance(ret, dict):
        scores, ids, paths = ret.get("scores"), ret.get("ids"), ret.get("paths")

    scores = to_list(scores)
    ids    = to_list(ids)
    paths  = to_list(paths)

    total = min(len(paths), len(ids))
    size  = max(int(ui["topk"]), 1)
    first = max(idx, 0) * size
    last  = min(first + size, total)

    out = []
    for i in range(first, last):
        s = to_float_scalar(scores[i], 0.0) if i < len(scores) else 0.0
        _id = ids[i]
        # ép id về int an toàn (kể cả numpy)
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

    has_more = last < total
    return render_template("home.html",
                           results=out, total=total, elapsed=0,
                           page=idx, has_more=has_more, ui=ui)
# ====== Route: Tìm kiếm text (visual/ocr/hybrid) ======
@app.route("/textsearch", methods=["GET"])
def textsearch():
    ui  = get_ui(request.args)                                     # lấy tham số UI (để giữ trạng thái)
    idx = int(request.args.get("index", 0))                        # trang hiện tại

    # Nếu FAISS chưa sẵn hoặc query rỗng -> render trống (HTTP 200)
    if (not FAISS_OK) or (ui["textquery"].strip() == ""):
        return render_template("home.html",
                               results=[], total=0, elapsed=0,
                               page=idx, has_more=False, ui=ui)

    # Gọi Myfaiss với "best-effort" — không quan tâm hàm có/không nhận search_type
    ret = call_text_search_best_effort(
        MF,
        text=ui["textquery"],
        k=ui["topk"],
        search_type=ui["search_type"]
    )

    # Quan trọng: nếu ret=None, ta vẫn không được unpack; chuẩn hoá an toàn trước
    scores, ids, paths = normalize_ret(ret)

    # Tính tổng & phân trang (không cho raise khi rỗng)
    total = min(len(paths), len(ids))
    size  = max(int(ui["topk"]), 1)
    first = max(idx, 0) * size
    last  = min(first + size, total)

    # Lắp dữ liệu ra template
    out = []
    for i in range(first, last):
        s  = float(scores[i]) if i < len(scores) else 0.0
        _id = ids[i]
        _id = int(_id) if isinstance(_id, (int, float, str)) and str(_id).isdigit() else _id
        out.append({"id": _id, "path": paths[i], "score": s, "ocr": ""})

    has_more = last < total
    return render_template("home.html",
                           results=out, total=total, elapsed=0,
                           page=idx, has_more=has_more, ui=ui)
# ====== Route: Trả file ảnh theo path tương đối ======
@app.route("/get_img", methods=["GET"])                              # endpoint tải ảnh
def get_img():
    rel = (request.args.get("fpath") or "").strip()                 # lấy tham số fpath
    if not rel:                                                     # nếu thiếu fpath
        abort(400, "Missing fpath")                                 # trả 400
    p = Path(rel)                                                   # tạo Path từ chuỗi
    if not p.is_absolute():                                         # nếu là path tương đối
        p = (BASE / rel).resolve()                                  # ghép với BASE để ra path tuyệt đối
    try:                                                            # giới hạn trong BASE (bảo mật)
        _ = p.relative_to(BASE)                                     # nếu vượt BASE sẽ ném lỗi
    except Exception:
        abort(403)                                                  # trả 403 nếu vượt
    if not p.exists():                                              # nếu file không tồn tại
        abort(404)                                                  # trả 404
    ext = p.suffix.lower()                                          # lấy đuôi file
    if ext in (".jpg", ".jpeg"): mime = "image/jpeg"                # đoán MIME
    elif ext == ".png":         mime = "image/png"                  # PNG
    elif ext == ".webp":        mime = "image/webp"                 # WEBP
    else:                       mime = "application/octet-stream"   # mặc định
    return send_file(p, mimetype=mime, conditional=True)            # trả file ảnh

# ====== Main ======
if __name__ == "__main__":                                          # chạy trực tiếp file
    app.run(debug=False, host="0.0.0.0", port=5001)                 # bật server 0.0.0.0:5001
