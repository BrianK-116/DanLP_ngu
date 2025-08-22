# create_ocr_json.py  (one-string-per-image: largest label by bbox area)
# v3.0 — GPU-first, speed-ups, shard support, vi/en, webp support

import json
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import easyocr
from tqdm import tqdm
from PIL import Image  # for WEBP fallback

# Các tối ưu tốc độ
import cv2
import numpy as np

# ==== 1) CONFIG (có thể override bằng CLI) ====
# Khuyên dùng raw-string hoặc Path để tránh lỗi backslash trên Windows
DEFAULT_IMAGE_PATH_JSON = "image_path_L22.json"
DEFAULT_OUTPUT_JSON_FILE = "id_to_ocr_text_L22.json"
DEFAULT_ERROR_LOG_FILE   = "ocr_errors11.log"

# Ngôn ngữ: thêm 'vi' để OCR tiếng Việt tốt hơn
LANGS = ['en']

# Chỉ lấy 1 dòng lớn nhất -> cần bbox từng dòng
PARAGRAPH = False     # không gộp đoạn
DETAIL    = 1         # 1 để có bbox & conf

# Checkpoint mỗi N ảnh để tránh mất dữ liệu nếu dở giữa chừng
DEFAULT_BATCH_SAVE_EVERY = 500

# Thư mục tạm cho ảnh chuyển đổi (webp->png)
TMP_DIR = Path(".ocr_tmp")
TMP_DIR.mkdir(exist_ok=True)

# Allowlist cho VN-EN (giúp recognizer chạy nhanh & ổn định hơn)
ALLOWED = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
    " -._/\""
)

# ==== 2) UTILS ====
def log_error(msg: str, error_log_file: str):
    with open(error_log_file, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def polygon_area(points):
    """
    Tính diện tích đa giác (shoelace). points = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    try:
        if not points or len(points) < 3:
            return 0.0
        s = 0.0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) * 0.5
    except Exception:
        return 0.0

def ensure_readable_path(img_path: str) -> str:
    """
    EasyOCR thường đọc được path trực tiếp. Một số bản OpenCV cũ lỗi với WEBP.
    Với .webp, chuyển tạm sang PNG để đảm bảo tương thích.
    """
    img_path = str(img_path)
    ext = Path(img_path).suffix.lower()
    if ext == ".webp":
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                tmp_png = TMP_DIR / (Path(img_path).stem + "_tmp.png")
                im.save(tmp_png, format="PNG")
                return str(tmp_png)
        except Exception:
            # Nếu chuyển thất bại, trả về path gốc để EasyOCR tự thử
            return img_path
    return img_path

def save_checkpoint(out_path: str, data: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_image_map(path_json: str):
    if not os.path.exists(path_json):
        print(f"[ERROR] Not found: {path_json}")
        sys.exit(1)

    print(f"Loading image paths from '{path_json}' ...")
    with open(path_json, "r", encoding="utf-8") as f:
        id_to_path = json.load(f)

    try:
        sorted_items = sorted(id_to_path.items(), key=lambda kv: int(kv[0]))
    except ValueError:
        print("[ERROR] Keys in JSON must be numeric-like (convertible to int).")
        sys.exit(1)
    print(f"Found {len(sorted_items)} images to process.")
    return sorted_items

# ==== 3) SPEED-UP HELPERS ====
def maybe_has_text(img_path, edge_th=0.015):
    """
    Pre-check rẻ tiền: downscale & đo mật độ cạnh. Ảnh quá “trơn” nhiều khả năng không có chữ.
    Trả về True nếu nên chạy OCR; False nếu có thể bỏ qua.
    """
    try:
        # Dùng imdecode + fromfile để tránh lỗi Unicode path trên Windows
        arr = np.fromfile(img_path, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return True
        h, w = im.shape[:2]
        scale = 256 / max(h, w)
        if scale < 1.0:
            im = cv2.resize(im, (int(w * scale), int(h * scale)))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        e = cv2.Canny(im, 50, 150)
        ratio = (e > 0).mean()
        return ratio > edge_th
    except Exception:
        return True

def iter_shard(items, k, i):
    """
    Chia danh sách theo shard: k tổng số shard, i là chỉ số shard (0..k-1)
    """
    if k <= 1:
        for it in items:
            yield it
        return
    for idx, it in enumerate(items):
        if idx % k == i:
            yield it

# ==== 4) INIT OCR MODEL (GPU-first, fallback CPU) ====
def init_reader():
    # Bật autotune cho conv → nhanh hơn khi ảnh có kích thước lặp lại
    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    try:
        print(f"Loading EasyOCR with LANGS={LANGS}, gpu={use_gpu} ...")
        reader_ = easyocr.Reader(LANGS, gpu=use_gpu, verbose=False)
        print("EasyOCR model loaded successfully.")
        if use_gpu:
            try:
                print("CUDA device:", torch.cuda.get_device_name(0))
            except Exception:
                pass
        return reader_
    except Exception as e:
        print(f"[WARN] GPU init failed or other issue: {e}")
        print("Retrying with gpu=False ...")
        reader_ = easyocr.Reader(LANGS, gpu=False, verbose=False)
        print("EasyOCR model loaded on CPU.")
        return reader_

# ==== 5) MAIN ====
def main():
    parser = argparse.ArgumentParser(description="Fast EasyOCR (VN/EN) with shard support.")
    parser.add_argument("--image-map", default=DEFAULT_IMAGE_PATH_JSON, help="Đường dẫn file JSON id->image_path")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_JSON_FILE, help="File JSON output")
    parser.add_argument("--err", default=DEFAULT_ERROR_LOG_FILE, help="File log lỗi")
    parser.add_argument("--batch-save-every", type=int, default=DEFAULT_BATCH_SAVE_EVERY, help="Checkpoint mỗi N ảnh")
    # Shard
    parser.add_argument("--k", type=int, default=1, help="Tổng số shard (mặc định 1 = không chia)")
    parser.add_argument("--i", type=int, default=0, help="Chỉ số shard (0..k-1)")
    # Tham số speed-up
    parser.add_argument("--edge-th", type=float, default=0.015, help="Ngưỡng mật độ cạnh pre-check")
    parser.add_argument("--canvas-size", type=int, default=1920, help="canvas_size cho detector (nhỏ hơn → nhanh hơn)")
    parser.add_argument("--text-th", type=float, default=0.7, help="text_threshold")
    parser.add_argument("--low-text", type=float, default=0.4, help="low_text")
    parser.add_argument("--link-th", type=float, default=0.5, help="link_threshold")
    args = parser.parse_args()

    reader = init_reader()
    all_items = load_image_map(args.image_map)

    # Áp dụng shard (nếu có)
    items = list(iter_shard(all_items, args.k, args.i))
    print(f"Shard info: k={args.k}, i={args.i} → this shard has {len(items)} images.")

    # Resume
    ocr_results = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
            print(f"[RESUME] Loaded {len(ocr_results)} entries from '{args.out}'.")
        except Exception as e:
            print(f"[WARN] Could not read existing output ({e}). Starting fresh.")

    print("Starting OCR processing ...")
    processed_since_save = 0
    start_time = time.time()

    pbar = tqdm(items, ncols=100)
    for image_id_str, image_path in pbar:
        # Skip nếu đã có kết quả
        if image_id_str in ocr_results:
            continue

        # Nếu mất file → gán rỗng
        if not os.path.exists(image_path):
            ocr_results[image_id_str] = ""
            log_error(f"[MISS] ID={image_id_str} path not found: {image_path}", args.err)
        else:
            try:
                safe_path = ensure_readable_path(image_path)

                # Pre-check: ảnh quá trơn có thể không có chữ → bỏ qua
                if not maybe_has_text(safe_path, edge_th=args.edge_th):
                    ocr_results[image_id_str] = ""
                else:
                    # EasyOCR detail=1: kết quả mỗi phần tử: [bbox, text, conf]
                    result = reader.readtext(
                        safe_path,
                        detail=DETAIL,
                        paragraph=PARAGRAPH,
                        # speed-up params
                        allowlist=ALLOWED,
                        canvas_size=args.canvas_size,
                        text_threshold=args.text_th,
                        low_text=args.low_text,
                        link_threshold=args.link_th,
                        mag_ratio=1.0,
                        width_ths=0.5,
                        height_ths=0.5,
                        ycenter_ths=0.5,
                        slope_ths=0.1,
                    )

                    best_text = ""
                    best_area = -1.0
                    best_conf = -1.0

                    if isinstance(result, list):
                        for item in result:
                            # Mỗi item: [bbox, text, conf]
                            if not (isinstance(item, (list, tuple)) and len(item) >= 3):
                                continue
                            bbox, text, conf = item[0], item[1], item[2]
                            if not isinstance(text, str) or not text.strip():
                                continue

                            area = polygon_area(bbox) if isinstance(bbox, (list, tuple)) else 0.0

                            # tiêu chí: diện tích lớn nhất -> nếu bằng nhau, lấy conf lớn hơn -> nếu vẫn bằng, lấy text dài hơn
                            better = False
                            if area > best_area:
                                better = True
                            elif area == best_area and conf > best_conf:
                                better = True
                            elif area == best_area and conf == best_conf and len(text) > len(best_text):
                                better = True

                            if better:
                                best_area = area
                                best_conf = conf
                                best_text = text.strip()

                    # chỉ 1 chuỗi lớn nhất (có thể rỗng nếu không có text hợp lệ)
                    ocr_results[image_id_str] = best_text

            except Exception as e:
                ocr_results[image_id_str] = ""
                log_error(f"[ERR] ID={image_id_str} path={image_path} err={repr(e)}", args.err)

        processed_since_save += 1
        if processed_since_save >= args.batch_save_every:
            save_checkpoint(args.out, ocr_results)
            processed_since_save = 0

    # Save cuối
    save_checkpoint(args.out, ocr_results)

    elapsed = time.time() - start_time
    print(f"--- All Done! --- {len(ocr_results)} items written to '{args.out}'.")
    print(f"Errors (if any) logged to '{args.err}'. Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()