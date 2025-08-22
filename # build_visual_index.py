import os, re, json
import numpy as np
import faiss

# ========= CONFIG =========
IMAGE_PATH_JSON = "F:\AIC25\code\AICute1-main\AICute1-main\image_path.json"          # map ID -> path (bạn đã có)
VIDEO_EMB_DIR   = "F:\AIC25\data\clip_features"        # chứa các file <VIDEO>.npy
OUT_FAISS       = "F:\AIC25\code\AICute1-main\AICute1-main/faiss_normal_ViT.bin"
OUT_JSON        = "F:\AIC25\code\AICute1-main\AICute1-main\image_path.json"          # có thể ghi đè lại cho “chắc kèo”
USE_IP          = True                       # True: IndexFlatIP (cosine), False: L2

# regex để bắt video & frame từ path, ví dụ: Keyframes/L01_V001/003.jpg
# sửa lại nếu cấu trúc path khác

# THAY thế dòng regex cũ bằng bộ đôi "thử đầy đủ rồi fallback":

VIDEO_FRAME_RE_STRICT = re.compile(
    "data[\\/]keyframes[\\/]Keyframes_(L\d+)[\\/](L\d+_V\d+)[\\/](\d+)\.(jpg|png|webp)$",
    re.IGNORECASE
)
VIDEO_FRAME_RE_FALLBACK = re.compile(
    "[\\/](L\d+_V\d+)[\\/](\d+)\.(jpg|png|webp)$",
    re.IGNORECASE
)

def parse_video_frame(path):
    p = path.replace("\\", "/")
    m = VIDEO_FRAME_RE_STRICT.search(p)
    if m:
        # m.group(1) = L21 (bucket), m.group(2) = L21_V001 (video), m.group(3) = frame
        video = m.group(2)
        frame = int(m.group(3))
        return video, frame
    # fallback (nếu map không có phần "Keyframes_Lxx")
    m = VIDEO_FRAME_RE_FALLBACK.search(p)
    if m:
        video = m.group(1)
        frame = int(m.group(2))
        return video, frame
    raise ValueError(f"Không parse được video/frame từ path: {path}")

def main():
    # 1) Load map ID -> path
    with open(IMAGE_PATH_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # keys có thể là string => chuyển sang int, rồi sort để đảm bảo ID tăng dần
    id_path_items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])

    # Cache embedding đã load theo video
    video_cache = {}

    X_list = []
    new_id2path = {}
    next_id = 0

    for orig_id, img_path in id_path_items:
        video, frame_idx = parse_video_frame(img_path)     # frame_idx: 1-based
        # nạp embedding video nếu chưa có
        if video not in video_cache:
            emb_path = os.path.join(VIDEO_EMB_DIR, f"{video}.npy")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Thiếu embedding video: {emb_path}")
            E = np.load(emb_path).astype(np.float32)       # (n_frames, d)
            video_cache[video] = E

        E = video_cache[video]
        local_idx = frame_idx - 1                           # chuyển về 0-based
        if not (0 <= local_idx < E.shape[0]):
            raise IndexError(f"Frame {frame_idx} vượt quá số frame của {video} ({E.shape[0]})")

        X_list.append(E[local_idx:local_idx+1, :])          # (1, d)
        new_id2path[next_id] = img_path                     # đảm bảo ID liên tiếp 0..N-1
        next_id += 1

    # 2) Ghép thành ma trận (N, D)
    X = np.vstack(X_list).astype(np.float32)
    print("Emb shape:", X.shape)

    # 3) Chuẩn hoá nếu dùng cosine (IP)
    if USE_IP:
        faiss.normalize_L2(X)

    # 4) Build FAISS
    D = X.shape[1]
    index = faiss.IndexFlatIP(D) if USE_IP else faiss.IndexFlatL2(D)
    index.add(X)
    faiss.write_index(index, OUT_FAISS)
    print("Wrote:", OUT_FAISS, "| ntotal =", index.ntotal)

    # 5) Ghi lại mapping (ID liên tiếp 0..N-1, khớp thứ tự add vào FAISS)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in new_id2path.items()}, f, indent=2, ensure_ascii=False)
    print("Wrote:", OUT_JSON)

    # 6) Sanity check
    scores, idx = index.search(X[0:1], k=5)
    print("Top5 of #0:", idx[0], scores[0])

if __name__ == "__main__":
    main()
