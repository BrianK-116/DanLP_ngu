# -*- coding: utf-8 -*-
"""
Clean VI OCR -> Fix common OCR errors -> Translate to EN -> CLIP encode -> Build FAISS index
- Giữ nguyên thứ tự ID
- Lưu thêm file đối chiếu EN để audit
- Batch + retry dịch để an toàn
"""

import os
import re
import json
import time
import unicodedata
from typing import List, Dict

import torch
import numpy as np
import faiss
import clip

# === OPTIONAL: dùng googletrans (cần Internet). Bạn có thể thay bằng deep_translator nếu muốn. ===
from googletrans import Translator
# Nếu gặp rate-limit, thử: Translator(service_urls=['translate.googleapis.com'])

# --------------------
# CONFIG
# --------------------
OCR_TEXT_FILE        = r"F:/AIC25/code/AICute1-main/AICute1-main/id_to_ocr_text.json"   # input VI
CLEANED_VI_JSON      = r"F:/AIC25/code/AICute1-main/AICute1-main/id_to_ocr_text_clean_vi.json"
TRANSLATED_EN_JSON   = r"F:/AIC25/code/AICute1-main/AICute1-main/id_to_ocr_text_en.json"
OUTPUT_INDEX_FILE    = r"F:/AIC25/code/AICute1-main/AICute1-main/faiss_ocr_ViT.bin"
CLIP_MODEL           = "ViT-B/32"
BATCH_SIZE_ENCODE    = 256
BATCH_SIZE_TRANSLATE = 40        # đừng để quá lớn kẻo dễ lỗi rate-limit
MAX_RETRY_TRANSLATE  = 5
SLEEP_BETWEEN_RETRY  = 2.5       # giây

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {device}")

# --------------------
# 1) Tiền xử lý & sửa lỗi OCR
# --------------------

# Các pattern/chuẩn hoá phổ biến (bạn có thể mở rộng bảng này theo dữ liệu của mình)
REPLACEMENTS = {
    # ký tự OCR sai hoặc bị lẫn:
    " HCMf": " HCM",
    "TP HCMf": "TP HCM",
    "dể": "để",
    "dến": "đến",
    "dơn": "đơn",
    "duối": "đuối",
    "Miên": "Miền",
    "Triêu": "Triệu",
    "ĐÔNG THÁPf": "ĐÔNG THÁP",
    "dại học": "đại học",
    "dặc": "đặc",
    "dược": "được",
    "dó": "đó",
    "dã": "đã",
    "dầu": "đầu",
    "dầu tư": "đầu tư",
    "Viêt": "Việt",
    "Tp. HCM": "TP. HCM",
    "Tp HCM": "TP HCM",
    "HCMf": "HCM",
    "Ha Noi": "Hà Nội",
    "TP HCM": "TP. HCM",
    # thêm tuỳ dữ liệu
}

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
YOUTUBE_PATTERN = re.compile(r"(?i)(kênh youtube|youtube|yt[\s:])\S*")
SOCIAL_PATTERN = re.compile(r"(?i)(facebook|zalo|tiktok|instagram|ig|@[\w._-]+)")
TIME_STAMP = re.compile(r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\b")  # 06:40:24, 06.40.24
NUM_NOISE = re.compile(r"\b\d{4,}\b")                        # số dài 4+ digits
MULTI_SPACE = re.compile(r"\s+")
DASH_LINE = re.compile(r"[-–—]{2,}")
DOTS = re.compile(r"\.{3,}")

def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def basic_clean_vi(s: str) -> str:
    s = normalize_unicode(s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    # gộp dòng -> câu
    s = s.replace("\r", " ").replace("\n", " ")
    # bỏ URL, social, youtube, timestamp, số nhiễu
    s = URL_PATTERN.sub(" ", s)
    s = YOUTUBE_PATTERN.sub(" ", s)
    s = SOCIAL_PATTERN.sub(" ", s)
    s = TIME_STAMP.sub(" ", s)
    s = NUM_NOISE.sub(" ", s)
    # thay lỗi OCR phổ biến
    for bad, good in REPLACEMENTS.items():
        s = s.replace(bad, good)
    # dọn dấu và khoảng trắng
    s = DASH_LINE.sub(" - ", s)
    s = DOTS.sub("…", s)
    s = MULTI_SPACE.sub(" ", s).strip(" -•—\t ")
    return s

def post_shorten(s: str, min_chars=8):
    # xoá câu rất ngắn vô nghĩa
    if len(s) < min_chars:
        return ""
    return s

# --------------------
# 2) Dịch VI -> EN (batch + retry)
# --------------------
def translate_batch_vi2en(texts: List[str], translator: Translator) -> List[str]:
    out = []
    # googletrans best-effort: dịch từng câu để tránh dính rate-limit packet lớn
    # (có thể gộp 5-10 câu một lần nếu ổn định hơn với mạng của bạn)
    for t in texts:
        tt = t
        ok = False
        for attempt in range(MAX_RETRY_TRANSLATE):
            try:
                if not tt:
                    out.append("")
                    ok = True
                    break
                res = translator.translate(tt, dest="en")
                out.append(res.text)
                ok = True
                break
            except Exception as e:
                wait = SLEEP_BETWEEN_RETRY * (attempt + 1)
                print(f"[translate] retry {attempt+1}/{MAX_RETRY_TRANSLATE} after error: {e}. Sleep {wait:.1f}s")
                time.sleep(wait)
        if not ok:
            out.append(tt)  # fallback: giữ nguyên VI nếu dịch thất bại
    return out

# --------------------
# 3) Encode CLIP + FAISS
# --------------------
def clip_encode_texts(texts: List[str], device: str, model_name: str = "ViT-B/32", batch: int = 256) -> np.ndarray:
    model, _ = clip.load(model_name, device=device)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i+batch]
            tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            feats = model.encode_text(tokens)
            embeddings.append(feats.cpu())
    arr = torch.cat(embeddings).numpy().astype("float32")
    return arr

# --------------------
# MAIN
# --------------------
def main():
    # 0) Load JSON VI
    with open(OCR_TEXT_FILE, "r", encoding="utf-8") as f:
        id2vi: Dict[str, str] = json.load(f)
    # Sắp xếp theo ID tăng dần (giữ mapping ổn định)
    items = sorted(id2vi.items(), key=lambda kv: int(kv[0]))
    ids = [int(k) for k, _ in items]
    vi_raw = [v if isinstance(v, str) else str(v) for _, v in items]

    # 1) Clean + sửa lỗi
    vi_clean = [post_shorten(basic_clean_vi(t)) for t in vi_raw]
    # nếu sau clean bị rỗng -> fallback dùng bản gốc (hoặc để rỗng tuỳ bạn)
    vi_final = [c if c else basic_clean_vi(t) for c, t in zip(vi_clean, vi_raw)]

    # Lưu VI đã clean để kiểm tra
    with open(CLEANED_VI_JSON, "w", encoding="utf-8") as f:
        json.dump({str(i): t for i, t in zip(ids, vi_final)}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Cleaned VI JSON -> {CLEANED_VI_JSON} (n={len(vi_final)})")

    # 2) Dịch sang EN
    translator = Translator()  # hoặc Translator(service_urls=['translate.googleapis.com'])
    en_texts = []
    # dịch theo batch nhỏ để kiểm soát tần suất
    for i in range(0, len(vi_final), BATCH_SIZE_TRANSLATE):
        chunk = vi_final[i:i+BATCH_SIZE_TRANSLATE]
        en_chunk = translate_batch_vi2en(chunk, translator)
        en_texts.extend(en_chunk)
        # chống bị chặn
        time.sleep(0.2)

    # Lưu EN để audit
    with open(TRANSLATED_EN_JSON, "w", encoding="utf-8") as f:
        json.dump({str(i): t for i, t in zip(ids, en_texts)}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Translated EN JSON -> {TRANSLATED_EN_JSON} (n={len(en_texts)})")

    # 3) Encode CLIP EN
    print(f"[CLIP] Loading model {CLIP_MODEL} on {device}")
    embeddings = clip_encode_texts(en_texts, device=device, model_name=CLIP_MODEL, batch=BATCH_SIZE_ENCODE)
    print(f"[CLIP] Embeddings shape: {embeddings.shape}")  # (N, D)

    # 4) Build FAISS (cosine dùng L2-normalize + IndexFlatL2 hoặc dùng IP + normalize)
    faiss.normalize_L2(embeddings)               # rất quan trọng!
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)                 # bạn có thể dùng IndexFlatIP + normalize cho cosine
    index.add(embeddings)
    print(f"[FAISS] ntotal = {index.ntotal}")

    # 5) Save index
    os.makedirs(os.path.dirname(OUTPUT_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, OUTPUT_INDEX_FILE)
    print(f"[DONE] Wrote FAISS index -> {OUTPUT_INDEX_FILE}")

if __name__ == "__main__":
    main()
