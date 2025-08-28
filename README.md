Video Search App

Ứng dụng tìm kiếm video theo hình ảnh và văn bản, được xây dựng bằng Flask + FAISS + CLIP.

📂 Cấu trúc thư mục
.
├── app.py                  # Flask backend chính
├── utils/                  # Tiện ích: FAISS, xử lý query
│   ├── faiss.py
│   └── query_processing.py
├── templates/              # Giao diện HTML (Jinja2)
│   ├── home.html
│   └── player.html
├── static/                 # CSS, JS, hình ảnh giao diện
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── visual.js
│   └── images/
├── data/                   # Dữ liệu video + keyframes
│   ├── videos/             # Video gốc
│   ├── keyframes/          # Keyframe trích xuất từ video
│   └── map_keyframes/      # CSV map frame_idx ↔ pts_time
├── faiss_normal_ViT.bin    # FAISS index cho đặc trưng CLIP
├── faiss_ocr_ViT.bin       # FAISS index cho OCR + CLIP
├── image_path.json         # Map ID → đường dẫn keyframe
└── dict/
    └── vietnamese-stopwords-dash.txt

🚀 Cách chạy
1. Clone repository
git clone https://github.com/BrianK-116/DanLP_ngu.git
cd Video_search_app

2. Tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate      # Windows

3. Cài đặt thư viện
pip install -r requirements.txt

4. Chuẩn bị dữ liệu

Giải nén thư mục data/ gồm:

videos/, keyframes/, map_keyframes/

Đặt các file FAISS index (faiss_normal_ViT.bin, faiss_ocr_ViT.bin) và image_path.json ở thư mục gốc.

5. Chạy ứng dụng
python app.py


Mặc định chạy tại: http://localhost:5001

✨ Tính năng chính

🔍 Tìm kiếm video bằng văn bản (Text → Keyframes)

📷 Tìm kiếm bằng hình ảnh (Image → Similar frames)

▶️ Xem video tại đúng frame tương ứng

📑 Xuất danh sách frame đã chọn ra CSV (KIS format)

📌 Ghi chú

Hệ thống hiện hỗ trợ tiếng Việt & tiếng Anh (tự động dịch query).

Yêu cầu GPU để tăng tốc FAISS + CLIP (khuyến nghị NVIDIA CUDA).

Dữ liệu mẫu có dung lượng lớn (~100GB), nên đặt data/ trên ổ SSD.