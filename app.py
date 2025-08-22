# === Import các thư viện & module cần thiết ===
from flask import Flask, render_template, Response, request, send_file, jsonify  # Flask web framework + tiện ích
import cv2                                  # OpenCV để đọc/hiển thị ảnh
import os                                   # Làm việc với hệ thống file, biến môi trường
os.environ['CUDA_VISIBLE_DEVICES'] = '1'    # Chọn GPU id=1 (nếu có). NOTE: Ở dưới Myfaiss đang chạy 'cpu'.
import numpy as np                           # Xử lý mảng số
import pandas as pd                          # (Hiện chưa dùng) – để xử lý bảng dữ liệu nếu cần
import easyocr                               # (Hiện chưa dùng trong file này) – OCR nếu tích hợp thêm
import glob                                  # (Hiện chưa dùng) – dò file theo pattern
import json                                  # Đọc/ghi JSON

# Import các module tự viết
from utils.query_processing import Translation  # Bộ xử lý/biên dịch query (VN/EN…)
from utils.faiss import Myfaiss                 # Lớp bao FAISS cho tìm kiếm ảnh/text

# Tham chiếu nhanh để test route: gọi kèm query index
# http://0.0.0.0:5001/home?index=0

# Khởi tạo ứng dụng Flask
# Nếu muốn tách static riêng: app = Flask(__name__, template_folder='templates', static_folder='static')
app = Flask(__name__, template_folder='templates')

####### CONFIG #########
# Đọc file ánh xạ id -> đường dẫn ảnh (được tạo sẵn khi build index)
with open('F:\\AIC25\\code\\AICute1-main\\AICute1-main\\image_path.json') as json_file:
    json_dict = json.load(json_file)

# Ép key từ string -> int để tiện truy cập theo chỉ số
DictImagePath = {}
for key, value in json_dict.items():
   DictImagePath[int(key)] = value

# Tổng số ảnh
LenDictPath = len(DictImagePath)

# Đường dẫn/ tên file index FAISS (visual & ocr)
bin_file_visual = 'faiss_normal_ViT.bin'  # index ảnh (CLIP/ViT)
bin_file_ocr    = 'faiss_ocr_ViT.bin'     # index OCR (vector hóa text)

print(bin_file_visual)  # Log tên index đang dùng

# Khởi tạo đối tượng FAISS helper
# Tham số:
#   visual_bin_file, ocr_bin_file, dict_id2path, device, translator, clip_model_name
# NOTE: device hiện là 'cpu' – nếu muốn dùng GPU thì cần index/trả về đúng kiểu GPU và sửa device.
MyFaiss = Myfaiss(bin_file_visual, bin_file_ocr, DictImagePath, 'cpu', Translation(), "ViT-B/32")
########################

# === ROUTES ===

# Trang chủ & alias /home
@app.route('/home')
@app.route('/')
def thumbnailimg():
    print("load_iddoc")  # Log khi vào trang

    pagefile = []  # Danh sách item (path + id) để render

    # Lấy tham số "index" từ URL (số trang – page index), mặc định 0
    index = request.args.get('index')
    if index == None:
        index = 0
    else:
        index = int(index)

    imgperindex = 100  # Số ảnh hiển thị mỗi trang

    # Danh sách đường dẫn ảnh & danh sách id tương ứng cho trang hiện tại
    page_filelist = []
    list_idx = []

    # Tính khoảng [first_index, last_index) của các id cho trang này
    # NOTE: Điều kiện dưới đây hơi lạ. Thường nên so với first_index + imgperindex < LenDictPath.
    if LenDictPath - 1 > index + imgperindex:
        first_index = index * imgperindex
        last_index  = index * imgperindex + imgperindex

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])  # Thêm đường dẫn ảnh
            list_idx.append(tmp_index)                      # Lưu id
            tmp_index += 1
    else:
        first_index = index * imgperindex
        last_index  = LenDictPath  # Trang cuối: cắt về tổng số ảnh

        tmp_index = first_index
        while tmp_index < last_index:
            page_filelist.append(DictImagePath[tmp_index])
            list_idx.append(tmp_index)
            tmp_index += 1

    # Gom vào cấu trúc để render: [{imgpath, id}, ...]
    for imgpath, id in zip(page_filelist, list_idx):
        pagefile.append({'imgpath': imgpath, 'id': id})

    # Tính tổng số trang (làm tròn lên)
    data = {'num_page': int(LenDictPath / imgperindex) + 1, 'pagefile': pagefile}

    # Render template 'home.html' với dữ liệu
    return render_template('home.html', data=data)

# Tìm kiếm theo ảnh query (id ảnh)
@app.route('/imgsearch')
def image_search():
    print("image search")

    pagefile = []

    # Lấy id ảnh query từ URL (?imgid=)
    id_query = int(request.args.get('imgid'))

    # Gọi tìm kiếm ảnh-ảnh: trả về danh sách id & path tương tự
    # Hàm image_search(id_query, k) -> (scores, list_ids, _, list_image_paths)
    _, list_ids, _, list_image_paths = MyFaiss.image_search(id_query, k=100)

    imgperindex = 100  # (Dùng để tính num_page cho giao diện)

    # Biến kết quả thành mảng [{imgpath, id}, ...]
    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath / imgperindex) + 1, 'pagefile': pagefile}

    return render_template('home.html', data=data)

# Tìm kiếm theo văn bản
@app.route('/textsearch')
def text_search():
    print("text search")

    pagefile = []

    # Lấy chuỗi text query từ URL (?textquery=)
    text_query = request.args.get('textquery')

    # --- PHẦN MỚI: cho phép chọn kiểu tìm kiếm qua tham số (?search_type=visual|ocr|hybrid) ---
    # Mặc định 'hybrid' nếu không truyền
    search_type = request.args.get('search_type', 'hybrid')

    # Gọi tìm kiếm text -> ảnh theo kiểu đã chọn
    # Hàm text_search(text, k, search_type) -> (scores, list_ids, _, list_image_paths)
    _, list_ids, _, list_image_paths = MyFaiss.text_search(text_query, k=100, search_type=search_type)
    # --- HẾT PHẦN MỚI ---

    imgperindex = 200

    # Biến kết quả thành mảng [{imgpath, id}, ...]
    for imgpath, id in zip(list_image_paths, list_ids):
        pagefile.append({'imgpath': imgpath, 'id': int(id)})

    data = {'num_page': int(LenDictPath / imgperindex) + 1, 'pagefile': pagefile}

    return render_template('home.html', data=data)

# Trả ảnh để hiển thị (kèm overlay tên file), dùng như một endpoint "stream" MJPEG
@app.route('/get_img')
def get_img():
    # Lấy full path của ảnh từ URL (?fpath=)
    fpath = request.args.get('fpath')

    # Lấy tên ảnh hiển thị gọn: ghép 2 cấp cuối cùng của đường dẫn (thư mục/video + file)
    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-2:])

    # Đọc ảnh: nếu không tồn tại thì trả ảnh 404 mặc định
    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jpg")
        img = cv2.imread("./static/images/404.jpg")

    # Resize ảnh về 1280x720 cho đồng nhất hiển thị
    img = cv2.resize(img, (1280, 720))

    # Vẽ text (tên ảnh rút gọn) lên ảnh để người dùng biết nguồn
    img = cv2.putText(
        img,
        image_name,
        (30, 80),                   # vị trí text (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,   # font
        3,                          # scale
        (255, 0, 0),                # màu (B, G, R): xanh dương đậm
        4,                          # độ dày nét
        cv2.LINE_AA                 # anti-aliased
    )

    # Mã hóa ảnh thành JPEG bytes
    ret, jpeg = cv2.imencode('.jpg', img)

    # Trả về theo dạng multipart/x-mixed-replace (stream khung hình)
    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# Điểm vào chương trình
if __name__ == '__main__':
    # Chạy Flask server ở 0.0.0.0:5001, tắt debug khi chạy production
    app.run(debug=False, host="0.0.0.0", port=5001)
