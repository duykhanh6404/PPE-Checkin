# Dự Án Hệ Thống Cảnh Báo An Toàn Lao Động (PPE) bằng YOLO11 + FastAPI

Đây là dự án Full-stack (Python FastAPI làm Backend xử lý Logic camera AI và phục vụ giao diện HTML Frontend tĩnh).

## Cấu trúc yêu cầu:
Sử dụng YOLO11 (`best.pt`) để kết nối camera nhận diện trạng thái Mũ, Áo bảo hộ. Web giao diện có endpoint xem trực tiếp, API thống kê excel, quét khuôn mặt, etc.

## ⚙ Hướng dẫn cài đặt và khởi động:

1. Mở Cửa sổ **Terminal** hoặc Command Prompt. Nhảy vào thư mục backend này.
```bash
cd c:\DA\KLTN\ppe-system
```

2. Cài đặt toàn bộ thư viện cần thiết sử dụng PIP (Có thể bật Môi trường ảo virtualenv tùy ý).
```bash
pip install -r requirements.txt
```

3. Bạn cần đặt file **`best.pt`** mà bạn có vào ngay tại thư mục gốc của Backend này (`c:\DA\KLTN\ppe-system\best.pt`). 
Nếu không, hệ thống sẽ tự động dùng model dự phòng thay thế (`yolo11n.pt`).

4. Hệ thống nhận diện đường dẫn chạy tĩnh của giao diện frontend trong thu mục `static/` (code backend của tôi đã auto copy các file html của bạn ở thư mục `..\static` ngoài vào đây rồi). Sẽ bao gồm file `index.html`.

5. Chạy Server Bằng **Uvicorn**:
```bash
uvicorn main:app --reload
```
Hoặc đơn giản là chạy file thẳng bằng python: 
```bash
python main.py
```

6. Truy cập kết quả hoàn thiện tại trình duyệt: **[http://localhost:8000]**
