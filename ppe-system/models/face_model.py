import insightface
import onnxruntime

# Khởi tạo mô hình insightface (Singleton trung tâm dùng chung để tránh tràn RAM)
print("[*] Khởi động model nhận diện khuôn mặt (InsightFace - buffalo_l)...")
available_providers = onnxruntime.get_available_providers()
print(f"[*] ONNX Providers khả dụng: {available_providers}")

app_face = insightface.app.FaceAnalysis(name='buffalo_l', providers=available_providers)

# ctx_id=0 sẽ dùng GPU nếu có, ở đây dùng mặc định để tránh lỗi thiết bị
app_face.prepare(ctx_id=0, det_size=(640, 640))
print("[*] Model InsightFace đã sẵn sàng!")
