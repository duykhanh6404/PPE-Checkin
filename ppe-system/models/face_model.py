import insightface

# Khởi tạo mô hình insightface (Singleton trung tâm dùng chung để tránh tràn RAM)
print("[*] Khởi động model nhận diện khuôn mặt (InsightFace - buffalo_l)...")
app_face = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

# ctx_id=0 sẽ dùng GPU nếu có, ở đây dùng mặc định để tránh lỗi thiết bị
app_face.prepare(ctx_id=0, det_size=(640, 640))
print("[*] Model InsightFace đã sẵn sàng!")
