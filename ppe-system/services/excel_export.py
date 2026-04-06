import pandas as pd
from database.database import SessionLocal, DetectionLog
import os

def export_history_to_excel():
    """Hàm trích xuất dữ liệu Logs từ bảng sqlite để xuất ra file Excel bằng Pandas"""
    db = SessionLocal()
    try:
        # Lấy mọi bản ghi lịch sử phát hiện
        logs = db.query(DetectionLog).order_by(DetectionLog.timestamp.desc()).all()
        data = []
        for log in logs:
            data.append({
                "ID": log.id,
                "Mã Nhân Viên": log.employee_id,
                "Tên Nhân Viên": log.employee_name,
                "Thời Gian Quét": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Trạng Thái An Toàn": "Hoàn toàn an toàn" if log.is_safe else "Không an toàn",
                "Chi Tiết Vi Phạm": log.details if log.details else "Không có",
                "Link Ảnh Cắt": log.image_path if not log.is_safe else "Không yêu cầu ảnh"
            })
            
        dfs = pd.DataFrame(data)
        
        # Thư mục lưu xuất kho file excel (bảo vệ khỏi việc sai CWD do Terminal)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        export_dir = os.path.join(BASE_DIR, "static", "exports")
        os.makedirs(export_dir, exist_ok=True)
        # Tên file xuất ra cuối cùng
        file_path = os.path.join(export_dir, "BaoCao_LichSu_PPE.xlsx")
        
        # Để thư viện này chạy chuẩn, trong environment user đã cần "pip install openpyxl"
        dfs.to_excel(file_path, index=False)
        
        # Trả về URL đường dẫn web để frontend render thành liên kết nút Tải về
        return {"status": "success", "file_url": f"/{file_path.replace(chr(92), '/')}"} # hỗ trợ dạng slash cho URL
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
