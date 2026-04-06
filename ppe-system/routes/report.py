from fastapi import APIRouter
from services.excel_export import export_history_to_excel

router = APIRouter(prefix="/api/report")

@router.get("/export-excel")
def export_excel():
    """Hàm gọi Logic thư viện pandas để lưu dữ liệu vào thư mục static/exports/"""
    result = export_history_to_excel()
    return result
