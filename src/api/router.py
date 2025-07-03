"""Router chính cho API với các module được tổ chức."""

from fastapi import APIRouter
from .careplan.routes import router as careplan_router
from .analyze.routes import router as analyze_router
from .system.routes import router as system_router
from .rag.routes import router as rag_router

# Tạo router chính
router = APIRouter()

# Bao gồm các router con với prefix và tags tương ứng
router.include_router(careplan_router, prefix="/careplan")
router.include_router(analyze_router, prefix="/analyze")
router.include_router(system_router, prefix="/system")
router.include_router(rag_router, prefix="/rag")
