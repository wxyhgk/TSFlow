from fastapi import APIRouter, Request

from app.contracts.models import ModelListResponse


router = APIRouter()


@router.get("", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    return ModelListResponse(items=request.app.state.model_registry.list_models())

