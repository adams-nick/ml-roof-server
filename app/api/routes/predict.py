from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/predict")
async def predict_roof_segments(image_data: dict):
    """
    Temporary simple prediction endpoint.
    Will be replaced with actual SAM integration.
    """
    try:
        # For now, just return a mock response
        return {
            "status": "success",
            "message": "This is a placeholder. SAM model integration coming soon!",
            "received_data": f"Received data for building ID: {image_data.get('building_id', 'unknown')}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
