import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import io

swap = FastAPI()

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize inswapper
swap_action = insightface.model_zoo.get_model('inswapper', download=False, download_zip=False)

# ERROR HANDLING: Define a standard error response model
class ErrorResponse(BaseModel):
    error_code: int
    error_message: str
    error_details: str = None

# ERROR HANDLING: Custom exception handler for HTTPException
@swap.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.status_code, "error_message": exc.detail}
    )

# ERROR HANDLING: Custom exception handler for unexpected exceptions
@swap.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error_code": 500, "error_message": "Internal server error", "error_details": str(exc)}
    )

@swap.post("/swapper")
async def swapper(source_image: UploadFile = File(...), dest_image: UploadFile = File(...)):
    try:
        # Read images
        source_img = await read_image(source_image)
        dest_img = await read_image(dest_image)

        # Get faces
        source_face = face_app.get(source_img)
        dest_face = face_app.get(dest_img)

        # ERROR HANDLING: Check if faces are detected in both images
        if not source_face:
            raise HTTPException(status_code=400, detail="No face detected in source image")
        if not dest_face:
            raise HTTPException(status_code=400, detail="No face detected in destination image")

        # Enhance image
        res = dest_img.copy()
        res = swap_action.get(res, dest_face[0], source_face[0], paste_back=True)

        # Convert result to PNG
        is_success, buffer = cv2.imencode(".png", res)
        # ERROR HANDLING: Check if image encoding was successful
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode resulting image")

        # Return image
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    # ERROR HANDLING: Catch and re-raise HTTPExceptions
    except HTTPException:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred.")

# ERROR HANDLING: Improved image reading function with error checks
async def read_image(file: UploadFile):
    # Check if the file type is supported
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if the image was successfully decoded
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    
    return img

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(swap, host="0.0.0.0", port=8000)