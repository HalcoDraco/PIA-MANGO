from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from feature_2 import run_replicate_model
from io import BytesIO
import base64
from settings_manager import SettingsManager

app = FastAPI()

@app.post("/generate")
async def generate_images(
    model_name: str = Form(...),
    input_json: str = Form(...)
):
    """
    Runs the specified Replicate model with given input and returns generated image(s) as base64.
    - model_name: the model string, e.g., 'stability-ai/stable-diffusion'
    - input_json: stringified JSON of the model input parameters
    """

    import json
    try:
        input_dict = json.loads(input_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid input JSON.")

    try:
        images = run_replicate_model(model_name, input_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(img_str)

    return JSONResponse(content={"images": base64_images})
