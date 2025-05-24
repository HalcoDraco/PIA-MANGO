from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
import shutil
import os
from settings_manager import SettingsManager
from io import BytesIO
import replicate
import zipfile

# feature imports
from features.feature_3 import relight_image
from features.feature_2 import run_replicate_model

settings = SettingsManager.get_settings()
REPLICATE_API_KEY = settings.get("replicate_api_key", None)

if not REPLICATE_API_KEY:
    raise ValueError("Replicate API key is missing. Please check your settings.")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

app = FastAPI()

@app.post("/run-model/")
async def run_model_endpoint(
    model_name: str = Body(..., embed=True),
    input_data: dict = Body(..., embed=True)
):
    try:
        images = run_replicate_model(model_name, input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution error: {e}")

    # Create an in-memory zip file of all images
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for idx, img in enumerate(images):
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            zip_file.writestr(f"output_{idx}.png", img_byte_arr.read())
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=outputs.zip"}
    )

# Endpoint for feature 3
@app.post("/relight/")
async def relight_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form("detailed face, light background, warm light, 8k, high quality, realistic, photo"),
    light_source: str = Form("Left Light"),
    height: int = Form(640),
    width: int = Form(512),
    n_images: int = Form(1),
    highres_scale: float = Form(1.5),
    steps: int = Form(25)
):
    os.makedirs("images/input", exist_ok=True)
    input_path = f"images/input/{image.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Call relight function
    output_paths = relight_image(
        subject_image=open(input_path, "rb"),
        prompt=prompt,
        light_source=light_source,
        height=height,
        width=width,
        n_images=n_images,
        highres_scale=highres_scale,
        steps=steps
    )

    return [FileResponse(path) for path in output_paths]
