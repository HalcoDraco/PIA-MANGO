# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import replicate
import shutil
import os
import uuid

app = FastAPI()

# Replicate client setup (optional if you use environment variables)
os.environ["REPLICATE_API_TOKEN"] = "your_api_token_here"

@app.post("/replace-background")
async def replace_background(image: UploadFile = File(...)):
    # Save the uploaded image
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Run the Replicate model
    output = replicate.run(
        "your-username/model-name:version", 
        input={"image": open(temp_filename, "rb")}
    )

    # Remove the temporary image
    os.remove(temp_filename)

    return {"output": output}

@app.post("/dreambooth-lora")
async def dreambooth_generate(prompt: str = Form(...)):
    output = replicate.run(
        "your-username/dreambooth-model:version", 
        input={"prompt": prompt}
    )
    return {"output": output}
