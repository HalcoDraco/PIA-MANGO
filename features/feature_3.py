import replicate
import os
from dotenv import load_dotenv

# Load API key from .env file
def relight_image(subject_image, 
                  prompt = "detailed face, light background, warm light, 8k, high quality, realistic, photo",
                  light_source = "Left Light",
                  height = 640,
                  width = 512,
                  n_images = 1,
                  highres_scale = 1.5,
                  steps = 25):
    load_dotenv()
    api_token = os.getenv("REPLICATE_API_TOKEN")
    replicate_client = replicate.Client(api_token)

    input = {
        "prompt": prompt,
        "subject_image": subject_image,
        "light_source": light_source,
        "height": height,
        "width": width,
        "number_of_images": n_images,
        "highres_scale": highres_scale,
        "steps": steps,
    }

    try:
        output = replicate.run(
            "zsxkib/ic-light:d41bcb10d8c159868f4cfbd7c6a2ca01484f7d39e4613419d5952c61562f1ba7",
            input=input
        )
    except replicate.exceptions.ModelError as e:
        print("Model error:", e)
        raise

    for index, item in enumerate(output):
        with open(f"images/output/output_{index}.jpg", "wb") as file:
            file.write(item.read())