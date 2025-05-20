import replicate
import os
from dotenv import load_dotenv
from settings_manager import SettingsManager

settings = SettingsManager.get_settings()

# Load API key from .env file
def relight_image(subject_image, 
                  prompt = "detailed face, light background, warm light, 8k, high quality, realistic, photo",
                  light_source = "Left Light",
                  height = 640,
                  width = 512,
                  n_images = 1,
                  highres_scale = 1.5,
                  steps = 25):

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
            settings["models"]["feature_3"]["model"],
            input=input
        )
    except replicate.exceptions.ModelError as e:
        print("Model error:", e)
        raise

    for index, item in enumerate(output):
        with open(f"images/output/output_{index}.jpg", "wb") as file:
            file.write(item.read())