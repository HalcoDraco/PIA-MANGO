import replicate
from PIL import Image
from io import BytesIO
from settings_manager import SettingsManager

settings = SettingsManager.get_settings()

MODELS = settings["models"]["feature_2"]

def run_replicate_model(model: str, input: dict):
    """
    Runs a specified model from Replicate with the given input parameters.

    Parameters
    ----------
    model : str
        The name of the model to run.
    input : dict
        Input parameters for the model.

    Returns
    -------
    Image or list[Image]
        The generated image(s) as PIL Image objects.
    """
    if model not in MODELS:
        raise ValueError(f"Model {model} is not supported.")
    
    result = replicate.run(
        model,
        input=input
    )

    if MODELS[model]["multiple_outputs"]:
        # Convert all images in the output to PIL Image objects
        return [Image.open(BytesIO(image.read())) for image in result]
    else:
        # Convert the single image in the output to a PIL Image object
        image_bytes = result.read()
        return Image.open(BytesIO(image_bytes))
