import replicate
from PIL import Image
from io import BytesIO
from settings_manager import SettingsManager

settings = SettingsManager.get_settings()

MODELS1 = settings["models"]["feature_1"]
MODELS2 = settings["models"]["feature_2"]

def run_replicate_model(model_str: str, input: dict):
    """
    Runs a specified model from Replicate with the given input parameters.

    Parameters
    ----------
    model_str : str
        The name of the model to run.
    input : dict
        Input parameters for the model.

    Returns
    -------
    list[Image]
        The generated image(s) as PIL Image objects.
    """

    if model_str not in MODELS1 and model_str not in MODELS2:
        raise ValueError(f"Model '{model_str}' is not available.")
    
    # Get the latest version of the model
    model = replicate.models.get(model_str)
    version = model.versions.list()[0]
    
    result = replicate.run(
        f"{model_str}:{version.id}",
        input=input
    )

    if type(result) == list:
        return [Image.open(BytesIO(image.read())) for image in result]
    else:
        return [Image.open(BytesIO(result.read()))]
