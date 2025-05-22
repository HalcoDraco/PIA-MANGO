import replicate
from settings_manager import SettingsManager

settings = SettingsManager.get_settings()

MODELS1 = settings["models"]["feature_1"]
MODELS2 = settings["models"]["feature_2"]

def run_replicate_model(model_str: str, input: dict, ):
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
    
    model = replicate.models.get(model_str)
    try:
        versions = model.versions.list()
    except Exception as e:
        versions = []

    if versions:
        version = versions[0]
        result = replicate.run(
            f"{model_str}:{version.id}",
            input=input
        )
    else:
        result = replicate.run(
            f"{model_str}",
            input=input
        )

    # result is expected to be file-like objects
    if isinstance(result, list):
        from PIL import Image
        from io import BytesIO
        return [Image.open(BytesIO(image.read())) for image in result]
    else:
        from PIL import Image
        from io import BytesIO
        return [Image.open(BytesIO(result.read()))]
