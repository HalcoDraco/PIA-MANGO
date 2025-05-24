import replicate
from settings_manager import SettingsManager
    
def _create_replicate_model(model_name: str, description: str):
    """
    Creates a new model on Replicate with the specified name and description.
    
    Parameters
    ----------
    model_name : str
        The name of the model to create.
    description : str
        The description of the model.
        
    Returns
    -------
    replicate.models.Model
        The created model object.
    """

    settings = SettingsManager.get_settings()

    model = replicate.models.create(
        owner=settings["replicate_username"],
        name=model_name.lower(),
        description=description,
        visibility=settings["new_replicate_models_visibility"],
        hardware="gpu-t4",
    )

    return model

def train_dreambooth_model(model_name: str, 
                           images_path: str, 
                           trigger_word: str, 
                           description: str,
                           steps: int = None,
                           lora_rank: int = None):
    """
    Trains a DreamBooth model on Replicate with the specified parameters.
    
    Parameters
    ----------
    model_name : str
        The name of the model to train.
    images_path : str
        The path to the training images.
    trigger_word : str
        The trigger word for the model.
    description : str
        The description of the model.
        
    Returns
    -------
    replicate.trainings.Training
        The training object for the model.
    """

    images_stream = open(images_path, "rb")
    settings = SettingsManager.get_settings()

    if steps is None:
        steps = settings["lora_dreambooth_params"]["steps"]
    if lora_rank is None:
        lora_rank = settings["lora_dreambooth_params"]["lora_rank"]
    
    model = _create_replicate_model(model_name, description)

    json_model = {
        "name": model_name,
        "author": settings["replicate_username"],
        "multiple_outputs": False,
        "description": description,
        "trigger_word": trigger_word,
        "lora_dreambooth_params": {
            "steps": steps,
            "lora_rank": lora_rank
        }
    }

    settings["models"]["feature_1"][f"{model.owner}/{model.name}"] = json_model
    SettingsManager.save_settings(settings)

    training = replicate.trainings.create(
        destination=f"{model.owner}/{model.name}",
        version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
        input={
            "steps": steps,
            "lora_rank": lora_rank,
            "input_images": images_stream,
            "trigger_word": trigger_word
        }
    )

    return training