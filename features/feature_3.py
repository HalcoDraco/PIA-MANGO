import replicate
from settings_manager import SettingsManager

settings = SettingsManager.get_settings()

def relight_image(replicate_client:str = None,
                  subject_image:None = None, 
                  prompt: str = "detailed face, light background, warm light, 8k, high quality, realistic, photo",
                  light_source: str = "Left Light",
                  height:int  = 640,
                  width:int  = 512,
                  n_images:int  = 1,
                  highres_scale: float = 1.5,
                  steps:int  = 25) -> list[str]: 
    
    """
    Relight an image using a specified model from Replicate.

    Parameters
    ----------
    replicate_client : str
        The Replicate client instance.

    subject_image : None
        The image to be relit.
    prompt : str
        The prompt for the model.
    light_source : str
        The light source direction.
        (e.g., "Left Light", "Right Light", etc.)
    height : int
        The height of the output image.
    width : int
        The width of the output image.
    n_images : int
        The number of images to generate.
    highres_scale : float
        The scale for high-resolution generation.
    steps : int
        The number of steps for the model to run.
    
    Returns
    -------

    list[str]
        A list of file paths to the generated images.
    """

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

    if replicate_client == None:
        try:
            output = replicate.run(
                settings["models"]["feature_3"]["model"],
                input=input
            )

        except replicate.exceptions.ModelError as e:
            print("Model error:", e)
            raise
    else:
        try:
            output = replicate_client.run(
                settings["models"]["feature_3"]["model"],
                input=input
            )
        except replicate.exceptions.ModelError as e:
            print("Model error:", e)
            raise

    for index, item in enumerate(output):
        with open(f"images/output/output_{index}.jpg", "wb") as file:
            file.write(item.read())

    return [f"images/output/output_{index}.jpg" for index in range(len(output))]