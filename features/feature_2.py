import replicate
import os
from constants import REPLICATE_API_KEY
from PIL import Image
from io import BytesIO

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

def stable_diffusion_3_5_large(input: dict) -> list:
    """
    Generates images using the Stable Diffusion 3.5 Large model.

    Parameters
    ----------
    input : dict
        Input parameters for the Stable Diffusion model.

    Returns
    -------
    list[Image]
        A list of generated images as PIL Image objects.
    """
    output = replicate.run(
        "stability-ai/stable-diffusion-3.5-large",
        input=input
    )
    # Convert all images in the output to PIL Image objects
    return [Image.open(BytesIO(image.read())) for image in output]

def flux_1_1_pro(input: dict) -> Image:
    """
    Generates an image using the Flux 1.1 Pro model.

    Parameters
    ----------
    input : dict
        Input parameters for the Flux model.

    Returns
    -------
    Image
        The generated image as a PIL Image object.
    """
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input=input
    )
    image_bytes = output.read()
    return Image.open(BytesIO(image_bytes))

def ideogram_v2(input: dict) -> Image:
    """
    Generates an image using the Ideogram V2 model.

    Parameters
    ----------
    input : dict
        Input parameters for the Ideogram model.

    Returns
    -------
    Image
        The generated image as a PIL Image object.
    """
    output = replicate.run(
        "ideogram-ai/ideogram-v2",
        input=input
    )
    image_bytes = output.read()
    return Image.open(BytesIO(image_bytes))

def recraft_v3(input: dict) -> Image:
    """
    Generates an image using the Recraft V3 model.

    Parameters
    ----------
    input : dict
        Input parameters for the Recraft model.

    Returns
    -------
    Image
        The generated image as a PIL Image object.
    """
    output = replicate.run(
        "recraft-ai/recraft-v3",
        input=input
    )
    image_bytes = output.read()
    return Image.open(BytesIO(image_bytes))

def photon(input: dict) -> Image:
    """
    Generates an image using the Photon model.

    Parameters
    ----------
    input : dict
        Input parameters for the Photon model.

    Returns
    -------
    Image
        The generated image as a PIL Image object.
    """
    output = replicate.run(
        "luma/photon",
        input=input
    )
    image_bytes = output.read()
    return Image.open(BytesIO(image_bytes))
