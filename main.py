import os
from settings_manager import SettingsManager
from features.feature_1 import train_dreambooth_model, create_replicate_model
from features.feature_2 import run_replicate_model


settings = SettingsManager.get_settings()

REPLICATE_API_KEY = settings.get("replicate_api_key", None)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

if __name__ == "__main__":

    model = list(settings["models"]["feature_1"].keys())[1]
    print(model)

    input = {
        "prompt": "A photo of a TOK on top of a rock in the middle of the forest, advertising, highly detailed, 8k, photorealistic",
    }
    output = run_replicate_model(model, input)
    output[0].show()

    # model_name = "flux-pingpong3"
    # images_path = "lora_dreambooth_tmp/Prueba_LoRA.zip"
    # trigger_word = "TOK"
    # description = "Flux fine-tuned with LoRA dreambooth conditioned on a pingpong ball case. Use the trigger word 'TOK' to generate images with this model."
    # training = train_dreambooth_model(model_name, images_path, trigger_word, description)