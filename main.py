import os
from settings_manager import SettingsManager
from features.feature_1 import train_dreambooth_model
from features.feature_2 import run_replicate_model


settings = SettingsManager.get_settings()

REPLICATE_API_KEY = settings.get("replicate_api_key", None)
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_KEY

if __name__ == "__main__":
    # model = list(settings["models"]["feature_2"].keys())[1]

    # input = {
    #     "prompt": "A fantasy landscape with mountains and a river"
    # }
    # output = run_replicate_model(model, input)
    # output.show()

    model_name = "flux-pingpong2"
    images_path = "lora_dreambooth_tmp/Prueba_LoRA.zip"
    trigger_word = "TOKPG"
    description = "Flux fine-tuned with LoRA dreambooth conditioned on a pingpong ball case. Use the trigger word 'TOKPG' to generate images with this model."
    training = train_dreambooth_model(model_name, images_path, trigger_word, description)
