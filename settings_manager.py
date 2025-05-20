import os
import json

class SettingsManager:
    """
    A class to manage settings for the application.

    It provides methods to load and save settings from/to a JSON file.
    The settings are loaded only once and cached for subsequent access.

    Attributes
    ----------
    _FILEPATH : str
        The path to the settings JSON file.
    _settings : dict
        The cached settings loaded from the JSON file.

    Methods
    -------
    get_settings()
        Loads the settings from the JSON file if not already loaded.
    save_settings(settings)
        Saves the settings to the JSON file.
    _load_settings()
        Loads the settings from the JSON file.
    """
    
    _FILEPATH = "settings.json"
    _IGNORED_KEY_PATH = "ignored_api_key.txt"
    _settings = None

    @classmethod
    def get_settings(cls) -> dict:
        """
        Loads the settings from a JSON file if not already loaded.
        
        Returns
        -------
        dict
            The settings loaded from the JSON file.
        """
        if cls._settings is None:
            cls._settings = cls._load_settings()
        return cls._settings

    @classmethod
    def save_settings(cls, settings):
        """
        Saves the settings to a JSON file.
        
        Parameters
        ----------
        settings : dict
            The settings to save.
        """
        with open(cls._FILEPATH, "w") as f:
            json.dump(settings, f, indent=4)
        
        cls._settings = settings

    @classmethod
    def _load_settings(cls):
        """
        Loads the settings from a JSON file.
        
        Returns
        -------
        dict
            The settings loaded from the JSON file.
        """
        if not os.path.exists(cls._FILEPATH):
            raise FileNotFoundError(f"Settings file {cls._FILEPATH} not found.")
        
        with open(cls._FILEPATH, "r") as f:
            settings_json = json.load(f)
        with open(cls._IGNORED_KEY_PATH, "r") as f:
            ignored_api_key = f.read().strip()
        settings_json["replicate_api_key"] = ignored_api_key
        return settings_json
    
