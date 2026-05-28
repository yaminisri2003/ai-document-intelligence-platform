# config.py
# This file is the central settings module for our entire project.
# It loads all configuration from the .env file in one place.
# Every other file in the project will import from here.

import os
from pathlib import Path
from dotenv import load_dotenv

# This line reads your .env file and loads everything into memory.
# After this runs, Python can access your API key via os.getenv()
load_dotenv()


class Settings:
    """
    Single source of truth for all project configuration.
    All values come from environment variables in your .env file.
    """

    # Your Groq API key — loaded from .env file
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Which AI model to use on Groq
    # llama3-70b-8192 means Llama 3, 70 billion parameters, 8192 token context
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # Which domain is currently active
    # This controls which config YAML file gets loaded
    ACTIVE_DOMAIN: str = os.getenv("ACTIVE_DOMAIN", "hr_enterprise")

    # Development or production environment
    APP_ENV: str = os.getenv("APP_ENV", "development")

    # How much detail to show in logs
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self):
        """
        Check that critical settings are present.
        Call this at startup so we fail immediately
        if something important is missing — rather than
        failing mysteriously later during a user request.
        This is called 'fail fast' in professional engineering.
        """
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Open your .env file and add your Groq API key. "
                "Get a free key at console.groq.com"
            )
        print("Settings validated successfully.")


# Create one single instance of Settings.
# Every other file imports this one object.
# This ensures configuration is loaded exactly once.
settings = Settings()