from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    boring_path: str

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# create a global instance
settings = Settings()