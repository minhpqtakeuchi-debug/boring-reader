from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    chatbot_base_url: str
    chatbot_instruction: str
    chatbot_model: str

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# create a global instance
settings = Settings()