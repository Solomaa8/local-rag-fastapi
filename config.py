"""
Конфигурация проекта считывается из файла .env с дефолтными значениями.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Параметры LLM и эмбеддингов
    llm_model: str = "llama3:8b"
    embeddings_model: str = "nomic-embed-text"
    ollama_host: str = "http://127.0.0.1:11434"
    llm_timeout: int = 120

    # Директория с исходными документами
    docs_dir: str = "./docs"

    # Параметры двухуровневого разбиения текста (Contextual Chunking)
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Настройки гибридного поиска и реранкинга
    retrieval_k: int = 10
    reranker_top_n: int = 3

    # Лимит сохраняемой истории сообщений в диалоге
    max_history: int = 10

    # Список разрешенных источников CORS
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5180,http://127.0.0.1:5180"

    openai_api_key: str = ""

    def get_allowed_origins(self) -> list[str]:
        """Возвращает массив строк разрешенных источников для CORS."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
