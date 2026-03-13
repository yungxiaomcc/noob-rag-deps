from pydantic_settings import (
    BaseSettings,
    TomlConfigSettingsSource,
    SettingsConfigDict
)
from pydantic_settings.sources import (
    PydanticBaseSettingsSource
)
from typing import Literal

from pydantic import BaseModel, Field

class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
        # env_prefix="spzn_api__",
        toml_file=["base.config.toml","config.toml","dev.config.toml","prod.config.toml"],  # 后续配置覆盖前者
    )
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, TomlConfigSettingsSource(settings_cls,deep_merge=True))


class DenseEmbeddingConfig(BaseModel):
    tei_base_url: str = "http://localhost:8080"
    max_concurrent_requests: int = 32
    timeout: float = 60.0
    truncate: bool = True
    max_tokens_per_request: int = 1024
    max_texts_per_request: int = 16

class SparseEmbeddingConfig(BaseModel):
    tei_base_url: str = "http://localhost:8080"
    max_concurrent_requests:int=32
    timeout: float = 60.0
    truncate: bool = True

class RerankingConfig(BaseModel):
    tei_base_url: str = "http://localhost:8080"
    max_concurrent_requests: int = 32
    timeout: float = 60.0
    truncate: bool = True
    max_tokens_per_request: int = 1024
    max_texts_per_request: int = 16


class MilvusConfig(BaseModel):
    uri: str
    dbname: str


class RagConfig(BaseModel):
    """RAG 服务配置。"""

    milvus: MilvusConfig
    collection_name: str = "noob_rag"
    dense_dim: int = 1024
    

class LogConfig(BaseModel):
    """日志配置。本地默认写文件；容器可设 file_path 为空仅 stdout。"""

    level: str = Field(default="INFO", description="日志级别")
    format: Literal["json", "text"] = Field(default="json", description="json 或 text")
    json_indent: int | None = Field(default=None, description="JSON 缩进，null 为单行")
    queue_size: int = Field(default=10000, description="QueueHandler 队列容量")
    otlp_endpoint: str | None = Field(default=None, description="配置则启用 OTLP 发送")
    service_name: str = Field(default="noob_rag_deps", description="服务名，用于日志平台")
    file_path: str = Field(
        default="logs/noob_rag_deps.log",
        description="本地默认写入文件；为空时仅 stdout",
    )
    file_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="单文件最大字节（默认 10MB），用于轮转",
    )
    file_backup_count: int = Field(default=5, description="保留的备份文件数量")


class NoobRagDepsConfig(BaseModel):

    dense: DenseEmbeddingConfig
    reranking: RerankingConfig
    rag: RagConfig
    log: LogConfig = Field(default_factory=LogConfig)

class Config(BaseConfig):

    noob_rag_deps:NoobRagDepsConfig

conf = Config().noob_rag_deps