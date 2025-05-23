##
import requests
import json
from chromadb import Documents, EmbeddingFunction, Embeddings
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
import os
@dataclass
class EmbeddingConfig:
    """Embedding模型配置
    Args:
        model_name (str): Embedding模型名称
        is_remote (bool): 是否使用远程Embedding模型
        url (str): Embedding模型API地址
        token (str): Embedding模型API密钥
        custome (bool): 是否使用自定义Embedding模型
    """
    model_name: str
    url: str
    token: str
    # 是否使用自定义embedding
    custom: bool = True
    is_remote: str = True
    is_vllm: bool = True

    def __post_init__(self):
        if not self.is_remote:
            return
        if self.is_vllm: # 暂时写固定了
            return
        if not self.url and self.is_remote:
            self.url = "https://api.siliconflow.cn/v1/embeddings"
        if not self.token and self.is_remote:
            self.token = os.getenv('SILICONFLOW_TOKEN')

class Embedding(EmbeddingFunction):
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

    def embed(self, text: str | list[str]) -> list[float]:
        if self.config.is_remote and not self.config.is_vllm:
            return self._embed_remote(text)
        elif self.config.is_remote and self.config.is_vllm:
            return self._embed_remote_vllm(text)
        else:
            return self._embed_local(text)

    def _embed_remote(self, text: str | list[str]) -> list[float]:
        assert self.config.is_remote, "Not remote"

        payload = {
            "model": self.config.model_name,
            "input": text,
            "encoding_format": "float"
        }

        headers = {
            "Authorization": "Bearer " + self.config.token,
            "Content-Type": "application/json"
        }

        response = requests.request(
            "POST", url=self.config.url, json=payload, headers=headers)
        return [d["embedding"] for d in json.loads(response.text)['data']]


    def _embed_remote_vllm(self, text: str | list[str]) -> list[float]:
        assert self.config.is_remote, "Not remote"
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8001/v1"



        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        model = models.data[0].id

        responses = client.embeddings.create(
            input=text,
            model=model,
            
        )  
        return [d.embedding for d in responses.data]


    
    def _embed_local(self, text: str | list[str]) -> list[float]:
        assert not self.config.is_remote, "Not local"

        return [1.0] * len(text)

    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        embeddings = self.embed(input)
        embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
        return embeddings
