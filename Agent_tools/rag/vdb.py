from abc import ABC, abstractmethod
import chromadb
# from pydantic import BaseModel
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Union
from dataclasses import dataclass
from hashlib import md5

def compute_mdhash_id(content, prefix: str = ""):
    """函数内部使用 md5 哈希算法对 content 进行哈希计算，并通过 hexdigest() 方法将结果转换为十六进制字符串。最后，将 prefix 添加到生成的哈希值前面，并返回结果。"""
    return prefix + md5(content.encode()).hexdigest()


@dataclass
class VdbConfig:
    namespace: str = 'default'
    vdb_path: str = "db/vdb/chunk"
    n_results: int = 5


@dataclass
class MetaData():
    source: str = ""
    url: str = ""
    question: str = ""
    summary: str = ""


@dataclass
class VectorDatabaseData():
    documents: List[str]
    ids: List[str] = None
    metadatas: List[dict] | List[MetaData] = None

    def __post_init__(self):
        if self.metadatas and not isinstance(self.metadatas[0], dict):
            self.metadatas = [asdict(d) for d in self.metadatas]


class VectorDatabase(ABC):
    @abstractmethod
    def client(self, db_path: str):
        """
        加载持久化数据库
        """
        raise NotImplementedError

    @abstractmethod
    def create_collection(self):
        """
        创建集合
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, new_data: dict):
        """
        向集合中添加数据
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, query_texts, n_results):
        """查询n_results个最相似的文档"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list):
        """
        删除集合中指定的文档
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, upodate_data: dict):
        """
        更新集合中指定的文档
        """
        raise NotImplementedError


class ChromaVectorDatabase(VectorDatabase):
    def __init__(self, vdb_config: VdbConfig, name_space: str = None, embed_func=None):
        self.vdb_config = vdb_config
        self.embed_func = embed_func
        self.vdb_config.namespace = name_space or self.vdb_config.namespace
        self.db_path = self.vdb_config.vdb_path
        self.client = self.client()
        self.collection = self.create_collection()

    def client(self):
        # db_path是确定持久化位置
        client = chromadb.PersistentClient(path=self.db_path)
        return client

    def create_collection(self):
        if self.embed_func:
            emb_fn = self.embed_func
            # name_space是确定集合的名字,可以理解为数据库的名字
            collection = self.client.get_or_create_collection(name=self.vdb_config.namespace, embedding_function=emb_fn, metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100})
            # try:
            #     collection = self.client.create_collection(
            #         name=self.vdb_config.namespace, embedding_function=emb_fn, metadata={
            #             "hnsw:space": "cosine",
            #             "hnsw:search_ef": 100
            #         })
            # except Exception as e:
            #     collection = self.client.get_collection(
            #         name=self.vdb_config.namespace, embedding_function=emb_fn)
        else:
            collection = self.client.get_or_create_collection(
                name=self.vdb_config.namespace, metadata={
                    "hnsw:space": "cosine",
                    "hnsw:search_ef": 100
                })
            # try:
            #     collection = self.client.create_collection(
            #         name=self.vdb_config.namespace, metadata={
            #             "hnsw:space": "cosine",
            #             "hnsw:search_ef": 100
            #         })
            # except Exception as e:
            #     collection = self.client.get_collection(
            #         name=self.vdb_config.namespace)
        return collection

    def add(self, new_data: VectorDatabaseData):
        self.collection.add(
            documents=new_data.documents,
            ids=new_data.ids,
            metadatas=[dict(d) for d in new_data.metadatas] if new_data.metadatas else None,
        )

    def query(self, query_texts: list, topk:int=None, **kwargs):
        return self.collection.query(
            query_texts=query_texts,
            n_results=self.vdb_config.n_results if not topk else topk,
            **kwargs
        )

    def delete(self, ids: list):
        self.collection.delete(ids=ids)

    def upsert(self, update_data: VectorDatabaseData):

        self.collection.upsert(
            documents=update_data.documents,
            ids=update_data.ids,
            metadatas=[dict(d) for d in update_data.metadatas]
        )


def construct_vdb(data:Union[Dict,VectorDatabaseData],vdb: VectorDatabase):
    """
    构建向量数据库
    """
    if isinstance(data, VectorDatabaseData):
            vdb.add(data)
    else:
        try:
            parsed_data = VectorDatabaseData(**data)
            vdb.add(parsed_data)
        except Exception as e:
            print(f"Error parsing data: {e}")
        
    return vdb


if __name__ == '__main__':
    new_data = {
        "documents": ["doc1", "doc2"],
        "ids": ["id1", "id2"],
        "metadatas": [{"source": "source1"}, {"source": "source2"}]
    }
    a = VectorDatabaseData(**new_data)
    print(a)
