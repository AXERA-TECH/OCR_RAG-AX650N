"""
============================================================
向量数据库存储模块
============================================================
嵌入模型: Qwen3-Embedding 系列
向量数据库: Chroma / FAISS

功能:
  1. 文档批量向量化入库
  2. 相似度检索 / MMR / 元数据过滤
  3. 持久化与增量更新
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores import Chroma, FAISS

from loguru import logger

import config
from embeddings import get_embedding_model


# ============================================================
# 向量数据库工厂
# ============================================================

class VectorStoreFactory:

    @staticmethod
    def create_chroma(
        persist_directory: Optional[str | Path] = None,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
    ) -> Chroma:
        persist_dir = str(persist_directory or config.VECTOR_DB_DIR / "chroma")
        embedding = embedding_function or get_embedding_model()

        logger.info(f"创建 Chroma 向量数据库: {persist_dir} (集合: {collection_name})")

        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir,
            collection_metadata={
                "hnsw:space": "cosine",  # Qwen3-Embedding 使用余弦相似度
                "hnsw:construction_ef": 200,
                "hnsw:M": 48,
            },
        )

    @staticmethod
    def create_faiss(
        embedding_function: Optional[Embeddings] = None,
    ) -> FAISS:
        embedding = embedding_function or get_embedding_model()
        logger.info("创建 FAISS 向量数据库 (flat L2 index)")
        # FAISS.from_documents 会创建合适的索引
        return FAISS(
            embedding_function=embedding,
            index=None,
            docstore=None,
            index_to_docstore_id={},
        )

    @staticmethod
    def create(store_type: Optional[str] = None, **kwargs) -> VectorStore:
        store_type = store_type or config.VECTOR_STORE_TYPE
        if store_type == "chroma":
            return VectorStoreFactory.create_chroma(**kwargs)
        elif store_type == "faiss":
            return VectorStoreFactory.create_faiss(**kwargs)
        else:
            raise ValueError(f"不支持的向量数据库: {store_type}. 可选: chroma, faiss")


# ============================================================
# 向量数据库管理器
# ============================================================

class VectorStoreManager:

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        store_type: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str | Path] = None,
    ):
        self.store_type = store_type or config.VECTOR_STORE_TYPE
        self.embedding_function = embedding_function or get_embedding_model()
        self.persist_directory = str(persist_directory or config.VECTOR_DB_DIR)
        self._store = vector_store or self._init_store()

    def _init_store(self) -> VectorStore:
        if self.store_type == "chroma":
            return self._init_chroma()
        elif self.store_type == "faiss":
            return self._init_faiss()
        else:
            raise ValueError(f"不支持的向量数据库: {self.store_type}")

    def _init_chroma(self) -> Chroma:
        persist_dir = Path(self.persist_directory) / "chroma"
        if persist_dir.exists() and any(persist_dir.iterdir()):
            logger.info(f"加载已有 Chroma 数据库: {persist_dir}")
            return Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embedding_function,
                collection_name=config.CHROMA_COLLECTION_NAME,
            )
        return VectorStoreFactory.create_chroma(
            persist_directory=str(persist_dir),
            embedding_function=self.embedding_function,
        )

    def _init_faiss(self) -> FAISS:
        index_path = Path(self.persist_directory) / "faiss_index"
        if index_path.exists():
            logger.info(f"加载已有 FAISS 数据库: {index_path}")
            return FAISS.load_local(
                str(index_path),
                self.embedding_function,
                allow_dangerous_deserialization=True,
            )
        return VectorStoreFactory.create_faiss(
            embedding_function=self.embedding_function,
        )

    @property
    def store(self) -> VectorStore:
        return self._store

    # ---- 入库 ----

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        if not documents:
            logger.warning("文档列表为空, 跳过入库")
            return 0

        total = len(documents)
        logger.info(f"开始向量化入库: {total} 个文档块 (批大小={batch_size})")

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            self._store.add_documents(batch)
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        self._persist()
        logger.info(f"向量化入库完成: {total} 个文档块")
        return total

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 50,
    ) -> List[str]:
        if not texts:
            return []
        all_ids = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size] if metadatas else None
            ids = self._store.add_texts(batch_texts, batch_metas)
            all_ids.extend(ids)
        self._persist()
        return all_ids

    # ---- 检索 ----

    def similarity_search(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        if filter and isinstance(self._store, Chroma):
            kwargs["filter"] = filter
        return self._store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.3,
        **kwargs,
    ) -> List[tuple]:
        if filter and isinstance(self._store, Chroma):
            kwargs["filter"] = filter
        raw = self._store.similarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        # Qwen3-Embedding 余弦相似度通常 > 0.5 为相关
        return [(doc, score) for doc, score in raw if score >= score_threshold]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if filter and isinstance(self._store, Chroma):
            return self._store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter,
            )
        return self._store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult,
        )

    # ---- 过滤查询 ----

    def search_by_document(
        self, query: str, document_name: str, k: int = config.RETRIEVAL_TOP_K
    ) -> List[Document]:
        return self.similarity_search(query, k=k, filter={"document_name": document_name})

    def search_by_page_range(
        self, query: str, start_page: int, end_page: int,
        k: int = config.RETRIEVAL_TOP_K,
    ) -> List[Document]:
        return self.similarity_search(
            query, k=k, filter={"page": {"$gte": start_page, "$lte": end_page}}
        )

    # ---- 管理 ----

    def _persist(self):
        if self.store_type == "faiss":
            index_path = Path(self.persist_directory) / "faiss_index"
            index_path.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(index_path))

    def clear(self):
        if self.store_type == "chroma":
            self._store.delete_collection()
            self._store = VectorStoreFactory.create_chroma(
                persist_directory=Path(self.persist_directory) / "chroma",
                embedding_function=self.embedding_function,
            )
        elif self.store_type == "faiss":
            self._store = VectorStoreFactory.create_faiss(
                embedding_function=self.embedding_function,
            )
        logger.info("向量数据库已清空")

    def get_document_count(self) -> int:
        try:
            if self.store_type == "chroma":
                return self._store._collection.count()
            elif self.store_type == "faiss":
                return self._store.index.ntotal if self._store.index else 0
        except Exception:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "store_type": self.store_type,
            "persist_directory": self.persist_directory,
            "document_count": self.get_document_count(),
            "embedding_model": config.EMBEDDING_MODEL_NAME,
        }


# ============================================================
# 便捷函数
# ============================================================

def build_vector_store(
    documents: List[Document],
    store_type: Optional[str] = None,
    embedding_model: Optional[Embeddings] = None,
    clear_existing: bool = False,
) -> VectorStoreManager:
    manager = VectorStoreManager(
        store_type=store_type,
        embedding_function=embedding_model,
    )
    if clear_existing:
        manager.clear()
    manager.add_documents(documents)
    return manager


def load_vector_store(
    store_type: Optional[str] = None,
    embedding_model: Optional[Embeddings] = None,
) -> VectorStoreManager:
    return VectorStoreManager(
        store_type=store_type,
        embedding_function=embedding_model,
    )
