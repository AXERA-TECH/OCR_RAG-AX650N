"""
============================================================
向量嵌入模块 (OpenAI 兼容 API)
============================================================
直接使用 openai 客户端, 兼容:
  - 阿里云 DashScope (text-embedding-v4 等)
  - vLLM 部署的 Qwen3-Embedding
  - 任意 OpenAI 兼容嵌入服务

用法:
    model = get_embedding_model()
    vec = model.embed_query("查询文本")
    vecs = model.embed_documents(["文本1", "文本2"])
"""

from typing import List, Optional
import numpy as np

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from loguru import logger

import config


# ============================================================
# 通用 OpenAI 兼容嵌入类
# ============================================================

class OpenAICompatEmbeddings(Embeddings):
    """
    轻量级 OpenAI 兼容嵌入类

    直接使用 openai 客户端发送请求, 避免 langchain_openai 的额外封装
    导致的 API 兼容性问题 (如 DashScope 的参数校验差异)。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
    ):
        self.model = model or config.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size if batch_size is not None else config.EMBEDDING_BATCH_SIZE
        self.dimensions = dimensions

        self._client = OpenAI(
            api_key=api_key or config.EMBEDDING_API_KEY,
            base_url=base_url or config.EMBEDDING_API_BASE,
        )

        logger.info(
            f"Embedding API 连接: model={self.model}, "
            f"base_url={base_url or config.EMBEDDING_API_BASE}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            kwargs = dict(model=self.model, input=batch)
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions

            response = self._client.embeddings.create(**kwargs)
            # response.data 按输入顺序返回
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            if len(texts) > self.batch_size:
                logger.debug(
                    f"嵌入进度: {min(i + self.batch_size, len(texts))}/{len(texts)}"
                )

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        kwargs = dict(model=self.model, input=text)
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding


# ============================================================
# 全局单例
# ============================================================

_embedding_model: Optional[Embeddings] = None


def get_embedding_model(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Embeddings:
    """获取全局嵌入模型单例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAICompatEmbeddings(
            model=model_name,
            base_url=api_base,
        )
    return _embedding_model


def reset_embedding_model():
    """重置嵌入模型单例"""
    global _embedding_model
    _embedding_model = None
    logger.info("嵌入模型已重置")


# ============================================================
# 工具函数
# ============================================================

def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    v1, v2 = np.array(vec1), np.array(vec2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def batch_embed(
    texts: List[str],
    model: Optional[Embeddings] = None,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> List[List[float]]:
    """批量嵌入文本 (支持自定义 batch_size)"""
    if model is None:
        model = get_embedding_model()

    all_embeddings = []
    total = len(texts)
    bs = batch_size or config.EMBEDDING_BATCH_SIZE

    for i in range(0, total, bs):
        batch = texts[i : i + bs]
        embeddings = model.embed_documents(batch)
        all_embeddings.extend(embeddings)

        if show_progress and i + bs < total:
            logger.debug(f"嵌入进度: {min(i + bs, total)}/{total}")

    return all_embeddings


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("测试 Embedding API 连接...\n")
    print(f"API: {config.EMBEDDING_API_BASE}")
    print(f"模型: {config.EMBEDDING_MODEL_NAME}")

    try:
        model = get_embedding_model()

        test_texts = [
            "这是第一段测试文本，用于验证嵌入API是否正常工作。",
            "这是第二段完全不同的文本内容，涉及人工智能话题。",
            "向量嵌入是自然语言处理中的基础技术。",
        ]

        print("\n测试单文本嵌入 (embed_query)...")
        query_vec = model.embed_query("嵌入模型测试")
        print(f"  维度: {len(query_vec)}")

        print("\n测试批量嵌入 (embed_documents)...")
        doc_vecs = model.embed_documents(test_texts)
        print(f"  数量: {len(doc_vecs)}, 维度: {len(doc_vecs[0])}")

        print("\n测试相似度计算...")
        sim1 = compute_similarity(doc_vecs[2], query_vec)
        sim2 = compute_similarity(doc_vecs[0], query_vec)
        print(f"  查询 vs 向量嵌入文本: {sim1:.4f}")
        print(f"  查询 vs 无关文本:      {sim2:.4f}")

        print(f"\n✓ Embedding API 测试通过")

    except Exception as e:
        print(f"\n✗ API 连接失败: {e}")
        print(f"  请确保 Embedding API 服务已启动: {config.EMBEDDING_API_BASE}")
