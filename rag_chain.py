"""
============================================================
RAG 检索增强生成问答链
============================================================
LLM: Qwen3-8B (通过 OpenAI 兼容 API 调用)
嵌入: Qwen3-Embedding (通过 OpenAI 兼容 API 调用)

所有模型均通过 API 调用, 无需本地推理:
  - Embedding API: /v1/embeddings
  - LLM API:       /v1/chat/completions

支持任意 OpenAI 兼容 API:
  - vLLM 部署的 Qwen3 / Llama / DeepSeek 等
  - 第三方 API (DeepSeek, 通义千问, 智谱 GLM 等)
  - OpenAI 官方 API

功能:
  1. LangChain LCEL RAG 问答链
  2. 多轮对话
  3. 流式输出
  4. 来源引用
"""

from typing import List, Optional, Dict, Any, Iterator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_openai import ChatOpenAI

from loguru import logger

import config
from vector_store import VectorStoreManager


# ============================================================
# LLM 工厂 (纯 API 模式)
# ============================================================

def create_llm(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    创建 OpenAI 兼容的 LLM 实例

    Args:
        model_name:  模型名称, 如 Qwen/Qwen3-8B
        api_base:    API 地址
        api_key:     API Key
        temperature: 生成温度
        max_tokens:  最大输出 token 数

    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=model_name or config.LLM_MODEL_NAME,
        api_key=api_key or config.LLM_API_KEY,
        base_url=api_base or config.LLM_API_BASE,
        temperature=temperature or config.LLM_TEMPERATURE,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
    )


# ============================================================
# RAG 问答链
# ============================================================

class RAGChain:
    """
    RAG 检索增强生成链

    流程:
      Query → Embedding API 检索 → 上下文格式化 →
      Prompt 模板 → LLM API 生成 → 结构化回答 (含来源)

    用法:
        rag = RAGChain(vector_store_manager)
        result = rag.query("文档主要内容是什么?")
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm: Optional[BaseChatModel] = None,
        top_k: int = config.RETRIEVAL_TOP_K,
        system_prompt: Optional[str] = None,
        search_type: str = "similarity",
    ):
        self.vector_store_manager = vector_store_manager
        self.llm = llm or create_llm()
        self.top_k = top_k
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.search_type = search_type
        self._chain = self._build_chain()

        logger.info(
            f"RAG 问答链初始化完成 (LLM={config.LLM_MODEL_NAME}, "
            f"top_k={top_k}, search={search_type})"
        )

    def _build_chain(self):
        """使用 LangChain LCEL 构建 RAG 链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", config.RAG_PROMPT_TEMPLATE),
        ])

        chain = (
            RunnableParallel({
                "context": lambda inputs: self._retrieve_and_format(inputs["query"]),
                "question": lambda inputs: inputs["query"],
                "system_prompt": lambda _: self.system_prompt,
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def _retrieve_and_format(self, query: str) -> str:
        docs = self._retrieve(query)
        return self._format_docs(docs)

    def _retrieve(self, query: str) -> List[Document]:
        if self.search_type == "mmr":
            return self.vector_store_manager.max_marginal_relevance_search(
                query, k=self.top_k
            )
        elif self.search_type == "similarity_score":
            results = self.vector_store_manager.similarity_search_with_score(
                query, k=self.top_k
            )
            return [doc for doc, _ in results]
        else:
            return self.vector_store_manager.similarity_search(query, k=self.top_k)

    MAX_CONTEXT_CHARS = 1800  # 总上下文字符上限 (适配小显存 1152 token 限制)

    @classmethod
    def _format_docs(cls, docs: List[Document]) -> str:
        if not docs:
            return "（未找到相关文档内容）"

        # 控制每个文档块长度，避免超过小显存的 token 限制
        max_chunk_chars = cls.MAX_CONTEXT_CHARS // max(len(docs), 1)

        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "未知")
            doc_name = doc.metadata.get("document_name", "未知文档")

            content = doc.page_content
            if len(content) > max_chunk_chars:
                content = content[:max_chunk_chars] + "..."

            header = f"[{i}] {doc_name} p{page}"
            parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(parts)

    # ---- 查询接口 ----

    def query(self, question: str) -> Dict[str, Any]:
        """
        单次问答

        Returns:
            {"query": str, "answer": str, "sources": [...], "context": str}
        """
        logger.info(f"RAG 查询: {question[:100]}...")

        retrieved_docs = self._retrieve(question)
        answer = self._chain.invoke({"query": question})

        sources = self._build_sources(retrieved_docs)

        logger.info(f"生成完成: {len(answer)} 字符, {len(sources)} 个来源")
        return {
            "query": question,
            "answer": answer,
            "sources": sources,
            "context": self._format_docs(retrieved_docs),
        }

    def query_stream(self, question: str) -> Iterator[str]:
        """流式问答"""
        logger.info(f"RAG 流式查询: {question[:100]}...")
        for chunk in self._chain.stream({"query": question}):
            yield chunk

    def query_with_history(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """带对话历史的多轮问答"""
        chat_history = chat_history or []

        history_context = self._format_history(chat_history)
        retrieved_docs = self._retrieve(question)
        context = self._format_docs(retrieved_docs)

        messages = [
            SystemMessage(content=(
                f"{self.system_prompt}\n\n"
                f"## 对话历史:\n{history_context}"
            )),
            HumanMessage(content=config.RAG_PROMPT_TEMPLATE.format(
                system_prompt="",
                context=context,
                question=question,
            )),
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        return {
            "query": question,
            "answer": answer,
            "sources": self._build_sources(retrieved_docs),
            "context": context,
        }

    @staticmethod
    def _build_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "rank": i,
                "content": doc.page_content[:300],
                "page": doc.metadata.get("page", "未知"),
                "document": doc.metadata.get("document_name", "未知"),
                "content_type": doc.metadata.get("content_type", "text"),
            }
            for i, doc in enumerate(docs, 1)
        ]

    @staticmethod
    def _format_history(chat_history: List[Dict[str, str]]) -> str:
        if not chat_history:
            return "（无历史对话）"
        parts = []
        for turn in chat_history[-8:]:  # 仅保留最近 4 轮对话
            role = "用户" if turn.get("role") == "user" else "助手"
            parts.append(f"{role}: {turn.get('content', '')}")
        return "\n".join(parts)


# ============================================================
# PDF 完整问答流水线
# ============================================================

class PDFRAGPipeline:
    """
    PDF 智能问答完整流水线 (全 API 模式)

    一步完成: 文档上传 → OCR → 清洗 → 分割 → API嵌入 → 入库 → API问答

    用法:
        pipeline = PDFRAGPipeline()
        pipeline.ingest("document.pdf")
        result = pipeline.ask("文档主要内容是什么?")
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        store_type: Optional[str] = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        verbose: bool = True,
    ):
        self.llm = llm or create_llm()
        self.store_type = store_type or config.VECTOR_STORE_TYPE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose

        self._vector_store_manager: Optional[VectorStoreManager] = None
        self._rag_chain: Optional[RAGChain] = None

    def ingest(self, file_path: str, clear_existing: bool = True) -> int:
        """
        处理文档并构建向量数据库

        支持格式: PDF / PNG / JPG / BMP / TIF
        """
        from ocr_loader import PaddleOCRLoader
        from text_processor import TextProcessingPipeline

        logger.info(f"开始入库: {file_path}")

        # Step 1: OCR
        self._log("Step 1/4: PaddleOCR-VL-1.5 识别...")
        loader = PaddleOCRLoader(file_path, verbose=False)
        raw_docs = loader.load()
        self._log(f"  ✓ 识别完成: {len(raw_docs)} 页/文档")

        # Step 2: 处理
        self._log("Step 2/4: 文本清洗与分割...")
        pipeline = TextProcessingPipeline(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = pipeline.process(raw_docs)
        self._log(f"  ✓ 分割完成: {len(chunks)} 个文本块")

        # Step 3: 向量化 (通过 Embedding API)
        self._log("Step 3/4: Embedding API 向量化...")
        self._vector_store_manager = VectorStoreManager(store_type=self.store_type)
        if clear_existing:
            self._vector_store_manager.clear()
        chunk_count = self._vector_store_manager.add_documents(chunks)
        self._log(f"  ✓ 入库完成: {chunk_count} 个文本块")

        # Step 4: 初始化 RAG
        self._log("Step 4/4: 初始化 RAG 引擎...")
        self._rag_chain = RAGChain(
            vector_store_manager=self._vector_store_manager,
            llm=self.llm,
        )
        self._log("  ✓ 问答引擎就绪")
        self._log("入库完成! 可以开始提问。")

        return chunk_count

    def ingest_multiple(self, file_paths: List[str], clear_existing: bool = True) -> int:
        total = 0
        for i, fp in enumerate(file_paths):
            total += self.ingest(fp, clear_existing=(clear_existing and i == 0))
        return total

    def ask(self, question: str) -> Dict[str, Any]:
        if self._rag_chain is None:
            self._vector_store_manager = VectorStoreManager(store_type=self.store_type)
            if self._vector_store_manager.get_document_count() == 0:
                raise RuntimeError("向量数据库为空! 请先调用 ingest() 处理文档。")
            self._rag_chain = RAGChain(
                vector_store_manager=self._vector_store_manager,
                llm=self.llm,
            )
        return self._rag_chain.query(question)

    def ask_stream(self, question: str) -> Iterator[str]:
        if self._rag_chain is None:
            raise RuntimeError("请先调用 ingest() 处理文档。")
        return self._rag_chain.query_stream(question)

    def ask_with_history(
        self, question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if self._rag_chain is None:
            raise RuntimeError("请先调用 ingest() 处理文档。")
        return self._rag_chain.query_with_history(question, chat_history)

    @property
    def is_ready(self) -> bool:
        try:
            if self._vector_store_manager is None:
                self._vector_store_manager = VectorStoreManager(store_type=self.store_type)
            return self._vector_store_manager.get_document_count() > 0
        except Exception:
            return False

    @property
    def stats(self) -> Dict[str, Any]:
        if self._vector_store_manager is None:
            return {"status": "not_initialized"}
        return self._vector_store_manager.get_stats()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ============================================================
# 便捷函数
# ============================================================

def quick_qa(file_path: str, question: str) -> Dict[str, Any]:
    """便捷函数: 直接对文档提问 (一次性)"""
    from ocr_loader import PaddleOCRLoader
    from text_processor import TextProcessingPipeline
    from vector_store import build_vector_store

    loader = PaddleOCRLoader(file_path, verbose=False)
    raw_docs = loader.load()
    pipeline = TextProcessingPipeline()
    chunks = pipeline.process(raw_docs)
    manager = build_vector_store(chunks, clear_existing=True)
    chain = RAGChain(vector_store_manager=manager)
    return chain.query(question)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(f"用法: python {__file__} <file_path> <question>")
        print(f"示例: python {__file__} document.pdf '文档主要内容是什么?'")
        sys.exit(1)

    file_path = sys.argv[1]
    question = sys.argv[2]

    print(f"\n{'='*60}")
    print(f"  PDF/文档 智能问答测试")
    print(f"  文件: {file_path}")
    print(f"  问题: {question}")
    print(f"{'='*60}")

    result = quick_qa(file_path, question)

    print(f"\n{'='*60}")
    print(f"  回答:")
    print(f"{'='*60}")
    print(result["answer"])

    print(f"\n{'='*60}")
    print(f"  参考来源:")
    print(f"{'='*60}")
    for src in result["sources"]:
        print(f"  [{src['rank']}] {src['document']} 第{src['page']}页 ({src['content_type']})")
        print(f"      {src['content'][:150]}...")
