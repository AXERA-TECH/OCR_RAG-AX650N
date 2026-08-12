#!/usr/bin/env python3
"""
============================================================
PDF OCR 智能问答系统 — 端到端运行脚本
============================================================

用法:
    # 交互模式: 处理文档后进入问答 REPL
    python run.py -f document.pdf

    # 单次问答
    python run.py -f document.pdf -q "文档主要内容是什么?"

    # 批量处理多个文档
    python run.py -f doc1.pdf doc2.png scan3.jpg

    # 指定分块参数
    python run.py -f document.pdf --chunk-size 1000 --chunk-overlap 200

    # 从已有向量库加载 (跳过 OCR, 直接问答)
    python run.py --load

    # 清空旧数据重新处理
    python run.py -f document.pdf --clear

    # 显示检索到的原文
    python run.py -f document.pdf -q "问题" --show-sources

环境变量 (或 .env 文件):
    EMBEDDING_API_BASE   Embedding API 地址
    EMBEDDING_MODEL_NAME Embedding 模型名
    LLM_API_BASE         LLM API 地址
    LLM_API_KEY          LLM API Key
    LLM_MODEL_NAME       LLM 模型名
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---- 环境补丁 (必须在其他导入之前) ----
def _patch():
    import types as _types
    if "langchain_text_splitters" not in sys.modules:
        m = _types.ModuleType("langchain_text_splitters")
        m.__path__ = []
        sys.modules["langchain_text_splitters"] = m
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


_patch()

# 项目导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ocr_loader import PaddleOCRLoader
from text_processor import TextProcessingPipeline, RecursiveCharacterTextSplitter
from embeddings import get_embedding_model
from vector_store import VectorStoreManager, build_vector_store
from rag_chain import RAGChain, create_llm, PDFRAGPipeline

# 将内置分割器注入到 mock 模块
import sys as _sys
_lts = _sys.modules.get("langchain_text_splitters")
if _lts is not None:
    _lts.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter

from loguru import logger

# ============================================================
# Banner
# ============================================================

BANNER = r"""
  ┌──────────────────────────────────────────────────────┐
  │         📄 PDF OCR 智能问答系统                        │
  │                                                      │
  │  OCR:   PaddleOCR-VL-1.5 (本地)                      │
  │  嵌入:  {emb_model}                                  │
  │  LLM:   {llm_model}                                  │
  │  向量库: {vec_store}                                  │
  └──────────────────────────────────────────────────────┘
"""


def print_banner():
    emb_name = config.EMBEDDING_MODEL_NAME
    llm_name = config.LLM_MODEL_NAME
    vs = config.VECTOR_STORE_TYPE
    # 截断过长的模型名
    if len(emb_name) > 35:
        emb_name = emb_name[:32] + "..."
    if len(llm_name) > 35:
        llm_name = llm_name[:32] + "..."
    print(BANNER.format(emb_model=emb_name, llm_model=llm_name, vec_store=vs))


# ============================================================
# 步骤函数
# ============================================================

def _save_documents(docs: list, path: Path, label: str = "文档"):
    """将 LangChain Document 列表保存为 JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for doc in docs:
        data.append({
            "page_content": doc.page_content,
            "metadata": {k: v for k, v in doc.metadata.items()
                         if isinstance(v, (str, int, float, bool, type(None)))}
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 {label}已保存: {path} ({len(data)} 条)")


def step_ocr(file_paths: List[str], output_dir: Optional[Path] = None) -> list:
    """Step 1: OCR 识别所有文件, 全部结果合并保存到一个文件"""
    all_docs = []
    for fp in file_paths:
        fp = Path(fp)
        if not fp.exists():
            logger.error(f"文件不存在: {fp}")
            continue
        suffix = fp.suffix.lower()
        if suffix not in config.SUPPORTED_FORMATS:
            logger.warning(f"跳过不支持格式: {fp} (支持: {config.SUPPORTED_FORMATS})")
            continue

        icon = "📄" if suffix == ".pdf" else "🖼️"
        print(f"  {icon} 正在识别: {fp.name} ...", end=" ", flush=True)
        t0 = time.time()
        loader = PaddleOCRLoader(str(fp), verbose=True)
        docs = loader.load()
        elapsed = time.time() - t0
        print(f"{len(docs)} 页/文档 ({elapsed:.1f}s)")
        all_docs.extend(docs)

    # 所有文件识别完后统一保存
    if output_dir and all_docs:
        save_path = output_dir / "ocr_results.json"
        _save_documents(all_docs, save_path, "OCR结果 ")

    return all_docs


def step_process(
    documents: list, chunk_size: int, chunk_overlap: int,
    output_dir: Optional[Path] = None
) -> list:
    """Step 2: 文本清洗 + 分割, 全部结果合并保存到一个文件"""
    print(f"  ✂️  正在分割: {len(documents)} 个文档 ...", end=" ", flush=True)
    t0 = time.time()
    pipeline = TextProcessingPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = pipeline.process(documents)
    elapsed = time.time() - t0
    print(f"→ {len(chunks)} 个文本块 ({elapsed:.1f}s)")

    if output_dir and chunks:
        save_path = output_dir / "chunks.json"
        _save_documents(chunks, save_path, "分块结果 ")

    return chunks


def step_embed(chunks: list) -> VectorStoreManager:
    """Step 3: 向量嵌入 + 入库"""
    print(f"  🧠 正在向量化: {len(chunks)} 个文本块 ...", end=" ", flush=True)
    t0 = time.time()
    manager = build_vector_store(chunks, clear_existing=True)
    elapsed = time.time() - t0
    print(f"完成 ({elapsed:.1f}s)")
    return manager


def step_rag(manager: VectorStoreManager):
    """Step 4: 初始化 RAG 链"""
    llm = create_llm()
    chain = RAGChain(vector_store_manager=manager, llm=llm)
    return chain


# ============================================================
# 核心流程
# ============================================================

def run_ingest(
    file_paths: List[str],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    clear: bool = True,
    output_dir: Optional[Path] = None,
) -> VectorStoreManager:
    """完整入库流程: OCR → 处理 → 嵌入 → 入库"""
    print("\n" + "─" * 55)
    print("  📥 阶段 1: 文档入库")
    print("─" * 55)

    # Step 1: OCR
    t_start = time.time()
    documents = step_ocr(file_paths, output_dir=output_dir)
    if not documents:
        logger.error("未识别到任何文本内容, 请检查文件是否包含可读文字")
        sys.exit(1)
    print(f"      总计: {len(documents)} 个原始文档页")

    # Step 2: 处理
    chunks = step_process(documents, chunk_size, chunk_overlap,
                          output_dir=output_dir)

    # Step 3: 嵌入入库
    manager = step_embed(chunks)

    total_time = time.time() - t_start
    print(f"\n  ✅ 入库完成 (总耗时 {total_time:.1f}s)")
    print(f"     文档: {len(documents)} 页 → {len(chunks)} 个文本块")
    print(f"     向量维度: {config.EMBEDDING_MODEL_NAME}")
    print(f"     存储: {config.VECTOR_STORE_TYPE} @ {config.VECTOR_DB_DIR}")

    return manager


def run_qa(chain: RAGChain, question: str, show_sources: bool = False):
    """执行单次问答"""
    print("\n" + "─" * 55)
    print(f"  ❓ 问题: {question}")
    print("─" * 55)

    t0 = time.time()
    result = chain.query(question)
    elapsed = time.time() - t0

    print(f"\n  🤖 回答 ({elapsed:.1f}s):")
    print("─" * 55)
    print(result["answer"])

    if show_sources:
        print(f"\n  📚 参考来源 ({len(result['sources'])} 条):")
        print("─" * 55)
        for src in result["sources"]:
            print(f"  [{src['rank']}] {src['document']} | 第{src['page']}页 "
                  f"| {src['content_type']}")
            print(f"      {src['content'][:120]}...")

    return result


def run_repl(chain: RAGChain):
    """交互式问答 REPL"""
    print("\n" + "─" * 55)
    print("  💬 交互问答模式")
    print("─" * 55)
    print("  输入问题后回车, 输入 :s 切换来源显示")
    print("  输入 :q 退出, :c 清屏, :h 帮助")
    print("─" * 55)

    chat_history = []
    show_sources = False

    while True:
        try:
            user_input = input("\n  🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见! 👋")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith(":"):
            cmd = user_input[1:].strip().lower()
            if cmd in ("q", "quit", "exit"):
                print("  再见! 👋")
                break
            elif cmd == "s":
                show_sources = not show_sources
                print(f"  来源显示: {'开启' if show_sources else '关闭'}")
            elif cmd == "c":
                os.system("clear" if os.name != "nt" else "cls")
            elif cmd == "h":
                print("  命令: :q 退出 | :s 切换来源 | :c 清屏 | :h 帮助")
            else:
                print(f"  未知命令: {user_input}")
            continue

        # 问答
        t0 = time.time()
        result = chain.query_with_history(user_input, chat_history)
        elapsed = time.time() - t0

        print(f"\n  🤖 ({elapsed:.1f}s):")
        print(f"  {result['answer']}")

        if show_sources:
            print(f"\n  📚 来源 ({len(result['sources'])} 条):")
            for src in result["sources"]:
                print(f"  [{src['rank']}] {src['document']} "
                      f"第{src['page']}页 | {src['content_type']}")

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": result["answer"]})


# ============================================================
# API 连通性检查
# ============================================================

def check_apis() -> bool:
    """检查 Embedding API 和 LLM API 是否可达"""
    import urllib.request

    all_ok = True

    # 检查 Embedding API
    emb_url = config.EMBEDDING_API_BASE.rstrip("/")
    try:
        req = urllib.request.Request(f"{emb_url}/models", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        print(f"  ✅ Embedding API: {emb_url}")
    except Exception as e:
        print(f"  ⚠️  Embedding API: {emb_url} — {e}")
        all_ok = False

    # 检查 LLM API
    llm_url = config.LLM_API_BASE.rstrip("/")
    try:
        req = urllib.request.Request(f"{llm_url}/models", method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        print(f"  ✅ LLM API: {llm_url}")
    except Exception as e:
        print(f"  ⚠️  LLM API: {llm_url} — {e}")
        all_ok = False

    return all_ok


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PDF OCR 智能问答系统 — 端到端运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py -f document.pdf                    # 交互问答
  python run.py -f doc.pdf -q "主要内容?"           # 单次问答
  python run.py -f a.pdf b.png --clear             # 批量处理
  python run.py --load                             # 加载已有向量库
        """,
    )
    parser.add_argument(
        "-f", "--files", nargs="+",
        default=["/data/huangjie/Project/dProject/pdfocr/过滤网modify.pdf",
                 "/data/huangjie/Project/dProject/pdfocr/videoagent.png",
                 "/data/huangjie/Project/dProject/pdfocr/biaozhun.jpg"],
        help="要处理的文档路径 (PDF/PNG/JPG/BMP/TIF)",
    )
    parser.add_argument(
        "-q", "--question",
        help="单次问答 (不进入交互模式)",
    )
    parser.add_argument(
        "--load", action="store_true",
        help="加载已有向量库, 跳过 OCR 处理",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="清空旧向量库数据后重新处理",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=config.CHUNK_SIZE,
        help=f"文本块大小 (默认: {config.CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=config.CHUNK_OVERLAP,
        help=f"块间重叠字符数 (默认: {config.CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--show-sources", action="store_true",
        help="在回答中显示参考来源",
    )
    parser.add_argument(
        "--top-k", type=int, default=config.RETRIEVAL_TOP_K,
        help=f"检索返回文档数 (默认: {config.RETRIEVAL_TOP_K})",
    )
    parser.add_argument(
        "--skip-api-check", action="store_true",
        help="跳过 API 连通性检查",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"中间结果保存目录 (默认: {config.OCR_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    # Banner
    print_banner()

    # API 检查
    if not args.skip_api_check:
        print("  🔌 API 连通性检查:")
        check_apis()
        print()

    # 模式判断
    if args.load:
        # 加载已有向量库
        print("  📂 加载已有向量库...")
        manager = VectorStoreManager(store_type=config.VECTOR_STORE_TYPE)
        count = manager.get_document_count()
        if count == 0:
            logger.error("向量库为空! 请先用 -f 指定文件进行入库")
            sys.exit(1)
        print(f"  ✅ 已加载: {count} 个文档块")
    elif args.files:
        # 处理文件
        output_dir = Path(args.output_dir) if args.output_dir else config.OCR_OUTPUT_DIR
        manager = run_ingest(
            args.files,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            clear=args.clear,
            output_dir=output_dir,
        )
    else:
        parser.print_help()
        print("\n  ❌ 请指定 -f/--files 或 --load")
        sys.exit(1)

    # 初始化 RAG 链
    print("\n" + "─" * 55)
    print("  🔗 阶段 2: 初始化 RAG 问答引擎")
    print("─" * 55)
    llm = create_llm()
    chain = RAGChain(
        vector_store_manager=manager,
        llm=llm,
        top_k=args.top_k,
    )
    print(f"  ✅ RAG 引擎就绪 (LLM={config.LLM_MODEL_NAME})")

    # 问答
    if args.question:
        run_qa(chain, args.question, show_sources=args.show_sources)
    else:
        run_repl(chain)


if __name__ == "__main__":
    main()
