"""
============================================================
PDF OCR 智能问答系统 - Web UI (FastAPI)
============================================================
模型栈: PaddleOCR-VL-1.5 + Embedding API + LLM API (OpenAI 兼容)
支持格式: PDF / PNG / JPG / JPEG / BMP / TIF / TIFF

启动:
    python app.py
访问: http://localhost:7860

前置依赖: 需先启动 Embedding API 和 LLM API 服务 (vLLM 或其他 OpenAI 兼容服务)
"""

import gc
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# ---- 环境补丁 ----
# 修复两个环境兼容性问题:
# 1. transformers + torch<2.4 → is_torch_available()=False → NameError
# 2. paddleocr → paddlex → langchain_text_splitters → sentence_transformers
#    → transformers 的损坏导入链
#
# 策略: 在触发导入链之前:
#   a) 提前导入 torch 并用 mock 模块替代 langchain_text_splitters
#   b) 提前导入我们的 RecursiveCharacterTextSplitter 并注入到 sys.modules


def _apply_env_patches():
    """尽早修复已知的环境兼容性问题"""
    import sys
    import types

    # Step 1: Mock `langchain_text_splitters` 以避免其 __init__.py
    #         触发 sentence_transformers → transformers 损坏链
    if "langchain_text_splitters" not in sys.modules:
        mock_lts = types.ModuleType("langchain_text_splitters")
        mock_lts.__path__ = []
        sys.modules["langchain_text_splitters"] = mock_lts

    # Step 2: 将我们的 RecursiveCharacterTextSplitter 注入到 mock 模块
    mock_lts = sys.modules["langchain_text_splitters"]
    from text_processor import RecursiveCharacterTextSplitter as OurSplitter
    mock_lts.RecursiveCharacterTextSplitter = OurSplitter

    # Step 3: 确保 torch 对 transformers 可用
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            pass


_apply_env_patches()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

import config
from rag_chain import PDFRAGPipeline, RAGChain
from vector_store import VectorStoreManager
from ocr_loader import PaddleOCRLoader
from text_processor import TextProcessingPipeline


# ============================================================
# 全局状态
# ============================================================

_pipeline: Optional[PDFRAGPipeline] = None
_processed_files: List[Dict[str, Any]] = []
_chat_history: List[Dict[str, str]] = []

# OCR 文本持久化目录
_OCR_OUTPUT_DIR = config.OCR_OUTPUT_DIR
_FILES_JSON = _OCR_OUTPUT_DIR / "_files.json"


def _load_files_from_disk():
    """启动时从磁盘恢复已处理文件列表"""
    global _processed_files
    if _FILES_JSON.exists():
        try:
            import json
            data = json.loads(_FILES_JSON.read_text(encoding="utf-8"))
            _processed_files = data.get("files", [])
            logger.info(f"从磁盘恢复 {len(_processed_files)} 个已处理文件")
        except Exception as e:
            logger.warning(f"恢复文件列表失败: {e}")


def _save_files_to_disk():
    """将已处理文件列表持久化到磁盘"""
    import json
    _FILES_JSON.parent.mkdir(parents=True, exist_ok=True)
    _FILES_JSON.write_text(
        json.dumps({"files": _processed_files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_ocr_text_path(filename: str) -> Path:
    """获取 OCR 文本的磁盘路径"""
    return _OCR_OUTPUT_DIR / f"{Path(filename).stem}.txt"


def _save_ocr_text(filename: str, text: str):
    """保存 OCR 文本到磁盘"""
    path = _get_ocr_text_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_ocr_text(filename: str) -> str:
    """从磁盘读取 OCR 文本"""
    path = _get_ocr_text_path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _delete_ocr_text(filename: str):
    """从磁盘删除 OCR 文本"""
    path = _get_ocr_text_path(filename)
    if path.exists():
        path.unlink()


def get_pipeline() -> PDFRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PDFRAGPipeline(verbose=False)
    return _pipeline


# ============================================================
# 核心处理逻辑 (从原 Gradio 回调中提取)
# ============================================================

def process_file_impl(
    file_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Tuple[Dict[str, Any], str]:
    """处理上传的文件: OCR → 分割 → 向量化入库"""
    global _pipeline, _processed_files, _chat_history

    suffix = file_path.suffix.lower()

    if suffix not in config.SUPPORTED_FORMATS:
        raise ValueError(
            f"不支持的文件格式: {suffix}\n支持: {', '.join(sorted(config.SUPPORTED_FORMATS))}"
        )

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        raise ValueError(f"文件过大: {file_size_mb:.1f}MB (限制: {config.MAX_FILE_SIZE_MB}MB)")

    # 复用 pipeline 对象避免重复创建 LLM 实例
    if _pipeline is None:
        _pipeline = PDFRAGPipeline(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            verbose=False,
        )

    loader = PaddleOCRLoader(str(file_path), verbose=False)
    raw_docs = loader.load()

    # 逐页写入 OCR 文本到磁盘，避免内存中构建完整副本
    ocr_path = _get_ocr_text_path(file_path.name)
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ocr_path, "w", encoding="utf-8") as ocr_f:
        preview_parts = []
        for i, doc in enumerate(raw_docs):
            page_num = doc.metadata.get("page", i + 1)
            ocr_f.write(f"--- 第 {page_num} 页 ---\n{doc.page_content}\n\n")
            if i < 3:
                preview_parts.append(
                    f"--- 第 {page_num} 页 ---\n{doc.page_content[:200]}..."
                )
        if len(raw_docs) > 3:
            preview_parts.append(f"\n... (共 {len(raw_docs)} 页/文档)")
    preview = "\n\n".join(preview_parts)

    # 文本分割
    pipeline = TextProcessingPipeline(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )
    chunks = pipeline.process(raw_docs)

    # 释放 raw_docs 引用，让 GC 可以回收
    raw_docs.clear()

    # 向量化入库
    _pipeline._vector_store_manager = VectorStoreManager(
        store_type=config.VECTOR_STORE_TYPE,
    )
    _pipeline._vector_store_manager.clear()
    _pipeline._vector_store_manager.add_documents(chunks)

    _pipeline._rag_chain = RAGChain(
        vector_store_manager=_pipeline._vector_store_manager,
        llm=_pipeline.llm,
    )

    _chat_history = []

    file_info = {
        "name": file_path.name,
        "format": suffix,
        "pages": len(raw_docs) if raw_docs else _count_ocr_pages(ocr_path),
        "chunks": len(chunks),
        "size_mb": round(file_size_mb, 2),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": str(file_path),
    }
    _processed_files.append(file_info)

    # 强制 GC 回收 OCR 过程中产生的临时对象
    del chunks
    gc.collect()

    logger.info(f"文件处理成功: {file_path.name}, {file_info['pages']} 页, {file_info['chunks']} 块")
    return file_info, preview


def _count_ocr_pages(ocr_path: Path) -> int:
    """从保存的 OCR 文件统计页数"""
    try:
        text = ocr_path.read_text(encoding="utf-8")
        return text.count("--- 第") or 1
    except Exception:
        return 1


def ask_question_impl(question: str) -> Dict[str, Any]:
    """执行 RAG 问答"""
    global _pipeline, _chat_history

    if _pipeline is None or not _pipeline.is_ready:
        raise RuntimeError("请先上传并处理文件")

    result = _pipeline.ask_with_history(question, _chat_history)

    _chat_history.append({"role": "user", "content": question})
    _chat_history.append({"role": "assistant", "content": result["answer"]})
    # 限制历史长度以防止内存无限增长 (保留最近 20 轮)
    if len(_chat_history) > 40:  # 20 pairs
        _chat_history = _chat_history[-40:]

    sources = []
    for src in result.get("sources", []):
        sources.append({
            "rank": src["rank"],
            "document": src["document"],
            "page": src["page"],
            "content_type": src.get("content_type", ""),
            "content": src["content"][:200],
        })

    return {"answer": result["answer"], "sources": sources}


def clear_chat_impl():
    global _chat_history
    _chat_history = []


def get_system_status_impl() -> Dict[str, Any]:
    global _pipeline, _processed_files

    def _mask_key(key: str) -> str:
        if not key or key == "not-needed":
            return ""
        if len(key) <= 8:
            return "*" * len(key)
        return key[:4] + "****" + key[-4:]

    status = {
        "embedding": {
            "model": config.EMBEDDING_MODEL_NAME,
            "api_base": config.EMBEDDING_API_BASE,
            "api_key": _mask_key(config.EMBEDDING_API_KEY),
        },
        "llm": {
            "model": config.LLM_MODEL_NAME,
            "api_base": config.LLM_API_BASE,
            "api_key": _mask_key(config.LLM_API_KEY),
        },
        "ocr": {
            "engine": config.OCR_ENGINE,
            "model": config.OCR_API_MODEL,
            "api_base": config.OCR_API_BASE,
            "api_key": _mask_key(config.OCR_API_KEY),
        },
        "vector_store": config.VECTOR_STORE_TYPE,
        "params": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
        },
        "document_count": 0,
        "files": _processed_files,
    }

    if _pipeline is not None:
        try:
            stats = _pipeline.stats
            status["document_count"] = stats.get("document_count", 0)
        except Exception:
            pass

    return status


def preload_ocr_engine():
    """启动时预热 OCR 引擎, 避免首次上传等待模型加载"""
    if config.OCR_ENGINE == "paddle":
        try:
            logger.info("预热 PaddleOCR-VL 引擎...")
            from ocr_loader import _get_ocr_vl_pipeline
            _get_ocr_vl_pipeline()
            logger.info("OCR 引擎预热完成 ✓")
        except Exception as e:
            logger.warning(f"OCR 引擎预热跳过: {e}")
    elif config.OCR_ENGINE == "api":
        logger.info(f"OCR API 模式, 跳过预热 (endpoint: {config.OCR_API_BASE})")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="PDF OCR 智能问答系统", version="2.0")

# Static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# ── Routes ──


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(150),
):
    """Upload and process multiple documents"""
    if not files or all(not f.filename for f in files):
        raise HTTPException(400, "No files provided")

    upload_dir = config.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_errors = []

    for file in files:
        if not file.filename:
            continue

        tmp_path = upload_dir / file.filename
        try:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            file_info, preview = process_file_impl(tmp_path, chunk_size, chunk_overlap)

            results.append({
                "success": True,
                "name": file_info["name"],
                "format": file_info["format"],
                "pages": file_info["pages"],
                "chunks": file_info["chunks"],
                "size_mb": file_info["size_mb"],
                "time": file_info["time"],
                "preview": preview,
                "message": "处理完成",
            })
        except ValueError as e:
            all_errors.append(f"{file.filename}: {e}")
        except Exception as e:
            logger.error(f"处理失败 {file.filename}: {e}")
            import traceback
            traceback.print_exc()
            all_errors.append(f"{file.filename}: {e}")

    if not results and all_errors:
        raise HTTPException(500, "; ".join(all_errors))

    _save_files_to_disk()

    return {
        "success": True,
        "results": results,
        "errors": all_errors,
        "total": len(results),
    }


@app.delete("/api/files/{index}")
async def delete_file(index: int):
    """Remove a processed file from the list by index"""
    global _processed_files
    if 0 <= index < len(_processed_files):
        removed = _processed_files.pop(index)
        _delete_ocr_text(removed["name"])
        _save_files_to_disk()
        logger.info(f"已移除文件: {removed['name']}")
        return {"success": True, "removed": removed["name"]}
    raise HTTPException(404, "File index not found")


@app.get("/api/preview/{index}")
async def get_preview(index: int):
    """Get full OCR text for a processed file (reads from disk)"""
    if 0 <= index < len(_processed_files):
        filename = _processed_files[index]["name"]
        text = _load_ocr_text(filename)
        if text:
            return {"success": True, "text": text, "index": index, "filename": filename}
        return {"success": False, "text": "", "message": "OCR text file not found on disk"}
    raise HTTPException(404, "File index out of range")


@app.get("/api/file/{index}")
async def get_original_file(index: int):
    """Serve the original uploaded file for preview"""
    if 0 <= index < len(_processed_files):
        filename = _processed_files[index]["name"]
        # 1) 尝试存储的路径
        file_path = _processed_files[index].get("path", "")
        if file_path and Path(file_path).exists():
            return FileResponse(file_path)
        # 2) 回退: 在 upload 目录中按文件名查找
        fallback = config.UPLOAD_DIR / filename
        if fallback.exists():
            return FileResponse(str(fallback))
        raise HTTPException(404, f"Original file not found: {filename}")
    raise HTTPException(404, f"File index {index} out of range (total: {len(_processed_files)})")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Ask a question about the processed document"""
    try:
        result = ask_question_impl(req.question)
        return ChatResponse(**result)
    except RuntimeError as e:
        return {"answer": str(e), "sources": []}
    except Exception as e:
        logger.error(f"问答失败: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": f"问答失败: {str(e)}", "sources": []}


@app.delete("/api/chat")
async def clear_chat():
    """Clear chat history"""
    clear_chat_impl()
    return {"success": True}


@app.get("/api/status")
async def get_status():
    """Get system status"""
    return get_system_status_impl()


# ── Config API ──

CONFIG_KEYS = {
    "EMBEDDING_API_BASE", "EMBEDDING_MODEL_NAME", "EMBEDDING_API_KEY",
    "LLM_API_BASE", "LLM_MODEL_NAME", "LLM_API_KEY",
    "OCR_API_BASE", "OCR_API_MODEL", "OCR_API_KEY", "OCR_ENGINE",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K",
}


def _update_env_file(updates: Dict[str, str]):
    """将配置变更写入 .env 文件"""
    env_path = config.BASE_DIR / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    for k, v in updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@app.get("/api/config")
async def get_config():
    """获取当前 API 配置"""
    return {
        "embedding": {
            "api_base": config.EMBEDDING_API_BASE,
            "model_name": config.EMBEDDING_MODEL_NAME,
            "api_key": config.EMBEDDING_API_KEY,
        },
        "llm": {
            "api_base": config.LLM_API_BASE,
            "model_name": config.LLM_MODEL_NAME,
            "api_key": config.LLM_API_KEY,
        },
        "ocr": {
            "engine": config.OCR_ENGINE,
            "api_base": config.OCR_API_BASE,
            "model_name": config.OCR_API_MODEL,
            "api_key": config.OCR_API_KEY,
        },
        "retrieval": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.RETRIEVAL_TOP_K,
        },
    }


@app.post("/api/config")
async def update_config(updates: Dict[str, str]):
    """更新 API 配置 (写入 .env 并即时生效)"""
    import os as _os

    applied = {}
    for key in updates:
        if key in CONFIG_KEYS:
            applied[key] = str(updates[key])
            _os.environ[key] = str(updates[key])

    if applied:
        _update_env_file(applied)

        # 重新加载 config 模块以生效
        import importlib
        importlib.reload(config)

        # 重置全局单例使新配置生效
        from embeddings import reset_embedding_model
        reset_embedding_model()

        logger.info(f"配置已更新: {list(applied.keys())}")

    return {"success": True, "updated": list(applied.keys())}


# ============================================================
# Main
# ============================================================

def main():
    import uvicorn

    logger.remove()
    logger.add(
        config.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        rotation="100 MB",
        retention="30 days",
        encoding="utf-8",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    logger.info("=" * 50)
    logger.info("  PDF OCR 智能问答系统 启动中...")
    logger.info("=" * 50)
    logger.info(f"  OCR: PaddleOCR-VL-1.5 ({config.OCR_VL_BACKEND})")
    logger.info(f"  嵌入: {config.EMBEDDING_MODEL_NAME} (API: {config.EMBEDDING_API_BASE})")
    logger.info(f"  LLM: {config.LLM_MODEL_NAME} (API: {config.LLM_API_BASE})")
    logger.info(f"  OCR: {config.OCR_ENGINE} ({config.OCR_API_BASE if config.OCR_ENGINE == 'api' else 'local'})")
    logger.info(f"  向量数据库: {config.VECTOR_STORE_TYPE}")
    logger.info(f"  支持格式: {sorted(config.SUPPORTED_FORMATS)}")

    # 从磁盘恢复已处理文件列表
    _load_files_from_disk()

    # 预热 OCR 引擎
    preload_ocr_engine()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
