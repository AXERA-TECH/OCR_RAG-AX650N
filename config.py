"""
============================================================
OCR RAG 智能问答系统 - 全局配置
============================================================

"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- 项目路径 ----
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OCR_OUTPUT_DIR = DATA_DIR / "ocr_output"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
LOG_DIR = DATA_DIR / "logs"

for d in [DATA_DIR, UPLOAD_DIR, OCR_OUTPUT_DIR, VECTOR_DB_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# PaddleOCR-VL-1.5 配置 
# ============================================================
# PaddleOCR-VL-1.5: 0.9B 视觉语言模型, OmniDocBench v1.5 94.5% 精度
# 支持: PDF / PNG / JPG / BMP / TIF
# OCR 引擎:
#   paddle       - PaddleOCR pipeline (默认, 版面分析 + 页面级解析, 推荐)
#   transformers - transformers v5 原生推理 (元素级识别, 轻量)
# PaddleOCR 后端 (仅 engine=paddle 时生效):
#   native          - 本地 PaddlePaddle 推理
#   vllm-server     - vLLM 服务端 (高吞吐)
#   llama-cpp-server - llama.cpp GGUF (边缘设备)

OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle")  # paddle / api

# OCR API 配置 (OCR_ENGINE=api, 通过 OpenAI 兼容 API 调用)
# vLLM 部署:
#   python -m vllm.entrypoints.openai.api_server \
#     --model PaddlePaddle/PaddleOCR-VL-1.5 --trust-remote-code --port 8002
OCR_API_BASE = os.getenv("OCR_API_BASE", "http://127.0.0.1:8002/v1")
OCR_API_KEY = os.getenv("OCR_API_KEY", "not-needed")
OCR_API_MODEL = os.getenv("OCR_API_MODEL", "PaddleOCR-VL-1.5")
OCR_TASK = os.getenv("OCR_TASK", "ocr")  # ocr / table / chart / formula / spotting / seal

OCR_VL_BACKEND = os.getenv("OCR_VL_BACKEND", "native")
OCR_VL_SERVER_URL = os.getenv("OCR_VL_SERVER_URL", "http://127.0.0.1:8080/v1")

OCR_USE_LAYOUT = os.getenv("OCR_USE_LAYOUT", "true").lower() == "true"
OCR_LAYOUT_THRESHOLD = float(os.getenv("OCR_LAYOUT_THRESHOLD", "0.5"))
OCR_USE_CHART = os.getenv("OCR_USE_CHART", "false").lower() == "true"

OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "4096"))
OCR_TEMPERATURE = float(os.getenv("OCR_TEMPERATURE", "0.0"))

PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "300"))

SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_FORMATS = {".pdf"} | SUPPORTED_IMAGE_FORMATS

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ============================================================
# 文本分割
# ============================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

# ============================================================
# Embedding API 配置 (OpenAI 兼容格式)
# ============================================================


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B"
)
EMBEDDING_API_BASE = os.getenv(
    "EMBEDDING_API_BASE", "http://127.0.0.1:8001/v1"
)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "not-needed")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

# ============================================================
# 向量数据库
# ============================================================
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_ocr_knowledge")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# ============================================================
# LLM API 配置 (OpenAI 兼容格式)
# ============================================================


LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ============================================================
# 系统 Prompt
# ============================================================
SYSTEM_PROMPT = """根据以下文档内容，简洁回答用户问题。只依据文档内容回答，不要编造。使用中文。"""

RAG_PROMPT_TEMPLATE = """{system_prompt}

## 参考文档内容:
{context}

## 用户问题:
{question}

## 回答:"""

# ============================================================
# 日志
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
