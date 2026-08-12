"""
============================================================
PaddleOCR-VL-1.5 文档加载器
============================================================
模型: PaddleOCR-VL-1.5 (0.9B 视觉语言模型, OmniDocBench v1.5 94.5% 精度)
支持格式: PDF / PNG / JPG / JPEG / BMP / TIF / TIFF

功能:
  1. 文档 (PDF/图片) → PaddleOCR-VL-1.5 端到端识别
  2. 输出 Markdown/JSON 结构化结果 (含版面/表格/公式/印章)
  3. 转换为 LangChain Document 对象
"""

import gc
import time
import warnings
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Any, Union
from dataclasses import dataclass, field

import fitz  # PyMuPDF: PDF 页面渲染和元数据提取
import numpy as np
from PIL import Image

from langchain_core.documents import Document

from loguru import logger

import config

warnings.filterwarnings("ignore")


# ============================================================
# PaddleOCR-VL-1.5 全局单例
# ============================================================

_ocr_vl_pipeline = None


def _get_ocr_vl_pipeline():
    """懒加载 PaddleOCR-VL-1.5 模型 (单例)"""
    global _ocr_vl_pipeline
    if _ocr_vl_pipeline is None:
        from paddleocr import PaddleOCRVL
        logger.info(
            f"正在初始化 PaddleOCR-VL-1.5 模型 "
            f"(backend={config.OCR_VL_BACKEND})..."
        )

        kwargs = dict(
            use_layout_detection=config.OCR_USE_LAYOUT,
            use_chart_recognition=config.OCR_USE_CHART,
            merge_layout_blocks=True,
            layout_threshold=config.OCR_LAYOUT_THRESHOLD,
        )

        if config.OCR_VL_BACKEND == "vllm-server":
            kwargs["vl_rec_backend"] = "vllm-server"
            kwargs["vl_rec_server_url"] = config.OCR_VL_SERVER_URL
        elif config.OCR_VL_BACKEND == "llama-cpp-server":
            kwargs["vl_rec_backend"] = "llama-cpp-server"
            kwargs["vl_rec_server_url"] = config.OCR_VL_SERVER_URL

        _ocr_vl_pipeline = PaddleOCRVL(**kwargs)
        logger.info("PaddleOCR-VL-1.5 模型初始化完成 ✓")
    return _ocr_vl_pipeline


# ============================================================
# 数据结构
# ============================================================

@dataclass
class OCRResult:
    """单页/单图 OCR 结果"""
    page_num: int = 0
    markdown_text: str = ""
    json_data: Optional[Dict[str, Any]] = None
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    images_in_page: List[Dict[str, Any]] = field(default_factory=list)
    layout_regions: List[Dict[str, Any]] = field(default_factory=list)
    ocr_time_ms: float = 0.0
    source_format: str = ""  # pdf / png / jpg / ...


# ============================================================
# PaddleOCR-VL-1.5 文本提取器
# ============================================================

class VLOCRExtractor:
    """使用 PaddleOCR-VL-1.5 从文档中提取结构化内容"""

    @staticmethod
    def extract(image_or_path: Union[str, Path, np.ndarray]) -> List[OCRResult]:
        """
        对单张图片或 PDF 执行 OCR 识别

        Args:
            image_or_path: 图片路径 / PDF路径 / numpy 数组

        Returns:
            OCRResult 列表 (PDF 为多页, 图片为单页)
        """
        pipeline = _get_ocr_vl_pipeline()
        start_time = time.time()

        logger.info("PaddleOCR-VL 正在推理中 (首次调用较慢, CPU 约 30-60s/页) ...")
        raw_output = pipeline.predict(image_or_path)
        logger.info(f"推理完成, 耗时 {time.time() - start_time:.1f}s")
        results = []
        for i, res in enumerate(raw_output):
            page_result = OCRResult(
                page_num=i + 1,
                ocr_time_ms=(time.time() - start_time) * 1000 / len(raw_output),
            )

            # 尝试获取 structured JSON
            try:
                json_data = res.json
                if json_data:
                    page_result.json_data = json_data
                    # 解析结构化内容
                    page_result.text_blocks = VLOCRExtractor._parse_text_blocks(json_data)
                    page_result.tables = VLOCRExtractor._parse_tables(json_data)
                    page_result.formulas = VLOCRExtractor._parse_formulas(json_data)
            except Exception as e:
                logger.debug(f"JSON 解析跳过: {e}")

            # 获取 Markdown 文本
            try:
                md = res.markdown
                if isinstance(md, dict):
                    page_result.markdown_text = md.get("text", "") or ""
                elif isinstance(md, str):
                    page_result.markdown_text = md
                else:
                    page_result.markdown_text = str(md) if md else ""
            except Exception:
                page_result.markdown_text = ""

            # 回退: markdown 为空时从 JSON blocks 构建文本
            if not page_result.markdown_text and page_result.json_data:
                page_result.markdown_text = VLOCRExtractor._build_text_from_blocks(
                    page_result.json_data
                )

            results.append(page_result)

        return results

    @staticmethod
    def extract_text(image_or_path: Union[str, Path, np.ndarray]) -> str:
        """便捷方法: 只返回纯文本 (合并所有页)"""
        results = VLOCRExtractor.extract(image_or_path)
        return "\n\n".join(r.markdown_text for r in results if r.markdown_text)

    @staticmethod
    def extract_to_markdown(image_or_path: Union[str, Path, np.ndarray]) -> str:
        """返回完整的 Markdown 格式文本"""
        return VLOCRExtractor.extract_text(image_or_path)

    @staticmethod
    def extract_to_json(
        image_or_path: Union[str, Path, np.ndarray],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """返回结构化 JSON 或保存到文件"""
        results = VLOCRExtractor.extract(image_or_path)
        output = {
            "pages": [],
            "total_pages": len(results),
        }
        for r in results:
            page_data = {
                "page_num": r.page_num,
                "markdown": r.markdown_text,
                "json": r.json_data,
                "tables": r.tables,
                "formulas": r.formulas,
            }
            output["pages"].append(page_data)

        if save_path:
            import json
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"OCR 结果已保存: {save_path}")

        return output

    # ---- 结构化解析辅助 ----

    @staticmethod
    def _get_parsing_list(json_data: Dict) -> List[Dict]:
        """从 PaddleOCR-VL JSON 中提取 parsing_res_list"""
        res = json_data.get("res", json_data)
        return res.get("parsing_res_list", [])

    @staticmethod
    def _parse_text_blocks(json_data: Dict) -> List[Dict[str, Any]]:
        """从 parsing_res_list 中提取文本块"""
        blocks = []
        for item in VLOCRExtractor._get_parsing_list(json_data):
            label = item.get("block_label", "")
            content = item.get("block_content", "")
            bbox = item.get("block_bbox", [])
            if content and label not in ("image",):
                blocks.append({
                    "type": label,
                    "text": content,
                    "bbox": bbox,
                })
        return blocks

    @staticmethod
    def _parse_tables(json_data: Dict) -> List[Dict[str, Any]]:
        """从 parsing_res_list 中提取表格"""
        tables = []
        for item in VLOCRExtractor._get_parsing_list(json_data):
            if item.get("block_label") == "table":
                tables.append({
                    "text": item.get("block_content", ""),
                    "html": item.get("block_html", ""),
                    "markdown": item.get("block_markdown", ""),
                    "bbox": item.get("block_bbox", []),
                })
        return tables

    @staticmethod
    def _parse_formulas(json_data: Dict) -> List[Dict[str, Any]]:
        """从 parsing_res_list 中提取公式"""
        formulas = []
        for item in VLOCRExtractor._get_parsing_list(json_data):
            if item.get("block_label") == "formula":
                formulas.append({
                    "latex": item.get("block_latex", ""),
                    "text": item.get("block_content", ""),
                    "bbox": item.get("block_bbox", []),
                })
        return formulas

    @staticmethod
    def _build_text_from_blocks(json_data: Dict) -> str:
        """从 parsing_res_list 构建纯文本"""
        lines = []
        for item in VLOCRExtractor._get_parsing_list(json_data):
            label = item.get("block_label", "")
            content = item.get("block_content", "")
            if not content:
                continue
            if label == "table":
                lines.append(f"[表格] {content}")
            elif label == "formula":
                lines.append(f"[公式] {content}")
            elif label in ("paragraph_title", "header"):
                lines.append(f"## {content}")
            elif label == "image":
                continue  # 跳过纯图片块
            else:
                lines.append(content)
        return "\n\n".join(lines)


# ============================================================
# OCR API 提取器 (OpenAI 兼容格式, 无需本地推理)
# ============================================================

_ocr_api_client = None


def _get_ocr_api_client():
    """懒加载 OCR API 客户端"""
    global _ocr_api_client
    if _ocr_api_client is None:
        from openai import OpenAI
        _ocr_api_client = OpenAI(
            api_key=config.OCR_API_KEY,
            base_url=config.OCR_API_BASE,
        )
        logger.info(
            f"OCR API 连接: model={config.OCR_API_MODEL}, "
            f"base_url={config.OCR_API_BASE}"
        )
    return _ocr_api_client


class OCRApiExtractor:
    """
    基于 OpenAI 兼容 API 的 PaddleOCR-VL-1.5 提取器

    通过 vLLM 或其他 OpenAI 兼容服务调用, 无需本地 GPU 推理。

    支持任务: ocr / table / formula / chart / spotting / seal
    """

    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }

    @staticmethod
    def extract(
        image_or_path: Union[str, Path, np.ndarray],
        task: Optional[str] = None,
        max_new_tokens: int = 2048,
    ) -> List[OCRResult]:
        """
        通过 API 执行 OCR 识别

        Args:
            image_or_path: 图片路径 / numpy 数组
            task: 任务类型
            max_new_tokens: 最大生成 token 数

        Returns:
            OCRResult 列表
        """
        import base64
        import io

        task = task or config.OCR_TASK
        client = _get_ocr_api_client()

        start_time = time.time()
        logger.info(f"OCR API 推理中 (task={task}) ...")

        # 图片 → base64 data URL
        if isinstance(image_or_path, (str, Path)):
            with open(image_or_path, "rb") as f:
                img_bytes = f.read()
        elif isinstance(image_or_path, np.ndarray):
            img = Image.fromarray(image_or_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        else:
            img_bytes = image_or_path

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64}"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": OCRApiExtractor.PROMPTS[task]},
            ],
        }]

        response = client.chat.completions.create(
            model=config.OCR_API_MODEL,
            messages=messages,
            max_tokens=max_new_tokens,
        )

        result_text = response.choices[0].message.content.strip()
        elapsed = (time.time() - start_time) * 1000

        result = OCRResult(
            page_num=1,
            markdown_text=result_text,
            ocr_time_ms=elapsed,
            source_format="image",
            text_blocks=[{"type": task, "text": result_text, "bbox": []}],
        )

        logger.info(f"OCR API 完成, 耗时 {elapsed:.0f}ms, {len(result_text)} 字符")
        return [result]

    @staticmethod
    def extract_text(
        image_or_path: Union[str, Path, np.ndarray],
        task: Optional[str] = None,
    ) -> str:
        """便捷方法: 只返回识别文本"""
        results = OCRApiExtractor.extract(image_or_path, task=task)
        return "\n".join(r.markdown_text for r in results)


# ============================================================
# 统一提取器入口
# ============================================================

def _extract_ocr(image_or_path: Union[str, Path, np.ndarray]) -> List[OCRResult]:
    """根据配置选择 OCR 引擎并执行识别"""
    if config.OCR_ENGINE == "api":
        return OCRApiExtractor.extract(image_or_path)
    else:
        return VLOCRExtractor.extract(image_or_path)


# ============================================================
# PDF 工具
# ============================================================

class PDFUtils:
    """PDF 处理工具: 渲染和元数据提取"""

    @staticmethod
    def render_page_to_image(page: fitz.Page, dpi: int = 300) -> np.ndarray:
        """将 PyMuPDF 页面渲染为 numpy 图片数组 (RGB)"""
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img)

    @staticmethod
    def get_page_count(pdf_path: Path) -> int:
        """获取 PDF 页数"""
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count

    @staticmethod
    def is_scanned_pdf(pdf_path: Path, sample_pages: int = 3) -> bool:
        """
        检测 PDF 是否为扫描版 (图片型 PDF)

        通过检查前几页是否包含可提取的文本层来判断
        """
        doc = fitz.open(str(pdf_path))
        text_chars = 0
        pages_to_check = min(sample_pages, len(doc))

        for i in range(pages_to_check):
            text_chars += len(doc[i].get_text().strip())

        doc.close()
        # 如果前几页几乎没有文本, 认为是扫描版
        return text_chars < 100 * pages_to_check

    @staticmethod
    def extract_text_layer(pdf_path: Path) -> List[Dict[str, Any]]:
        """
        提取 PDF 内嵌文本层 (非 OCR, 用于数字原生 PDF)
        返回每页的文本和元数据
        """
        doc = fitz.open(str(pdf_path))
        pages = []

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "char_count": len(text),
                    "has_text_layer": True,
                })

        doc.close()
        return pages


# ============================================================
# LangChain PaddleOCR-VL-1.5 文档加载器
# ============================================================

class PaddleOCRLoader:
    """
    LangChain 兼容的 PaddleOCR-VL-1.5 文档加载器

    支持格式: PDF / PNG / JPG / JPEG / BMP / TIF / TIFF

    用法:
        # 加载 PDF
        loader = PaddleOCRLoader("document.pdf")
        documents = loader.load()

        # 加载图片
        loader = PaddleOCRLoader("scan.png")
        documents = loader.load()

        # 延迟加载 (大文件推荐)
        for doc in loader.lazy_load():
            process(doc)
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        dpi: int = config.PDF_RENDER_DPI,
        verbose: bool = True,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        self.suffix = self.file_path.suffix.lower()
        if self.suffix not in config.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {self.suffix}. "
                f"支持: {config.SUPPORTED_FORMATS}"
            )

        self.dpi = dpi
        self.verbose = verbose
        self._doc_name = self.file_path.stem
        self._is_pdf = (self.suffix == ".pdf")

    def load(self) -> List[Document]:
        """完整加载文档, 返回 LangChain Document 列表"""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """逐页延迟加载"""

        if self._is_pdf:
            yield from self._load_pdf()
        else:
            yield from self._load_image()

    def _load_pdf(self) -> Iterator[Document]:
        """加载 PDF 文件"""
        total_start = time.time()
        page_count = PDFUtils.get_page_count(self.file_path)
        self._log(f"开始处理 PDF: {self.file_path.name} ({page_count} 页, DPI={self.dpi})")

        pdf_doc = fitz.open(str(self.file_path))

        for page_idx in range(page_count):
            page_start = time.time()

            # 渲染页面为高清图片
            page = pdf_doc[page_idx]
            image = PDFUtils.render_page_to_image(page, dpi=self.dpi)

            # PaddleOCR-VL-1.5 识别
            results = _extract_ocr(image)

            # 释放页面图像内存 (高DPI图片可能占用数百MB)
            del image

            ocr_time = (time.time() - page_start) * 1000

            for ocr_result in results:
                ocr_result.page_num = page_idx + 1
                ocr_result.source_format = "pdf"

                text = ocr_result.markdown_text
                if not text and ocr_result.json_data:
                    text = self._extract_text_from_json(ocr_result.json_data)

                if isinstance(text, dict):
                    text = text.get("text", "") or ""
                if not text or not str(text).strip():
                    self._log(f"  第 {page_idx + 1} 页: 未检测到文本")
                    continue

                # 构建元数据
                metadata = {
                    "source": str(self.file_path),
                    "document_name": self._doc_name,
                    "page": page_idx + 1,
                    "total_pages": page_count,
                    "ocr_text_length": len(text),
                    "ocr_time_ms": round(ocr_time, 1),
                    "dpi": self.dpi,
                    "source_format": "pdf",
                    "tables_count": len(ocr_result.tables),
                    "formulas_count": len(ocr_result.formulas),
                    "text_blocks_count": len(ocr_result.text_blocks),
                }

                # 附加表格/公式数据
                if ocr_result.tables:
                    metadata["tables_markdown"] = [
                        t.get("markdown", "") for t in ocr_result.tables
                    ]
                    metadata["tables_html"] = [
                        t.get("html", "") for t in ocr_result.tables
                    ]
                if ocr_result.formulas:
                    metadata["formulas_latex"] = [
                        f.get("latex", "") for f in ocr_result.formulas
                    ]

                doc = Document(page_content=text, metadata=metadata)

                self._log(
                    f"  第 {page_idx + 1}/{page_count} 页: "
                    f"{len(text)} 字符, "
                    f"表格={metadata['tables_count']}, "
                    f"公式={metadata['formulas_count']}, "
                    f"耗时 {ocr_time:.0f}ms"
                )

                yield doc

        pdf_doc.close()
        gc.collect()  # 强制回收页面渲染残留内存
        self._log(f"PDF 处理完成, 总耗时 {time.time() - total_start:.1f}s")

    def _load_image(self) -> Iterator[Document]:
        """加载单张图片"""
        total_start = time.time()
        self._log(f"开始处理图片: {self.file_path.name}")

        # 验证图片可读
        try:
            img = Image.open(self.file_path)
            img.verify()
            img = Image.open(self.file_path)  # verify 后需重新打开
        except Exception as e:
            raise ValueError(f"无法读取图片文件: {e}")

        # PaddleOCR-VL-1.5 可以直接接受图片路径
        results = _extract_ocr(str(self.file_path))
        ocr_time = (time.time() - total_start) * 1000

        for ocr_result in results:
            ocr_result.source_format = self.suffix.lstrip(".")
            # print("ocr_result: ",ocr_result)
            text = ocr_result.markdown_text

            if not text and ocr_result.json_data:
                text = self._extract_text_from_json(ocr_result.json_data)

            if isinstance(text, dict):
                text = text.get("text", "") or ""
            if not text or not str(text).strip():
                self._log("  未检测到文本")
                continue

            metadata = {
                "source": str(self.file_path),
                "document_name": self._doc_name,
                "page": 1,
                "total_pages": 1,
                "ocr_text_length": len(text),
                "ocr_time_ms": round(ocr_time, 1),
                "dpi": self.dpi,
                "source_format": self.suffix.lstrip("."),
                "image_width": img.width,
                "image_height": img.height,
                "tables_count": len(ocr_result.tables),
                "formulas_count": len(ocr_result.formulas),
                "text_blocks_count": len(ocr_result.text_blocks),
            }

            if ocr_result.tables:
                metadata["tables_markdown"] = [
                    t.get("markdown", "") for t in ocr_result.tables
                ]
                metadata["tables_html"] = [
                    t.get("html", "") for t in ocr_result.tables
                ]
            if ocr_result.formulas:
                metadata["formulas_latex"] = [
                    f.get("latex", "") for f in ocr_result.formulas
                ]

            doc = Document(page_content=text, metadata=metadata)
            yield doc

        self._log(f"图片处理完成, 耗时 {time.time() - total_start:.1f}s")

    def load_with_ocr_results(self) -> List[OCRResult]:
        """返回 OCRResult 对象列表 (包含更丰富的结构化信息)"""
        if self._is_pdf:
            pdf_doc = fitz.open(str(self.file_path))
            all_results = []
            for page_idx in range(len(pdf_doc)):
                page = pdf_doc[page_idx]
                image = PDFUtils.render_page_to_image(page, dpi=self.dpi)
                results = _extract_ocr(image)
                for r in results:
                    r.page_num = page_idx + 1
                    r.source_format = "pdf"
                all_results.extend(results)
            pdf_doc.close()
            return all_results
        else:
            results = _extract_ocr(str(self.file_path))
            for r in results:
                r.source_format = self.suffix.lstrip(".")
            return results

    @staticmethod
    def _extract_text_from_json(json_data: Dict) -> str:
        """从 PaddleOCR-VL JSON 结构中提取所有文本"""
        return VLOCRExtractor._build_text_from_blocks(json_data)

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)


# ============================================================
# 批量加载器
# ============================================================

class PaddleOCRDirectoryLoader:
    """批量加载目录下的所有支持的文档文件"""

    def __init__(
        self,
        directory: Union[str, Path],
        glob_patterns: Optional[List[str]] = None,
        **loader_kwargs,
    ):
        self.directory = Path(directory)
        self.glob_patterns = glob_patterns or [
            "**/*.pdf", "**/*.png", "**/*.jpg", "**/*.jpeg",
            "**/*.bmp", "**/*.tif", "**/*.tiff",
        ]
        self.loader_kwargs = loader_kwargs

    def load(self) -> List[Document]:
        """加载目录下所有支持的文档"""
        all_docs = []
        files = []
        for pattern in self.glob_patterns:
            files.extend(self.directory.glob(pattern))
        files = sorted(set(files))

        if not files:
            logger.warning(f"目录 {self.directory} 中未找到支持的文档文件")
            return all_docs

        logger.info(f"在 {self.directory} 中找到 {len(files)} 个文件")

        for file_path in files:
            try:
                loader = PaddleOCRLoader(file_path, **self.loader_kwargs)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"  ✓ {file_path.name}: {len(docs)} 页/块")
            except Exception as e:
                logger.error(f"  ✗ {file_path.name}: {e}")

        logger.info(f"批量加载完成, 共 {len(all_docs)} 个文档块")
        return all_docs

    def lazy_load(self) -> Iterator[Document]:
        """延迟加载"""
        files = []
        for pattern in self.glob_patterns:
            files.extend(self.directory.glob(pattern))
        files = sorted(set(files))

        for file_path in files:
            try:
                loader = PaddleOCRLoader(file_path, **self.loader_kwargs)
                yield from loader.lazy_load()
            except Exception as e:
                logger.error(f"加载失败 {file_path.name}: {e}")


# ============================================================
# 便捷函数
# ============================================================

def load_document(file_path: Union[str, Path], **kwargs) -> List[Document]:
    """便捷函数: 加载单个文档 (自动识别格式)"""
    loader = PaddleOCRLoader(file_path, **kwargs)
    return loader.load()


def load_directory(directory: Union[str, Path], **kwargs) -> List[Document]:
    """便捷函数: 加载目录下所有文档"""
    loader = PaddleOCRDirectoryLoader(directory, **kwargs)
    return loader.load()


def ocr_to_markdown(file_path: Union[str, Path]) -> str:
    """便捷函数: OCR 识别并返回 Markdown"""
    return VLOCRExtractor.extract_to_markdown(file_path)


def ocr_to_json(file_path: Union[str, Path], save_path: Optional[str] = None) -> Dict:
    """便捷函数: OCR 识别并返回 JSON"""
    return VLOCRExtractor.extract_to_json(file_path, save_path)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"用法: python {__file__} <file_path> [--json] [--md]")
        print(f"支持格式: {config.SUPPORTED_FORMATS}")
        sys.exit(1)

    file_path = sys.argv[1]
    output_mode = "doc"  # doc / json / md
    if "--json" in sys.argv:
        output_mode = "json"
    elif "--md" in sys.argv:
        output_mode = "md"

    loader = PaddleOCRLoader(file_path, verbose=True)

    if output_mode == "json":
        result = ocr_to_json(file_path)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2)[:5000])
    elif output_mode == "md":
        md = ocr_to_markdown(file_path)
        print(md[:5000])
    else:
        documents = loader.load()
        print(f"\n{'='*60}")
        print(f"共加载 {len(documents)} 页/文档")
        print(f"{'='*60}")
        for i, doc in enumerate(documents):
            print(f"\n--- 第 {doc.metadata.get('page', '?')} 页 "
                  f"({len(doc.page_content)} 字符) ---")
            print(doc.page_content[:500])
            if len(doc.page_content) > 500:
                print("...")
            print(f"  元数据: source={doc.metadata.get('document_name')}, "
                  f"tables={doc.metadata.get('tables_count', 0)}, "
                  f"formulas={doc.metadata.get('formulas_count', 0)}")
