"""
============================================================
文本处理模块: Markdown 清洗 + 智能分割 (Chunking)
============================================================
适配 PaddleOCR-VL-1.5 输出的 Markdown 格式文本

功能:
  1. Markdown 文本清洗 (保留表格/公式结构)
  2. 基于 LangChain 的语义感知分割
  3. 表格/公式专项处理
"""

import re
from typing import List, Optional, Callable

from langchain_core.documents import Document

from loguru import logger

import config


# ============================================================
# 内置递归文本分割器 (替代 langchain_text_splitters)
# ============================================================
# 避免 langchain_text_splitters → sentence_transformers → transformers
# 的传递依赖链在部分环境中导致的兼容性问题


class RecursiveCharacterTextSplitter:
    """
    递归字符文本分割器

    与 langchain_text_splitters.RecursiveCharacterTextSplitter 接口兼容,
    按分隔符优先级逐级分割, 保持语义完整性。
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
        add_start_index: bool = True,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        self.add_start_index = add_start_index
        self.length_function = length_function
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割 Document 列表"""
        chunks = []
        for doc in documents:
            doc_chunks = self.split_text(doc.page_content, doc.metadata)
            chunks.extend(doc_chunks)
        return chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """分割单个文本, 返回 Document 列表"""
        metadata = metadata or {}
        splits = self._split(text, self.separators)
        chunks = self._merge(splits)

        docs = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {**metadata}
            if self.add_start_index:
                chunk_meta["start_index"] = text.find(chunk) if chunk in text else 0
            docs.append(Document(page_content=chunk, metadata=chunk_meta))
        return docs

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """从文本列表创建 Document 列表"""
        metadatas = metadatas or [{}] * len(texts)
        docs = []
        for text, meta in zip(texts, metadatas):
            docs.extend(self.split_text(text, meta))
        return docs

    def _split(self, text: str, separators: List[str]) -> List[str]:
        """递归分割"""
        # 使用最合适的分隔符
        sep = separators[-1]  # 默认用最后一个 (空字符串, 按字符分割)
        for s in separators:
            if s == "":
                sep = s
                break
            if s in text:
                sep = s
                break

        # 按分隔符分割
        if sep == "":
            # 按字符分割
            splits = list(text)
        else:
            if self.keep_separator:
                # 保留分隔符在片段末尾
                parts = text.split(sep)
                splits = []
                for i, part in enumerate(parts):
                    if i > 0:
                        splits.append(sep + part)
                    else:
                        splits.append(part)
            else:
                splits = text.split(sep)

        # 去除空白并过滤空字符串
        if self.strip_whitespace:
            splits = [s.strip() for s in splits]
        splits = [s for s in splits if s]

        # 递归处理超长片段
        final_splits = []
        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                final_splits.append(split)
            else:
                # 片段仍超长, 用下一级分隔符递归分割
                if len(separators) > 1:
                    next_seps = separators[separators.index(sep) + 1 :]
                    final_splits.extend(self._split(split, next_seps))
                else:
                    # 无法再分, 强制按字符切分
                    forced = self._force_split(split)
                    final_splits.extend(forced)

        return final_splits

    def _force_split(self, text: str) -> List[str]:
        """强制按字符数切分 (兜底)"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if self.strip_whitespace:
                chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _merge(self, splits: List[str]) -> List[str]:
        """合并短片段为 chunk_size 大小的块"""
        if not splits:
            return []

        chunks = []
        current = ""
        current_len = 0

        for split in splits:
            split_len = self.length_function(split)

            if current_len + split_len <= self.chunk_size:
                if current:
                    current += "\n\n" + split
                    current_len += 2 + split_len
                else:
                    current = split
                    current_len = split_len
            else:
                if current:
                    chunks.append(current)
                # 重叠: 保留前一块的尾部
                if self.chunk_overlap > 0 and current:
                    overlap_text = current[-self.chunk_overlap:]
                    current = overlap_text + "\n\n" + split
                    current_len = self.length_function(current)
                else:
                    current = split
                    current_len = split_len

        if current:
            chunks.append(current)

        return chunks


# ============================================================
# Markdown 文本清洗器
# ============================================================

class MarkdownTextCleaner:
    """PaddleOCR-VL-1.5 Markdown 输出清洗"""

    @staticmethod
    def clean(text: str, preserve_structure: bool = True) -> str:
        """
        清洗 Markdown 文本
        - 保留表格 (|...|) 和公式 ($...$ / $$...$$)
        - 规范化空白和换行
        - 移除 OCR 残留噪声
        """
        if not text:
            return ""

        cleaned = text.strip()

        # 移除控制字符 (保留换行和制表符)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)

        # 统一换行符
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')

        # 规范化空白 (但不影响表格结构)
        if preserve_structure:
            # 保护表格行和代码块
            lines = cleaned.split('\n')
            cleaned_lines = []
            in_table = False
            in_code = False

            for line in lines:
                # 检测 Markdown 表格
                if line.strip().startswith('|') and '|' in line.strip()[1:]:
                    in_table = True
                    cleaned_lines.append(line.rstrip())
                elif in_table and re.match(r'^[\s\|:\-]+$', line):
                    # 表格分隔行
                    cleaned_lines.append(line.rstrip())
                elif in_table and not line.strip().startswith('|'):
                    in_table = False
                    if line.strip():
                        cleaned_lines.append(line.strip())
                    elif cleaned_lines and cleaned_lines[-1] != '':
                        cleaned_lines.append('')
                elif line.strip().startswith('```'):
                    in_code = not in_code
                    cleaned_lines.append(line.rstrip())
                elif in_code:
                    cleaned_lines.append(line.rstrip())
                else:
                    # 普通行: 去除首尾空白, 合并多个空格
                    stripped = re.sub(r' +', ' ', line.strip())
                    if stripped:
                        cleaned_lines.append(stripped)
                    elif cleaned_lines and cleaned_lines[-1] != '':
                        cleaned_lines.append('')

            cleaned = '\n'.join(cleaned_lines)
        else:
            cleaned = re.sub(r' +', ' ', cleaned)
            cleaned = re.sub(r' *\n *', '\n', cleaned)

        # 压缩过多连续空行
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)

        return cleaned.strip()

    @staticmethod
    def clean_documents(documents: List[Document]) -> List[Document]:
        """批量清洗 Document 列表"""
        cleaned_docs = []
        for doc in documents:
            original_len = len(doc.page_content)
            cleaned_text = MarkdownTextCleaner.clean(doc.page_content)
            cleaned_len = len(cleaned_text)

            if cleaned_text:
                cleaned_doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        **doc.metadata,
                        "cleaned": True,
                        "original_length": original_len,
                        "cleaned_length": cleaned_len,
                    },
                )
                cleaned_docs.append(cleaned_doc)
            else:
                logger.debug(
                    f"页面 {doc.metadata.get('page', '?')} 清洗后为空, 已跳过"
                )

        logger.info(
            f"文本清洗: {len(documents)} → {len(cleaned_docs)} 个文档 "
            f"(移除 {len(documents) - len(cleaned_docs)} 个空白页)"
        )
        return cleaned_docs

    @staticmethod
    def extract_tables_as_chunks(documents: List[Document]) -> List[Document]:
        """
        将 Markdown 表格提取为独立的文本块
        PaddleOCR-VL-1.5 已输出标准 Markdown 表格格式
        """
        table_docs = []
        for doc in documents:
            tables_html = doc.metadata.get("tables_html", [])
            tables_md = doc.metadata.get("tables_markdown", [])

            for i, (html, md) in enumerate(
                zip(tables_html, tables_md or [""] * len(tables_html))
            ):
                content = md or html
                if content.strip():
                    table_doc = Document(
                        page_content=f"[表格数据]\n{content}",
                        metadata={
                            **doc.metadata,
                            "content_type": "table",
                            "table_index": i,
                            "table_html": html,
                            "table_markdown": md,
                        },
                    )
                    table_docs.append(table_doc)

        if table_docs:
            logger.info(f"提取了 {len(table_docs)} 个表格块")
        return table_docs

    @staticmethod
    def extract_formulas_as_chunks(documents: List[Document]) -> List[Document]:
        """将 LaTeX 公式提取为独立块"""
        formula_docs = []
        for doc in documents:
            formulas_latex = doc.metadata.get("formulas_latex", [])
            for i, latex in enumerate(formulas_latex):
                if latex.strip():
                    formula_doc = Document(
                        page_content=f"[公式]\n$${latex}$$",
                        metadata={
                            **doc.metadata,
                            "content_type": "formula",
                            "formula_index": i,
                            "formula_latex": latex,
                        },
                    )
                    formula_docs.append(formula_doc)

        if formula_docs:
            logger.info(f"提取了 {len(formula_docs)} 个公式块")
        return formula_docs


# ============================================================
# 智能文本分割器
# ============================================================

class DocumentSplitter:
    """
    文档智能分割器

    针对 PaddleOCR-VL-1.5 的 Markdown 输出优化:
      - 在 Markdown 标题处分段
      - 保护表格完整性
      - 保护代码块完整性
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or config.SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            add_start_index=True,
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        if not documents:
            return []

        chunks = self._splitter.split_documents(documents)
        logger.info(
            f"文本分割: {len(documents)} → {len(chunks)} 个文本块 "
            f"(块大小={self.chunk_size}, 重叠={self.chunk_overlap})"
        )
        return chunks

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """分割单个文本"""
        return self._splitter.create_documents(
            [text], metadatas=[metadata or {}]
        )


class MarkdownAwareSplitter:
    """
    Markdown 感知分割器

    在 Markdown 结构边界处分割:
      - ## 标题 → 新段
      - 表格 → 保持完整
      - 代码块 → 保持完整
    """

    def __init__(
        self,
        target_chunk_size: int = config.CHUNK_SIZE,
        min_chunk_size: int = 100,
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """基于 Markdown 结构分割"""
        all_chunks = []

        for doc in documents:
            sections = self._split_by_headers(doc.page_content)
            chunks = self._merge_sections(
                sections, doc.metadata, self.target_chunk_size, self.min_chunk_size
            )
            all_chunks.extend(chunks)

        logger.info(
            f"Markdown 感知分割: {len(documents)} → {len(all_chunks)} 个文本块"
        )
        return all_chunks

    @staticmethod
    def _split_by_headers(text: str) -> List[str]:
        """
        按 Markdown 标题 (# ## ###) 和段落分割
        保护表格和代码块完整性
        """
        # 先在代码块和表格处做保护标记
        protected = []
        protection_map = {}

        def protect(match):
            key = f"__PROTECTED_{len(protected)}__"
            protected.append(match.group(0))
            protection_map[key] = match.group(0)
            return key

        # 保护代码块
        text = re.sub(r'```[\s\S]*?```', protect, text)
        # 保护表格 (连续的 | 行)
        text = re.sub(
            r'(?:^\|.+\|\n)+(?:^\|[\s\-:]+\|\n)?(?:^\|.+\|\n?)+',
            protect,
            text,
            flags=re.MULTILINE,
        )

        # 按 Markdown 标题分割
        raw_sections = re.split(r'\n(?=#{1,3}\s)', text)

        # 恢复保护的内容
        sections = []
        for section in raw_sections:
            for key, original in protection_map.items():
                section = section.replace(key, original)
            section = section.strip()
            if section:
                sections.append(section)

        return sections

    @staticmethod
    def _merge_sections(
        sections: List[str],
        base_metadata: dict,
        target_size: int,
        min_size: int,
    ) -> List[Document]:
        """将段落合并为目标大小的块"""
        chunks = []
        current = ""
        start_idx = 0

        for i, section in enumerate(sections):
            if not current:
                current = section
                start_idx = i
            elif len(current) + len(section) + 2 <= target_size:
                current += "\n\n" + section
            else:
                if len(current) >= min_size:
                    meta = {
                        **base_metadata,
                        "chunk_sections": f"{start_idx}-{i - 1}",
                        "chunk_type": "markdown_semantic",
                    }
                    chunks.append(Document(page_content=current, metadata=meta))
                current = section
                start_idx = i

        # 最后一个块
        if current and len(current) >= min_size:
            meta = {
                **base_metadata,
                "chunk_sections": f"{start_idx}-{len(sections) - 1}",
                "chunk_type": "markdown_semantic",
            }
            chunks.append(Document(page_content=current, metadata=meta))
        elif current and chunks:
            chunks[-1].page_content += "\n\n" + current

        return chunks


# ============================================================
# 完整处理流水线
# ============================================================

class TextProcessingPipeline:
    """
    文本处理流水线

    用法:
        pipeline = TextProcessingPipeline()
        chunks = pipeline.process(raw_documents)
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        split_method: str = "recursive",
        extract_tables: bool = True,
        extract_formulas: bool = False,
        clean_text: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.extract_tables = extract_tables
        self.extract_formulas = extract_formulas
        self.clean_text = clean_text

        if split_method == "markdown":
            self.splitter = MarkdownAwareSplitter(
                target_chunk_size=chunk_size,
                min_chunk_size=max(50, chunk_size // 4),
            )
        else:
            self.splitter = DocumentSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    def process(self, documents: List[Document]) -> List[Document]:
        """
        完整处理流水线:
          原始文档 → 清洗 → 提取表格/公式 → 分割 → 最终块
        """
        docs = list(documents)
        logger.info(f"文本处理流水线启动: {len(docs)} 个原始文档")

        # Step 1: 文本清洗
        if self.clean_text:
            docs = MarkdownTextCleaner.clean_documents(docs)

        # Step 2: 提取表格和公式为独立块
        extra_docs = []
        if self.extract_tables:
            extra_docs.extend(MarkdownTextCleaner.extract_tables_as_chunks(docs))
        if self.extract_formulas:
            extra_docs.extend(MarkdownTextCleaner.extract_formulas_as_chunks(docs))

        # Step 3: 分割
        chunks = self.splitter.split_documents(docs)

        # Step 4: 合并特殊内容块
        if extra_docs:
            chunks.extend(extra_docs)
            logger.info(f"合并特殊块后总计: {len(chunks)} 个文本块")

        # Step 5: 添加块 ID
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{i:06d}"

        logger.info(f"文本处理完成: {len(documents)} 页 → {len(chunks)} 个文本块")
        return chunks


# ============================================================
# 便捷函数
# ============================================================

def process_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    **kwargs,
) -> List[Document]:
    """便捷函数: 一键文本处理"""
    pipeline = TextProcessingPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    return pipeline.process(documents)
