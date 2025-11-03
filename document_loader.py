from docx import Document
import os
from typing import TypedDict


class DocxChunk(TypedDict):
    """文档数据块类型定义，包含文本内容和元数据"""
    text: str
    metadata: dict


def load_and_chunk_docx(file_path: str) -> list[DocxChunk]:
    """
    加载并分块处理 .docx 文件
    使用智能分块策略：将标题和后续内容合并为一个逻辑块
    
    Args:
        file_path: Word 文档的文件路径
        
    Returns:
        包含文本和元数据的数据块列表
    """
    # 提取文件名
    file_name = os.path.basename(file_path)
    
    # 打开 Word 文档
    doc = Document(file_path)
    
    # 创建空列表用于存放结果
    chunks: list[DocxChunk] = []
    
    # 当前正在构建的块
    current_chunk_lines: list[str] = []
    current_chunk_start_id = 0
    
    # 遍历文档中的所有段落
    for i, p in enumerate(doc.paragraphs):
        # 获取段落文本并去除前后空白
        text = p.text.strip()
        
        # 跳过空段落
        if not text:
            continue
        
        # 判断是否是新的标题（以数字+顿号+中文开头，如 "15、金稻田面包屋"）
        is_title = False
        if text and len(text) > 2:
            # 检查是否以 "数字、" 开头
            parts = text.split('、', 1)
            if len(parts) == 2 and parts[0].strip().replace('.', '').isdigit():
                is_title = True
        
        # 如果是新标题且当前已有内容，保存当前块
        if is_title and current_chunk_lines:
            # 合并当前块的所有行
            combined_text = '\n'.join(current_chunk_lines)
            chunk: DocxChunk = {
                "text": combined_text,
                "metadata": {
                    "source_file": file_name,
                    "type": "docx",
                    "paragraph_id": current_chunk_start_id
                }
            }
            chunks.append(chunk)
            
            # 重置当前块
            current_chunk_lines = []
            current_chunk_start_id = i
        
        # 如果是新标题，记录起始ID
        if is_title and not current_chunk_lines:
            current_chunk_start_id = i
        
        # 将当前行添加到当前块
        current_chunk_lines.append(text)
    
    # 处理最后一个块
    if current_chunk_lines:
        combined_text = '\n'.join(current_chunk_lines)
        chunk: DocxChunk = {
            "text": combined_text,
            "metadata": {
                "source_file": file_name,
                "type": "docx",
                "paragraph_id": current_chunk_start_id
            }
        }
        chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    # 定义文档路径
    file_path = "光明区光明街道旅游手册.docx"
    
    # 加载并处理文档
    chunks = load_and_chunk_docx(file_path)
    
    # 打印统计信息
    print(f"数据块总数: {len(chunks)}")
    print("\n前 5 个数据块:")
    
    # 打印前 5 个数据块
    for idx, chunk in enumerate(chunks[:5], 1):
        print(f"\n--- 数据块 {idx} ---")
        print(f"文本: {chunk['text']}")
        print(f"元数据: {chunk['metadata']}")
