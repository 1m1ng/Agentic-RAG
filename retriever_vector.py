"""
向量检索引擎
使用 FAISS 和 BGE 模型进行语义向量检索，支持元数据溯源
在 CPU 上运行，无需 GPU
支持索引持久化，避免重复计算向量
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from document_loader import load_and_chunk_docx, DocxChunk
from transformers.utils import logging

# 设置 Hugging Face 缓存目录为当前目录下的 `.cache` 文件夹
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
logging.set_verbosity_error()  # 静默日志输出

# 全局变量：文档路径
DOCX_FILE_PATH = "光明区光明街道旅游手册.docx"

# BGE 模型名称（CPU 友好的中文嵌入模型）
MODEL_NAME = 'BAAI/bge-m3'

# 索引保存路径
INDEX_DIR = ".vector_index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")


class VectorRetriever:
    """基于 FAISS 和 BGE 的向量检索器类，支持索引持久化"""
    
    def __init__(self, force_rebuild: bool = False):
        """
        初始化向量检索器
        
        Args:
            force_rebuild: 是否强制重建索引（忽略已保存的索引）
        """
        # 加载 BGE 嵌入模型（CPU 模式）
        print("正在加载 BGE 嵌入模型...")
        self.model = SentenceTransformer(MODEL_NAME, device='cpu')
        
        # 初始化空索引
        self.chunks: list[DocxChunk] = []
        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)
        
        # 检查是否存在已保存的索引
        if not force_rebuild and DOCX_FILE_PATH and self._check_index_exists():
            print("发现已保存的索引，正在加载...")
            self._load_index()
            print(f"索引加载完成，共 {len(self.chunks)} 个文档块。")
        elif DOCX_FILE_PATH and os.path.exists(DOCX_FILE_PATH):
            print("未找到已保存的索引或强制重建，正在创建新索引...")
            self._build_index()
            self._save_index()
            print("索引创建并保存完成。")
    
    def add_document(self, text: str, metadata: dict):
        """添加文档到索引"""
        chunk: DocxChunk = {"text": text, "metadata": metadata}
        self.chunks.append(chunk)
        embedding = self.model.encode([text])
        self.index.add(embedding.astype('float32'))  # type: ignore
    
    def clear(self):
        """清空索引"""
        self.chunks = []
        d = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(d)
    
    def save(self, path: str):
        """保存索引到文件"""
        import pickle
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
    
    def load(self, path: str):
        """从文件加载索引"""
        import pickle
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
    
    def _check_index_exists(self) -> bool:
        """检查索引文件是否存在"""
        return os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH)
    
    def _build_index(self):
        """构建新的 FAISS 索引"""
        # 加载和分块文档
        print("正在加载和分块文档...")
        self.chunks: list[DocxChunk] = load_and_chunk_docx(DOCX_FILE_PATH)
        
        # 提取所有文本
        texts = [chunk['text'] for chunk in self.chunks]
        
        # 生成向量嵌入（显示进度条）
        print(f"正在为 {len(texts)} 个文档块生成向量嵌入 (这在 CPU 上可能需要1-2分钟)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 获取向量维度
        d = embeddings.shape[1]
        
        # 创建 FAISS 索引（使用 L2 距离）
        self.index = faiss.IndexFlatL2(d)
        
        # 将向量添加到索引中（FAISS 需要 float32 类型）
        self.index.add(embeddings.astype('float32'))  # type: ignore
    
    def _save_index(self):
        """保存 FAISS 索引和文档块到本地"""
        # 创建索引目录
        os.makedirs(INDEX_DIR, exist_ok=True)
        
        # 保存 FAISS 索引
        print(f"正在保存 FAISS 索引到 {FAISS_INDEX_PATH}...")
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        
        # 保存文档块
        print(f"正在保存文档块到 {CHUNKS_PATH}...")
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print("索引保存成功。")
    
    def _load_index(self):
        """从本地加载 FAISS 索引和文档块"""
        # 加载 FAISS 索引
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        # 加载文档块
        with open(CHUNKS_PATH, 'rb') as f:
            self.chunks: list[DocxChunk] = pickle.load(f)
    
    def search(self, query: str, top_k: int = 3) -> list[DocxChunk]:
        """
        使用向量相似度检索相关文档块
        
        Args:
            query: 查询字符串
            top_k: 返回的最相关文档数量
            
        Returns:
            按相似度排序的文档块列表（包含文本和元数据）
        """
        # 对查询生成向量嵌入
        query_embedding = self.model.encode([query])
        
        # 在 FAISS 索引中搜索最相似的向量
        # D: 距离数组, I: 索引数组
        D, I = self.index.search(query_embedding.astype('float32'), top_k)  # type: ignore
        
        # 根据索引提取对应的文档块
        results = [self.chunks[i] for i in I[0]]
        
        return results


if __name__ == "__main__":
    # 初始化检索器
    print("正在初始化 Vector 检索器...")
    retriever = VectorRetriever()
    
    # 测试查询
    test_query = "光明区有哪些非遗美食？"
    results = retriever.search(test_query)
    
    # 打印检索结果
    print(f"\n--- 查询 '{test_query}' 的检索结果 ---\n")
    
    for idx, chunk in enumerate(results, 1):
        print(f"结果 {idx}:")
        print(f"  文本: {chunk['text']}")
        print(f"  元数据: {chunk['metadata']}")
        print()
