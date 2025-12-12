"""
BM25 检索引擎
使用 BM25 算法对中文文档进行检索，支持元数据溯源
"""

import os
import jieba
from rank_bm25 import BM25Okapi
from document_loader import load_and_chunk_docx, DocxChunk

# 全局变量：文档路径
DOCX_FILE_PATH = "input/光明区光明街道旅游手册.docx"


class BM25Retriever:
    """BM25 检索器类"""
    
    def __init__(self):
        """
        初始化 BM25 检索器
        加载文档、分词并构建 BM25 索引
        """
        # 关闭 jieba 的调试日志
        jieba.setLogLevel(20)
        
        # 初始化空列表
        self.chunks: list[DocxChunk] = []
        self.tokenized_corpus = []
        self.bm25 = None
        
        # 如果文件存在则加载
        if DOCX_FILE_PATH and os.path.exists(DOCX_FILE_PATH):
            self.chunks = load_and_chunk_docx(DOCX_FILE_PATH)
            corpus = [chunk['text'] for chunk in self.chunks]
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def add_document(self, text: str, metadata: dict):
        """添加文档到索引"""
        chunk: DocxChunk = {"text": text, "metadata": metadata}
        self.chunks.append(chunk)
        tokens = list(jieba.cut(text))
        self.tokenized_corpus.append(tokens)
        # 重建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def clear(self):
        """清空索引"""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
    
    def save(self, path: str):
        """保存索引到文件"""
        import pickle
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(path, "tokenized_corpus.pkl"), "wb") as f:
            pickle.dump(self.tokenized_corpus, f)
    
    def load(self, path: str):
        """从文件加载索引"""
        import pickle
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        with open(os.path.join(path, "tokenized_corpus.pkl"), "rb") as f:
            self.tokenized_corpus = pickle.load(f)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query: str, top_k: int = 3) -> list[DocxChunk]:
        """
        使用 BM25 算法检索相关文档块
        
        Args:
            query: 查询字符串
            top_k: 返回的最相关文档数量
            
        Returns:
            按相关性排序的文档块列表（包含文本和元数据）
        """
        # 检查 BM25 索引是否已初始化
        if self.bm25 is None:
            return []
        
        # 对查询进行中文分词
        tokenized_query = list(jieba.cut(query))
        
        # 使用 BM25 检索 top_k 个最相关的文档块
        results = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        
        return results


if __name__ == "__main__":
    # 初始化检索器
    print("正在初始化 BM25 检索器...")
    retriever = BM25Retriever()
    print("初始化完成。")
    
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
