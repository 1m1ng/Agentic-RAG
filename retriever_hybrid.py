"""
混合检索引擎（Hybrid Retrieval + Rerank）
结合 BM25 和向量检索，使用 BGE Reranker 进行重排序
支持元数据溯源，在 CPU 上运行
"""

from FlagEmbedding import FlagReranker
from retriever_bm25 import BM25Retriever
from retriever_vector import VectorRetriever
from document_loader import DocxChunk

# BGE Reranker 模型名称
RERANKER_MODEL = 'BAAI/bge-reranker-base'


class HybridRetriever:
    """混合检索器类：BM25 + Vector + Rerank"""
    
    def __init__(self):
        """
        初始化混合检索器
        加载 BM25、Vector 检索器和 Reranker 模型
        """
        # 初始化 BM25 检索器
        print("Initializing BM25...")
        self.bm25 = BM25Retriever()
        
        # 初始化向量检索器
        print("Initializing Vector...")
        self.vector = VectorRetriever()
        
        # 加载 BGE Reranker 模型（CPU 模式）
        print("Loading BGE Reranker model (CPU)...")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
        
        print("Hybrid Retriever initialized.")
    
    def add_document(self, text: str, metadata: dict):
        """
        添加文档到检索器
        
        Args:
            text: 文档文本
            metadata: 文档元数据
        """
        self.bm25.add_document(text, metadata)
        self.vector.add_document(text, metadata)
    
    def clear(self):
        """清空所有索引"""
        self.bm25.clear()
        self.vector.clear()
    
    def save_index(self, index_dir: str):
        """
        保存索引到指定目录
        
        Args:
            index_dir: 索引保存目录
        """
        import os
        os.makedirs(index_dir, exist_ok=True)
        self.bm25.save(os.path.join(index_dir, "bm25"))
        self.vector.save(os.path.join(index_dir, "vector"))
    
    def load_index(self, index_dir: str):
        """
        从指定目录加载索引
        
        Args:
            index_dir: 索引加载目录
        """
        import os
        self.bm25.load(os.path.join(index_dir, "bm25"))
        self.vector.load(os.path.join(index_dir, "vector"))
    
    def search(self, query: str, top_k: int = 3) -> list[DocxChunk]:
        """
        使用混合检索策略检索相关文档块
        
        Args:
            query: 查询字符串
            top_k: 最终返回的文档数量
            
        Returns:
            经过重排序的文档块列表（包含文本和元数据）
        """
        # 执行 BM25 检索
        print("Executing BM25 search...")
        bm25_results = self.bm25.search(query, top_k=5)
        
        # 执行向量检索
        print("Executing Vector search...")
        vector_results = self.vector.search(query, top_k=5)
        
        # 合并并去重候选文档
        combined_chunks: dict[int, DocxChunk] = {}
        
        # 添加 BM25 结果
        for chunk in bm25_results:
            para_id = chunk['metadata']['paragraph_id']
            combined_chunks[para_id] = chunk
        
        # 添加 Vector 结果（自动去重）
        for chunk in vector_results:
            para_id = chunk['metadata']['paragraph_id']
            combined_chunks[para_id] = chunk
        
        # 转换为列表
        candidate_chunks = list(combined_chunks.values())
        print(f"Recall phase found {len(candidate_chunks)} unique candidates.")
        
        # 准备重排序数据
        pairs = [[query, chunk['text']] for chunk in candidate_chunks]
        
        # 使用 Reranker 重新排序
        print("Reranking candidates on CPU...")
        scores = self.reranker.compute_score(pairs, batch_size=4)  # type: ignore
        
        # 组合分数和文档块
        scored_chunks = list(zip(scores, candidate_chunks))  # type: ignore
        
        # 按分数降序排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)  # type: ignore
        
        # 提取 top_k 个结果
        final_results = [chunk for score, chunk in scored_chunks[:top_k]]  # type: ignore
        
        return final_results


if __name__ == "__main__":
    # 初始化混合检索器
    print("Initializing Hybrid Retriever (this may take 1-2 minutes)...")
    retriever = HybridRetriever()
    
    # 测试查询
    test_query = "光明区有哪些非遗美食？"
    results = retriever.search(test_query)
    
    # 打印最终重排序结果
    print(f"\n--- Final Reranked Results for '{test_query}' ---\n")
    
    for idx, chunk in enumerate(results, 1):
        print(f"结果 {idx}:")
        print(f"  文本: {chunk['text']}")
        print(f"  元数据: {chunk['metadata']}")
        print()
