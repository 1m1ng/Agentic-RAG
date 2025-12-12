"""
Agentic RAG 系统核心
实现完整的查询重写 -> 混合检索 -> 重排序 -> 生成答案 -> 评测流水线
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from retriever_hybrid import HybridRetriever
from document_loader import DocxChunk


def get_kimi_client() -> OpenAIChat:
    """
    获取配置好的 Kimi 客户端
    
    Returns:
        配置好的 OpenAIChat 客户端
    """
    # 加载环境变量
    load_dotenv()
    
    # 获取 API 密钥
    api_key = os.getenv("KIMI_API_KEY")
    
    if not api_key:
        raise ValueError("错误：未找到 KIMI_API_KEY 环境变量")
    
    # 返回配置好的 Kimi 客户端
    return OpenAIChat(
        id="kimi-k2-0905-preview",
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )


async def rewrite_query(client: OpenAIChat, user_query: str) -> list[str]:
    """
    使用 Kimi LLM 重写查询为多个变体
    
    Args:
        client: Kimi OpenAIChat 客户端
        user_query: 用户的原始查询
        
    Returns:
        包含原始查询和重写变体的查询列表
    """
    # 系统提示词 - 指示 Kimi 充当查询重写专家
    system_prompt = """你是一个专业的中文搜索查询重写专家。

你的任务是将用户的单个查询重写为 3 个不同的变体，以提高在旅游手册中的召回率。

重写时请考虑：
1. 同义词和近义词
2. 缩写和全称
3. 不同的搜索意图和表达方式
4. 保持查询的语义相关性

**严格要求**：你必须只返回一个 JSON 对象，格式为：
{
  "queries": ["query1", "query2", "query3"]
}

不要返回任何其他内容，只返回 JSON 对象。"""

    # 用户提示词
    user_prompt = f"请重写这个查询：'{user_query}'"
    
    # 创建 Agent 实例
    agent = Agent(
        model=client,
        markdown=False
    )
    
    # 合并提示词
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # 调用 Kimi 进行查询重写
    response = agent.run(combined_prompt, stream=False)
    
    # 解析响应
    try:
        # 获取响应内容
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # 确保 content 不为 None
        if content is None:
            content = "{}"
        
        # 解析 JSON
        result = json.loads(content)
        queries = result.get('queries', [])
        
        # 追加原始查询到列表中
        queries.append(user_query)
        
        return queries
        
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print(f"原始响应: {content}")
        # 如果解析失败，至少返回原始查询
        return [user_query]


async def evaluate_answer(client: OpenAIChat, query: str, generated_answer: str, context: str) -> dict:
    """
    使用 Kimi AI 评测生成答案的质量
    
    Args:
        client: Kimi OpenAIChat 客户端
        query: 用户问题
        generated_answer: 生成的答案
        context: 提供的上下文
        
    Returns:
        包含 score 和 reasoning 的评测结果字典
    """
    # 系统提示词 - 指示 Kimi 充当评测员
    system_prompt = """你是一个专业的 RAG 系统评测员。你的任务是评估 [生成答案] 的质量。

评测维度：
1. **准确性 (Accuracy)**: 判断 [生成答案] 是否正确回答了 [用户问题]。
2. **忠实度 (Faithfulness)**: 判断 [生成答案] 中的信息是否**完全**基于 [上下文]，没有捏造或添加上下文中不存在的信息。
3. **完整性 (Completeness)**: 判断 [生成答案] 是否充分利用了 [上下文] 中的相关信息。

评分标准：
- 1.0: 完全准确、完全忠实、信息完整
- 0.7-0.9: 基本准确且忠实，但可能有轻微不完整
- 0.4-0.6: 部分正确但有明显遗漏或轻微偏差
- 0.0-0.3: 错误答案、严重捏造或严重偏离上下文

**重要**：你必须返回一个 JSON 对象，包含以下字段：
- "score": 浮点数，范围 0.0 到 1.0
- "reasoning": 字符串，详细解释评分理由

只返回 JSON 对象，不要包含其他内容。"""

    # 用户提示词
    user_prompt = f"""请评测以下答案：

[用户问题]
{query}

[上下文]
{context}

[生成答案]
{generated_answer}

请根据准确性、忠实度和完整性进行评分，并返回 JSON 格式的评测结果。"""

    # 创建 Agent 实例
    agent = Agent(
        model=client,
        markdown=False
    )
    
    # 合并提示词
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # 调用 Kimi 进行评测
    response = agent.run(combined_prompt, stream=False)
    
    # 解析响应
    try:
        # 获取响应内容
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # 确保 content 不为 None
        if content is None:
            content = "{}"
        
        # 解析 JSON
        result = json.loads(content)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print(f"原始响应: {content}")
        return {"score": 0.0, "reasoning": "评测失败：无法解析 JSON 响应"}


class AgenticRAG:
    """Agentic RAG 系统核心类"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化 RAG Agent
        
        Args:
            max_workers: 线程池最大工作线程数
        """
        print("正在初始化 RAG Agent...")
        
        # 初始化 Kimi 客户端
        self.kimi_client = get_kimi_client()
        
        # 初始化混合检索器
        print("正在初始化 Hybrid Retriever (这可能需要1-2分钟)...")
        self.retriever = HybridRetriever()
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.chunks_lock = Lock()
        
        print("Agent 初始化完成。")
    
    def _search_single_query(self, args: tuple) -> list:
        """
        单个查询的检索（用于多线程执行）
        
        Args:
            args: (query_index, query) 元组
            
        Returns:
            检索结果列表
        """
        query_index, query = args
        print(f"  - 正在检索子查询 {query_index}/[总数]: '{query}'")
        return self.retriever.search(query, top_k=10)
    
    def _parallel_search(self, rewritten_queries: list[str]) -> dict:
        """
        并行执行多个查询的检索
        
        Args:
            rewritten_queries: 重写后的查询列表
            
        Returns:
            去重后的候选块字典
        """
        all_candidate_chunks: dict = {}
        
        # 准备任务列表
        tasks = [(i + 1, query) for i, query in enumerate(rewritten_queries)]
        
        # 使用线程池并行执行检索
        futures = []
        for task in tasks:
            future = self.thread_pool.submit(self._search_single_query, task)
            futures.append(future)
        
        # 收集结果并去重
        for future in as_completed(futures):
            try:
                results = future.result()
                for chunk in results:
                    para_id = chunk['metadata']['paragraph_id']
                    with self.chunks_lock:
                        all_candidate_chunks[para_id] = chunk
            except Exception as e:
                print(f"检索任务出错: {e}")
        
        return all_candidate_chunks
    
    async def run(self, user_query: str, enable_evaluation: bool = True):
        """
        执行完整的 RAG 流水线（使用多线程加速）
        
        Args:
            user_query: 用户的原始查询
            enable_evaluation: 是否启用答案评测
        """
        print(f"\n{'=' * 80}")
        print(f"用户问题: {user_query}")
        print(f"{'=' * 80}\n")
        
        # ==================== 步骤 1: 查询重写 ====================
        print("--- 步骤 1: Agent 正在重写查询 ---")
        rewritten_queries = await rewrite_query(self.kimi_client, user_query)
        
        for idx, query in enumerate(rewritten_queries, 1):
            print(f"  {idx}. {query}")
        print()
        
        # ==================== 步骤 2: 混合检索（多线程并行执行） ====================
        print(f"--- 步骤 2: Agent 正在执行混合检索（使用多线程并行处理 {len(rewritten_queries)} 个查询）---")
        
        # 并行执行检索
        all_candidate_chunks = self._parallel_search(rewritten_queries)
        
        # 获取所有唯一候选块
        final_candidates = list(all_candidate_chunks.values())
        print(f"\n--- 步骤 3: Agent 正在对 {len(final_candidates)} 个唯一候选块进行最终重排 ---")
        
        # ==================== 步骤 3: 最终重排 ====================
        # 使用原始查询对所有候选块进行最终重排
        pairs = [[user_query, chunk['text']] for chunk in final_candidates]
        scores = self.retriever.reranker.compute_score(pairs, batch_size=4)  # type: ignore
        
        # 组合分数和文档块
        scored_chunks = list(zip(scores, final_candidates))  # type: ignore
        
        # 按分数降序排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)  # type: ignore
        
        # 获取 top 3 个最相关的文档块
        top_k_chunks = [chunk for score, chunk in scored_chunks[:3]]  # type: ignore
        
        # 打印最终检索到的上下文
        print("\n--- 最终检索到的上下文 (Context) ---\n")
        context_str = ""
        
        for idx, chunk in enumerate(top_k_chunks, 1):
            print(f"上下文块 {idx}:")
            print(f"  文本: {chunk['text']}")
            print(f"  元数据: {chunk['metadata']}")
            print()
            
            # 拼接上下文字符串
            context_str += f"[文档片段 {idx}]\n{chunk['text']}\n\n"
        
        
        # ==================== 步骤 4: 生成答案 ====================
        print("--- 步骤 4: Agent 正在基于上下文生成答案 ---\n")
        
        # 系统提示词
        system_prompt = """你是一个问答助手。请根据下面提供的 [上下文]，用中文回答 [用户问题]。

你的回答必须严格基于 [上下文] 包含的信息，禁止捏造。如果上下文中没有足够的信息，请明确说明。"""
        
        # 用户提示词
        user_prompt = f"""[上下文]:
{context_str}

[用户问题]:
{user_query}"""
        
        # 创建 Agent 实例并流式生成答案
        agent = Agent(
            model=self.kimi_client,
            markdown=False
        )
        
        # 合并提示词
        combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
        
        # 打印最终答案标题
        print("--- 最终答案 ---\n")
        
        # 流式打印答案并收集完整答案
        final_answer = ""
        for chunk in agent.run(combined_prompt, stream=True):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)
            
            if content:
                print(content, end="", flush=True)
                final_answer += content
        
        print(f"\n\n{'=' * 80}\n")
        
        # ==================== 步骤 5: 评测答案 ====================
        if enable_evaluation:
            print("--- 步骤 5: Agent 正在评测答案质量 ---\n")
            
            evaluation = await evaluate_answer(
                self.kimi_client,
                user_query,
                final_answer,
                context_str
            )
            
            print(f"📊 评测结果:")
            print(f"   得分: {evaluation.get('score', 0.0):.2f}")
            print(f"   理由: {evaluation.get('reasoning', '无')}")
            print(f"\n{'=' * 80}\n")
        
        return final_answer
    
    def close(self):
        """关闭线程池资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            print("\n线程池已关闭")


async def main():
    """主函数：运行 Agentic RAG 系统"""
    # 初始化 RAG Agent（设置 max_workers=4 进行并行处理）
    rag = AgenticRAG(max_workers=4)
    
    try:
        # 定义测试查询
        test_query = "光明区有什么文化建筑？因什么而闻名？"
        
        # 运行完整的 RAG 流水线
        await rag.run(test_query)
    finally:
        # 确保关闭线程池资源
        rag.close()


if __name__ == "__main__":
    asyncio.run(main())
