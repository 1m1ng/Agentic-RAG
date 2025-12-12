"""
批量评测脚本
读取评测集 Excel 文件，运行 RAG Agent，并将结果与标准答案对比评分
"""

import asyncio
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from rag_agent import AgenticRAG, get_kimi_client


async def evaluate_with_standard(
    client: OpenAIChat, 
    query: str, 
    generated_answer: str, 
    standard_answer: str,
    context: str
) -> dict:
    """
    使用 Kimi AI 评测生成答案与标准答案的匹配度
    
    Args:
        client: Kimi OpenAIChat 客户端
        query: 用户问题
        generated_answer: RAG 生成的答案
        standard_answer: 标准答案
        context: RAG 使用的上下文
        
    Returns:
        包含 score 和 reasoning 的评测结果字典
    """
    system_prompt = """你是一个专业的 RAG 系统评测员。你的任务是评估 [RAG生成答案] 相对于 [标准答案] 的质量。

评测维度：
1. **准确性 (Accuracy)**: [RAG生成答案] 是否正确回答了 [用户问题]，与 [标准答案] 的一致性如何。
2. **忠实度 (Faithfulness)**: [RAG生成答案] 是否基于 [上下文]，没有捏造信息。
3. **完整性 (Completeness)**: [RAG生成答案] 是否包含了 [标准答案] 中的关键信息。

评分标准：
- 1.0: 与标准答案完全一致或更好，准确且完整
- 0.8-0.9: 与标准答案基本一致，有轻微差异但不影响准确性
- 0.6-0.7: 包含标准答案的主要内容，但有明显遗漏或不够准确
- 0.4-0.5: 部分正确，但有重要信息缺失或偏差
- 0.0-0.3: 与标准答案严重不符，错误或严重不完整

**重要**：你必须返回一个 JSON 对象，包含以下字段：
- "score": 浮点数，范围 0.0 到 1.0
- "reasoning": 字符串，详细解释评分理由

只返回 JSON 对象，不要包含其他内容。"""

    user_prompt = f"""请评测以下答案：

[用户问题]
{query}

[标准答案]
{standard_answer}

[RAG生成答案]
{generated_answer}

[上下文]
{context}

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
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        if content is None:
            content = "{}"
        
        result = json.loads(content)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print(f"原始响应: {content}")
        return {"score": 0.0, "reasoning": "评测失败：无法解析 JSON 响应"}


async def run_rag_and_evaluate(
    rag: AgenticRAG,
    kimi_client: OpenAIChat,
    query: str,
    standard_answer: str
) -> dict:
    """
    运行 RAG 并评测结果
    
    Args:
        rag: RAG Agent 实例
        kimi_client: Kimi 客户端
        query: 用户问题
        standard_answer: 标准答案
        
    Returns:
        包含生成答案、评测结果等信息的字典
    """
    print(f"\n{'=' * 100}")
    print(f"正在处理查询: {query}")
    print(f"{'=' * 100}\n")
    
    # 执行 RAG（禁用自动评测）
    try:
        # 修改 run 方法以返回更多信息
        from rag_agent import rewrite_query
        
        # 1. 查询重写
        print("--- 步骤 1: 查询重写 ---")
        rewritten_queries = await rewrite_query(kimi_client, query)
        print(f"重写查询数: {len(rewritten_queries)}")
        
        # 2. 混合检索
        print("\n--- 步骤 2: 混合检索 ---")
        from document_loader import DocxChunk
        all_candidate_chunks: dict[int, DocxChunk] = {}
        
        for i, q in enumerate(rewritten_queries, 1):
            results = rag.retriever.search(q, top_k=10)
            for chunk in results:
                para_id = chunk['metadata']['paragraph_id']
                all_candidate_chunks[para_id] = chunk
        
        final_candidates = list(all_candidate_chunks.values())
        print(f"候选块数: {len(final_candidates)}")
        
        # 3. 最终重排
        print("\n--- 步骤 3: 最终重排 ---")
        pairs = [[query, chunk['text']] for chunk in final_candidates]
        scores = rag.retriever.reranker.compute_score(pairs, batch_size=4)  # type: ignore
        scored_chunks = list(zip(scores, final_candidates))  # type: ignore
        scored_chunks.sort(key=lambda x: x[0], reverse=True)  # type: ignore
        top_k_chunks = [chunk for score, chunk in scored_chunks[:3]]  # type: ignore
        
        # 构建上下文
        context_str = ""
        for idx, chunk in enumerate(top_k_chunks, 1):
            context_str += f"[文档片段 {idx}]\n{chunk['text']}\n\n"
        
        print(f"最终上下文块数: {len(top_k_chunks)}")
        
        # 4. 生成答案
        print("\n--- 步骤 4: 生成答案 ---")
        system_prompt = """你是一个问答助手。请根据下面提供的 [上下文]，用中文回答 [用户问题]。

你的回答必须严格基于 [上下文] 包含的信息，禁止捏造。如果上下文中没有足够的信息，请明确说明。"""
        
        user_prompt = f"""[上下文]:
{context_str}

[用户问题]:
{query}"""
        
        agent = Agent(
            model=kimi_client,
            markdown=False
        )
        
        combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
        
        # 非流式获取完整答案
        response = agent.run(combined_prompt, stream=False)
        
        if hasattr(response, 'content'):
            generated_answer = response.content or ""
        else:
            generated_answer = str(response)
        
        if not generated_answer:
            generated_answer = ""
        
        print(f"生成答案长度: {len(generated_answer)} 字符")
        
        # 5. 评测
        print("\n--- 步骤 5: 评测答案 ---")
        evaluation = await evaluate_with_standard(
            kimi_client,
            query,
            generated_answer,
            standard_answer,
            context_str
        )
        
        print(f"评测得分: {evaluation.get('score', 0.0):.2f}")
        print(f"评测理由: {evaluation.get('reasoning', '无')[:100]}...")
        
        return {
            "query": query,
            "standard_answer": standard_answer,
            "generated_answer": generated_answer,
            "context": context_str,
            "rewritten_queries_count": len(rewritten_queries),
            "candidate_chunks_count": len(final_candidates),
            "score": evaluation.get('score', 0.0),
            "reasoning": evaluation.get('reasoning', '无'),
            "status": "成功"
        }
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        return {
            "query": query,
            "standard_answer": standard_answer,
            "generated_answer": "",
            "context": "",
            "rewritten_queries_count": 0,
            "candidate_chunks_count": 0,
            "score": 0.0,
            "reasoning": f"处理失败: {str(e)}",
            "status": "失败"
        }


async def batch_evaluate(input_file: str = "评测集.xlsx", output_file: str | None = None):
    """
    批量评测主函数
    
    Args:
        input_file: 输入的评测集 Excel 文件路径
        output_file: 输出的结果 Excel 文件路径（默认自动生成带时间戳的文件名）
    """
    print("=" * 100)
    print("批量评测开始")
    print("=" * 100)
    
    # 读取评测集
    print(f"\n正在读取评测集: {input_file}")
    try:
        df = pd.read_excel(input_file)
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {input_file}")
        return
    
    # 检查列名
    if 'query' not in df.columns or 'standard_answer' not in df.columns:
        print("❌ 错误：Excel 文件必须包含 'query' 和 'standard_answer' 列")
        print(f"当前列名: {df.columns.tolist()}")
        return
    
    print(f"✅ 成功读取 {len(df)} 条评测数据")
    
    # 初始化 RAG Agent
    print("\n正在初始化 RAG Agent...")
    rag = AgenticRAG()
    kimi_client = get_kimi_client()
    
    # 存储结果
    results = []
    
    # 逐条处理
    for idx, row in df.iterrows():
        query = str(row['query'])
        standard_answer = str(row['standard_answer'])
        
        print(f"\n{'#' * 100}")
        print(f"进度: {int(idx) + 1}/{len(df)}")  # type: ignore
        print(f"{'#' * 100}")
        
        # 运行 RAG 并评测
        result = await run_rag_and_evaluate(
            rag,
            kimi_client,
            query,
            standard_answer
        )
        
        results.append(result)
    
    # 创建结果 DataFrame
    results_df = pd.DataFrame(results)
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"评测结果_{timestamp}.xlsx"
    
    # 保存结果
    print(f"\n正在保存结果到: {output_file}")
    results_df.to_excel(output_file, index=False)
    
    # 打印统计信息
    print("\n" + "=" * 100)
    print("批量评测完成")
    print("=" * 100)
    print(f"\n📊 统计信息:")
    print(f"   总问题数: {len(results)}")
    print(f"   成功处理: {sum(1 for r in results if r['status'] == '成功')}")
    print(f"   处理失败: {sum(1 for r in results if r['status'] == '失败')}")
    print(f"   平均得分: {results_df['score'].mean():.2f}")
    print(f"   最高得分: {results_df['score'].max():.2f}")
    print(f"   最低得分: {results_df['score'].min():.2f}")
    print(f"\n✅ 结果已保存到: {output_file}")
    print("=" * 100)


async def main():
    """主函数"""
    # 批量评测
    await batch_evaluate(
        input_file="评测集.xlsx",
        output_file=None  # 自动生成带时间戳的文件名
    )


if __name__ == "__main__":
    asyncio.run(main())
