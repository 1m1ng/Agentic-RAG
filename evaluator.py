"""
RAG 系统评测脚本
使用 Kimi AI 作为评测员，评估答案的准确性和忠实度
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from agno.models.openai import OpenAIChat
from eval_questions import EVAL_QUESTIONS


async def evaluate_answer(query: str, ground_truth: str, actual_answer: str) -> dict:
    """
    使用 Kimi AI 评测答案的准确性和忠实度
    
    Args:
        query: 用户问题
        ground_truth: 标准答案
        actual_answer: 实际生成的答案
        
    Returns:
        包含 score 和 reasoning 的评测结果字典
    """
    # 系统提示词 - 指示 Kimi 充当评测员
    system_prompt = """你是一个专业的 RAG 系统评测员。你的任务是根据 [标准答案] 评估 [生成答案] 的质量。

评测维度：
1. **准确性 (Accuracy)**: 判断 [生成答案] 是否正确回答了 [用户问题]。
2. **忠实度 (Faithfulness)**: 判断 [生成答案] 中的信息是否**完全**包含在 [标准答案] 中，没有捏造或添加额外信息。

评分标准：
- 1.0: 完全准确且完全忠实
- 0.7-0.9: 基本准确但可能不完整，或有轻微偏差
- 0.4-0.6: 部分正确但有明显遗漏或偏差
- 0.0-0.3: 错误答案或严重偏离标准答案

**重要**：你必须返回一个 JSON 对象，包含以下字段：
- "score": 浮点数，范围 0.0 到 1.0
- "reasoning": 字符串，详细解释评分理由

只返回 JSON 对象，不要包含其他内容。"""

    # 用户提示词 - 包含具体的问题和答案
    user_prompt = f"""请评测以下答案：

[用户问题]
{query}

[标准答案]
{ground_truth}

[生成答案]
{actual_answer}

请根据准确性和忠实度进行评分，并返回 JSON 格式的评测结果。"""

    # 加载环境变量
    load_dotenv()
    
    # 实例化 OpenAIChat 模型，连接到 Kimi API
    kimi_model = OpenAIChat(
        id="kimi-k2-0905-preview",
        api_key=os.getenv("KIMI_API_KEY"),
        base_url="https://api.moonshot.cn/v1"
    )
    
    # 创建 Agent 实例
    from agno.agent import Agent
    
    agent = Agent(
        model=kimi_model,
        markdown=False
    )
    
    # 合并 system prompt 和 user prompt
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # 调用 Kimi API 进行评测
    response = agent.run(combined_prompt, stream=False)
    
    # 解析响应内容
    try:
        # 获取响应文本内容
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # 确保 content 不为 None
        if content is None:
            content = ""
        
        result = json.loads(content)
        return result
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        print(f"原始响应: {content}")
        return {"score": 0.0, "reasoning": "评测失败：无法解析 JSON 响应"}


async def main():
    """主函数：遍历评估问题集并进行评测"""
    print("=" * 100)
    print("开始评测 RAG 系统")
    print("=" * 100)
    
    total_score = 0.0
    
    for idx, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n{'=' * 100}")
        print(f"评测问题 {idx}/{len(EVAL_QUESTIONS)}")
        print(f"{'=' * 100}")
        
        query = item["query"]
        ground_truth = item["ground_truth"]
        simulated_answer = item["simulated_answer"]
        
        print(f"\n📝 用户问题:")
        print(f"   {query}")
        
        print(f"\n✅ 标准答案:")
        print(f"   {ground_truth}")
        
        print(f"\n🤖 模拟答案:")
        print(f"   {simulated_answer}")
        
        print(f"\n⏳ 正在调用 Kimi AI 评测员...")
        
        # 调用评测函数
        evaluation = await evaluate_answer(query, ground_truth, simulated_answer)
        
        print(f"\n📊 评测结果:")
        print(f"   得分: {evaluation.get('score', 0.0):.2f}")
        print(f"   理由: {evaluation.get('reasoning', '无')}")
        
        total_score += evaluation.get('score', 0.0)
    
    # 计算平均分
    avg_score = total_score / len(EVAL_QUESTIONS) if EVAL_QUESTIONS else 0.0
    
    print(f"\n{'=' * 100}")
    print(f"评测完成")
    print(f"{'=' * 100}")
    print(f"总问题数: {len(EVAL_QUESTIONS)}")
    print(f"平均得分: {avg_score:.2f}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    asyncio.run(main())
