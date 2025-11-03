import asyncio
from os import getenv
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat


async def main():
    """主函数：使用 Kimi AI 模型运行 Agent"""
    # 加载环境变量
    load_dotenv()
    
    # 从环境变量读取 API 密钥
    api_key = getenv("KIMI_API_KEY")
    
    if not api_key:
        print("错误：未找到 KIMI_API_KEY 环境变量。")
        print("请在 .env 文件中设置 KIMI_API_KEY=your_api_key")
        return
    
    # 配置 OpenAI 兼容的 Kimi 模型
    kimi_model = OpenAIChat(
        id="kimi-k2-0905-preview",
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )
    
    # 创建 Agent 实例
    agent = Agent(
        model=kimi_model
    )
    
    # 发送测试消息并打印响应
    agent.print_response(
        "你好！请用中文介绍一下你自己。",
        stream=True
    )


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
