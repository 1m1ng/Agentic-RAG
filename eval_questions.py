"""
评估问题集 - 基于《光明区光明街道旅游手册》真实目录信息
用于测试 RAG 系统的准确性、完整性和可溯源性
"""

EVAL_QUESTIONS = [
    {
        "query": "光明区有哪些非遗美食？",
        "ground_truth": "根据目录，光明区的非遗美食包括下村烧猪和旧街濑粉。",
        "simulated_answer": "有下村烧猪和旧街濑粉。"
    },
    {
        "query": "光明招待所（光明乳鸽老字号）在哪个片区？",
        "ground_truth": "根据目录，光明招待所（光明乳鸽老字号）位于光明中心片区。",
        "simulated_answer": "光明招待所在迳口古村。"
    },
    {
        "query": "光明区有哪些民宿？",
        "ground_truth": "根据目录，光明区的民宿有诺花里民宿和耕读别院。",
        "simulated_answer": "光明区有诺花里民宿。"
    }
]


if __name__ == "__main__":
    print("评估问题集统计:")
    print(f"总问题数: {len(EVAL_QUESTIONS)}")
    print("\n" + "=" * 80)
    
    for idx, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n问题 {idx}:")
        print(f"Query: {item['query']}")
        print(f"\nGround Truth: {item['ground_truth']}")
        print(f"\nSimulated Answer: {item['simulated_answer']}")
        print("\n" + "-" * 80)
