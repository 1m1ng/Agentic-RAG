"""
Gradio WebUI for Agentic RAG System
提供知识库管理、文档上传、评测集上传和问答功能的Web界面
"""

import asyncio
import gradio as gr
import pandas as pd
import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from document_loader import load_and_chunk_docx
from rag_agent import AgenticRAG, get_kimi_client, rewrite_query, evaluate_answer
from batch_evaluation import evaluate_with_standard
from agno.agent import Agent

# 加载环境变量
load_dotenv()


def parse_evaluation_json(content: str) -> dict:
    """
    解析评测结果 JSON
    
    Args:
        content: LLM 生成的 JSON 字符串
        
    Returns:
        包含 score 和 reasoning 的字典
    """
    import re
    try:
        # 尝试直接解析
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        # 尝试提取 JSON 部分
        try:
            # 查找 JSON 对象
            json_match = re.search(r'\{[^{}]*"score"[^{}]*"reasoning"[^{}]*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # 尝试另一种模式
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"score": 0.0, "reasoning": "评测失败：无法解析 JSON 响应"}


def stream_evaluate_with_standard(
    client,
    query: str,
    generated_answer: str,
    standard_answer: str,
    context: str
):
    """
    流式评测生成答案与标准答案的匹配度
    
    Yields:
        评测内容的流式更新
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

    agent = Agent(
        model=client,
        markdown=False
    )
    
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # 流式生成评测结果
    eval_content = ""
    for chunk in agent.run(combined_prompt, stream=True):
        if hasattr(chunk, 'content'):
            content = chunk.content
        else:
            content = str(chunk)
        
        if content:
            eval_content += content
            yield eval_content


def stream_evaluate_answer(
    client,
    query: str,
    generated_answer: str,
    context: str
):
    """
    流式评测生成答案的质量（无标准答案，基于上下文自评）
    
    Yields:
        评测内容的流式更新
    """
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

    user_prompt = f"""请评测以下答案：

[用户问题]
{query}

[上下文]
{context}

[生成答案]
{generated_answer}

请根据准确性、忠实度和完整性进行评分，并返回 JSON 格式的评测结果。"""

    agent = Agent(
        model=client,
        markdown=False
    )
    
    combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
    
    # 流式生成评测结果
    eval_content = ""
    for chunk in agent.run(combined_prompt, stream=True):
        if hasattr(chunk, 'content'):
            content = chunk.content
        else:
            content = str(chunk)
        
        if content:
            eval_content += content
            yield eval_content

# 全局变量
KNOWLEDGE_BASE_DIR = Path("knowledge_bases")
EVAL_SETS_DIR = Path("eval_sets")
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
EVAL_SETS_DIR.mkdir(exist_ok=True)

# 全局 RAG 实例缓存（标记知识库是否已加载）
rag_instances: Dict[str, bool] = {}
kimi_client = None

# 全局共享的检索器（预加载模型）
shared_retriever = None

# 全局线程池（用于并行检索）
thread_pool = None
chunks_lock = Lock()


def init_shared_retriever():
    """初始化共享的 HybridRetriever（预加载模型）"""
    global shared_retriever
    if shared_retriever is None:
        from retriever_hybrid import HybridRetriever
        print("正在预加载检索模型（这可能需要1-2分钟）...")
        shared_retriever = HybridRetriever()
        print("✅ 检索模型预加载完成！")
    return shared_retriever


def init_thread_pool(max_workers: int = 4):
    """初始化线程池"""
    global thread_pool
    if thread_pool is None:
        thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return thread_pool


def _search_single_query(retriever, args: tuple) -> list:
    """单个查询的检索（用于多线程执行）"""
    query_index, query = args
    return retriever.search(query, top_k=10)


def parallel_search(retriever, rewritten_queries: list) -> dict:
    """并行执行多个查询的检索"""
    global thread_pool, chunks_lock
    all_candidate_chunks = {}
    
    pool = init_thread_pool()
    tasks = [(i + 1, query) for i, query in enumerate(rewritten_queries)]
    
    futures = []
    for task in tasks:
        future = pool.submit(_search_single_query, retriever, task)
        futures.append(future)
    
    for future in as_completed(futures):
        try:
            results = future.result()
            for chunk in results:
                para_id = chunk['metadata']['paragraph_id']
                with chunks_lock:
                    all_candidate_chunks[para_id] = chunk
        except Exception as e:
            print(f"检索任务出错: {e}")
    
    return all_candidate_chunks


def init_kimi_client():
    """初始化 Kimi 客户端"""
    global kimi_client
    if kimi_client is None:
        kimi_client = get_kimi_client()
    return kimi_client


def get_knowledge_bases() -> List[str]:
    """获取所有知识库名称"""
    if not KNOWLEDGE_BASE_DIR.exists():
        return []
    return [d.name for d in KNOWLEDGE_BASE_DIR.iterdir() if d.is_dir()]


def get_eval_sets() -> List[str]:
    """获取所有评测集名称"""
    if not EVAL_SETS_DIR.exists():
        return []
    return [f.stem for f in EVAL_SETS_DIR.glob("*.xlsx")]


def create_knowledge_base(name: str) -> str:
    """创建新的知识库"""
    if not name or not name.strip():
        return "❌ 错误：知识库名称不能为空"
    
    name = name.strip()
    kb_path = KNOWLEDGE_BASE_DIR / name
    
    if kb_path.exists():
        return f"❌ 错误：知识库 '{name}' 已存在"
    
    try:
        kb_path.mkdir(parents=True)
        (kb_path / "documents").mkdir()
        (kb_path / "index").mkdir()
        
        # 创建元数据文件
        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "document_count": 0,
            "indexed": False
        }
        with open(kb_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return f"✅ 成功创建知识库 '{name}'"
    except Exception as e:
        return f"❌ 创建知识库失败：{str(e)}"


def delete_knowledge_base(name: str) -> str:
    """删除知识库"""
    if not name:
        return "❌ 错误：请选择要删除的知识库"
    
    kb_path = KNOWLEDGE_BASE_DIR / name
    
    if not kb_path.exists():
        return f"❌ 错误：知识库 '{name}' 不存在"
    
    try:
        # 清理缓存标记
        if name in rag_instances:
            del rag_instances[name]
        
        shutil.rmtree(kb_path)
        return f"✅ 成功删除知识库 '{name}'"
    except Exception as e:
        return f"❌ 删除知识库失败：{str(e)}"


def upload_document(kb_name: str, files: List) -> str:
    """上传文档到知识库"""
    if not kb_name:
        return "❌ 错误：请先选择知识库"
    
    if not files:
        return "❌ 错误：请选择要上传的文档"
    
    kb_path = KNOWLEDGE_BASE_DIR / kb_name
    if not kb_path.exists():
        return f"❌ 错误：知识库 '{kb_name}' 不存在"
    
    doc_dir = kb_path / "documents"
    uploaded_count = 0
    
    try:
        for file in files:
            if not file.name.endswith(('.docx', '.doc')):
                continue
            
            # 复制文件到知识库文档目录
            file_name = Path(file.name).name
            dest_path = doc_dir / file_name
            shutil.copy(file.name, dest_path)
            uploaded_count += 1
        
        # 更新元数据
        metadata_path = kb_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        metadata["document_count"] = len(list(doc_dir.glob("*.docx"))) + len(list(doc_dir.glob("*.doc")))
        metadata["indexed"] = False
        metadata["last_upload"] = datetime.now().isoformat()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return f"✅ 成功上传 {uploaded_count} 个文档到知识库 '{kb_name}'"
    except Exception as e:
        return f"❌ 上传文档失败：{str(e)}"


def build_knowledge_base(kb_name: str, progress=gr.Progress()) -> str:
    """构建知识库索引"""
    if not kb_name:
        return "❌ 错误：请先选择知识库"
    
    kb_path = KNOWLEDGE_BASE_DIR / kb_name
    if not kb_path.exists():
        return f"❌ 错误：知识库 '{kb_name}' 不存在"
    
    doc_dir = kb_path / "documents"
    index_dir = kb_path / "index"
    
    try:
        progress(0, desc="正在加载文档...")
        
        # 加载所有文档
        all_chunks = []
        doc_files = list(doc_dir.glob("*.docx")) + list(doc_dir.glob("*.doc"))
        
        if not doc_files:
            return f"❌ 错误：知识库 '{kb_name}' 中没有文档"
        
        for i, doc_file in enumerate(doc_files):
            progress((i + 1) / len(doc_files), desc=f"正在处理文档 {i+1}/{len(doc_files)}")
            chunks = load_and_chunk_docx(str(doc_file))
            all_chunks.extend(chunks)
        
        progress(0.8, desc="正在构建索引...")
        
        # 使用共享的检索器（已预加载模型）
        retriever = init_shared_retriever()
        
        # 清空现有索引
        retriever.clear()
        
        # 添加文档块到检索器
        for chunk in all_chunks:
            retriever.add_document(chunk['text'], chunk['metadata'])
        
        # 保存索引到知识库目录
        retriever.save_index(str(index_dir))
        
        # 更新元数据
        metadata_path = kb_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        metadata["indexed"] = True
        metadata["chunk_count"] = len(all_chunks)
        metadata["last_indexed"] = datetime.now().isoformat()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 标记该知识库已加载（索引已在共享检索器中）
        rag_instances[kb_name] = True
        
        progress(1.0, desc="完成")
        
        return f"✅ 成功构建知识库 '{kb_name}'\n处理文档数: {len(doc_files)}\n文档块数: {len(all_chunks)}"
    except Exception as e:
        return f"❌ 构建知识库失败：{str(e)}"


def upload_eval_set(file) -> str:
    """上传评测集"""
    if not file:
        return "❌ 错误：请选择要上传的评测集文件"
    
    if not file.name.endswith('.xlsx'):
        return "❌ 错误：评测集文件必须是 .xlsx 格式"
    
    try:
        # 验证文件格式
        df = pd.read_excel(file.name)
        if 'query' not in df.columns or 'standard_answer' not in df.columns:
            return "❌ 错误：评测集必须包含 'query' 和 'standard_answer' 列"
        
        # 保存文件
        file_name = Path(file.name).name
        dest_path = EVAL_SETS_DIR / file_name
        shutil.copy(file.name, dest_path)
        
        return f"✅ 成功上传评测集 '{file_name}'\n包含 {len(df)} 条评测数据"
    except Exception as e:
        return f"❌ 上传评测集失败：{str(e)}"


def load_knowledge_base_index(kb_name: str) -> bool:
    """
    加载知识库索引到共享检索器
    
    Returns:
        是否成功加载
    """
    if not kb_name:
        return False
    
    # 如果已加载，直接返回
    if kb_name in rag_instances:
        return True
    
    kb_path = KNOWLEDGE_BASE_DIR / kb_name
    if not kb_path.exists():
        return False
    
    index_dir = kb_path / "index"
    if not index_dir.exists():
        return False
    
    # 检查索引文件是否存在
    bm25_path = index_dir / "bm25" / "chunks.pkl"
    vector_path = index_dir / "vector" / "faiss.index"
    if not bm25_path.exists() or not vector_path.exists():
        return False
    
    try:
        # 使用共享检索器加载索引
        retriever = init_shared_retriever()
        retriever.load_index(str(index_dir))
        
        # 标记已加载
        rag_instances[kb_name] = True
        return True
    except Exception as e:
        print(f"加载知识库索引失败：{str(e)}")
        return False


def answer_question_sync(
    kb_name: str,
    eval_set_name: str,
    question: str,
    enable_eval: bool
):
    """
    回答问题并评测（流式输出，同步生成器）
    
    Yields:
        (answer, context, evaluation_result, time_stats) 的流式更新
    """
    if not kb_name:
        yield "❌ 请先选择知识库", "", "", ""
        return
    
    if not question or not question.strip():
        yield "❌ 请输入问题", "", "", ""
        return
    
    # 加载知识库索引
    if not load_knowledge_base_index(kb_name):
        yield f"❌ 无法加载知识库 '{kb_name}'，请先构建索引", "", "", ""
        return
    
    # 获取共享检索器
    retriever = init_shared_retriever()
    
    # 初始化 Kimi 客户端
    client = init_kimi_client()
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 初始状态
        yield "⏳ 正在重写查询...", "", "", "⏳ 计时中..."
        
        # 1. 查询重写（在事件循环中运行异步函数）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewritten_queries = loop.run_until_complete(rewrite_query(client, question))
        finally:
            loop.close()
        
        yield "⏳ 正在检索文档（多线程并行）...", "", "", "⏳ 计时中..."
        
        # 2. 混合检索（多线程并行）
        all_candidate_chunks = parallel_search(retriever, rewritten_queries)
        final_candidates = list(all_candidate_chunks.values())
        
        yield "⏳ 正在重排序...", "", "", "⏳ 计时中..."
        
        # 3. 最终重排
        pairs = [(question, chunk['text']) for chunk in final_candidates]
        scores = retriever.reranker.compute_score(pairs, batch_size=4)
        
        # Handle case where scores might be None
        if scores is None:
            scores = [0.0] * len(final_candidates)
        
        scored_chunks = list(zip(scores, final_candidates))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_k_chunks = [chunk for score, chunk in scored_chunks[:3]]
        
        # 构建上下文
        context_str = ""
        for idx, chunk in enumerate(top_k_chunks, 1):
            context_str += f"[文档片段 {idx}]\n{chunk['text']}\n\n"
        
        # 流式显示上下文
        yield "⏳ 正在生成答案...", context_str, "", "⏳ 计时中..."
        
        # 4. 生成答案（流式）
        system_prompt = """你是一个问答助手。请根据下面提供的 [上下文]，用中文回答 [用户问题]。

你的回答必须严格基于 [上下文] 包含的信息，禁止捏造。如果上下文中没有足够的信息，请明确说明。"""
        
        user_prompt = f"""[上下文]:
{context_str}

[用户问题]:
{question}"""
        
        agent = Agent(
            model=client,
            markdown=False
        )
        
        combined_prompt = f"""{system_prompt}

---

{user_prompt}"""
        
        # 流式生成答案
        generated_answer = ""
        for chunk in agent.run(combined_prompt, stream=True):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)
            
            if content:
                generated_answer += content
                elapsed = time.time() - start_time
                yield generated_answer, context_str, "", f"⏳ 生成中... ({elapsed:.1f}s)"
        
        # 5. 评测（如果启用）
        eval_result = ""
        
        # 计算答案生成完成的时间
        answer_time = time.time() - start_time
        
        if enable_eval:
            yield generated_answer, context_str, "⏳ 正在评测答案...", f"⏳ 答案生成耗时: {answer_time:.1f}s"
            
            standard_answer = None
            
            # 如果选择了评测集，尝试查找匹配的标准答案
            if eval_set_name:
                eval_file = EVAL_SETS_DIR / f"{eval_set_name}.xlsx"
                if eval_file.exists():
                    df = pd.read_excel(eval_file)
                    
                    # 标准化问题文本用于匹配（去除空格、标点差异）
                    def normalize_text(text):
                        import re
                        if pd.isna(text):
                            return ""
                        text = str(text).strip()
                        # 移除所有空白字符
                        text = re.sub(r'\s+', '', text)
                        # 统一中英文标点
                        text = text.replace('？', '?').replace('。', '.').replace('，', ',')
                        return text.lower()
                    
                    user_question_normalized = normalize_text(question)
                    
                    # 查找匹配的行
                    matching_rows = df[df['query'].apply(normalize_text) == user_question_normalized]
                    
                    # 如果精确匹配找不到，尝试包含匹配
                    if matching_rows.empty:
                        matching_rows = df[df['query'].apply(lambda x: normalize_text(x) in user_question_normalized or user_question_normalized in normalize_text(x))]
                    
                    if not matching_rows.empty:
                        standard_answer = matching_rows.iloc[0]['standard_answer']
            
            # 根据是否有标准答案选择评测方式（流式输出）
            if standard_answer:
                # 有标准答案：使用标准答案进行评测
                eval_header = f"""📊 评测结果（对比标准答案）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 标准答案:
{standard_answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 AI 评测中...
"""
                yield generated_answer, context_str, eval_header, f"⏳ 答案生成耗时: {answer_time:.1f}s"
                
                # 流式评测
                eval_content = ""
                for eval_chunk in stream_evaluate_with_standard(
                    client, question, generated_answer, standard_answer, context_str
                ):
                    eval_content = eval_chunk
                    yield generated_answer, context_str, eval_header + eval_content, f"⏳ 评测中..."
                
                # 解析评测结果
                evaluation = parse_evaluation_json(eval_content)
                eval_result = f"""📊 评测结果（对比标准答案）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 标准答案:
{standard_answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ 得分: {evaluation.get('score', 0.0):.2f} / 1.0

💡 评测理由:
{evaluation.get('reasoning', '无')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            else:
                # 无标准答案：使用基于上下文的自我评测
                eval_header = """📊 评测结果（基于上下文自评）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️ 评测模式: 未找到标准答案，使用上下文自评

🤖 AI 评测中...
"""
                yield generated_answer, context_str, eval_header, f"⏳ 答案生成耗时: {answer_time:.1f}s"
                
                # 流式评测
                eval_content = ""
                for eval_chunk in stream_evaluate_answer(
                    client, question, generated_answer, context_str
                ):
                    eval_content = eval_chunk
                    yield generated_answer, context_str, eval_header + eval_content, f"⏳ 评测中..."
                
                # 解析评测结果
                evaluation = parse_evaluation_json(eval_content)
                eval_result = f"""📊 评测结果（基于上下文自评）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ️ 评测模式: 未找到标准答案，使用上下文自评

⭐ 得分: {evaluation.get('score', 0.0):.2f} / 1.0

💡 评测理由:
{evaluation.get('reasoning', '无')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 耗时统计单独输出
        time_stats = f"⏱️ 耗时统计:\n━━━━━━━━━━━━━━━━━━━━\n• 答案生成: {answer_time:.1f}s\n• 总耗时: {total_time:.1f}s"
        
        # 最终结果
        yield generated_answer, context_str, eval_result, time_stats
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ 处理失败：{str(e)}", "", "", "❌ 计时失败"


def refresh_kb_list():
    """刷新知识库列表"""
    choices = get_knowledge_bases()
    return (
        gr.update(choices=choices),
        gr.update(choices=choices),
        gr.update(choices=choices),
        gr.update(choices=choices)
    )


def refresh_eval_list():
    """刷新评测集列表"""
    choices = get_eval_sets()
    return (
        gr.update(choices=choices),
        gr.update(choices=choices)
    )


# 创建 Gradio 界面
def create_webui():
    """创建 Gradio WebUI"""
    
    with gr.Blocks(title="Agentic RAG System") as app:
        gr.Markdown("""
        # 🤖 Agentic RAG System
        
        智能检索增强生成系统 - 知识库管理与问答平台
        """)
        
        with gr.Tabs():
            # Tab 1: 知识库管理
            with gr.Tab("📚 知识库管理"):
                gr.Markdown("### 创建与管理知识库")
                
                with gr.Row():
                    with gr.Column():
                        kb_name_input = gr.Textbox(
                            label="知识库名称",
                            placeholder="输入新知识库的名称..."
                        )
                        create_kb_btn = gr.Button("创建知识库", variant="primary")
                        create_kb_output = gr.Textbox(label="操作结果", lines=2)
                    
                    with gr.Column():
                        kb_list = gr.Dropdown(
                            label="选择知识库",
                            choices=get_knowledge_bases(),
                            interactive=True
                        )
                        refresh_kb_btn = gr.Button("刷新列表")
                        delete_kb_btn = gr.Button("删除选中的知识库", variant="stop")
                        delete_kb_output = gr.Textbox(label="操作结果", lines=2)
                
                gr.Markdown("---")
                gr.Markdown("### 上传文档与构建索引")
                
                with gr.Row():
                    with gr.Column():
                        kb_select_for_upload = gr.Dropdown(
                            label="选择知识库",
                            choices=get_knowledge_bases(),
                            interactive=True
                        )
                        file_upload = gr.Files(
                            label="上传 Word 文档 (.docx)",
                            file_types=[".docx", ".doc"]
                        )
                        upload_btn = gr.Button("上传文档", variant="primary")
                        upload_output = gr.Textbox(label="上传结果", lines=3)
                    
                    with gr.Column():
                        build_kb_select = gr.Dropdown(
                            label="选择知识库",
                            choices=get_knowledge_bases(),
                            interactive=True
                        )
                        build_btn = gr.Button("构建知识库索引", variant="primary")
                        build_output = gr.Textbox(label="构建结果", lines=3)
            
            # Tab 2: 评测集管理
            with gr.Tab("📊 评测集管理"):
                gr.Markdown("""
                ### 上传评测集
                
                评测集必须是 Excel (.xlsx) 格式，包含以下列：
                - `query`: 问题
                - `standard_answer`: 标准答案
                """)
                
                with gr.Row():
                    with gr.Column():
                        eval_file_upload = gr.File(
                            label="上传评测集 Excel 文件 (.xlsx)",
                            file_types=[".xlsx"]
                        )
                        upload_eval_btn = gr.Button("上传评测集", variant="primary")
                        upload_eval_output = gr.Textbox(label="上传结果", lines=3)
                    
                    with gr.Column():
                        eval_list = gr.Dropdown(
                            label="已上传的评测集",
                            choices=get_eval_sets(),
                            interactive=True
                        )
                        refresh_eval_btn = gr.Button("刷新评测集列表")
            
            # Tab 3: 问答与评测
            with gr.Tab("💬 问答与评测"):
                gr.Markdown("### 智能问答系统")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        qa_kb_select = gr.Dropdown(
                            label="选择知识库",
                            choices=get_knowledge_bases(),
                            interactive=True
                        )
                        qa_eval_select = gr.Dropdown(
                            label="选择评测集（可选）",
                            choices=get_eval_sets(),
                            interactive=True,
                            value=None
                        )
                        enable_eval_checkbox = gr.Checkbox(
                            label="启用答案评测（需选择评测集）",
                            value=True
                        )
                        refresh_qa_lists_btn = gr.Button("刷新列表")
                    
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="输入问题",
                            placeholder="在这里输入您的问题...",
                            lines=3
                        )
                        ask_btn = gr.Button("提问", variant="primary", size="lg")
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="🤖 生成答案",
                            lines=10
                        )
                    
                    with gr.Column():
                        context_output = gr.Textbox(
                            label="📄 检索上下文",
                            lines=10
                        )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        eval_output = gr.Textbox(
                            label="📊 评测结果",
                            lines=8
                        )
                    
                    with gr.Column(scale=1):
                        time_output = gr.Textbox(
                            label="⏱️ 耗时统计",
                            lines=8
                        )
        
        # 事件绑定
        create_kb_btn.click(
            fn=create_knowledge_base,
            inputs=[kb_name_input],
            outputs=[create_kb_output]
        )
        
        delete_kb_btn.click(
            fn=delete_knowledge_base,
            inputs=[kb_list],
            outputs=[delete_kb_output]
        )
        
        refresh_kb_btn.click(
            fn=refresh_kb_list,
            outputs=[kb_list, kb_select_for_upload, build_kb_select, qa_kb_select]
        )
        
        upload_btn.click(
            fn=upload_document,
            inputs=[kb_select_for_upload, file_upload],
            outputs=[upload_output]
        )
        
        build_btn.click(
            fn=build_knowledge_base,
            inputs=[build_kb_select],
            outputs=[build_output]
        )
        
        upload_eval_btn.click(
            fn=upload_eval_set,
            inputs=[eval_file_upload],
            outputs=[upload_eval_output]
        )
        
        refresh_eval_btn.click(
            fn=refresh_eval_list,
            outputs=[eval_list, qa_eval_select]
        )
        
        refresh_qa_lists_btn.click(
            fn=lambda: (gr.update(choices=get_knowledge_bases()), gr.update(choices=get_eval_sets())),
            outputs=[qa_kb_select, qa_eval_select]
        )
        
        ask_btn.click(
            fn=answer_question_sync,
            inputs=[qa_kb_select, qa_eval_select, question_input, enable_eval_checkbox],
            outputs=[answer_output, context_output, eval_output, time_output]
        )
    
    return app


def main():
    """主函数"""
    print("=" * 80)
    print("正在启动 Agentic RAG WebUI...")
    print("=" * 80)
    
    # 预加载模型（启动时加载，避免后续重复加载）
    print("\n📦 预加载模型...")
    init_shared_retriever()
    init_kimi_client()
    init_thread_pool(max_workers=4)
    print("\n✅ 所有模型预加载完成！\n")
    
    app = create_webui()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
