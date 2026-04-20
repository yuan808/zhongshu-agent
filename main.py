"""
种薯 Agent 后端 - 最小可行版
=============================
功能：
1. /api/session - 创建会话（按角色维护对话历史）
2. /api/chat    - 核心对话（LLM + Function Calling + 工具调用）
3. /api/rewrite - 文案改写（标题/正文/标签优化）

工具：
- fetch_product_info: 抓取商品链接，提取名称/价格/卖点
- search_trending_tags: 搜索小红书热门话题标签
- analyze_image: 识别商品图片内容（预留）

部署：
  pip install -r requirements.txt
  cp .env.example .env  # 填入 DEEPSEEK_API_KEY
  python main.py
"""

import os
import json
import uuid
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"  # DeepSeek-V3，支持 function calling

app = FastAPI(title="种薯 Agent API", version="1.0.0")

# CORS - 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改成具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# 会话存储（内存，重启丢失；生产环境用 Redis）
# ══════════════════════════════════════════════════════════════
sessions: dict = {}  # session_id -> {role, messages, created_at}


# ══════════════════════════════════════════════════════════════
# System Prompts - 三个角色的人格定义
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPTS = {
    "brand": """你是「种薯」的品牌种草助手，专门帮品牌方在小红书做内容营销。

你的能力：
1. 分析商品链接/图片，提取核心卖点
2. 根据品牌定位生成种草文案（符合小红书调性）
3. 推荐高流量话题标签
4. 规划内容矩阵（多角度系列文案）
5. 给出聚光投放建议

工作流程：
- 用户发来商品链接时，先调用 fetch_product_info 工具获取真实商品信息
- 根据商品信息分析卖点、目标人群
- 生成符合小红书调性的种草文案
- 推荐话题标签（可调用 search_trending_tags 获取实时热门标签）

输出格式要求：
当生成笔记时，必须返回 JSON 格式：
{"text": "你的分析过程和建议", "note": {"title": "笔记标题", "body": "笔记正文", "tags": ["标签1", "标签2", ...]}}

风格要求：
- 语气专业但亲和，像一个有经验的小红书营销顾问
- 文案要有真实感，不要太硬广
- 善用 emoji 但不过度""",

    "seller": """你是「种薯」的带货文案助手，专门帮个人卖家写高转化的种草笔记。

你的能力：
1. 分析商品链接/图片，提取卖点和用户痛点
2. 生成高转化的带货笔记（真实用户分享风格）
3. 优化标题提升点击率
4. 推荐精准话题标签
5. 给出薯条加热建议

工作流程：
- 用户发来商品链接时，先调用 fetch_product_info 工具获取真实商品信息
- 从「用户使用体验」角度切入，生成有真实感的种草笔记
- 标题突出利益点和好奇心
- 推荐话题标签（可调用 search_trending_tags 获取实时热门标签）

输出格式要求：
当生成笔记时，必须返回 JSON 格式：
{"text": "你的分析和建议", "note": {"title": "笔记标题", "body": "笔记正文", "tags": ["标签1", "标签2", ...]}}

风格要求：
- 像闺蜜分享好物一样真实、有温度
- 多用第一人称，强调真实使用体验
- 标题要有紧迫感或利益点
- 适当用 emoji 增加亲切感""",

    "kol": """你是「种薯」的达人创作助手，专门帮内容达人写有个人风格的种草文案。

你的能力：
1. 分析商品链接/图片，提取适合内容创作的角度
2. 生成有达人个人风格的种草文案
3. 兼顾内容质量和商业价值
4. 推荐能涨粉的话题标签
5. 给出账号内容规划建议

工作流程：
- 用户发来商品链接时，先调用 fetch_product_info 工具获取真实商品信息
- 从达人专业测评/生活分享角度切入
- 平衡内容价值和商业转化
- 推荐话题标签（可调用 search_trending_tags 获取实时热门标签）

输出格式要求：
当生成笔记时，必须返回 JSON 格式：
{"text": "你的建议", "note": {"title": "笔记标题", "body": "笔记正文", "tags": ["标签1", "标签2", ...]}}

风格要求：
- 有专业度和个人见解，像一个真正的测评博主
- 内容有深度，不是简单堆砌卖点
- 善于用对比、故事、数据增加说服力
- 平衡种草和真实感，避免硬广""",
}


# ══════════════════════════════════════════════════════════════
# 工具定义 - Function Calling Schema
# ══════════════════════════════════════════════════════════════
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_product_info",
            "description": "抓取电商商品页面，提取商品名称、价格、核心卖点、用户评价等信息。支持淘宝、天猫、京东、拼多多等主流平台链接。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "商品页面 URL，例如 https://item.taobao.com/item.htm?id=xxx"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_trending_tags",
            "description": "搜索小红书上当前热门的话题标签，帮助提升笔记曝光。根据商品类目或关键词，返回相关的高热度标签。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "搜索关键词，例如 '护肤'、'美白精华'、'平价好物'"
                    },
                    "category": {
                        "type": "string",
                        "description": "商品类目，例如 '美妆护肤'、'食品饮料'、'数码家电'",
                    }
                },
                "required": ["keyword"]
            }
        }
    },
]


# ══════════════════════════════════════════════════════════════
# 工具实现
# ══════════════════════════════════════════════════════════════
async def fetch_product_info(url: str) -> str:
    """
    用 Jina Reader API 抓取商品页面内容。
    Jina Reader 免费、不需要 API Key、支持 JS 渲染。
    """
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                jina_url,
                headers={
                    "Accept": "text/plain",
                    "X-Return-Format": "text",
                }
            )
            if resp.status_code == 200:
                content = resp.text[:3000]  # 限制长度，避免 token 爆炸
                return f"✅ 成功抓取商品页面内容：\n\n{content}"
            else:
                return f"⚠️ 抓取失败（HTTP {resp.status_code}），请检查链接是否正确。将基于你的描述生成文案。"
    except Exception as e:
        return f"⚠️ 网络请求失败：{str(e)}。将基于你的描述生成文案。"


async def search_trending_tags(keyword: str, category: str = "") -> str:
    """
    搜索热门标签。
    当前实现：用 Jina Search 搜索小红书相关话题。
    未来可接入新红数据/千瓜数据等专业 API。
    """
    search_query = f"小红书 热门话题标签 {keyword} {category}".strip()
    jina_search_url = f"https://s.jina.ai/{search_query}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                jina_search_url,
                headers={"Accept": "text/plain"}
            )
            if resp.status_code == 200:
                content = resp.text[:1500]  # 限制长度避免 token 过多
                return f"✅ 搜索到与「{keyword}」相关的热门话题：\n\n{content}"
            else:
                # 降级：返回通用建议
                return _fallback_tags(keyword)
    except Exception:
        return _fallback_tags(keyword)


def _fallback_tags(keyword: str) -> str:
    """降级标签建议"""
    return f"""基于「{keyword}」的通用标签建议：
高流量标签：#{keyword} #好物推荐 #真实测评 #分享日常
精准人群标签：#{keyword}推荐 #{keyword}测评 #平价{keyword}
场景标签：#日常分享 #购物分享 #好物分享
建议每篇笔记选5-8个标签，高流量+精准混搭效果最好。"""


# 工具调度器
TOOL_HANDLERS = {
    "fetch_product_info": fetch_product_info,
    "search_trending_tags": search_trending_tags,
}


# ══════════════════════════════════════════════════════════════
# API 接口
# ══════════════════════════════════════════════════════════════

class SessionRequest(BaseModel):
    role: str  # brand / seller / kol
    nickname: Optional[str] = "小薯"


class ChatRequest(BaseModel):
    session_id: str
    role: str
    message: str


class RewriteRequest(BaseModel):
    action: str  # "rewrite_title" / "rewrite_body" / "rewrite_tags" / "polish"
    title: Optional[str] = ""
    body: Optional[str] = ""
    tags: Optional[list] = []
    instruction: Optional[str] = ""  # 用户的改写要求


@app.post("/api/session")
async def create_session(req: SessionRequest):
    """创建会话，返回 session_id"""
    if req.role not in SYSTEM_PROMPTS:
        raise HTTPException(400, f"不支持的角色：{req.role}，可选：brand/seller/kol")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "role": req.role,
        "nickname": req.nickname,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[req.role]}
        ],
        "created_at": time.time(),
    }
    return {"session_id": session_id}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    核心对话接口 - Agent ReAct 循环：
    1. 用户消息加入历史
    2. 调用 LLM（带 function calling）
    3. 如果 LLM 决定调用工具 → 执行工具 → 把结果反馈给 LLM → 再次调用
    4. 循环直到 LLM 直接回复用户
    5. 解析回复，提取结构化笔记
    """
    if req.session_id not in sessions:
        raise HTTPException(404, "会话不存在或已过期，请重新创建")

    session = sessions[req.session_id]

    # 加入用户消息
    session["messages"].append({"role": "user", "content": req.message})

    # Agent 循环（最多 5 轮工具调用，防止无限循环）
    max_tool_rounds = 5
    for _ in range(max_tool_rounds):
        # 调用 LLM
        llm_response = await call_deepseek(
            messages=session["messages"],
            tools=TOOLS,
        )

        choice = llm_response["choices"][0]
        message = choice["message"]

        # 如果 LLM 决定调用工具
        if message.get("tool_calls"):
            # 记录 assistant 的工具调用意图
            session["messages"].append(message)

            # 执行每个工具调用
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])

                # 执行工具
                handler = TOOL_HANDLERS.get(func_name)
                if handler:
                    result = await handler(**func_args)
                else:
                    result = f"未知工具：{func_name}"

                # 把工具结果加入历史
                session["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })

            # 继续循环，让 LLM 看到工具结果后再决定
            continue

        # LLM 直接回复用户（没有工具调用），退出循环
        assistant_text = message.get("content", "")
        session["messages"].append({"role": "assistant", "content": assistant_text})
        break
    else:
        # 超过最大轮数，强制结束
        assistant_text = "抱歉，处理过程太复杂了，请简化你的需求再试一次。"

    # 解析回复：尝试提取结构化笔记
    response = parse_agent_response(assistant_text)
    return response


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest):
    """
    文案改写接口 - 对已生成的笔记做局部优化
    """
    action_prompts = {
        "rewrite_title": f"请为以下笔记重新生成5个更有吸引力的标题备选：\n\n原标题：{req.title}\n正文摘要：{req.body[:200]}",
        "rewrite_body": f"请优化以下笔记正文，使其更符合小红书调性，更有感染力：\n\n标题：{req.title}\n原文：{req.body}\n\n用户要求：{req.instruction or '优化语言表达，增加真实感'}",
        "rewrite_tags": f"请为以下笔记重新推荐10个更精准的话题标签：\n\n标题：{req.title}\n正文：{req.body[:300]}\n原标签：{', '.join(req.tags or [])}",
        "polish": f"请全面润色以下笔记（标题+正文+标签都优化）：\n\n标题：{req.title}\n正文：{req.body}\n标签：{', '.join(req.tags or [])}\n\n用户要求：{req.instruction or '整体提升质量'}",
    }

    prompt = action_prompts.get(req.action)
    if not prompt:
        raise HTTPException(400, f"不支持的改写动作：{req.action}")

    system_msg = """你是小红书文案优化专家。根据用户要求优化笔记内容。
必须返回 JSON 格式：{"text": "你的优化说明", "note": {"title": "新标题", "body": "新正文", "tags": ["新标签1", ...]}}
如果只改写标题，body和tags保持原文；如果只改标签，title和body保持原文。"""

    llm_response = await call_deepseek(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        tools=None,
    )

    assistant_text = llm_response["choices"][0]["message"].get("content", "")
    return parse_agent_response(assistant_text)


# ══════════════════════════════════════════════════════════════
# LLM 调用
# ══════════════════════════════════════════════════════════════
async def call_deepseek(messages: list, tools: Optional[list] = None) -> dict:
    """调用 DeepSeek API（兼容 OpenAI 格式）"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(500, "未配置 DEEPSEEK_API_KEY，请在 .env 文件中设置")

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 2000,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"  # 让模型自己决定是否调用工具

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        if resp.status_code != 200:
            error_text = resp.text
            raise HTTPException(502, f"DeepSeek API 错误：{resp.status_code} - {error_text}")

        return resp.json()


# ══════════════════════════════════════════════════════════════
# 响应解析
# ══════════════════════════════════════════════════════════════
def parse_agent_response(text: str) -> dict:
    """
    尝试从 LLM 回复中提取结构化笔记。
    LLM 可能返回：
    1. 纯 JSON（理想情况）
    2. Markdown 包裹的 JSON（```json ... ```）
    3. 带有自然语言 + JSON 混合的文本
    4. 纯文本（没有笔记，只是对话）
    """
    # 尝试提取 JSON
    note_preview = None

    # 方法1：整体是 JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "note" in data:
            note_preview = data["note"]
            return {"text": data.get("text", ""), "note": note_preview}
    except (json.JSONDecodeError, TypeError):
        pass

    # 方法2：从 ```json ``` 块中提取
    import re
    json_match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, dict) and "note" in data:
                note_preview = data["note"]
                clean_text = text[:json_match.start()].strip()
                return {"text": clean_text or data.get("text", ""), "note": note_preview}
        except (json.JSONDecodeError, TypeError):
            pass

    # 方法3：从文本中找 {"note": ...} 模式
    json_match = re.search(r'\{[^{}]*"note"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "note" in data:
                note_preview = data["note"]
                clean_text = text[:json_match.start()].strip()
                return {"text": clean_text or data.get("text", ""), "note": note_preview}
        except (json.JSONDecodeError, TypeError):
            pass

    # 方法4：纯文本回复
    return {"text": text, "note": None}


# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"""
╔══════════════════════════════════════════════════╗
║  🌱 种薯 Agent 后端已启动                        ║
║  地址: http://localhost:{port}                    ║
║  文档: http://localhost:{port}/docs               ║
║                                                  ║
║  前端连接方式:                                    ║
║  打开 zhongshu/index.html?api=http://localhost:{port} ║
╚══════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=port)
