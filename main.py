"""
种薯 Agent 后端 - 最小可行版
=============================
功能：
1. /api/session - 创建会话（按角色维护对话历史）
2. /api/chat    - 核心对话（LLM + Function Calling + 工具调用）
3. /api/chat/stream - 流式对话（SSE，逐字输出）
4. /api/rewrite - 文案改写（标题/正文/标签优化）

工具：
- fetch_product_info: 抓取商品链接，提取名称/价格/卖点
- search_trending_tags: 搜索小红书热门话题标签

部署：
  pip install -r requirements.txt
  cp .env.example .env  # 填入 DEEPSEEK_API_KEY
  python main.py
"""

import os
import re
import json
import uuid
import time
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

app = FastAPI(title="种薯 Agent API", version="2.0.0")

# CORS - 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    "brand": """你是「种薯」品牌种草助手。用户发来商品关键词或链接，立即生成种草笔记，绝对不要反问。

工作流程：收到链接先调用 fetch_product_info，收到关键词直接写。可调用 search_trending_tags 获取标签。

██ 最重要的规则 ██
body 正文必须 ≤ 70个汉字！只写4句短句，用\\n分隔。每句不超过18个字。像发朋友圈一样简短有力。超过70字视为失败输出！

输出格式（只返回JSON，无其他内容）：
{"text":"","note":{"title":"15字以内","body":"4句短句用\\n连接","tags":["标签1","标签2","标签3","标签4","标签5"]}}

完整示例（请严格模仿这个长度）：
{"text":"","note":{"title":"这防晒霜也太绝了吧","body":"之前做代购被假货坑惨了\\n自己跑工厂才找到这款宝藏\\n油皮上脸秒变哑光清爽不闷痘\\n现在只要79真的闭眼入","tags":["#防晒霜","#好物推荐","#平价好物","#护肤","#真实测评"]}}

再次强调：body ≤ 70字，4句话，多一个字都不行！""",

    "seller": """你是「种薯」带货文案助手。用户发来商品关键词或链接，立即生成种草笔记，绝对不要反问。像闺蜜安利好物一样写。

工作流程：收到链接先调用 fetch_product_info，收到关键词直接写。可调用 search_trending_tags 获取标签。

██ 最重要的规则 ██
body 正文必须 ≤ 70个汉字！只写4句短句，用\\n分隔。每句不超过18个字。像发朋友圈一样简短有力。超过70字视为失败输出！

输出格式（只返回JSON，无其他内容）：
{"text":"","note":{"title":"15字以内","body":"4句短句用\\n连接","tags":["标签1","标签2","标签3","标签4","标签5"]}}

完整示例（请严格模仿这个长度）：
{"text":"","note":{"title":"姐妹快冲这个袜子太绝了","body":"之前买的袜子不是勒脚就是掉跟\\n这款穿上脚感软fufu的巨舒服\\n透气不闷脚洗了N次不变形\\n一盒才29块闭眼囤不心疼","tags":["#袜子","#好物推荐","#平价好物","#真实分享","#闺蜜推荐"]}}

再次强调：body ≤ 70字，4句话，多一个字都不行！""",

    "kol": """你是「种薯」达人创作助手。用户发来商品关键词或链接，立即生成种草笔记，绝对不要反问。像专业测评博主一样写，数据说话。

工作流程：收到链接先调用 fetch_product_info，收到关键词直接写。可调用 search_trending_tags 获取标签。

██ 最重要的规则 ██
body 正文必须 ≤ 70个汉字！只写4句短句，用\\n分隔。每句不超过18个字。像发朋友圈一样简短有力。超过70字视为失败输出！

输出格式（只返回JSON，无其他内容）：
{"text":"","note":{"title":"15字以内","body":"4句短句用\\n连接","tags":["标签1","标签2","标签3","标签4","标签5"]}}

完整示例（请严格模仿这个长度）：
{"text":"","note":{"title":"测了20款耳机这款封神","body":"测了20款蓝牙耳机这款音质吊打千元\\n降噪拉满地铁上终于听得清\\n续航40小时出差党福音\\n200出头的价格真的离谱","tags":["#蓝牙耳机","#数码好物","#真实测评","#性价比","#好物推荐"]}}

再次强调：body ≤ 70字，4句话，多一个字都不行！""",
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
                content = resp.text[:3000]
                return f"✅ 成功抓取商品页面内容：\n\n{content}"
            else:
                return f"⚠️ 抓取失败（HTTP {resp.status_code}），请检查链接是否正确。将基于你的描述生成文案。"
    except Exception as e:
        return f"⚠️ 网络请求失败：{str(e)}。将基于你的描述生成文案。"


async def search_trending_tags(keyword: str, category: str = "") -> str:
    search_query = f"小红书 热门话题标签 {keyword} {category}".strip()
    jina_search_url = f"https://s.jina.ai/{search_query}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                jina_search_url,
                headers={"Accept": "text/plain"}
            )
            if resp.status_code == 200:
                content = resp.text[:1500]
                return f"✅ 搜索到与「{keyword}」相关的热门话题：\n\n{content}"
            else:
                return _fallback_tags(keyword)
    except Exception:
        return _fallback_tags(keyword)


def _fallback_tags(keyword: str) -> str:
    return f"""基于「{keyword}」的通用标签建议：
高流量标签：#{keyword} #好物推荐 #真实测评 #分享日常
精准人群标签：#{keyword}推荐 #{keyword}测评 #平价{keyword}
场景标签：#日常分享 #购物分享 #好物分享
建议每篇笔记选5-8个标签，高流量+精准混搭效果最好。"""


TOOL_HANDLERS = {
    "fetch_product_info": fetch_product_info,
    "search_trending_tags": search_trending_tags,
}


# ══════════════════════════════════════════════════════════════
# SSE 辅助函数
# ══════════════════════════════════════════════════════════════
def sse_event(event: str, data: dict) -> str:
    """格式化一条 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
    action: str
    title: Optional[str] = ""
    body: Optional[str] = ""
    tags: Optional[list] = []
    instruction: Optional[str] = ""


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


# ── 原有非流式接口（保持兼容） ──────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "会话不存在或已过期，请重新创建")

    session = sessions[req.session_id]
    session["messages"].append({"role": "user", "content": req.message})

    max_tool_rounds = 5
    assistant_text = ""
    for _ in range(max_tool_rounds):
        llm_response = await call_deepseek(
            messages=session["messages"],
            tools=TOOLS,
        )
        choice = llm_response["choices"][0]
        message = choice["message"]

        if message.get("tool_calls"):
            session["messages"].append(message)
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                handler = TOOL_HANDLERS.get(func_name)
                if handler:
                    result = await handler(**func_args)
                else:
                    result = f"未知工具：{func_name}"
                session["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result,
                })
            continue

        assistant_text = message.get("content", "")
        session["messages"].append({"role": "assistant", "content": assistant_text})
        break
    else:
        assistant_text = "抱歉，处理过程太复杂了，请简化你的需求再试一次。"

    return parse_agent_response(assistant_text)


# ── 新增：流式对话接口（SSE） ──────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口 - SSE (Server-Sent Events)
    事件类型:
    - tool_start: 开始调用工具 {name, args}
    - tool_done:  工具调用完成 {name, result_preview}
    - delta:      文本增量 {content}
    - done:       完成 {full_text}
    - error:      出错 {message}
    """
    if req.session_id not in sessions:
        async def error_gen():
            yield sse_event("error", {"message": "会话不存在或已过期，请重新创建"})
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    session = sessions[req.session_id]
    session["messages"].append({"role": "user", "content": req.message})

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            # Phase 1: Agent 工具调用循环（非流式，但发事件通知前端）
            max_tool_rounds = 5
            for _ in range(max_tool_rounds):
                llm_response = await call_deepseek(
                    messages=session["messages"],
                    tools=TOOLS,
                    stream=False,
                )
                choice = llm_response["choices"][0]
                message = choice["message"]

                if not message.get("tool_calls"):
                    break  # 没有工具调用，进入流式生成阶段

                session["messages"].append(message)
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])

                    # 通知前端：开始调用工具
                    yield sse_event("tool_start", {
                        "name": func_name,
                        "args": func_args,
                    })

                    handler = TOOL_HANDLERS.get(func_name)
                    if handler:
                        result = await handler(**func_args)
                    else:
                        result = f"未知工具：{func_name}"

                    session["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    })

                    # 通知前端：工具调用完成
                    yield sse_event("tool_done", {
                        "name": func_name,
                        "result_preview": result[:100],
                    })
            else:
                yield sse_event("delta", {"content": "抱歉，处理过程太复杂了，请简化你的需求再试一次。"})
                yield sse_event("done", {"full_text": ""})
                return

            # Phase 2: 流式生成最终回复
            full_text = ""
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": MODEL,
                    "messages": session["messages"],
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "stream": True,
                }

                async with client.stream(
                    "POST",
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        yield sse_event("error", {"message": f"DeepSeek API 错误：{resp.status_code}"})
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                yield sse_event("delta", {"content": content})
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            # 保存到会话历史
            session["messages"].append({"role": "assistant", "content": full_text})

            # 发送完成事件（包含完整文本，前端用于最终解析笔记结构）
            yield sse_event("done", {"full_text": full_text})

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁止 Nginx/Render 代理缓冲
        },
    )


@app.post("/api/rewrite")
async def rewrite(req: RewriteRequest):
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
# LLM 调用（非流式，用于工具调用阶段）
# ══════════════════════════════════════════════════════════════
async def call_deepseek(messages: list, tools: Optional[list] = None, stream: bool = False) -> dict:
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
        payload["tool_choice"] = "auto"

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
    note_preview = None

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "note" in data:
            note_preview = data["note"]
            return {"text": data.get("text", ""), "note": note_preview}
    except (json.JSONDecodeError, TypeError):
        pass

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
