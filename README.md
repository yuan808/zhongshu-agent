# 种薯 Agent 后端

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 3. 启动
python main.py
```

启动后访问 http://localhost:8000/docs 查看 API 文档。

## 连接前端

在浏览器打开种薯前端时，URL 加上 `?api=` 参数：

```
zhongshu/index.html?api=http://localhost:8000
```

或部署后：
```
zhongshu/index.html?api=https://your-vercel-app.vercel.app
```

## 部署到 Vercel（免费）

```bash
# 1. 安装 Vercel CLI
npm i -g vercel

# 2. 在 server/ 目录下执行
cd zhongshu/server
vercel

# 3. 设置环境变量
vercel env add DEEPSEEK_API_KEY
```

## API 接口

### POST /api/session
创建会话。

```json
{"role": "brand", "nickname": "小薯"}
```
返回：`{"session_id": "uuid"}`

### POST /api/chat
发送消息，获取 Agent 回复。

```json
{"session_id": "xxx", "role": "brand", "message": "帮我分析这个商品 https://..."}
```
返回：`{"text": "分析说明", "note": {"title": "...", "body": "...", "tags": [...]}}`

### POST /api/rewrite
改写已有文案。

```json
{"action": "rewrite_title", "title": "原标题", "body": "原正文", "tags": ["原标签"]}
```

## 架构说明

```
用户消息 → LLM（带 Function Calling）
              ↓ 决定调用工具？
         ┌─── YES → 执行工具 → 结果反馈给 LLM → 再次决定
         └─── NO  → 直接生成回复 → 解析结构化笔记 → 返回前端
```

这就是 Agent 和普通聊天机器人的核心区别：LLM 自主决定何时调用工具，可以多轮循环。
