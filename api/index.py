"""
Vercel Serverless 入口文件
将 main.py 的 FastAPI app 暴露给 Vercel
"""
import sys
import os

# 将 server 根目录加入 path，这样能 import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
