"""全局配置"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM 配置
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 数据路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, os.getenv("DATA_DIR", "data/cases"))

# 流程控制
MAX_ITERATIONS = 3  # 情报回溯最大轮次

# ChromaDB 持久化路径
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data/chroma_db")

# 知识图谱持久化路径
GRAPH_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data/graphs")
