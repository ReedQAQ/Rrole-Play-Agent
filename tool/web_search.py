import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"


@tool
def web_search(query: str) -> str:
    """使用百度 AI Search 搜索网页并返回搜索结果文本。"""
    api_key = os.getenv("BAIDU_API_KEY")
    if not api_key:
        return "联网搜索不可用：未配置 BAIDU_API_KEY。"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "messages": [{"role": "user", "content": query}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": 5}],
    }

    response = requests.post(URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return str(response.json())
