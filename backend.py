"""HTTP API for the Assistant and Role Play graphs."""

import json
import os
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from graph.assistant_graph import build_assistant_graph
from graph.character_graph import build_character_graph
from graph.state import PersonalityProfile
from memory.memory import MemoryRAG
from tool.create_Conversation import create_role_play_conversation
from tool.get_character import (
    create_character_messages_tool,
    create_character_state_tool,
    get_existing_characters,
)
from tool.get_skill import get_skill
from tool.memory_search import create_memory_search_tool
from tool.web_search import web_search

ROOT = Path(__file__).resolve().parent
load_dotenv()
DB_PATH = Path(os.getenv("APP_DB_PATH", ROOT / "data" / "app.sqlite3"))


class SendMessageRequest(BaseModel):
    content: str


def now():
    return datetime.now(timezone.utc).isoformat()


def load_conversation(conversation_id: str):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return row


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            character_id TEXT,
            character_name TEXT,
            config_json TEXT,
            parent_conversation_id TEXT,
            announced INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            turn_count INTEGER NOT NULL DEFAULT 0
        )""")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在项目根目录的 .env 中配置。")
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )
    memory_rag = MemoryRAG(db_name=os.getenv("MEMORY_DB_NAME", "role_play_memory"))

    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        app.state.character_graph = build_character_graph(
            llm=llm,
            memory_rag=memory_rag,
            checkpointer=checkpointer,
        )
        app.state.assistant_graph = build_assistant_graph(
            agent_llm=llm,
            tools=[
                create_role_play_conversation,
                get_existing_characters,
                create_character_messages_tool(app.state.character_graph),
                create_character_state_tool(app.state.character_graph),
                get_skill,
                web_search,
                create_memory_search_tool(memory_rag, owner="assistant"),
            ],
            checkpointer=checkpointer,
            memory_rag=memory_rag,
        )
        app.state.conversation_locks = {}
        yield


app = FastAPI(title="Role Play Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/conversations")
def create_conversation():
    conversation_id = str(uuid.uuid4())
    created_at = now()

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """INSERT INTO conversations
            (conversation_id, mode, character_id, character_name, config_json,
             created_at, updated_at, parent_conversation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL)""",
            (conversation_id, "assistant", None, None, None, created_at, created_at),
        )
    return {
        "conversation_id": conversation_id,
        "mode": "assistant",
        "character_id": None,
        "character_name": None,
        "created_at": created_at,
    }


@app.get("/conversations")
def list_conversations():
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT conversation_id, mode, character_id, character_name,
            created_at, updated_at, turn_count FROM conversations
            ORDER BY updated_at DESC"""
        ).fetchall()
    return {"conversations": [dict(row) for row in rows]}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    row = load_conversation(conversation_id)
    graph = (app.state.assistant_graph if row["mode"] == "assistant"
             else app.state.character_graph)
    snapshot = graph.get_state({"configurable": {"thread_id": conversation_id}})
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    return {
        "conversation_id": conversation_id,
        "mode": row["mode"],
        "character_id": row["character_id"],
        "character_name": row["character_name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "messages": [
            {"role": "user" if isinstance(message, HumanMessage) else "assistant",
             "content": message.content}
            for message in messages
            if isinstance(message, HumanMessage)
            or (isinstance(message, AIMessage) and not message.tool_calls)
        ],
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    load_conversation(conversation_id)
    lock = app.state.conversation_locks.setdefault(conversation_id, threading.Lock())
    with lock, sqlite3.connect(DB_PATH) as db:
        # LangGraph's SQLite checkpointer stores state in tables keyed by thread_id.
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        for (table_name,) in tables:
            columns = {row[1] for row in db.execute(f'PRAGMA table_info("{table_name}")')}
            if "thread_id" in columns:
                db.execute(f'DELETE FROM "{table_name}" WHERE thread_id = ?', (conversation_id,))
        db.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    return {"conversation_id": conversation_id, "deleted": True}


@app.post("/conversations/{conversation_id}/messages")
def send_message(conversation_id: str, request: SendMessageRequest):
    load_conversation(conversation_id)
    lock = app.state.conversation_locks.setdefault(conversation_id, threading.Lock())
    with lock:
        row = load_conversation(conversation_id)
        graph = (app.state.assistant_graph if row["mode"] == "assistant"
                 else app.state.character_graph)
        state = {"messages": [HumanMessage(content=request.content)]}

        if row["turn_count"] == 0 and row["mode"] == "assistant":
            state["summary"] = ""
        elif row["turn_count"] == 0:
            role = json.loads(row["config_json"])
            state.update({
                "summary": f"初始情景：{role['initial_context']}",
                "retry_count": 0,
                "character_id": role["character_id"],
                "character_name": role["character_name"],
                "character_identity": role["character_identity"],
                "personality": PersonalityProfile(**role["personality"]),
            })

        result = graph.invoke(
            state,
            config={"configurable": {
                "thread_id": conversation_id,
                "conversation_turn_count": row["turn_count"],
            }},
        )
        answer = next(
            (message.content for message in reversed(result.get("messages", []))
             if isinstance(message, AIMessage) and not message.tool_calls),
            "",
        )
        updated_at = now()
        with sqlite3.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            db.execute(
                "UPDATE conversations SET turn_count = turn_count + 1, updated_at = ? WHERE conversation_id = ?",
                (updated_at, conversation_id),
            )
            created = db.execute(
                """SELECT conversation_id, character_id, character_name, created_at
                FROM conversations
                WHERE parent_conversation_id = ? AND announced = 0""",
                (conversation_id,),
            ).fetchall()
            db.execute(
                "UPDATE conversations SET announced = 1 WHERE parent_conversation_id = ? AND announced = 0",
                (conversation_id,),
            )

    return {
        "conversation_id": conversation_id,
        "reply": answer,
        "check_result": result.get("check_result"),
        "memory_update_status": result.get("memory_update_status"),
        "created_conversations": [dict(item) for item in created],
    }


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
