import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from graph.state import PersonalityProfile

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DB_PATH = Path(os.getenv("APP_DB_PATH", ROOT / "data" / "app.sqlite3"))


@tool("create_role_play_conversation")
def create_role_play_conversation(
    initial_context: str,
    character_id: str,
    character_name: str,
    character_identity: str,
    personality: PersonalityProfile,
    config: RunnableConfig,
) -> str:
    """Create a Role Play conversation and return its internal conversation ID.

    Args:
        initial_context: Opening situation for the Role Play conversation.
        character_id: Stable ID for this character, or an empty string for a new ID.
        character_name: Character name.
        character_identity: Character identity and background.
        personality: Character personality profile.
    """
    character_id = character_id or str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    config_json = json.dumps({
        "character_id": character_id,
        "character_name": character_name,
        "character_identity": character_identity,
        "initial_context": initial_context,
        "personality": personality.model_dump(),
    }, ensure_ascii=False)
    parent_id = config.get("configurable", {}).get("thread_id")

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """INSERT INTO conversations
            (conversation_id, mode, character_id, character_name, config_json,
             created_at, updated_at, parent_conversation_id)
            VALUES (?, 'role-play', ?, ?, ?, ?, ?, ?)""",
            (conversation_id, character_id, character_name, config_json,
             created_at, created_at, parent_id),
        )

    return f"角色对话已创建，conversation_id={conversation_id}。"
