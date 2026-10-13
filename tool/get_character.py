import json
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from tool.create_Conversation import DB_PATH


@tool("get_existing_characters")
def get_existing_characters() -> str:
    """List existing Role Play characters for identity matching.

    Returns each character's internal ID, name, and identity description.
    Use this before creating a character to decide whether an existing ID
    should be reused, and before reading a character's history or state to
    resolve the exact character_id. The ID is internal and should not be shown
    to the user.
    """
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute(
            """SELECT character_id, character_name, config_json
            FROM conversations
            WHERE mode = 'role-play' AND character_id IS NOT NULL
            ORDER BY updated_at DESC"""
        ).fetchall()

    characters = {}
    for character_id, character_name, config_json in rows:
        if character_id not in characters:
            config = json.loads(config_json)
            characters[character_id] = {
                "character_id": character_id,
                "character_name": character_name,
                "character_identity": config["character_identity"],
            }
    return json.dumps(list(characters.values()), ensure_ascii=False)


def _get_latest_character_conversation(character_id: str):
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            """SELECT conversation_id, character_name, config_json,
                      created_at, updated_at
               FROM conversations
               WHERE mode = 'role-play' AND character_id = ?
               ORDER BY updated_at DESC
               LIMIT 1""",
            (character_id,),
        ).fetchone()
    return row


def create_character_messages_tool(character_graph):
    @tool("get_character_messages")
    def get_character_messages(character_id: str, limit: int = 20) -> str:
        """查看指定角色最近更新的角色扮演对话的上下文和近期消息。

        character_id 必须是 get_existing_characters 返回的内部 ID，不能传角色名称。
        若返回 found=false，表示数据库中没有匹配的角色对话；若 found=true 但消息列表为空，表示会话存在但还没有可读取的消息。
        上下文包含较早对话摘要、初始情境和 checkpoint 中保留的近期消息。
        Args:
            character_id: 角色 ID。
            limit: 最多返回多少条近期消息，默认 20。
        """
        row = _get_latest_character_conversation(character_id)
        if row is None:
            return json.dumps({"found": False, "character_id": character_id,
                               "message": "没有找到该角色的角色扮演对话。"}, ensure_ascii=False)

        snapshot = character_graph.get_state(
            {"configurable": {"thread_id": row["conversation_id"]}}
        )
        values = snapshot.values or {}
        config = json.loads(row["config_json"] or "{}")
        messages = []
        recent_messages = values.get("messages", [])[-limit:] if limit > 0 else []
        for message in recent_messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage) and not message.tool_calls:
                role = "character"
            else:
                continue
            messages.append({"role": role, "content": message.content})

        return json.dumps({
            "found": True,
            "character_id": character_id,
            "character_name": row["character_name"],
            "conversation_id": row["conversation_id"],
            "initial_context": config.get("initial_context", ""),
            "summary": values.get("summary", ""),
            "recent_messages": messages,
        }, ensure_ascii=False, default=str)

    return get_character_messages


def create_character_state_tool(character_graph):
    @tool("get_character_state")
    def get_character_state(character_id: str) -> str:
        """查看指定角色最近更新的角色扮演对话中的心情、能量和内心想法。

        character_id 必须是 get_existing_characters 返回的内部 ID，不能传角色名称。
        若返回 found=false，只表示没有找到匹配的角色对话记录。

        Args:
            character_id: 角色 ID。
        """
        row = _get_latest_character_conversation(character_id)
        if row is None:
            return json.dumps({"found": False, "character_id": character_id,
                               "message": "没有找到该角色的角色扮演对话。"}, ensure_ascii=False)

        snapshot = character_graph.get_state(
            {"configurable": {"thread_id": row["conversation_id"]}}
        )
        values = snapshot.values or {}
        config = json.loads(row["config_json"] or "{}")
        personality = values.get("personality") or config.get("personality", {})
        if hasattr(personality, "model_dump"):
            personality = personality.model_dump()
        dynamic = personality.get("dynamic_states", {}) if isinstance(personality, dict) else {}
        if hasattr(dynamic, "model_dump"):
            dynamic = dynamic.model_dump()

        return json.dumps({
            "found": True,
            "character_id": character_id,
            "character_name": row["character_name"],
            "conversation_id": row["conversation_id"],
            "mood": dynamic.get("mood", ""),
            "energy_level": dynamic.get("energy_level"),
            "inner_thoughts": values.get(
                "last_cognitive_workspace",
                values.get("cognitive_workspace", ""),
            ),
            "inner_thoughts_user_message": values.get(
                "last_cognitive_workspace_user_message", ""
            ),
            "inner_thoughts_scope": "最近一次复杂轮次；不是当前简单轮次的即时思考。",
        }, ensure_ascii=False, default=str)

    return get_character_state
