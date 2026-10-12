from langchain_core.tools import tool

from memory.memory import MemoryType


def create_memory_search_tool(memory_rag, owner: str, top_k: int = 2):
    """Create a scoped memory search tool for one Assistant or character owner.

    Experiences use semantic search capped at two records. Impression and
    relationship are current snapshots, so they are fetched directly rather
    than ranked with historical experiences.
    """
    @tool("memory_search")
    def memory_search(query: str, related_to: str) -> str:
        """Search up to two relevant experiences and read current user state.

        Args:
            query: The experience or fact to search for.
            related_to: The entity associated with these memories, usually user.
        """
        results = memory_rag.search(
            query=query,
            top_k=min(max(top_k, 0), 2),
            where={"$and": [
                {"owner": owner},
                {"related_to": related_to},
                {"memory_type": MemoryType.EXPERIENCE.value},
            ]},
        )
        documents = (results or {}).get("documents", [[]])
        experiences = documents[0] if documents else []
        formatted = []
        for text in experiences[:2]:
            formatted.append(f"- [experience] {text}")

        for memory_type in (MemoryType.IMPRESSION, MemoryType.RELATIONSHIP):
            snapshot = memory_rag.get_current_memory(
                memory_type=memory_type,
                related_to=related_to,
                owner=owner,
            )
            if snapshot:
                formatted.append(f"- [{memory_type.value}:current] {snapshot['text']}")

        return "\n".join(formatted) if formatted else "没有找到相关记忆。"

    return memory_search
