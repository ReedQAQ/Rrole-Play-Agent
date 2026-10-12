
from pathlib import Path
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKILLS_DIR=(PROJECT_ROOT/"skills").resolve()

SKILL_REGISTRY = {
    "create_conversation": Path("create_conversation.md"),
}


@tool
def get_skill(skill_name:str):
    """读取项目中已登记的 Skill 指南。"""

    relative_path = SKILL_REGISTRY.get(skill_name)

    if relative_path is None:
        available = ", ".join(SKILL_REGISTRY)
        return f"未知 Skill：{skill_name}。可用 Skill：{available}"

    skill_path = (SKILLS_DIR / relative_path).resolve()


    if not skill_path.is_relative_to(SKILLS_DIR):
        return "无效的 Skill 路径。"

    try:
        content = skill_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Skill 文件不存在：{skill_name}"
    except OSError:
        return f"读取 Skill 失败：{skill_name}"

    if not content.strip():
        return f"Skill 文件为空：{skill_name}"

    return content
