"""Role-play 的 Trigger、Cognitive Workspace 与 Answer 节点。"""
from typing import Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from graph.context_summary import prepare_context_summary
from graph.state import CharacterState, PersonalityProfile


def _last_user_message(messages: list[AnyMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _message_content(message: AnyMessage | None) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _conversation_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """只保留真实用户输入和角色回复，排除工具及系统消息。"""
    conversation = []
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation.append(message)
        elif isinstance(message, AIMessage) and not message.tool_calls:
            conversation.append(message)
    return conversation


def trigger_node(state: CharacterState) -> dict[str, object]:
    """标记本轮走简单回复还是复杂认知路径。"""
    messages = state.get("messages", [])
    action = _message_content(_last_user_message(messages))
    profile: PersonalityProfile = state["personality"]

    emotional_keywords = (
        "喜欢", "讨厌", "生气", "开心", "难过", "愤怒", "失望", "惊喜", "爱", "恨",
        "查抄", "紧张", "咳血", "病", "婚事", "离开", "死", "走", "担心", "委屈",
        "议论", "指责", "质疑", "威胁", "危机", "紧急", "重大", "关键",
        "love", "hate", "angry", "happy", "sad", "disappointed", "surprised",
        "sick", "blood", "marriage", "leave", "die", "worry", "threat", "crisis", "urgent",
    )
    action_lower = action.lower()
    is_complex = any(keyword in action_lower for keyword in emotional_keywords)
    if not is_complex:
        is_complex = any(
            interest and interest.lower() in action_lower
            for interest in profile.interests
        )

    return {
        "trigger_result": "complex" if is_complex else "simple",
        "filtered_memories": "",
        "cognitive_workspace": "",
        "check_result": None,
        "check_reason": "",
        "retry_count": 0,
    }


def route_after_trigger(state: CharacterState) -> Literal["memory_processing", "answer"]:
    """Trigger 后的 LangGraph 条件路由函数。"""
    result = state.get("trigger_result")
    if result == "complex":
        return "memory_processing"
    if result == "simple":
        return "answer"
    raise ValueError("CharacterState 缺少有效的 trigger_result")


def cognitive_workspace_node(
    state: CharacterState,
    llm: BaseLanguageModel,
    context_window_tokens: int = 1_048_576,
) -> dict[str, object]:
    """基于角色状态、对话和记忆过滤结果生成内部认知。"""
    prepared_state, removed_messages, summary_updated = prepare_context_summary(
        state,
        llm,
        context_window_tokens,
    )
    profile: PersonalityProfile = state["personality"]
    trait_names = {
        "openness": "开放性",
        "conscientiousness": "尽责性",
        "extraversion": "外向性",
        "agreeableness": "宜人性",
        "neuroticism": "神经质",
    }
    traits = "，".join(
        f"{trait_names[name]}={value:.2f}"
        for name, value in profile.core_traits.model_dump().items()
    )
    dynamic = profile.dynamic_states
    context = prepared_state.get("summary", "")
    filtered_memories = prepared_state.get("filtered_memories", "")
    conversation = _conversation_messages(prepared_state.get("messages", []))

    action_maker_name = "用户"
    action_detail = _message_content(_last_user_message(conversation))
    system_prompt = f"""你是{state["character_name"]}。
你的大五人格是：
{traits}

当前状态：
心情={dynamic.mood}，能量={dynamic.energy_level}/100。

{action_maker_name}说：
“{action_detail}”

角色相关记忆为：
【{filtered_memories or "没有可用的相关记忆。"}】

生成你的内部想法（不对外说出）。

思考以下问题：
1. 根据你的人格，这件事让你有什么感受？
2. 如果你感到压力，你的防御机制（{profile.defense_mechanism}）会如何影响你的认知？
3. 你真正想表达什么、为什么这样想？

请根据你的性格生成一段**内心独白**（不要输出给用户看，仅用于生成下一步行为）。


规则：
- 神经质高于 0.7：关注威胁与焦虑。
- 宜人性低于 0.4：允许出现批评性想法。
- 外向性高于 0.7：想法积极、主动。
- 能量较低：想法简短，体现疲惫。

只输出内部想法。不要生成对用户说出的回复。"""
    result = llm.invoke([SystemMessage(content=system_prompt), *conversation])
    content = getattr(result, "content", result)
    if not isinstance(content, str):
        content = str(content)
    update: dict[str, object] = {
        "cognitive_workspace": content.strip(),
        "messages": removed_messages,
    }
    if summary_updated:
        update["summary"] = prepared_state["summary"]
    return update


def answer_node(
    state: CharacterState,
    llm: BaseLanguageModel,
    context_window_tokens: int = 1_048_576,
) -> dict[str, object]:
    """把认知工作区中的内部想法转化为角色对外回复。"""
    prepared_state, removed_messages, summary_updated = prepare_context_summary(
        state,
        llm,
        context_window_tokens,
    )
    profile: PersonalityProfile = state["personality"]
    style = profile.speaking_style
    style_names = {
        "short": "短句为主", "medium": "中等长度", "long": "长句为主", "mixed": "长短句混合",
    }
    vocabulary_names = {
        "academic": "学术或正式", "poem": "诗意表达", "casual": "口语化",
        "netword": "网络用语", "mixed": "混合",
    }
    sentence_length = style_names.get(style.sentencelength, style.sentencelength)
    vocabulary = vocabulary_names.get(style.VocabulartLevel, style.VocabulartLevel)
    conversation = _conversation_messages(prepared_state.get("messages", []))
    latest_user_index = next(
        (index for index in range(len(conversation) - 1, -1, -1)
         if isinstance(conversation[index], HumanMessage)),
        None,
    )
    action_detail = _message_content(
        conversation[latest_user_index] if latest_user_index is not None else None
    )
    history_messages = (
        conversation[:latest_user_index] if latest_user_index is not None else conversation
    )
    history = "\n".join(
        f"{state['character_name']}：{_message_content(message)}"
        if isinstance(message, AIMessage)
        else f"用户：{_message_content(message)}"
        for message in history_messages
    ) or "暂无历史对话。"
    action_maker_name = "用户"
    inner_monologue = prepared_state.get("cognitive_workspace", "") or "本轮没有额外的内心想法。"
    summary = prepared_state.get("summary", "") or "暂无更早对话摘要。"
    check_reason = state.get("check_reason", "") or "无。"

    system_prompt = f"""你正在扮演{state["character_name"]}。
角色身份与背景：
{state["character_identity"]}

你的内心想法是：
“{inner_monologue}”

现在请将其转化为回复给{action_maker_name}。

严格遵守以下语言风格：
- 句长：{sentence_length}
- 词汇风格：{vocabulary}

禁止事项（非常重要）：
- 禁止使用心理学术语，例如“神经质”“外向性”“宜人性”“尽责性”“开放性”“防御机制”。
- 禁止提及能量值、心情值等内部状态信息。
- 禁止提及 MBTI 类型或大五人格。
- 避免翻译腔和过于正式的词汇。
- 禁止出现“内心独白”字样。
- 不要向用户透露上面的内心想法或任何内部分析，只将其转化为自然的角色回复。

较早对话摘要：
{summary}

上次检查指出的问题（首次生成时为空）：
{check_reason}

当前上下文：
{action_maker_name}说：“{action_detail}”

历史对话：
{history}

请生成符合上述角色身份和语言风格的回复。只输出角色说的话，不要添加心理分析或元描述。"""
    result = llm.invoke([SystemMessage(content=system_prompt)])
    content = getattr(result, "content", result)
    if not isinstance(content, str):
        content = str(content)
    update: dict[str, object] = {
        "messages": [*removed_messages, AIMessage(content=content.strip())]
    }
    if summary_updated:
        update["summary"] = prepared_state["summary"]
    return update
