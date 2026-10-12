"""Assistant agent 的 LangGraph。"""
from collections.abc import Sequence
from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.assistant_node.agent import agent_node
from graph.assistant_node.check import check_node
from graph.memory_update import assistant_memory_background_node, signal_memory_check
from graph.state import AssistantState


def build_assistant_graph(
    agent_llm: BaseChatModel,
    tools: Sequence[BaseTool] = (),
    check_llm: BaseChatModel | None = None,
    max_tool_rounds: int = 5,
    checkpointer=None,
    context_window_tokens: int = 1_048_576,
    memory_rag=None,
    memory_owner: str = "assistant",
    memory_related_to: str = "user",
):
    """构建 Assistant 图：agent -> tools（按需循环）-> check -> END。

    Assistant Check 只记录质量结论，不在图内重试或分流。
    """
    tool_list = list(tools)
    checker_model = check_llm if check_llm is not None else agent_llm
    graph = StateGraph(AssistantState)
    graph.add_node(
        "agent",
        partial(
            agent_node,
            agent_llm=agent_llm,
            tools=tool_list,
            max_tool_rounds=max_tool_rounds,
            context_window_tokens=context_window_tokens,
        ),
    )
    graph.add_node("check", partial(check_node, llm=checker_model))
    graph.add_edge(START, "agent")

    if memory_rag is not None:
        graph.add_node(
            "memory_start",
            partial(
                assistant_memory_background_node,
                llm=agent_llm,
                memory_rag=memory_rag,
                owner=memory_owner,
                related_to=memory_related_to,
            ),
        )
        graph.add_node(
            "memory_signal",
            lambda state: signal_memory_check(
                state,
                passed=all((state.get("check_result") or {}).get(key, False)
                           for key in ("task_completed", "information_sufficient", "safety")),
            ),
        )
        # The background worker extracts while the complete agent/tool/check
        # loop runs. The main graph only releases it after check and never joins it.
        graph.add_edge(START, "memory_start")
        graph.add_edge("check", "memory_signal")
        graph.add_edge("memory_signal", END)

    if tool_list:
        tool_executor = ToolNode(tool_list)

        def execute_tools(state: AssistantState) -> dict:
            result = tool_executor.invoke(state)
            return {
                "messages": result.get("messages", []),
                "assistant_tool_rounds": state.get("assistant_tool_rounds", 0) + 1,
            }

        def route_after_agent(state: AssistantState) -> str:
            latest = state.get("messages", [])[-1]
            if (
                isinstance(latest, AIMessage)
                and latest.tool_calls
                and state.get("assistant_tool_rounds", 0) < max_tool_rounds
            ):
                return "tools"
            return "check"

        graph.add_node("tools", execute_tools)
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {"tools": "tools", "check": "check"},
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", "check")

    if memory_rag is None:
        graph.add_edge("check", END)
    return graph.compile(checkpointer=checkpointer)
