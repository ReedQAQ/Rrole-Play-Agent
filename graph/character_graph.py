"""Role-play LangGraph 主流程。"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from graph.character_node.dual_process_agent import (
    answer_node,
    cognitive_workspace_node,
    route_after_trigger,
    trigger_node,
)
from graph.character_node.memory_product import memory_processing_node
from graph.character_node.check import check_node, route_after_check, route_after_check_retry
from graph.character_node.dynamic_state import dynamic_state_update_node
from graph.memory_update import role_memory_background_node, signal_memory_check
from graph.state import CharacterState


def build_character_graph(
    llm,
    memory_rag=None,
    checkpointer=None,
    context_window_tokens: int = 1_048_576,
    memory_related_to: str = "user",
):
    """构建角色扮演图。

    简单路径：START -> trigger -> answer -> check
    复杂路径：START -> trigger -> memory_processing -> cognitive_workspace
               -> answer -> check -> dynamic_state_update（终态时，与记忆信号并行）

    Args:
        llm: 支持 LangChain 工具调用的 ChatModel。
        memory_rag: 可选的角色记忆库实例；为 None 时跳过长期记忆检索。
        checkpointer: 可选 LangGraph checkpointer。
        context_window_tokens: 模型上下文窗口 token 上限；历史摘要在达到其一半时更新。
    """
    graph = StateGraph(CharacterState)
    graph.add_node("trigger", trigger_node)
    graph.add_node(
        "memory_processing",
        partial(memory_processing_node, llm=llm, memory_rag=memory_rag),
    )
    graph.add_node(
        "cognitive_workspace",
        partial(
            cognitive_workspace_node,
            llm=llm,
            context_window_tokens=context_window_tokens,
        ),
    )
    graph.add_node(
        "answer",
        partial(answer_node, llm=llm, context_window_tokens=context_window_tokens),
    )
    graph.add_node("check", partial(check_node, llm=llm))
    graph.add_node(
        "dynamic_state_update",
        partial(dynamic_state_update_node, llm=llm),
    )

    if memory_rag is not None:
        graph.add_node(
            "memory_start",
            partial(
                role_memory_background_node,
                llm=llm,
                memory_rag=memory_rag,
                related_to=memory_related_to,
            ),
        )
        graph.add_node(
            "memory_signal",
            lambda state: signal_memory_check(
                state,
                passed=all((state.get("check_result") or {}).get(key, False)
                           for key in (
                               "role_consistency", "information_reliability",
                               "safety", "anthropomorphism",
                           )),
            ),
        )

    def prepare_retry(state):
        return {"retry_count": state.get("retry_count", 0) + 1}

    graph.add_node("prepare_retry", prepare_retry)
    if memory_rag is not None:
        # Re-enter the background branch on retry; duplicate task IDs are ignored.
        graph.add_node("retry_answer", lambda state: {})
        graph.add_edge("retry_answer", "answer")
        graph.add_edge("retry_answer", "memory_start")

    graph.add_edge(START, "trigger")
    if memory_rag is not None:
        def route_after_trigger_with_memory(state: CharacterState):
            if state.get("trigger_result") == "complex":
                return "memory_processing"
            if state.get("trigger_result") == "simple":
                return ["answer", "memory_start"]
            raise ValueError("CharacterState 缺少有效的 trigger_result")

        graph.add_conditional_edges("trigger", route_after_trigger_with_memory)
        graph.add_edge("cognitive_workspace", "answer")
        graph.add_edge("cognitive_workspace", "memory_start")
        graph.add_edge("answer", "check")
    else:
        graph.add_conditional_edges(
            "trigger",
            route_after_trigger,
            {
                "memory_processing": "memory_processing",
                "answer": "answer",
            },
        )
        graph.add_edge("answer", "check")

    graph.add_edge("memory_processing", "cognitive_workspace")
    if memory_rag is None:
        graph.add_edge("cognitive_workspace", "answer")
    def route_after_check_terminal(state: CharacterState):
        outcome = route_after_check(state)
        if outcome == "retry":
            return "prepare_retry"

        # Simple turns have no cognitive workspace and do not update dynamic state.
        if state.get("trigger_result") == "complex":
            if memory_rag is not None:
                # State update and memory decision signaling are independent branches.
                return ["dynamic_state_update", "memory_signal"]
            return "dynamic_state_update"

        return "memory_signal" if memory_rag is not None else END

    graph.add_conditional_edges("check", route_after_check_terminal)

    graph.add_edge("dynamic_state_update", END)

    if memory_rag is not None:
        graph.add_edge("memory_signal", END)
    graph.add_conditional_edges(
        "prepare_retry",
        route_after_check_retry,
        {
            "memory_processing": "memory_processing",
            "cognitive_workspace": "cognitive_workspace",
            "answer": "retry_answer" if memory_rag is not None else "answer",
        },
    )

    return graph.compile(checkpointer=checkpointer)
