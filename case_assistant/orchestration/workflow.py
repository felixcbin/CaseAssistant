"""LangGraph 状态机编排

管理案件状态、调度 Agent、处理条件分支与人工审核节点。
支持 LangGraph 不可用时的降级顺序执行。
"""
import logging
from typing import Dict, Any, Optional

from case_assistant.state import CaseState
from case_assistant.config import MAX_ITERATIONS
from case_assistant.data import loader
from case_assistant.agents.intelligence_agent import IntelligenceAgent
from case_assistant.agents.clue_agent import ClueAgent
from case_assistant.agents.profile_agent import ProfileAgent
from case_assistant.agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)

# Agent 实例
_intelligence_agent = IntelligenceAgent()
_clue_agent = ClueAgent()
_profile_agent = ProfileAgent()
_report_agent = ReportAgent()


def init_case_node(state: CaseState) -> dict:
    """初始化案件状态，从数据加载器加载案件基础信息"""
    case_id = state.get("case_id", "")
    errors = list(state.get("errors", []) or [])

    if not case_id:
        errors.append("init_case: case_id 缺失")
        return {"errors": errors, "current_phase": "init_case"}

    dossier = loader.load_dossier(case_id)
    if not dossier:
        errors.append("init_case: 案件卷宗加载失败 case_id={}".format(case_id))
        return {"errors": errors, "current_phase": "init_case"}

    return {
        "case_id": case_id,
        "case_type": dossier.get("case_type", ""),
        "case_brief": dossier.get("case_brief", ""),
        "known_suspects": dossier.get("known_suspects", []) or [],
        "intelligence_cards": [],
        "clues": [],
        "profiles": [],
        "report": None,
        "current_phase": "init_case",
        "iteration": 0,
        "human_feedback": None,
        "needs_more_intel": False,
        "errors": errors,
    }


def intelligence_gathering_node(state: CaseState) -> dict:
    """情报收集 Agent 执行节点"""
    return _intelligence_agent.run(state)


def clue_analysis_node(state: CaseState) -> dict:
    """线索分析 Agent 执行节点"""
    return _clue_agent.run(state)


def suspect_profiling_node(state: CaseState) -> dict:
    """嫌疑人画像 Agent 执行节点"""
    return _profile_agent.run(state)


def human_review_node(state: CaseState) -> dict:
    """人工审核节点

    此节点为人工干预点，实际反馈通过 UI 注入 state。
    节点本身仅做状态传递，人工反馈在 UI 层通过更新 human_feedback 和
    needs_more_intel 字段实现。
    """
    return {"current_phase": "human_review"}


def report_generation_node(state: CaseState) -> dict:
    """报告生成 Agent 执行节点"""
    return _report_agent.run(state)


def route_after_clue(state: CaseState) -> str:
    """线索分析后路由：线索不足时先补充情报"""
    iteration = state.get("iteration", 0)
    needs_more = state.get("needs_more_intel", False)
    clues = state.get("clues", []) or []

    if needs_more or len(clues) < 1:
        if iteration >= MAX_ITERATIONS:
            logger.info("达到最大迭代轮次 %d，强制进入嫌疑人画像", MAX_ITERATIONS)
            return "suspect_profiling"
        return "intelligence_gathering"
    return "suspect_profiling"


def route_after_review(state: CaseState) -> str:
    """人工审核后路由"""
    needs_more = state.get("needs_more_intel", False)
    feedback = state.get("human_feedback")

    if needs_more or feedback:
        iteration = state.get("iteration", 0)
        if iteration >= MAX_ITERATIONS:
            logger.info("达到最大迭代轮次 %d，强制进入报告生成", MAX_ITERATIONS)
            return "report_generation"
        return "intelligence_gathering"
    return "report_generation"


def build_workflow():
    """构建 LangGraph 状态机工作流

    使用 interrupt_before 在人工审核节点前暂停，等待人工输入。
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.warning("langgraph 未安装，将使用降级顺序执行器")
        return None

    workflow = StateGraph(CaseState)

    workflow.add_node("init_case", init_case_node)
    workflow.add_node("intelligence_gathering", intelligence_gathering_node)
    workflow.add_node("clue_analysis", clue_analysis_node)
    workflow.add_node("suspect_profiling", suspect_profiling_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("report_generation", report_generation_node)

    workflow.set_entry_point("init_case")
    workflow.add_edge("init_case", "intelligence_gathering")
    workflow.add_edge("intelligence_gathering", "clue_analysis")
    workflow.add_conditional_edges("clue_analysis", route_after_clue)
    workflow.add_edge("suspect_profiling", "human_review")
    workflow.add_conditional_edges("human_review", route_after_review)
    workflow.add_edge("report_generation", END)

    # 在人工审核节点前暂停
    app = workflow.compile(interrupt_before=["human_review"])
    return app


def run_sequential(initial_state: Dict[str, Any],
                   human_feedback: Optional[str] = None,
                   max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """降级顺序执行器（langgraph 不可用时使用）

    执行完整流程，支持人工反馈触发的情报回溯。
    """
    state = dict(initial_state)

    # 初始化
    state.update(init_case_node(state))

    for iteration in range(max_iterations + 1):
        state["iteration"] = iteration

        # 情报收集
        state.update(_intelligence_agent.run(state))

        # 线索分析
        state.update(_clue_agent.run(state))

        # 判断是否需要补充情报
        if state.get("needs_more_intel") and iteration < max_iterations:
            if human_feedback:
                state["human_feedback"] = human_feedback
            continue

        break

    # 嫌疑人画像
    state.update(_profile_agent.run(state))

    # 人工审核（降级模式下直接使用传入的反馈）
    if human_feedback:
        state["human_feedback"] = human_feedback
        state["needs_more_intel"] = True
        # 有反馈时再跑一轮情报补充
        state["iteration"] = state.get("iteration", 0) + 1
        state.update(_intelligence_agent.run(state))
        state.update(_clue_agent.run(state))
        state.update(_profile_agent.run(state))

    # 报告生成
    state.update(_report_agent.run(state))

    return state


def run_case(case_id: str,
             human_feedback: Optional[str] = None,
             max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """运行案件侦查全流程

    优先使用 LangGraph，不可用时降级为顺序执行。
    """
    initial_state = {"case_id": case_id}

    app = build_workflow()
    if app is None:
        return run_sequential(initial_state, human_feedback, max_iterations)

    # LangGraph 模式：先跑到 human_review 暂停
    state = dict(initial_state)
    config = {"configurable": {"thread_id": case_id}}

    try:
        # 第一阶段：跑到 human_review 暂停
        for event in app.stream(state, config=config):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    state.update(node_output)

        # 如果有人工反馈，注入后继续
        if human_feedback:
            state["human_feedback"] = human_feedback
            state["needs_more_intel"] = True
            state["iteration"] = state.get("iteration", 0) + 1
            # 继续执行（从 human_review 后的路由开始）
            for event in app.stream(state, config=config):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state.update(node_output)
        else:
            # 无反馈，直接继续到报告生成
            state["human_feedback"] = None
            state["needs_more_intel"] = False
            for event in app.stream(state, config=config):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        state.update(node_output)

    except Exception as e:
        logger.error("LangGraph 执行异常，降级为顺序执行: %s", e)
        return run_sequential(initial_state, human_feedback, max_iterations)

    return state


def run_case_to_review(case_id: str) -> Dict[str, Any]:
    """运行案件侦查到人工审核节点（暂停等待人工输入）

    返回到 human_review 之前的状态，供 UI 展示并收集用户反馈。
    """
    initial_state = {"case_id": case_id}

    app = build_workflow()
    if app is None:
        # 降级模式：顺序执行到画像完成
        state = dict(initial_state)
        state.update(init_case_node(state))

        for iteration in range(max(MAX_ITERATIONS, 1)):
            state["iteration"] = iteration
            state.update(_intelligence_agent.run(state))
            state.update(_clue_agent.run(state))

            if state.get("needs_more_intel") and iteration < MAX_ITERATIONS - 1:
                continue
            break

        state.update(_profile_agent.run(state))
        state["current_phase"] = "human_review"
        return state

    # LangGraph 模式
    state = dict(initial_state)
    config = {"configurable": {"thread_id": case_id}}

    try:
        for event in app.stream(state, config=config):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    state.update(node_output)
    except Exception as e:
        logger.error("LangGraph 执行异常，降级为顺序执行: %s", e)
        state = dict(initial_state)
        state.update(init_case_node(state))
        state.update(_intelligence_agent.run(state))
        state.update(_clue_agent.run(state))
        state.update(_profile_agent.run(state))
        state["current_phase"] = "human_review"

    return state


def resume_after_review(state: Dict[str, Any],
                        human_feedback: Optional[str] = None,
                        needs_more_intel: bool = False) -> Dict[str, Any]:
    """人工审核后恢复执行

    根据人工反馈决定是回溯到情报收集还是直接生成报告。
    """
    if human_feedback:
        state["human_feedback"] = human_feedback
    state["needs_more_intel"] = needs_more_intel

    if needs_more_intel or human_feedback:
        iteration = state.get("iteration", 0)
        if iteration < MAX_ITERATIONS:
            state["iteration"] = iteration + 1
            # 回溯：重新情报收集 -> 线索分析 -> 画像
            state.update(_intelligence_agent.run(state))
            state.update(_clue_agent.run(state))
            state.update(_profile_agent.run(state))

    # 报告生成
    state.update(_report_agent.run(state))

    return state
