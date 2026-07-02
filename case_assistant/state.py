"""LangGraph 全局状态定义"""
from typing import TypedDict, List, Optional


class CaseState(TypedDict):
    # 案件基础信息
    case_id: str
    case_type: str                  # smuggling/drug/vice/juvenile
    case_brief: str                 # 案件简述
    known_suspects: List[str]       # 已知嫌疑人 ID

    # 各 Agent 产出
    intelligence_cards: List[dict]  # 情报收集产出
    clues: List[dict]               # 线索分析产出
    profiles: List[dict]            # 嫌疑人画像产出
    report: Optional[dict]          # 最终报告

    # 流程控制
    current_phase: str              # 当前阶段
    iteration: int                  # 迭代轮次（情报不足时回溯）
    human_feedback: Optional[str]   # 人工干预反馈
    needs_more_intel: bool          # 是否需要补充情报
    errors: List[str]               # 异常记录
