"""嫌疑人画像 Agent

为每位已知嫌疑人构建多维度画像与关系网络，输出符合 Profile 模型结构的画像列表。
画像覆盖基础信息、行为模式、关系网络、风险评估四个维度。
"""
import logging
from typing import Any, Dict, List

from case_assistant.state import CaseState
from case_assistant.tools.profile_tools import (
    aggregate_person_data,
    build_behavior_pattern,
    draw_relation_graph,
    assess_risk,
)

logger = logging.getLogger(__name__)


class ProfileAgent:
    """嫌疑人画像 Agent - 为每位嫌疑人构建多维度画像与关系网络"""

    def run(self, state: CaseState) -> dict:
        """执行嫌疑人画像

        - 遍历 known_suspects，为每位嫌疑人构建画像
        - 调用 aggregate_person_data, build_behavior_pattern,
          draw_relation_graph, assess_risk 四个工具
        - 生成 Profile 列表
        - 返回 {profiles: [...], current_phase: "suspect_profiling"}
        """
        case_id = state.get("case_id", "")
        known_suspects: List[str] = list(state.get("known_suspects") or [])
        intelligence_cards = state.get("intelligence_cards") or []
        clues = state.get("clues") or []
        errors: List[str] = list(state.get("errors") or [])

        profiles: List[Dict[str, Any]] = []

        if not known_suspects:
            logger.warning("ProfileAgent: known_suspects 为空，跳过画像生成")
            return {
                "profiles": profiles,
                "current_phase": "suspect_profiling",
                "errors": errors,
            }

        for person_id in known_suspects:
            try:
                profile = self._build_single_profile(
                    case_id=case_id,
                    person_id=person_id,
                    intelligence_cards=intelligence_cards,
                    clues=clues,
                )
                if profile:
                    profiles.append(profile)
            except Exception as e:
                msg = f"ProfileAgent: 嫌疑人 {person_id} 画像生成失败 - {e}"
                logger.exception(msg)
                errors.append(msg)

        return {
            "profiles": profiles,
            "current_phase": "suspect_profiling",
            "errors": errors,
        }

    def _build_single_profile(self, case_id: str, person_id: str,
                              intelligence_cards: list,
                              clues: list) -> Dict[str, Any]:
        """为单个嫌疑人构建画像字典。"""
        # 聚合个人数据
        aggregated = aggregate_person_data(
            case_id, person_id, intelligence_cards
        )
        basic_info = aggregated.get("basic_info") or {}

        # 行为模式建模
        behavior_pattern = build_behavior_pattern(case_id, person_id)

        # 关系图谱
        relation_network = draw_relation_graph(case_id, person_id)

        # 风险评估
        risk = assess_risk(
            case_id, person_id,
            intelligence_cards=intelligence_cards,
            clues=clues,
        )

        return {
            "person_id": person_id,
            "case_id": case_id,
            "basic_info": basic_info,
            "behavior_pattern": behavior_pattern,
            "relation_network": relation_network,
            "risk_level": risk.get("risk_level", "low"),
            "risk_score": risk.get("risk_score", 0),
            "risk_reasoning": risk.get("reasoning", ""),
            "risk_factors": risk.get("factors", []),
        }
