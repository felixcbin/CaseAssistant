"""情报收集 Agent

从多源数据中检索、整合结构化情报，输出 IntelligenceCard 列表。
LangGraph 节点函数：接收 CaseState，返回 state 更新字典。
"""
import logging
from typing import List

from case_assistant.state import CaseState
from case_assistant.tools.intel_tools import collect_intelligence

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    """情报收集 Agent - 从多源数据中检索、整合结构化情报"""

    def run(self, state: CaseState) -> dict:
        """执行情报收集

        - 调用 collect_intelligence 收集情报
        - 如果有 human_feedback，针对性补充
        - 返回 state 更新字典：
          {intelligence_cards: [...], known_suspects: [...],
           current_phase: "intelligence_gathering"}
        """
        case_id = state.get("case_id", "")
        known_suspects = list(state.get("known_suspects", []) or [])
        human_feedback = state.get("human_feedback")
        existing_cards: List[dict] = list(state.get("intelligence_cards", []) or [])
        errors: List[str] = list(state.get("errors", []) or [])

        if not case_id:
            errors.append("IntelligenceAgent: case_id 缺失，无法收集情报")
            return {
                "intelligence_cards": existing_cards,
                "known_suspects": known_suspects,
                "current_phase": "intelligence_gathering",
                "errors": errors,
            }

        try:
            new_cards = collect_intelligence(
                case_id=case_id,
                known_suspects=known_suspects,
                human_feedback=human_feedback,
            )
        except Exception as e:
            msg = "IntelligenceAgent: 情报收集异常 {cid}: {err}".format(cid=case_id, err=e)
            logger.exception(msg)
            errors.append(msg)
            new_cards = []

        # 迭代回溯时追加而非覆盖：按 card_id 去重合并
        merged_cards = self._merge_cards(existing_cards, new_cards)

        # 更新 known_suspects：从情报卡片中识别新关联人员
        updated_suspects = self._extract_persons(merged_cards, known_suspects)

        # 评估是否仍存在信息缺失
        missing_summary = self._summarize_missing(merged_cards)

        return {
            "intelligence_cards": merged_cards,
            "known_suspects": updated_suspects,
            "current_phase": "intelligence_gathering",
            "errors": errors,
            "needs_more_intel": bool(missing_summary),
        }

    @staticmethod
    def _merge_cards(existing: List[dict], new: List[dict]) -> List[dict]:
        """按 card_id 去重合并情报卡片（已有卡片不被新卡片覆盖）"""
        seen = set()
        merged: List[dict] = []
        for card in existing:
            cid = card.get("card_id")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            merged.append(card)
        for card in new:
            cid = card.get("card_id")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            merged.append(card)
        return merged

    @staticmethod
    def _extract_persons(cards: List[dict], known: List[str]) -> List[str]:
        """从情报卡片 involved_persons 中识别所有涉及人员，合并入已知嫌疑人"""
        result = list(known)
        seen = set(result)
        for card in cards:
            for pid in card.get("involved_persons", []) or []:
                if pid and pid not in seen:
                    seen.add(pid)
                    result.append(pid)
        return result

    @staticmethod
    def _summarize_missing(cards: List[dict]) -> List[str]:
        """汇总情报卡片中标注的信息缺失项"""
        missing: List[str] = []
        for card in cards:
            for item in card.get("missing_info", []) or []:
                if item and item not in missing:
                    missing.append(item)
        return missing
