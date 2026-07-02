"""线索分析 Agent

对情报卡片进行关联分析、时空碰撞、异常识别，输出可追溯的 Clue 列表。
LangGraph 节点函数：接收 CaseState，返回 state 更新字典。
"""
import logging
from typing import Dict, List

from case_assistant.state import CaseState
from case_assistant.tools.clue_tools import (
    build_relation_graph,
    detect_anomaly,
    score_clue,
    spatiotemporal_collision,
)

logger = logging.getLogger(__name__)


class ClueAgent:
    """线索分析 Agent - 关联分析、时空碰撞、异常识别"""

    def run(self, state: CaseState) -> dict:
        """执行线索分析

        - 调用 build_relation_graph 构建关系图谱
        - 调用 spatiotemporal_collision 进行时空碰撞
        - 调用 detect_anomaly 识别异常
        - 综合以上结果生成 Clue 列表
        - 评估是否需要补充情报（needs_more_intel）
        - 返回 state 更新字典
        """
        case_id = state.get("case_id", "")
        cards: List[dict] = list(state.get("intelligence_cards", []) or [])
        errors: List[str] = list(state.get("errors", []) or [])

        if not cards:
            errors.append("ClueAgent: 无情报卡片可分析，需先执行情报收集")
            return {
                "clues": [],
                "needs_more_intel": True,
                "current_phase": "clue_analysis",
                "errors": errors,
            }

        clues: List[dict] = []

        # 1. 构建关系图谱
        try:
            graph = build_relation_graph(cards, case_id=case_id)
        except Exception as e:
            msg = "ClueAgent: 构建关系图谱异常: {err}".format(err=e)
            logger.exception(msg)
            errors.append(msg)
            graph = {"nodes": [], "edges": []}

        # 2. 时空碰撞分析
        try:
            collisions = spatiotemporal_collision(cards, case_id=case_id)
        except Exception as e:
            msg = "ClueAgent: 时空碰撞分析异常: {err}".format(err=e)
            logger.exception(msg)
            errors.append(msg)
            collisions = []

        # 3. 异常识别
        try:
            anomalies = detect_anomaly(cards, case_id=case_id)
        except Exception as e:
            msg = "ClueAgent: 异常识别异常: {err}".format(err=e)
            logger.exception(msg)
            errors.append(msg)
            anomalies = []

        # 4. 综合生成 Clue 列表
        clue_index = 0

        # 4.1 关系图谱线索：识别核心节点
        clues.extend(self._clues_from_graph(graph, case_id, clue_index))
        clue_index += len(clues)

        # 4.2 时空碰撞线索
        for col in collisions:
            card_ids = col.get("card_ids", []) or []
            description = (
                "{p1} 与 {p2} 于 {loc} 共现（时间差 {diff} 秒），"
                "存在同行或接头可能".format(
                    p1=col.get("person1"),
                    p2=col.get("person2"),
                    loc=col.get("location"),
                    diff=col.get("time_diff"),
                )
            )
            score = score_clue(
                description=description,
                evidence_count=max(len(card_ids), 1),
                anomaly_severity="medium",
            )
            clues.append({
                "clue_id": "CLUE-ST-{idx:03d}".format(idx=clue_index),
                "case_id": case_id,
                "description": description,
                "related_card_ids": card_ids,
                "related_persons": [p for p in (col.get("person1"), col.get("person2")) if p],
                "score": score,
                "reasoning": "时空轨迹碰撞：两人在相近时间出现在同一地点，"
                             "结合案件背景存在共同行动嫌疑。",
                "needs_more_intel": False,
            })
            clue_index += 1

        # 4.3 异常行为线索
        for anomaly in anomalies:
            evidence = anomaly.get("evidence", []) or []
            persons = anomaly.get("persons", []) or []
            description = anomaly.get("description", "")
            severity = anomaly.get("severity", "medium")
            score = score_clue(
                description=description,
                evidence_count=max(len(evidence), 1),
                anomaly_severity=severity,
            )
            clues.append({
                "clue_id": "CLUE-AN-{idx:03d}".format(idx=clue_index),
                "case_id": case_id,
                "description": description,
                "related_card_ids": evidence,
                "related_persons": persons,
                "score": score,
                "reasoning": "异常行为识别：{atype}，严重程度 {sev}，"
                             "基于 {n} 条情报证据。".format(
                                 atype=anomaly.get("type", "unknown"),
                                 sev=severity,
                                 n=len(evidence),
                             ),
                "needs_more_intel": False,
            })
            clue_index += 1

        # 5. 按可信度降序排列
        clues.sort(key=lambda c: c.get("score", 0), reverse=True)

        # 6. 评估是否需要补充情报
        needs_more = self._evaluate_needs_more(clues, cards, collisions, anomalies)

        return {
            "clues": clues,
            "needs_more_intel": needs_more,
            "current_phase": "clue_analysis",
            "errors": errors,
        }

    @staticmethod
    def _clues_from_graph(
        graph: dict, case_id: str, start_index: int
    ) -> List[dict]:
        """从关系图谱中提取线索：识别高连接度节点与多关系聚合"""
        clues: List[dict] = []
        nodes = graph.get("nodes", []) or []
        edges = graph.get("edges", []) or []
        if not nodes:
            return clues

        # 统计每个人员的关系数（度）
        degree: Dict[str, int] = {}
        relation_kinds: Dict[str, set] = {}
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            rel = edge.get("relation", "related")
            for pid in (src, tgt):
                if not pid:
                    continue
                degree[pid] = degree.get(pid, 0) + 1
                relation_kinds.setdefault(pid, set()).add(rel)

        # 找出高连接度节点（度 >= 3 或具有多种关系类型）
        for node in nodes:
            pid = node.get("id")
            if not pid:
                continue
            deg = degree.get(pid, 0)
            kinds = relation_kinds.get(pid, set())
            if deg < 3 and len(kinds) < 2:
                continue
            related_edges = [
                e for e in edges
                if e.get("source") == pid or e.get("target") == pid
            ]
            card_ids = [e.get("card_id") for e in related_edges if e.get("card_id")]
            related_persons = list({
                other for e in related_edges
                for other in (e.get("source"), e.get("target"))
                if other and other != pid
            })
            description = (
                "{pid} 在关系网络中连接度较高（度={deg}，关系类型：{kinds}），"
                "疑为案件核心节点".format(
                    pid=pid, deg=deg,
                    kinds="/".join(sorted(kinds)) if kinds else "unknown",
                )
            )
            score = score_clue(
                description=description,
                evidence_count=max(len(card_ids), deg),
                anomaly_severity="high" if deg >= 5 else "medium",
            )
            clues.append({
                "clue_id": "CLUE-GR-{idx:03d}".format(idx=start_index + len(clues)),
                "case_id": case_id,
                "description": description,
                "related_card_ids": card_ids,
                "related_persons": related_persons,
                "score": score,
                "reasoning": "关系图谱分析：该人员与多人存在多种关系（通讯/资金/共现），"
                             "在网络中处于核心位置，需重点侦查。",
                "needs_more_intel": False,
            })
        return clues

    @staticmethod
    def _evaluate_needs_more(
        clues: List[dict],
        cards: List[dict],
        collisions: List[dict],
        anomalies: List[dict],
    ) -> bool:
        """评估是否需要补充情报

        判定条件：
        - 线索数量不足（< 2）
        - 高分线索缺失（无 score >= 60 的线索）
        - 情报卡片中大量标注信息缺失
        """
        if len(clues) < 2:
            return True
        if not any(c.get("score", 0) >= 60 for c in clues):
            return True
        # 检查情报卡片中信息缺失比例
        if cards:
            missing_count = sum(
                1 for c in cards if c.get("missing_info")
            )
            if missing_count / len(cards) > 0.5:
                return True
        return False
