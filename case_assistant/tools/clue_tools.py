"""线索分析工具函数

纯 Python 函数，不依赖 LLM。基于情报卡片进行关系图谱构建、时空碰撞、
异常识别与线索评分。
"""
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from case_assistant.data import loader
from case_assistant.memory.knowledge_graph import get_knowledge_graph

logger = logging.getLogger(__name__)


# 边境/跨境相关关键词
_BORDER_KEYWORDS = [
    "边境", "口岸", "缅甸", "缅北", "瑞丽", "出境", "入境", "跨境", "木姐", "南坎",
]

# 深夜时段（23:00-05:00，含 23 与 0-4）
_LATE_NIGHT_HOURS = {23, 0, 1, 2, 3, 4}

# 大额转账阈值（元）
_LARGE_AMOUNT_THRESHOLD = 50000.0


def _parse_dt(value: Any) -> Optional[datetime]:
    """安全解析时间为 datetime 对象"""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _normalize_location(loc: str) -> str:
    """规范化地点字符串，去除括号内细节以便聚合"""
    if not loc:
        return ""
    s = str(loc).strip()
    # 去除中英文括号内的地址细节，保留核心地点名
    s = re.sub(r"[（(][^）)]*[）)]", "", s)
    return s.strip()


def _time_diff_seconds(t1: Any, t2: Any) -> Optional[int]:
    """计算两个时间的绝对差（秒），解析失败返回 None"""
    d1 = _parse_dt(t1)
    d2 = _parse_dt(t2)
    if not d1 or not d2:
        return None
    return abs(int((d1 - d2).total_seconds()))


def _extract_amount(content: str) -> Optional[float]:
    """从情报卡片 content 中提取金额"""
    if not content:
        return None
    match = re.search(r"([\d.]+)\s*元", str(content))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def build_relation_graph(intelligence_cards: list, case_id: str = None) -> dict:
    """构建关系图谱
    - 从情报卡片中提取涉及人员
    - 根据通讯、资金、同行等关系构建边
    - 返回 {nodes: [...], edges: [...]} 格式
    """
    kg = get_knowledge_graph()
    cards = intelligence_cards or []

    # 1. 收集所有涉及人员
    person_ids = set()
    for card in cards:
        for pid in card.get("involved_persons", []) or []:
            if pid:
                person_ids.add(pid)

    # 2. 尝试加载人员姓名
    person_names: Dict[str, str] = {}
    if case_id:
        try:
            for p in loader.load_persons(case_id):
                pid = p.get("person_id")
                if pid:
                    person_names[pid] = p.get("name", pid)
        except Exception as e:
            logger.warning("加载人员姓名失败 %s: %s", case_id, e)

    # 3. 添加人员节点
    for pid in person_ids:
        name = person_names.get(pid, pid)
        kg.add_node(pid, "person", name=name, label=name, group="person")

    # 4. 通讯关系边
    for card in cards:
        if card.get("source_type") != "comm":
            continue
        involved = card.get("involved_persons", []) or []
        if len(involved) >= 2:
            kg.add_edge(
                involved[0], involved[1], "communication", weight=1.0,
                card_id=card.get("card_id"),
            )

    # 5. 资金关系边
    for card in cards:
        if card.get("source_type") != "finance":
            continue
        involved = card.get("involved_persons", []) or []
        if len(involved) >= 2:
            kg.add_edge(
                involved[0], involved[1], "finance", weight=1.0,
                card_id=card.get("card_id"),
            )

    # 6. 同行/共现关系边（基于轨迹时空碰撞）
    try:
        collisions = spatiotemporal_collision(cards, case_id)
        for col in collisions:
            p1 = col.get("person1")
            p2 = col.get("person2")
            if p1 and p2 and p1 != p2:
                kg.add_edge(
                    p1, p2, "co-occurrence", weight=2.0,
                    location=col.get("location"),
                    time=col.get("time1"),
                )
    except Exception as e:
        logger.warning("构建共现边失败: %s", e)

    # 7. 返回涉及人员的子图
    if not person_ids:
        return {"nodes": [], "edges": []}
    return kg.get_subgraph(list(person_ids))


def spatiotemporal_collision(intelligence_cards: list, case_id: str = None) -> list:
    """时空轨迹碰撞分析
    - 加载所有人员的轨迹数据
    - 找出不同人员在相近时间出现在相同地点的情况
    - 返回碰撞结果列表 [{person1, person2, location, time1, time2, time_diff, note}]
    """
    # 1. 从情报卡片中提取轨迹数据
    trajectories: List[Dict[str, Any]] = []
    for card in intelligence_cards or []:
        if card.get("source_type") != "trajectory":
            continue
        involved = card.get("involved_persons", []) or []
        if not involved:
            continue
        trajectories.append({
            "person_id": involved[0],
            "location": card.get("location"),
            "time": card.get("occurred_at"),
            "note": card.get("content", ""),
            "card_id": card.get("card_id"),
        })

    # 2. 若卡片中无轨迹数据且提供 case_id，则从数据源加载
    if not trajectories and case_id:
        try:
            for t in loader.load_trajectories(case_id):
                trajectories.append({
                    "person_id": t.get("person_id"),
                    "location": t.get("location"),
                    "time": t.get("time"),
                    "note": t.get("note", ""),
                    "card_id": None,
                })
        except Exception as e:
            logger.warning("加载轨迹数据失败 %s: %s", case_id, e)

    # 3. 按规范化地点分组
    location_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trajectories:
        loc = _normalize_location(t.get("location"))
        if not loc:
            continue
        t["norm_location"] = loc
        location_groups[loc].append(t)

    # 4. 在每个地点内寻找不同人员的相近时间共现
    collisions: List[dict] = []
    time_threshold = 3600  # 1 小时内视为相近时间
    for loc, group in location_groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                t1 = group[i]
                t2 = group[j]
                if t1["person_id"] == t2["person_id"]:
                    continue
                diff = _time_diff_seconds(t1.get("time"), t2.get("time"))
                if diff is None or diff > time_threshold:
                    continue
                collisions.append({
                    "person1": t1["person_id"],
                    "person2": t2["person_id"],
                    "location": loc,
                    "time1": t1.get("time"),
                    "time2": t2.get("time"),
                    "time_diff": diff,
                    "note": "共现于 {loc}，时间差 {diff} 秒".format(loc=loc, diff=diff),
                    "card_ids": [c for c in (t1.get("card_id"), t2.get("card_id")) if c],
                })
    return collisions


def detect_anomaly(intelligence_cards: list, case_id: str = None) -> list:
    """异常行为识别
    - 深夜通讯聚集（23:00-05:00）
    - 频繁大额转账或分散转入集中转出模式
    - 边境往返轨迹
    - 返回异常列表 [{type, description, persons, evidence, severity}]
    """
    cards = intelligence_cards or []
    anomalies: List[dict] = []

    # 1. 深夜通讯聚集
    try:
        _detect_late_night_comm(cards, anomalies)
    except Exception as e:
        logger.warning("深夜通讯聚集检测失败: %s", e)

    # 2. 资金异常（频繁大额转账 / 分散转入集中转出）
    try:
        _detect_finance_anomaly(cards, anomalies)
    except Exception as e:
        logger.warning("资金异常检测失败: %s", e)

    # 3. 边境往返轨迹
    try:
        _detect_border_trajectory(cards, anomalies)
    except Exception as e:
        logger.warning("边境轨迹检测失败: %s", e)

    return anomalies


def _detect_late_night_comm(cards: List[dict], anomalies: List[dict]) -> None:
    """深夜通讯聚集检测：23:00-05:00 时段通讯按人员对聚合"""
    # 按 (caller, callee) 有序对聚合
    pair_records: Dict[tuple, List[dict]] = defaultdict(list)
    for card in cards:
        if card.get("source_type") != "comm":
            continue
        dt = _parse_dt(card.get("occurred_at"))
        if not dt or dt.hour not in _LATE_NIGHT_HOURS:
            continue
        involved = card.get("involved_persons", []) or []
        if len(involved) < 2:
            continue
        key = tuple(sorted([involved[0], involved[1]]))
        pair_records[key].append(card)

    for (p1, p2), recs in pair_records.items():
        if len(recs) < 2:
            continue
        severity = "high" if len(recs) >= 5 else "medium"
        evidence = [r.get("card_id") for r in recs if r.get("card_id")]
        anomalies.append({
            "type": "late_night_communication",
            "description": "{p1} 与 {p2} 在深夜时段（23:00-05:00）通讯 {n} 次，存在异常联络".format(
                p1=p1, p2=p2, n=len(recs),
            ),
            "persons": [p1, p2],
            "evidence": evidence,
            "severity": severity,
        })


def _detect_finance_anomaly(cards: List[dict], anomalies: List[dict]) -> None:
    """资金异常检测：频繁大额转账 / 分散转入集中转出"""
    # 按主账户人聚合资金卡片
    person_txns: Dict[str, List[dict]] = defaultdict(list)
    for card in cards:
        if card.get("source_type") != "finance":
            continue
        involved = card.get("involved_persons", []) or []
        if not involved:
            continue
        person_txns[involved[0]].append(card)

    for person, txns in person_txns.items():
        ins: List[dict] = []
        outs: List[dict] = []
        large_txns: List[dict] = []
        for t in txns:
            content = t.get("content", "") or ""
            amount = _extract_amount(content) or 0.0
            if amount >= _LARGE_AMOUNT_THRESHOLD:
                large_txns.append(t)
            if "转入" in content:
                ins.append(t)
            elif "转出" in content:
                outs.append(t)

        # 频繁大额转账
        if len(large_txns) >= 2:
            evidence = [t.get("card_id") for t in large_txns if t.get("card_id")]
            anomalies.append({
                "type": "frequent_large_transfer",
                "description": "{p} 存在 {n} 笔大额转账（≥{thr:.0f}元），需核查资金来源与用途".format(
                    p=person, n=len(large_txns), thr=_LARGE_AMOUNT_THRESHOLD,
                ),
                "persons": [person],
                "evidence": evidence,
                "severity": "high",
            })

        # 分散转入集中转出：≥3 笔转入且有单笔大额转出
        if len(ins) >= 3 and outs:
            large_outs = [t for t in outs if (_extract_amount(t.get("content", "")) or 0) >= _LARGE_AMOUNT_THRESHOLD]
            if large_outs:
                evidence = [t.get("card_id") for t in ins + large_outs if t.get("card_id")]
                anomalies.append({
                    "type": "distributed_in_centralized_out",
                    "description": "{p} 多笔分散转入（{n_in}笔）后集中大额转出（{n_out}笔），符合资金清洗特征".format(
                        p=person, n_in=len(ins), n_out=len(large_outs),
                    ),
                    "persons": [person],
                    "evidence": evidence,
                    "severity": "high",
                })


def _detect_border_trajectory(cards: List[dict], anomalies: List[dict]) -> None:
    """边境往返轨迹检测"""
    person_border_trajs: Dict[str, List[dict]] = defaultdict(list)
    for card in cards:
        if card.get("source_type") != "trajectory":
            continue
        text = " ".join([
            str(card.get("location", "")),
            str(card.get("content", "")),
        ])
        if not any(kw in text for kw in _BORDER_KEYWORDS):
            continue
        involved = card.get("involved_persons", []) or []
        if not involved:
            continue
        person_border_trajs[involved[0]].append(card)

    for person, trajs in person_border_trajs.items():
        evidence = [t.get("card_id") for t in trajs if t.get("card_id")]
        if len(trajs) >= 2:
            desc = "{p} 存在 {n} 条边境/跨境相关轨迹，疑似边境往返活动".format(
                p=person, n=len(trajs),
            )
            severity = "high"
        else:
            desc = "{p} 存在边境/跨境相关轨迹，需关注".format(p=person, n=1)
            severity = "medium"
        anomalies.append({
            "type": "border_activity",
            "description": desc,
            "persons": [person],
            "evidence": evidence,
            "severity": severity,
        })


def score_clue(description: str, evidence_count: int, anomaly_severity: str = "medium") -> int:
    """线索可信度评分（0-100）
    - 基于证据数量、异常严重程度、描述完整性综合评分
    """
    # 证据数量评分（每条 15 分，上限 50）
    evidence_score = min(int(evidence_count or 0) * 15, 50)
    # 异常严重程度加分
    severity_bonus = {"high": 25, "medium": 15, "low": 5}.get(
        str(anomaly_severity or "").lower(), 15
    )
    # 描述完整性加分（每 10 字符 1 分，上限 25）
    desc_len = len(description or "")
    description_bonus = min(desc_len // 10, 25)
    # 总分并约束到 0-100
    total = evidence_score + severity_bonus + description_bonus
    return max(0, min(100, total))
