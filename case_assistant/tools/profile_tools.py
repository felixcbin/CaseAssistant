"""嫌疑人画像工具函数

提供聚合个人数据、行为模式建模、关系图谱构建、风险评估等纯 Python 工具，
供 ProfileAgent 调用。所有函数均不依赖 LLM，仅基于案件原始数据做规则化分析。
"""
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from case_assistant.data import loader
from case_assistant.memory.knowledge_graph import get_knowledge_graph


# 深夜时段定义：22:00 - 次日 06:00
_LATE_NIGHT_HOURS = set(list(range(22, 24)) + list(range(0, 6)))

# 大额交易阈值
_LARGE_AMOUNT_THRESHOLD = 50000.0

# 边境/境外关键词
_BORDER_KEYWORDS = ["瑞丽", "缅北", "缅甸", "边境", "口岸", "木姐", "南坎", "姐告"]


def _parse_hour(time_str: str) -> Optional[int]:
    """从 ISO 时间字符串中提取小时，解析失败返回 None。"""
    if not time_str or not isinstance(time_str, str):
        return None
    # 兼容 "2026-02-18T23:42:00" 这类格式
    match = re.search(r"T(\d{2}):\d{2}:\d{2}", time_str)
    if not match:
        match = re.search(r"(\d{2}):\d{2}:\d{2}", time_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return None


def _parse_day_of_month(time_str: str) -> Optional[int]:
    """从 ISO 时间字符串中提取日期（月中第几日），解析失败返回 None。"""
    if not time_str or not isinstance(time_str, str):
        return None
    match = re.search(r"-(\d{2})T", time_str)
    if not match:
        match = re.search(r"-(\d{2})\s", time_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (ValueError, IndexError):
        return None


def _parse_counterparty_name(remark: str) -> Optional[str]:
    """从流水备注中解析交易对手姓名。

    备注通常形如 "转入，对方周八，备注\"货款\""，提取 "对方" 后的姓名。
    """
    if not remark or not isinstance(remark, str):
        return None
    match = re.search(r"对方([\u4e00-\u9fa5A-Za-z]{1,10})", remark)
    if not match:
        return None
    return match.group(1)


def _is_border_location(location: str) -> bool:
    """判断地点是否涉及边境/境外。"""
    if not location or not isinstance(location, str):
        return False
    return any(kw in location for kw in _BORDER_KEYWORDS)


def _build_name_to_id_map(case_id: str) -> Dict[str, str]:
    """构建 姓名 -> person_id 的映射。"""
    mapping: Dict[str, str] = {}
    for person in loader.load_persons(case_id):
        name = person.get("name")
        pid = person.get("person_id")
        if name and pid:
            mapping[name] = pid
    return mapping


def _person_status_group(status: Optional[str]) -> str:
    """将人员 status 字段映射为关系图谱节点 group。

    suspect -> suspect；witness -> witness；其余统一为 associate。
    """
    if status == "suspect":
        return "suspect"
    if status == "witness":
        return "witness"
    return "associate"


def aggregate_person_data(case_id: str, person_id: str,
                          intelligence_cards: Optional[list] = None) -> dict:
    """聚合个人数据

    合并人员基础信息、前科、通讯、银行、轨迹，并附加与该人员相关的情报卡片。
    所有加载异常被降级处理为空值，保证调用方不中断。
    """
    try:
        basic_info = loader.query_person(case_id, person_id) or {}
    except Exception:
        basic_info = {}

    try:
        records = loader.query_person_records(case_id, person_id) or []
    except Exception:
        records = []

    try:
        communications = loader.query_person_communications(case_id, person_id) or []
    except Exception:
        communications = []

    try:
        finance = loader.query_person_finance(case_id, person_id) or []
    except Exception:
        finance = []

    try:
        trajectories = loader.query_person_trajectories(case_id, person_id) or []
    except Exception:
        trajectories = []

    # 过滤与该人员相关的情报卡片
    related_cards: List[Dict[str, Any]] = []
    if intelligence_cards:
        for card in intelligence_cards:
            if not isinstance(card, dict):
                continue
            involved = card.get("involved_persons") or []
            if person_id in involved:
                related_cards.append(card)

    return {
        "person_id": person_id,
        "case_id": case_id,
        "basic_info": basic_info,
        "criminal_records": records,
        "communications": communications,
        "finance": finance,
        "trajectories": trajectories,
        "related_intelligence_cards": related_cards,
    }


def build_behavior_pattern(case_id: str, person_id: str) -> dict:
    """行为模式建模

    分别从通讯、资金、轨迹三个维度刻画行为规律：
    - 通讯规律：通话时段分布、频繁联系人、深夜通话比例
    - 资金往来特征：转入转出模式、大额交易、固定日期交易
    - 活动轨迹：频繁出入地点、出行频率、特殊地点（边境）
    """
    pattern: Dict[str, Any] = {
        "communication": {},
        "finance": {},
        "trajectory": {},
    }

    # ---------- 通讯维度 ----------
    try:
        comms = loader.query_person_communications(case_id, person_id) or []
    except Exception:
        comms = []

    hour_distribution: Dict[int, int] = defaultdict(int)
    contact_counter: Counter = Counter()
    late_night_count = 0
    total_comms = 0
    for record in comms:
        total_comms += 1
        # 通话时段分布
        hour = _parse_hour(record.get("time", ""))
        if hour is not None:
            hour_distribution[hour] += 1
            if hour in _LATE_NIGHT_HOURS:
                late_night_count += 1
        # 频繁联系人：对方号码对应的 person_id
        caller = record.get("caller")
        callee = record.get("callee")
        counterpart = callee if caller == person_id else caller
        if counterpart:
            contact_counter[counterpart] += 1

    frequent_contacts = [
        {"person_id": pid, "count": cnt}
        for pid, cnt in contact_counter.most_common(5)
    ]
    pattern["communication"] = {
        "total_records": total_comms,
        "hour_distribution": dict(sorted(hour_distribution.items())),
        "frequent_contacts": frequent_contacts,
        "late_night_count": late_night_count,
        "late_night_ratio": round(late_night_count / total_comms, 3) if total_comms else 0.0,
    }

    # ---------- 资金维度 ----------
    try:
        finance = loader.query_person_finance(case_id, person_id) or []
    except Exception:
        finance = []

    inflow_total = 0.0
    outflow_total = 0.0
    inflow_count = 0
    outflow_count = 0
    large_transactions: List[Dict[str, Any]] = []
    day_counter: Counter = Counter()

    for txn in finance:
        amount = float(txn.get("amount") or 0.0)
        remark = txn.get("remark") or ""
        time_str = txn.get("time", "")
        # 通过 remark 判断转入/转出
        is_inflow = "转入" in remark
        is_outflow = "转出" in remark or "取现" in remark
        if is_inflow:
            inflow_total += amount
            inflow_count += 1
        elif is_outflow:
            outflow_total += amount
            outflow_count += 1
        # 大额交易
        if amount >= _LARGE_AMOUNT_THRESHOLD:
            large_transactions.append({
                "transaction_id": txn.get("transaction_id"),
                "amount": amount,
                "time": time_str,
                "remark": remark,
            })
        # 固定日期交易
        day = _parse_day_of_month(time_str)
        if day is not None:
            day_counter[day] += 1

    # 固定日期判定：同一日出现 >= 2 次交易视为固定日期交易
    fixed_days = [
        {"day": d, "count": c}
        for d, c in day_counter.items() if c >= 2
    ]

    pattern["finance"] = {
        "total_records": len(finance),
        "inflow": {
            "count": inflow_count,
            "total_amount": round(inflow_total, 2),
        },
        "outflow": {
            "count": outflow_count,
            "total_amount": round(outflow_total, 2),
        },
        "large_transactions": large_transactions,
        "large_transaction_count": len(large_transactions),
        "fixed_date_transactions": fixed_days,
    }

    # ---------- 轨迹维度 ----------
    try:
        trajectories = loader.query_person_trajectories(case_id, person_id) or []
    except Exception:
        trajectories = []

    location_counter: Counter = Counter()
    border_locations: List[Dict[str, Any]] = []
    for traj in trajectories:
        location = traj.get("location") or ""
        if location:
            location_counter[location] += 1
        if _is_border_location(location):
            border_locations.append({
                "trajectory_id": traj.get("trajectory_id"),
                "location": location,
                "time": traj.get("time"),
                "note": traj.get("note"),
            })

    frequent_locations = [
        {"location": loc, "count": cnt}
        for loc, cnt in location_counter.most_common(5)
    ]

    pattern["trajectory"] = {
        "total_records": len(trajectories),
        "frequent_locations": frequent_locations,
        "travel_frequency": len(trajectories),
        "border_locations": border_locations,
        "has_border_activity": len(border_locations) > 0,
    }

    return pattern


def draw_relation_graph(case_id: str, person_id: str) -> dict:
    """个人关系图谱

    从通讯、资金、轨迹数据中提取该人员的关系网络，返回 {nodes, edges} 格式
    供前端渲染。节点包含 group（suspect/associate/witness）与 size（基于度数）。
    """
    name_to_id = _build_name_to_id_map(case_id)

    # 边聚合：(source, target, relation) -> weight
    edge_agg: Dict[tuple, int] = defaultdict(int)
    # 节点 status 缓存（person_id -> status）
    node_status: Dict[str, Optional[str]] = {person_id: None}

    # 读取主人员的 status
    try:
        main_person = loader.query_person(case_id, person_id) or {}
        node_status[person_id] = main_person.get("status")
    except Exception:
        pass

    # ---------- 通讯关系 ----------
    try:
        comms = loader.query_person_communications(case_id, person_id) or []
    except Exception:
        comms = []

    for record in comms:
        caller = record.get("caller")
        callee = record.get("callee")
        counterpart = callee if caller == person_id else caller
        if not counterpart:
            continue
        node_status.setdefault(counterpart, None)
        edge_agg[(person_id, counterpart, "communication")] += 1

    # ---------- 资金关系 ----------
    try:
        finance = loader.query_person_finance(case_id, person_id) or []
    except Exception:
        finance = []

    for txn in finance:
        remark = txn.get("remark") or ""
        counterparty_name = _parse_counterparty_name(remark)
        if not counterparty_name:
            continue
        counterparty_id = name_to_id.get(counterparty_name)
        if not counterparty_id:
            # 未在人员表中匹配到，仍以姓名作为临时节点 ID
            counterparty_id = f"NAME::{counterparty_name}"
        node_status.setdefault(counterparty_id, None)
        edge_agg[(person_id, counterparty_id, "finance")] += 1

    # ---------- 轨迹共现关系 ----------
    try:
        all_trajectories = loader.load_trajectories(case_id) or []
    except Exception:
        all_trajectories = []

    main_trajs = [t for t in all_trajectories if t.get("person_id") == person_id]
    # 按地点分组，查找共现
    location_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for traj in all_trajectories:
        loc = traj.get("location")
        if loc:
            location_groups[loc].append(traj)

    for main_traj in main_trajs:
        loc = main_traj.get("location")
        main_time = main_traj.get("time", "")
        if not loc:
            continue
        for other in location_groups.get(loc, []):
            other_pid = other.get("person_id")
            if not other_pid or other_pid == person_id:
                continue
            # 简单共现判定：同一地点（不做严格时间窗口，因为演示数据规模小）
            node_status.setdefault(other_pid, None)
            edge_agg[(person_id, other_pid, "co-occurrence")] += 1

    # ---------- 同步人员 status ----------
    try:
        persons = loader.load_persons(case_id) or []
    except Exception:
        persons = []
    status_lookup = {p.get("person_id"): p.get("status") for p in persons if p.get("person_id")}
    for nid in list(node_status.keys()):
        if node_status.get(nid) is None:
            node_status[nid] = status_lookup.get(nid)

    # ---------- 计算节点度数 ----------
    degree: Counter = Counter()
    for (src, tgt, _rel) in edge_agg.keys():
        degree[src] += 1
        degree[tgt] += 1
    # 主节点至少为 1，避免被忽略
    degree[person_id] = max(degree[person_id], 1)

    # ---------- 构建节点 ----------
    nodes: List[Dict[str, Any]] = []
    for nid, status in node_status.items():
        label = nid
        if not nid.startswith("NAME::"):
            try:
                p = loader.query_person(case_id, nid)
                if p:
                    label = p.get("name", nid)
            except Exception:
                pass
        else:
            label = nid.replace("NAME::", "")
        deg = degree.get(nid, 1)
        nodes.append({
            "id": nid,
            "label": label,
            "group": _person_status_group(status),
            "size": 20 + min(deg, 10) * 3,  # 30~50 区间
        })

    # ---------- 构建边 ----------
    relation_label_map = {
        "communication": "通讯",
        "finance": "资金往来",
        "co-occurrence": "同行共现",
    }
    edges: List[Dict[str, Any]] = []
    for (src, tgt, rel), weight in edge_agg.items():
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation_label_map.get(rel, rel),
            "weight": weight,
        })

    return {"nodes": nodes, "edges": edges}


def assess_risk(case_id: str, person_id: str,
                intelligence_cards: Optional[list] = None,
                clues: Optional[list] = None) -> dict:
    """风险评估

    基于前科记录、异常行为、关系网络、涉案线索综合评估。
    评估因素与权重：
      - 有前科(+30)
      - 深夜通讯频繁(+15)
      - 大额异常资金(+20)
      - 边境轨迹(+15)
      - 与高风险人员关联(+10)
      - 未成年人涉案(+20)
    """
    risk_score = 0
    factors: List[str] = []

    # ---------- 前科记录 ----------
    try:
        records = loader.query_person_records(case_id, person_id) or []
    except Exception:
        records = []
    if records:
        risk_score += 30
        crime_types = [r.get("crime_type", "") for r in records if r.get("crime_type")]
        factors.append(f"有前科记录({len(records)}条)，涉及：{', '.join(crime_types)}")

    # ---------- 行为模式 ----------
    try:
        pattern = build_behavior_pattern(case_id, person_id)
    except Exception:
        pattern = {}

    comm = pattern.get("communication", {}) or {}
    fin = pattern.get("finance", {}) or {}
    traj = pattern.get("trajectory", {}) or {}

    # 深夜通讯频繁：深夜通话 >= 3 次，且占比 >= 0.3
    late_night_count = comm.get("late_night_count", 0)
    late_night_ratio = comm.get("late_night_ratio", 0.0)
    if late_night_count >= 3 and late_night_ratio >= 0.3:
        risk_score += 15
        factors.append(
            f"深夜通讯频繁({late_night_count}次，占比{late_night_ratio:.0%})"
        )

    # 大额异常资金：存在大额交易
    large_txn_count = fin.get("large_transaction_count", 0)
    if large_txn_count > 0:
        risk_score += 20
        factors.append(f"存在大额异常资金往来({large_txn_count}笔)")

    # 边境轨迹
    if traj.get("has_border_activity"):
        border_count = len(traj.get("border_locations", []))
        risk_score += 15
        factors.append(f"有边境/境外活动轨迹({border_count}条)")

    # ---------- 与高风险人员关联 ----------
    # 高风险人员：有前科的其他人员
    try:
        all_persons = loader.load_persons(case_id) or []
    except Exception:
        all_persons = []
    try:
        all_records = loader.load_criminal_records(case_id) or []
    except Exception:
        all_records = []
    persons_with_records = {r.get("person_id") for r in all_records if r.get("person_id")}
    persons_with_records.discard(person_id)

    try:
        comms = loader.query_person_communications(case_id, person_id) or []
    except Exception:
        comms = []
    related_persons = {
        (rec.get("callee") if rec.get("caller") == person_id else rec.get("caller"))
        for rec in comms
    }
    related_persons.discard(None)
    related_persons.discard(person_id)

    high_risk_links = related_persons & persons_with_records
    if high_risk_links:
        risk_score += 10
        factors.append(
            f"与高风险(有前科)人员关联：{', '.join(sorted(high_risk_links))}"
        )

    # ---------- 未成年人涉案 ----------
    minor_involved = False
    # 情况1：本人是未成年人
    main_person = loader.query_person(case_id, person_id) or {}
    age = main_person.get("age")
    if isinstance(age, (int, float)) and age < 18:
        minor_involved = True
        factors.append(f"本人为未成年人({age}岁)")
    # 情况2：与未成年人有关联（通讯/资金/线索）
    if not minor_involved:
        minor_ids = {
            p.get("person_id") for p in all_persons
            if isinstance(p.get("age"), (int, float)) and p.get("age", 99) < 18
        }
        minor_ids.discard(person_id)
        if related_persons & minor_ids:
            minor_involved = True
            factors.append("与未成年人存在通讯/资金关联")
        # 情况3：线索或情报卡片中提及未成年人
        if not minor_involved:
            minor_keywords = ["未成年", "小孩", "少年", "学生"]
            check_items = list(intelligence_cards or []) + list(clues or [])
            for item in check_items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("content", "")) + str(item.get("description", "")) + str(item.get("reasoning", ""))
                if any(kw in text for kw in minor_keywords):
                    minor_involved = True
                    factors.append("线索/情报中提及未成年人涉案")
                    break

    if minor_involved:
        risk_score += 20

    # ---------- 风险等级 ----------
    if risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    if factors:
        reasoning = f"综合评估得分 {risk_score} 分，定级为 {risk_level}。主要依据：" + "；".join(factors) + "。"
    else:
        reasoning = f"综合评估得分 {risk_score} 分，定级为 {risk_level}。未发现明显风险因素。"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reasoning": reasoning,
        "factors": factors,
    }
