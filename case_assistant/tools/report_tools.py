"""报告生成工具函数

提供证据链梳理、案件时间线生成、Markdown 报告渲染等纯 Python 工具，
供 ReportAgent 调用。所有函数均不依赖 LLM，仅做规则化数据组织与格式化输出。
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

# 可选依赖：clue_tools 可能由其他子代理创建，未就绪时降级跳过
try:  # pragma: no cover - 依赖外部模块，不做覆盖测试
    from case_assistant.tools.clue_tools import detect_anomaly  # noqa: F401
except Exception:  # pylint: disable=broad-except
    detect_anomaly = None  # type: ignore


# 可信度排序权重，便于聚合时取最严格等级
_CREDIBILITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def _safe_str(value: Any) -> str:
    """安全转字符串，None 返回空串。"""
    if value is None:
        return ""
    return str(value)


def _parse_iso_time(time_str: Any) -> Optional[str]:
    """提取并规范化时间字段，返回可排序的字符串。

    保留原始 ISO 字符串以便排序；解析失败返回 None。
    """
    if not time_str:
        return None
    text = _safe_str(time_str)
    # 兼容 "2026-02-18T23:42:00" / "2026-02-18 23:42:00" / "2026-02-18"
    match = re.search(r"(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)", text)
    return match.group(1) if match else None


def format_evidence_chain(clues: list, intelligence_cards: list) -> list:
    """梳理证据链

    将线索与情报卡片关联，形成完整的证据链。每条证据链包含：
    - clue_id：线索ID
    - description：线索描述
    - related_card_ids：关联情报卡片ID列表
    - sources：证据来源列表（去重）
    - credibility：综合可信度
    - related_persons：关联人员
    - reasoning：推理过程
    """
    if not clues:
        return []

    # 构建 card_id -> card 的索引
    card_index: Dict[str, Dict[str, Any]] = {}
    if intelligence_cards:
        for card in intelligence_cards:
            if not isinstance(card, dict):
                continue
            cid = card.get("card_id")
            if cid:
                card_index[cid] = card

    evidence_chains: List[Dict[str, Any]] = []
    for clue in clues:
        if not isinstance(clue, dict):
            continue
        related_card_ids: List[str] = list(clue.get("related_card_ids") or [])
        related_cards: List[Dict[str, Any]] = [
            card_index[cid] for cid in related_card_ids
            if cid in card_index
        ]

        # 证据来源（去重）
        sources: List[str] = []
        for card in related_cards:
            src = card.get("source")
            if src and src not in sources:
                sources.append(src)

        # 综合可信度：取关联卡片中最严格等级；若无卡片则基于线索评分推断
        credibility = _aggregate_credibility(related_cards, clue.get("score"))

        evidence_chains.append({
            "clue_id": clue.get("clue_id"),
            "description": clue.get("description", ""),
            "related_card_ids": related_card_ids,
            "sources": sources,
            "credibility": credibility,
            "related_persons": list(clue.get("related_persons") or []),
            "reasoning": clue.get("reasoning", ""),
            "score": clue.get("score", 0),
        })

    # 按线索评分降序排列
    evidence_chains.sort(key=lambda x: x.get("score", 0), reverse=True)
    return evidence_chains


def _aggregate_credibility(cards: List[Dict[str, Any]],
                           fallback_score: Any) -> str:
    """根据关联卡片可信度综合判定证据链可信度。

    取最严格等级（权重最低者）作为综合可信度，体现"木桶效应"。
    若无卡片，则按线索评分映射：>=70 -> high，>=40 -> medium，其余 -> low。
    """
    if cards:
        weights = [
            _CREDIBILITY_WEIGHT.get(_safe_str(c.get("credibility")).lower(), 2)
            for c in cards
        ]
        if weights:
            min_weight = min(weights)
            for level, w in _CREDIBILITY_WEIGHT.items():
                if w == min_weight:
                    return level
    # 无卡片时基于评分推断
    try:
        score = int(fallback_score or 0)
    except (TypeError, ValueError):
        score = 0
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def generate_timeline(intelligence_cards: list, case_id: str = None) -> list:
    """生成案件时间线

    从情报卡片和案件数据中提取时间事件，按时间升序排列。
    每条事件：{time, event, source, persons}
    """
    events: List[Dict[str, Any]] = []

    # ---------- 情报卡片事件 ----------
    if intelligence_cards:
        for card in intelligence_cards:
            if not isinstance(card, dict):
                continue
            time_str = _parse_iso_time(card.get("occurred_at"))
            if not time_str:
                continue
            events.append({
                "time": time_str,
                "event": card.get("content", ""),
                "source": card.get("source", "情报卡片"),
                "persons": list(card.get("involved_persons") or []),
            })

    # ---------- 案件原始数据事件 ----------
    if case_id:
        _collect_case_events(case_id, events)

    # 按时间排序（字符串 ISO 格式可直接字典序排序）
    events.sort(key=lambda e: e.get("time", ""))
    return events


def _collect_case_events(case_id: str, events: List[Dict[str, Any]]) -> None:
    """从案件原始数据中提取事件并追加到 events 列表。"""
    # 延迟导入，避免循环依赖
    from case_assistant.data import loader

    # 通讯记录
    try:
        comms = loader.load_communications(case_id) or []
    except Exception:
        comms = []
    for rec in comms:
        time_str = _parse_iso_time(rec.get("time"))
        if not time_str:
            continue
        persons = [p for p in [rec.get("caller"), rec.get("callee")] if p]
        event_text = f"通讯记录({rec.get('type', '未知')})：{rec.get('content', '')}"
        events.append({
            "time": time_str,
            "event": event_text,
            "source": "通讯记录",
            "persons": persons,
        })

    # 资金流水
    try:
        finance = loader.load_finance(case_id) or []
    except Exception:
        finance = []
    for txn in finance:
        time_str = _parse_iso_time(txn.get("time"))
        if not time_str:
            continue
        amount = txn.get("amount", 0)
        event_text = f"资金流水：金额{amount}元，{txn.get('remark', '')}"
        persons = [p for p in [txn.get("person_id")] if p]
        events.append({
            "time": time_str,
            "event": event_text,
            "source": "银行流水",
            "persons": persons,
        })

    # 出行轨迹
    try:
        trajectories = loader.load_trajectories(case_id) or []
    except Exception:
        trajectories = []
    for traj in trajectories:
        time_str = _parse_iso_time(traj.get("time"))
        if not time_str:
            continue
        event_text = f"出行轨迹：{traj.get('location', '')}（{traj.get('transport', '')}）{traj.get('note', '')}"
        persons = [p for p in [traj.get("person_id")] if p]
        events.append({
            "time": time_str,
            "event": event_text,
            "source": "出行轨迹",
            "persons": persons,
        })

    # 卷宗事件（案发时间）
    try:
        dossier = loader.load_dossier(case_id) or {}
    except Exception:
        dossier = {}
    occurred_at = _parse_iso_time(dossier.get("occurred_at"))
    if occurred_at:
        events.append({
            "time": occurred_at,
            "event": f"案件发生：{dossier.get('case_brief', '')}",
            "source": "案件卷宗",
            "persons": list(dossier.get("known_suspects") or []),
        })


def render_markdown(report_data: dict) -> str:
    """渲染报告为 Markdown

    接收 InvestigationReport 格式的字典，输出包含全部章节的 Markdown 报告字符串。
    章节：案件概述、侦查过程（时间线）、关键线索与证据链、嫌疑人画像摘要、
          侦查建议、证据清单附录。
    """
    if not isinstance(report_data, dict):
        return ""

    lines: List[str] = []
    case_id = report_data.get("case_id", "")
    report_id = report_data.get("report_id", "")
    generated_at = report_data.get("generated_at", "")

    # ---------- 标题 ----------
    lines.append("# 案件侦查报告")
    lines.append("")
    if report_id:
        lines.append(f"- 报告编号：{report_id}")
    if case_id:
        lines.append(f"- 案件编号：{case_id}")
    if generated_at:
        lines.append(f"- 生成时间：{generated_at}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---------- 一、案件概述 ----------
    lines.append("## 一、案件概述")
    lines.append("")
    overview = report_data.get("overview", "")
    if overview:
        lines.append(overview)
    else:
        lines.append("（暂无案件概述）")
    lines.append("")

    # ---------- 二、侦查过程（时间线） ----------
    lines.append("## 二、侦查过程（时间线）")
    lines.append("")
    timeline = report_data.get("timeline") or []
    if timeline:
        lines.append("| 时间 | 事件 | 来源 | 涉及人员 |")
        lines.append("|------|------|------|----------|")
        for event in timeline:
            time_str = _safe_str(event.get("time", ""))
            event_text = _safe_str(event.get("event", "")).replace("|", "\\|").replace("\n", " ")
            source = _safe_str(event.get("source", "")).replace("|", "\\|")
            persons = ", ".join(_safe_str(p) for p in (event.get("persons") or []))
            persons = persons.replace("|", "\\|")
            lines.append(f"| {time_str} | {event_text} | {source} | {persons} |")
    else:
        lines.append("（暂无时间线数据）")
    lines.append("")

    # ---------- 三、关键线索与证据链 ----------
    lines.append("## 三、关键线索与证据链")
    lines.append("")
    key_clues = report_data.get("key_clues") or []
    evidence_appendix = report_data.get("evidence_appendix") or []

    if key_clues:
        for idx, clue in enumerate(key_clues, start=1):
            if not isinstance(clue, dict):
                continue
            desc = clue.get("description", "")
            score = clue.get("score", "")
            credibility = clue.get("credibility", "")
            related_cards = clue.get("related_card_ids") or []
            related_persons = clue.get("related_persons") or []
            reasoning = clue.get("reasoning", "")

            lines.append(f"### 线索 {idx}：{desc}")
            lines.append("")
            if score != "":
                lines.append(f"- 可信度评分：{score}")
            if credibility:
                lines.append(f"- 综合可信度：{credibility}")
            if related_cards:
                lines.append(f"- 关联情报卡片：{', '.join(_safe_str(c) for c in related_cards)}")
            if related_persons:
                lines.append(f"- 关联人员：{', '.join(_safe_str(p) for p in related_persons)}")
            if reasoning:
                lines.append(f"- 推理过程：{reasoning}")
            lines.append("")
    else:
        lines.append("（暂无线索数据）")
        lines.append("")

    # ---------- 四、嫌疑人画像摘要 ----------
    lines.append("## 四、嫌疑人画像摘要")
    lines.append("")
    suspect_profiles = report_data.get("suspect_profiles") or []
    if suspect_profiles:
        for idx, profile in enumerate(suspect_profiles, start=1):
            if not isinstance(profile, dict):
                continue
            person_id = profile.get("person_id", "")
            basic_info = profile.get("basic_info") or {}
            name = basic_info.get("name", person_id)
            risk_level = profile.get("risk_level", "")
            risk_reasoning = profile.get("risk_reasoning", "")

            lines.append(f"### 嫌疑人 {idx}：{name}（{person_id}）")
            lines.append("")
            lines.append(f"- 嫌疑等级：{risk_level}")
            # 基础信息
            if basic_info:
                info_lines = []
                for key in ["gender", "age", "occupation", "address", "phone", "status", "remark"]:
                    val = basic_info.get(key)
                    if val:
                        info_lines.append(f"{key}={val}")
                if info_lines:
                    lines.append(f"- 基础信息：{', '.join(info_lines)}")
            # 行为模式摘要
            behavior = profile.get("behavior_pattern") or {}
            if behavior:
                comm = behavior.get("communication") or {}
                fin = behavior.get("finance") or {}
                traj = behavior.get("trajectory") or {}
                behavior_summary_parts = []
                if comm:
                    behavior_summary_parts.append(
                        f"通讯{comm.get('total_records', 0)}条"
                    )
                if fin:
                    behavior_summary_parts.append(
                        f"资金{fin.get('total_records', 0)}条"
                    )
                if traj:
                    behavior_summary_parts.append(
                        f"轨迹{traj.get('total_records', 0)}条"
                    )
                if behavior_summary_parts:
                    lines.append(f"- 行为模式：{', '.join(behavior_summary_parts)}")
            # 关系网络摘要
            relation = profile.get("relation_network") or {}
            if relation:
                node_count = len(relation.get("nodes") or [])
                edge_count = len(relation.get("edges") or [])
                lines.append(f"- 关系网络：节点{node_count}个，边{edge_count}条")
            if risk_reasoning:
                lines.append(f"- 风险依据：{risk_reasoning}")
            lines.append("")
    else:
        lines.append("（暂无嫌疑人画像数据）")
        lines.append("")

    # ---------- 五、侦查建议 ----------
    lines.append("## 五、侦查建议")
    lines.append("")
    recommendations = report_data.get("recommendations") or []
    if recommendations:
        for idx, rec in enumerate(recommendations, start=1):
            lines.append(f"{idx}. {rec}")
    else:
        lines.append("（暂无侦查建议）")
    lines.append("")

    # ---------- 六、证据清单附录 ----------
    lines.append("## 六、证据清单附录")
    lines.append("")
    if evidence_appendix:
        for idx, evidence in enumerate(evidence_appendix, start=1):
            if not isinstance(evidence, dict):
                continue
            desc = evidence.get("description", "")
            related_cards = evidence.get("related_card_ids") or []
            sources = evidence.get("sources") or []
            credibility = evidence.get("credibility", "")
            related_persons = evidence.get("related_persons") or []
            score = evidence.get("score", "")

            lines.append(f"### 证据 {idx}：{desc}")
            lines.append("")
            if credibility:
                lines.append(f"- 综合可信度：{credibility}")
            if score != "":
                lines.append(f"- 评分：{score}")
            if sources:
                lines.append(f"- 来源：{', '.join(_safe_str(s) for s in sources)}")
            if related_cards:
                lines.append(f"- 关联情报卡片：{', '.join(_safe_str(c) for c in related_cards)}")
            if related_persons:
                lines.append(f"- 关联人员：{', '.join(_safe_str(p) for p in related_persons)}")
            lines.append("")
    else:
        lines.append("（暂无证据清单）")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> 本报告由 CaseAssistant 多智能体案件侦查系统自动生成，仅供办案参考，"
                 "所有结论需经办案人员核实。")
    lines.append("")

    return "\n".join(lines)
