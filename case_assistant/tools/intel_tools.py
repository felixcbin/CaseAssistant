"""情报收集工具函数

纯 Python 函数，不依赖 LLM。从多源数据（卷宗、人员、通讯、银行流水、
轨迹、前科）中检索并整合结构化情报卡片 (IntelligenceCard)。
"""
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from case_assistant.data import loader
from case_assistant.memory.knowledge_graph import get_knowledge_graph

logger = logging.getLogger(__name__)


# 各数据类型对应的默认可信度
_CREDIBILITY_MAP = {
    "dossier": "high",          # 案件卷宗：官方文书
    "person": "high",           # 户籍信息：官方登记
    "criminal_record": "high",  # 前科记录：司法档案
    "finance": "high",          # 银行流水：金融机构记录
    "comm": "medium",           # 通讯记录：内容可能不完整
    "trajectory": "medium",     # 轨迹数据：定位精度有限
    "web": "low",               # 网络信息：未经验证
    "knowledge_base": "medium", # 向量库检索：来自历史数据
}


def _parse_datetime(value: Any) -> Optional[str]:
    """安全解析时间字符串，统一返回 ISO 格式字符串"""
    if not value:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").isoformat()
            except ValueError:
                return value  # 无法解析则原样返回
    return str(value)


def _make_card_id(source_type: str, identifier: str) -> str:
    """生成情报卡片 ID"""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(identifier))
    return "IC-{}-{}".format(source_type.upper(), safe_id)


def search_dossier(case_id: str) -> dict:
    """检索案件卷宗"""
    try:
        dossier = loader.load_dossier(case_id)
        return dossier if dossier else {}
    except Exception as e:
        logger.warning("检索案件卷宗失败 %s: %s", case_id, e)
        return {}


def query_person_info(case_id: str, person_id: str) -> dict:
    """查询人员户籍/前科信息，合并人员基础信息和前科记录"""
    try:
        person = loader.query_person(case_id, person_id)
        if not person:
            return {}
        records = loader.query_person_records(case_id, person_id)
        return {
            "basic_info": person,
            "criminal_records": records if records else [],
        }
    except Exception as e:
        logger.warning("查询人员信息失败 %s/%s: %s", case_id, person_id, e)
        return {}


def query_communication(case_id: str, person_id: str) -> list:
    """查询通讯记录"""
    try:
        return loader.query_person_communications(case_id, person_id)
    except Exception as e:
        logger.warning("查询通讯记录失败 %s/%s: %s", case_id, person_id, e)
        return []


def query_finance(case_id: str, person_id: str) -> list:
    """查询银行流水"""
    try:
        return loader.query_person_finance(case_id, person_id)
    except Exception as e:
        logger.warning("查询银行流水失败 %s/%s: %s", case_id, person_id, e)
        return []


def query_trajectories(case_id: str, person_id: str) -> list:
    """查询出行轨迹"""
    try:
        return loader.query_person_trajectories(case_id, person_id)
    except Exception as e:
        logger.warning("查询出行轨迹失败 %s/%s: %s", case_id, person_id, e)
        return []


def search_knowledge_base(query: str, n_results: int = 5) -> list:
    """向量库语义检索（如果 ChromaDB 不可用则返回空列表）"""
    if not query:
        return []
    try:
        from case_assistant.memory.vector_store import (
            CHROMA_AVAILABLE,
            get_vector_store,
        )
    except ImportError as e:
        logger.warning("向量库模块导入失败: %s", e)
        return []
    if not CHROMA_AVAILABLE:
        logger.info("chromadb 不可用，跳过向量库检索")
        return []
    try:
        store = get_vector_store()
        results: List[dict] = []
        for collection_name in ("dossiers", "person_profiles", "case_knowledge"):
            try:
                hits = store.search(collection_name, query, n_results=n_results)
                for hit in hits:
                    hit["collection"] = collection_name
                    results.append(hit)
            except Exception as e:
                logger.warning("检索 collection %s 失败: %s", collection_name, e)
        return results[:n_results] if results else []
    except Exception as e:
        logger.warning("向量库检索失败 %s: %s", query, e)
        return []


def web_search(query: str) -> str:
    """外部公开信息检索（演示用 mock，返回模拟结果）"""
    if not query:
        return ""
    return (
        "[mock web search] 检索关键词: {q}。"
        "公开信息检索为演示 mock，未连接真实搜索引擎。"
        "可对接公安公开通报、新闻媒体、社交媒体公开数据等。"
    ).format(q=query)


# ---------------------------------------------------------------------------
# 以下为 collect_intelligence 的内部辅助函数
# ---------------------------------------------------------------------------


def _build_account_person_map(case_id: str) -> Dict[str, str]:
    """构建账户号 -> person_id 的映射，用于解析银行流水的对手方

    根据 remark 中的"转入"/"转出"判断哪个账户属于该人员：
    - 转入：to_account 属于 person_id（收款方）
    - 转出：from_account 属于 person_id（付款方）
    - 其他（如取现）：from_account 属于 person_id
    """
    mapping: Dict[str, str] = {}
    try:
        all_finance = loader.load_finance(case_id)
    except Exception:
        return mapping
    for txn in all_finance:
        pid = txn.get("person_id")
        if not pid:
            continue
        remark = str(txn.get("remark", ""))
        if "转入" in remark:
            own = txn.get("to_account")
        elif "转出" in remark:
            own = txn.get("from_account")
        else:
            # 取现、其他类型，默认 from_account 属于该人
            own = txn.get("from_account")
        if own and own != "CASH" and own not in mapping:
            mapping[own] = pid
    return mapping


def _card_from_dossier(case_id: str, dossier: dict) -> Optional[dict]:
    """从案件卷宗生成情报卡片"""
    if not dossier:
        return None
    suspects = dossier.get("known_suspects", []) or []
    content_parts = []
    brief = dossier.get("case_brief")
    if brief:
        content_parts.append(brief)
    desc = dossier.get("incident_description")
    if desc:
        content_parts.append(desc)
    content = " | ".join(content_parts) if content_parts else "案件卷宗摘要"
    return {
        "card_id": _make_card_id("dossier", case_id),
        "case_id": case_id,
        "source": "案件卷宗 dossier.json",
        "source_type": "dossier",
        "occurred_at": _parse_datetime(dossier.get("occurred_at")),
        "location": dossier.get("location"),
        "involved_persons": suspects,
        "content": content,
        "credibility": _CREDIBILITY_MAP["dossier"],
        "missing_info": [],
    }


def _card_from_person(case_id: str, info: dict) -> Optional[dict]:
    """从人员信息生成情报卡片"""
    if not info:
        return None
    basic = info.get("basic_info", {}) or {}
    records = info.get("criminal_records", []) or []
    person_id = basic.get("person_id", "UNKNOWN")
    parts = [
        "姓名: {name}, 性别: {gender}, 年龄: {age}".format(
            name=basic.get("name", "未知"),
            gender=basic.get("gender", "未知"),
            age=basic.get("age", "未知"),
        ),
        "身份证: {id_card}".format(id_card=basic.get("id_card", "未知")),
        "住址: {addr}".format(addr=basic.get("address", "未知")),
        "职业: {occ}".format(occ=basic.get("occupation", "未知")),
        "电话: {phone}".format(phone=basic.get("phone", "未知")),
        "状态: {status}".format(status=basic.get("status", "未知")),
    ]
    remark = basic.get("remark")
    if remark:
        parts.append("备注: {r}".format(r=remark))
    if records:
        crimes = ", ".join(
            "{ctype}({sentence})".format(
                ctype=r.get("crime_type", "未知"),
                sentence=r.get("sentence", ""),
            )
            for r in records
        )
        parts.append("前科: {c}".format(c=crimes))
    missing: List[str] = []
    if not basic.get("id_card"):
        missing.append("身份证号缺失")
    if not basic.get("phone"):
        missing.append("电话缺失")
    if not basic.get("address"):
        missing.append("住址缺失")
    if not records:
        missing.append("无前科记录或前科数据缺失")
    return {
        "card_id": _make_card_id("person", person_id),
        "case_id": case_id,
        "source": "人员信息 persons.json + 前科 criminal_records.json",
        "source_type": "person",
        "occurred_at": None,
        "location": basic.get("address"),
        "involved_persons": [person_id] if person_id != "UNKNOWN" else [],
        "content": "; ".join(parts),
        "credibility": _CREDIBILITY_MAP["person"],
        "missing_info": missing,
    }


def _card_from_comm(case_id: str, record: dict) -> Optional[dict]:
    """从通讯记录生成情报卡片"""
    if not record:
        return None
    rid = record.get("record_id", "")
    caller = record.get("caller", "")
    callee = record.get("callee", "")
    involved = [p for p in (caller, callee) if p]
    content = "{t} {caller} -> {callee} ({dur}s): {c}".format(
        t=record.get("type", "call"),
        caller=caller,
        callee=callee,
        dur=record.get("duration", "未知"),
        c=record.get("content", ""),
    )
    return {
        "card_id": _make_card_id("comm", rid),
        "case_id": case_id,
        "source": "通讯记录 communications.json [{rid}]".format(rid=rid),
        "source_type": "comm",
        "occurred_at": _parse_datetime(record.get("time")),
        "location": None,
        "involved_persons": involved,
        "content": content,
        "credibility": _CREDIBILITY_MAP["comm"],
        "missing_info": [],
    }


def _card_from_finance(
    case_id: str, txn: dict, counterparty: Optional[str]
) -> Optional[dict]:
    """从银行流水生成情报卡片"""
    if not txn:
        return None
    tid = txn.get("transaction_id", "")
    person_id = txn.get("person_id", "")
    involved = [person_id] if person_id else []
    if counterparty and counterparty not in involved:
        involved.append(counterparty)
    content = "{bank} {amt:.2f}元 ({time}): {remark}".format(
        bank=txn.get("bank", "未知银行"),
        amt=float(txn.get("amount", 0) or 0),
        time=txn.get("time", "未知时间"),
        remark=txn.get("remark", ""),
    )
    return {
        "card_id": _make_card_id("finance", tid),
        "case_id": case_id,
        "source": "银行流水 finance.json [{tid}]".format(tid=tid),
        "source_type": "finance",
        "occurred_at": _parse_datetime(txn.get("time")),
        "location": None,
        "involved_persons": involved,
        "content": content,
        "credibility": _CREDIBILITY_MAP["finance"],
        "missing_info": [],
    }


def _card_from_trajectory(case_id: str, traj: dict) -> Optional[dict]:
    """从出行轨迹生成情报卡片"""
    if not traj:
        return None
    tid = traj.get("trajectory_id", "")
    person_id = traj.get("person_id", "")
    content = "{loc} ({time}, {trans}): {note}".format(
        loc=traj.get("location", "未知地点"),
        time=traj.get("time", "未知时间"),
        trans=traj.get("transport", "未知交通方式"),
        note=traj.get("note", ""),
    )
    return {
        "card_id": _make_card_id("trajectory", tid),
        "case_id": case_id,
        "source": "出行轨迹 trajectories.json [{tid}]".format(tid=tid),
        "source_type": "trajectory",
        "occurred_at": _parse_datetime(traj.get("time")),
        "location": traj.get("location"),
        "involved_persons": [person_id] if person_id else [],
        "content": content,
        "credibility": _CREDIBILITY_MAP["trajectory"],
        "missing_info": [],
    }


def _find_comm_counterparties(records: List[dict], self_id: str) -> List[str]:
    """从通讯记录中识别对手方人员 ID"""
    others: List[str] = []
    seen = set()
    for r in records:
        for field in ("caller", "callee"):
            pid = r.get(field)
            if pid and pid != self_id and pid not in seen:
                seen.add(pid)
                others.append(pid)
    return others


def _find_finance_counterparties(
    records: List[dict], self_id: str, account_map: Dict[str, str]
) -> List[str]:
    """从银行流水中识别对手方人员 ID

    通过账户号->person_id 映射反查对手方。对于 CASH（取现）等无对手方账户
    的情况不返回。
    """
    others: List[str] = []
    seen = set()
    self_accounts = {acct for acct, pid in account_map.items() if pid == self_id}
    for txn in records:
        for acct in (txn.get("from_account"), txn.get("to_account")):
            if not acct or acct == "CASH":
                continue
            if acct in self_accounts:
                continue
            counter_pid = account_map.get(acct)
            if counter_pid and counter_pid != self_id and counter_pid not in seen:
                seen.add(counter_pid)
                others.append(counter_pid)
    return others


def _keyword_match(text: str, keywords: List[str]) -> bool:
    """简单关键词匹配，任一命中即返回 True"""
    if not text or not keywords:
        return False
    lower = text.lower()
    for kw in keywords:
        if kw and kw.lower() in lower:
            return True
    return False


def collect_intelligence(
    case_id: str,
    known_suspects: list,
    human_feedback: str = None,
) -> list:
    """主函数：为已知嫌疑人收集全部情报，生成 IntelligenceCard 列表

    - 遍历 known_suspects，收集每个人的卷宗/人员/通讯/银行/轨迹/前科信息
    - 将每条信息转化为 IntelligenceCard 格式的 dict
    - 根据数据类型设置 source_type 和 credibility
    - 识别通讯和银行流水中涉及的新人员，加入返回结果
    - 如果有 human_feedback，针对性补充相关情报
    """
    cards: List[dict] = []
    seen_card_ids = set()
    discovered_persons: List[str] = []

    def _add(card: Optional[dict]) -> None:
        if not card:
            return
        cid = card.get("card_id")
        if cid and cid in seen_card_ids:
            return
        if cid:
            seen_card_ids.add(cid)
        cards.append(card)

    # 1. 案件卷宗（全局，仅一次）
    try:
        dossier = search_dossier(case_id)
        _add(_card_from_dossier(case_id, dossier))
    except Exception as e:
        logger.warning("生成卷宗情报卡片失败 %s: %s", case_id, e)

    # 解析人工反馈关键词
    feedback_keywords: List[str] = []
    if human_feedback:
        feedback_keywords = [
            k.strip()
            for k in re.split(r"[,，;；\s]+", human_feedback)
            if k.strip()
        ]

    # 预构建账户映射，用于解析银行流水对手方
    account_map = _build_account_person_map(case_id)

    suspects = list(known_suspects or [])
    processed = set()

    def _collect_for_person(person_id: str) -> None:
        if person_id in processed:
            return
        processed.add(person_id)
        # 人员 + 前科
        try:
            info = query_person_info(case_id, person_id)
            _add(_card_from_person(case_id, info))
        except Exception as e:
            logger.warning("人员情报收集失败 %s/%s: %s", case_id, person_id, e)
        # 通讯
        try:
            comms = query_communication(case_id, person_id)
            for r in comms:
                _add(_card_from_comm(case_id, r))
            # 识别新人员
            for new_pid in _find_comm_counterparties(comms, person_id):
                if new_pid not in suspects and new_pid not in discovered_persons:
                    discovered_persons.append(new_pid)
        except Exception as e:
            logger.warning("通讯情报收集失败 %s/%s: %s", case_id, person_id, e)
        # 银行流水
        try:
            txns = query_finance(case_id, person_id)
            for t in txns:
                # 解析对手方
                counter = None
                for acct in (t.get("from_account"), t.get("to_account")):
                    if not acct or acct == "CASH":
                        continue
                    if acct in account_map and account_map[acct] != person_id:
                        counter = account_map[acct]
                        break
                _add(_card_from_finance(case_id, t, counter))
            # 识别新人员
            for new_pid in _find_finance_counterparties(txns, person_id, account_map):
                if new_pid not in suspects and new_pid not in discovered_persons:
                    discovered_persons.append(new_pid)
        except Exception as e:
            logger.warning("银行流水情报收集失败 %s/%s: %s", case_id, person_id, e)
        # 轨迹
        try:
            trajs = query_trajectories(case_id, person_id)
            for t in trajs:
                _add(_card_from_trajectory(case_id, t))
        except Exception as e:
            logger.warning("轨迹情报收集失败 %s/%s: %s", case_id, person_id, e)

    # 2. 遍历已知嫌疑人收集情报
    for pid in suspects:
        _collect_for_person(pid)

    # 3. 处理通讯/银行流水中发现的新关联人员
    # 使用 BFS 逐层扩散（限制深度避免无限扩散），确保多跳关系也能被采集
    max_rounds = 3
    round_num = 0
    queue = [p for p in discovered_persons if p not in processed]
    while queue and round_num < max_rounds:
        current_batch = list(queue)
        queue = []
        for pid in current_batch:
            if pid in processed:
                continue
            before = len(discovered_persons)
            _collect_for_person(pid)
            # 收集过程中若又发现新人员，加入下一轮队列
            for p in discovered_persons[before:]:
                if p not in processed:
                    queue.append(p)
        round_num += 1

    # 4. 人工反馈：针对性补充相关情报
    if feedback_keywords:
        # 4.1 向量库语义检索
        try:
            kb_hits = search_knowledge_base(human_feedback, n_results=5)
            for i, hit in enumerate(kb_hits):
                card = {
                    "card_id": _make_card_id("kb", "fb%d" % i),
                    "case_id": case_id,
                    "source": "向量库检索 collection={c}".format(
                        c=hit.get("collection", "unknown")
                    ),
                    "source_type": "knowledge_base",
                    "occurred_at": None,
                    "location": None,
                    "involved_persons": [],
                    "content": hit.get("document", ""),
                    "credibility": _CREDIBILITY_MAP["knowledge_base"],
                    "missing_info": [],
                }
                _add(card)
        except Exception as e:
            logger.warning("向量库补充检索失败: %s", e)
        # 4.2 全量数据中按关键词过滤补充情报卡片
        try:
            all_comms = loader.load_communications(case_id)
            for r in all_comms:
                text = " ".join([
                    str(r.get("content", "")),
                    str(r.get("caller", "")),
                    str(r.get("callee", "")),
                ])
                if _keyword_match(text, feedback_keywords):
                    _add(_card_from_comm(case_id, r))
            all_finance = loader.load_finance(case_id)
            for t in all_finance:
                text = " ".join([
                    str(t.get("remark", "")),
                    str(t.get("person_id", "")),
                ])
                if _keyword_match(text, feedback_keywords):
                    counter = None
                    for acct in (t.get("from_account"), t.get("to_account")):
                        if not acct or acct == "CASH":
                            continue
                        if acct in account_map:
                            counter = account_map.get(acct)
                            break
                    _add(_card_from_finance(case_id, t, counter))
            all_trajs = loader.load_trajectories(case_id)
            for t in all_trajs:
                text = " ".join([
                    str(t.get("location", "")),
                    str(t.get("note", "")),
                    str(t.get("person_id", "")),
                ])
                if _keyword_match(text, feedback_keywords):
                    _add(_card_from_trajectory(case_id, t))
        except Exception as e:
            logger.warning("人工反馈补充情报收集失败: %s", e)
        # 4.3 外部公开信息检索（mock）
        try:
            web_result = web_search(human_feedback)
            if web_result:
                card = {
                    "card_id": _make_card_id("web", "feedback"),
                    "case_id": case_id,
                    "source": "外部公开信息检索 (mock)",
                    "source_type": "web",
                    "occurred_at": None,
                    "location": None,
                    "involved_persons": [],
                    "content": web_result,
                    "credibility": _CREDIBILITY_MAP["web"],
                    "missing_info": ["外部信息未经验证"],
                }
                _add(card)
        except Exception as e:
            logger.warning("外部信息检索失败: %s", e)

    # 5. 同步至知识图谱（便于后续图查询，幂等操作）
    try:
        _sync_to_knowledge_graph(case_id, cards)
    except Exception as e:
        logger.warning("同步知识图谱失败: %s", e)

    return cards


def _sync_to_knowledge_graph(case_id: str, cards: List[dict]) -> None:
    """将情报卡片中的人员与关系同步至全局知识图谱（幂等）"""
    kg = get_knowledge_graph()
    # 添加人员节点
    person_ids = set()
    for card in cards:
        for pid in card.get("involved_persons", []) or []:
            if pid:
                person_ids.add(pid)
    for pid in person_ids:
        kg.add_node(pid, "person", case_id=case_id)
    # 添加关系边
    for card in cards:
        stype = card.get("source_type")
        involved = card.get("involved_persons", []) or []
        if len(involved) < 2:
            continue
        if stype == "comm":
            kg.add_edge(involved[0], involved[1], "communication", weight=1.0,
                        case_id=case_id, card_id=card.get("card_id"))
        elif stype == "finance":
            kg.add_edge(involved[0], involved[1], "finance", weight=1.0,
                        case_id=case_id, card_id=card.get("card_id"))
