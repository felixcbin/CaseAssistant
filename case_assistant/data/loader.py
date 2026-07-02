"""案件数据加载器

从 data/cases/<case_id>/ 目录加载虚拟案件数据，并提供按人员维度的查询能力。
所有函数在文件缺失或异常时返回空字典/空列表，保证调用方不会因数据缺失而中断。
"""
import json
import os
from typing import List, Dict, Any, Optional

from case_assistant.config import DATA_DIR


def _case_dir(case_id: str) -> str:
    """返回指定案件的数据目录绝对路径。"""
    return os.path.join(DATA_DIR, case_id)


def _load_json(case_id: str, filename: str, default: Any) -> Any:
    """安全加载某个案件目录下的 JSON 文件。

    文件不存在或解析失败时返回 default，不抛出异常。
    """
    file_path = os.path.join(_case_dir(case_id), filename)
    if not os.path.isfile(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def load_case(case_id: str) -> Dict[str, Any]:
    """加载案件完整数据，返回包含所有数据类型的字典。"""
    return {
        "dossier": load_dossier(case_id),
        "persons": load_persons(case_id),
        "communications": load_communications(case_id),
        "finance": load_finance(case_id),
        "trajectories": load_trajectories(case_id),
        "criminal_records": load_criminal_records(case_id),
    }


def load_dossier(case_id: str) -> Dict[str, Any]:
    """加载案件卷宗。"""
    result = _load_json(case_id, "dossier.json", {})
    return result if isinstance(result, dict) else {}


def load_persons(case_id: str) -> List[Dict[str, Any]]:
    """加载人员信息。"""
    result = _load_json(case_id, "persons.json", [])
    return result if isinstance(result, list) else []


def load_communications(case_id: str) -> List[Dict[str, Any]]:
    """加载通讯记录。"""
    result = _load_json(case_id, "communications.json", [])
    return result if isinstance(result, list) else []


def load_finance(case_id: str) -> List[Dict[str, Any]]:
    """加载银行流水。"""
    result = _load_json(case_id, "finance.json", [])
    return result if isinstance(result, list) else []


def load_trajectories(case_id: str) -> List[Dict[str, Any]]:
    """加载出行轨迹。"""
    result = _load_json(case_id, "trajectories.json", [])
    return result if isinstance(result, list) else []


def load_criminal_records(case_id: str) -> List[Dict[str, Any]]:
    """加载前科信息。"""
    result = _load_json(case_id, "criminal_records.json", [])
    return result if isinstance(result, list) else []


def query_person(case_id: str, person_id: str) -> Optional[Dict[str, Any]]:
    """查询单个人员信息。"""
    for person in load_persons(case_id):
        if person.get("person_id") == person_id:
            return person
    return None


def query_person_communications(case_id: str, person_id: str) -> List[Dict[str, Any]]:
    """查询某人的通讯记录（作为caller或callee）。"""
    return [
        record
        for record in load_communications(case_id)
        if record.get("caller") == person_id or record.get("callee") == person_id
    ]


def query_person_finance(case_id: str, person_id: str) -> List[Dict[str, Any]]:
    """查询某人的银行流水。"""
    return [
        txn
        for txn in load_finance(case_id)
        if txn.get("person_id") == person_id
    ]


def query_person_trajectories(case_id: str, person_id: str) -> List[Dict[str, Any]]:
    """查询某人的出行轨迹。"""
    return [
        traj
        for traj in load_trajectories(case_id)
        if traj.get("person_id") == person_id
    ]


def query_person_records(case_id: str, person_id: str) -> List[Dict[str, Any]]:
    """查询某人的前科记录。"""
    return [
        record
        for record in load_criminal_records(case_id)
        if record.get("person_id") == person_id
    ]


def list_cases() -> List[str]:
    """列出所有可用案件ID。

    扫描 DATA_DIR 下所有包含 dossier.json 的子目录，视为有效案件。
    """
    if not os.path.isdir(DATA_DIR):
        return []
    cases: List[str] = []
    for name in sorted(os.listdir(DATA_DIR)):
        case_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(case_path) and os.path.isfile(
            os.path.join(case_path, "dossier.json")
        ):
            cases.append(name)
    return cases
