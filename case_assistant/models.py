"""核心数据模型 (Pydantic)"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CaseType(str, Enum):
    SMUGGLING = "smuggling"        # 走私
    DRUG = "drug"                  # 贩毒
    VICE = "vice"                  # 扫黄
    JUVENILE = "juvenile"          # 未成年人犯罪


class Credibility(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Case(BaseModel):
    case_id: str
    case_type: CaseType
    case_brief: str
    occurred_at: Optional[datetime] = None
    location: Optional[str] = None
    known_suspects: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class IntelligenceCard(BaseModel):
    card_id: str
    case_id: str
    source: str                    # 数据来源
    source_type: str               # dossier/person/comm/finance/web
    occurred_at: Optional[datetime] = None
    location: Optional[str] = None
    involved_persons: List[str] = Field(default_factory=list)
    content: str                   # 情报内容摘要
    credibility: Credibility
    missing_info: List[str] = Field(default_factory=list)


class Clue(BaseModel):
    clue_id: str
    case_id: str
    description: str
    related_card_ids: List[str]    # 关联情报
    related_persons: List[str] = Field(default_factory=list)
    score: int = Field(ge=0, le=100)
    reasoning: str                 # 推理过程
    needs_more_intel: bool = False


class Profile(BaseModel):
    person_id: str
    case_id: str
    basic_info: Dict[str, Any]
    behavior_pattern: Dict[str, Any]
    relation_network: Dict[str, Any]  # 节点 + 边，供图谱渲染
    risk_level: RiskLevel
    risk_reasoning: str


class InvestigationReport(BaseModel):
    report_id: str
    case_id: str
    overview: str
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    key_clues: List[Dict[str, Any]] = Field(default_factory=list)
    suspect_profiles: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence_appendix: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
