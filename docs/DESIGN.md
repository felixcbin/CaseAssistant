# CaseAssistant 多智能体案件侦查系统 - 详细设计文档

> 版本：v1.0 ｜ 日期：2026-07-02 ｜ 阶段：演示原型（MVP）

---

## 一、项目概述

### 1.1 项目背景

传统案件侦查依赖人工梳理卷宗、关联线索、刻画嫌疑人，存在信息分散、关联维度有限、效率瓶颈等问题。本项目通过多智能体协作，辅助办案人员完成情报收集、线索分析、嫌疑人画像、报告生成等环节，提升侦查效率与线索发现能力。

### 1.2 项目目标

| 维度 | 目标 |
|---|---|
| **功能目标** | 4 个核心 Agent 协作，完成从案件录入到侦查报告输出的全链路辅助 |
| **形态目标** | 可演示的 MVP 原型，含可视化界面，可输入虚拟案件并输出结构化报告 |
| **案件范围** | 刑事侦查：走私、贩毒、扫黄、未成年人犯罪 |
| **非目标** | 不替代办案人员决策；不直接对接真实公安系统；不处理涉密数据 |

### 1.3 系统定位

- **角色**：办案辅助工具，输出建议与线索，由办案人员决策采信
- **使用者**：侦查人员、情报分析员
- **部署**：演示原型阶段，本地运行，使用模拟数据

---

## 二、系统架构

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                  用户交互层 (Streamlit UI)                │
│        案件录入 / 进度查看 / 报告下载 / 人工干预           │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              编排层 (LangGraph State Machine)             │
│     状态管理 / Agent 调度 / 条件路由 / 人机协同节点        │
└────────────────────────┬─────────────────────────────────┘
                         │
   ┌─────────────┬───────┴────────┬─────────────┐
   ▼             ▼                ▼             ▼
┌──────────┐ ┌──────────┐  ┌───────────┐ ┌──────────┐
│ 情报收集  │ │ 线索分析  │  │ 嫌疑人画像 │ │ 报告生成  │
│ Agent    │ │ Agent    │  │ Agent     │ │ Agent    │
│ (CrewAI) │ │ (CrewAI) │  │ (CrewAI)  │ │ (CrewAI) │
└────┬─────┘ └────┬─────┘  └─────┬─────┘ └────┬─────┘
     │            │              │            │
     ▼            ▼              ▼            ▼
┌──────────────────────────────────────────────────────────┐
│                  共享记忆层 (Memory)                       │
│  向量库(ChromaDB)  +  知识图谱(NetworkX)  +  案件状态库    │
└──────────────────────────────────────────────────────────┘
```

### 2.2 分层职责

- **用户交互层**：案件录入、流程可视化、报告展示、人工审核干预
- **编排层**：基于 LangGraph 的状态机，管理案件状态、调度 Agent、处理条件分支与人工节点
- **智能体层**：4 个基于 CrewAI 的角色化 Agent，各自承担专项任务
- **共享记忆层**：向量库（语义检索）、知识图谱（关系推理）、案件状态库（流程数据）

### 2.3 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 编排框架 | LangGraph ≥ 0.2 | 状态机式调度，支持条件路由与人工节点 |
| Agent 框架 | CrewAI ≥ 0.4 | 角色化 Agent 定义，工具集成便捷 |
| LLM | DeepSeek / Qwen | 中文场景效果好；原型可用 API，生产可本地化 |
| 向量库 | ChromaDB | 轻量本地，原型足够 |
| 知识图谱 | NetworkX | 原型阶段；生产可换 Neo4j |
| UI | Streamlit | 快速搭建演示界面 |
| 数据格式 | Pydantic | 结构化数据校验 |
| 语言 | Python 3.11+ | |

---

## 三、LangGraph 状态机设计

### 3.1 全局状态定义 (State)

```python
from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel

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
```

### 3.2 节点 (Nodes)

| 节点 | 类型 | 职责 |
|---|---|---|
| `init_case` | 自动 | 初始化案件状态，加载已有数据 |
| `intelligence_gathering` | Agent | 情报收集 Agent 执行 |
| `clue_analysis` | Agent | 线索分析 Agent 执行 |
| `suspect_profiling` | Agent | 嫌疑人画像 Agent 执行 |
| `human_review` | 人工 | 办案人员审核线索，可要求补充情报 |
| `report_generation` | Agent | 报告生成 Agent 执行 |
| `end` | 终止 | 流程结束 |

### 3.3 状态流转图

```
        START
          │
          ▼
    ┌──────────────┐
    │  init_case   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────┐
    │ intelligence_gathering│ ◄──────────┐
    └──────┬───────────────┘            │
           │                            │
           ▼                            │
    ┌──────────────────────┐            │
    │   clue_analysis      │            │
    └──────┬───────────────┘            │
           │                            │
           ▼                            │
    ┌──────────────────────┐            │
    │  suspect_profiling   │            │
    └──────┬───────────────┘            │
           │                            │
           ▼                            │
    ┌──────────────────────┐            │
    │    human_review      │ ───────────┘
    └──────┬───────────────┘  (needs_more_intel=true)
           │ (needs_more_intel=false)
           ▼
    ┌──────────────────────┐
    │ report_generation    │
    └──────┬───────────────┘
           │
           ▼
         END
```

### 3.4 条件路由 (Conditional Edges)

```python
def route_after_review(state: CaseState) -> str:
    """人工审核后路由"""
    if state.get("needs_more_intel") or state.get("human_feedback"):
        return "intelligence_gathering"   # 回到情报收集补充
    return "report_generation"

def route_after_clue(state: CaseState) -> str:
    """线索分析后路由：线索不足时先补充情报"""
    if len(state.get("clues", [])) < 1 or state.get("needs_more_intel"):
        if state.get("iteration", 0) >= 3:   # 防止死循环
            return "suspect_profiling"
        return "intelligence_gathering"
    return "suspect_profiling"
```

### 3.5 编排代码骨架

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(CaseState)

workflow.add_node("init_case", init_case_node)
workflow.add_node("intelligence_gathering", intelligence_agent.run)
workflow.add_node("clue_analysis", clue_agent.run)
workflow.add_node("suspect_profiling", profile_agent.run)
workflow.add_node("human_review", human_review_node)
workflow.add_node("report_generation", report_agent.run)

workflow.set_entry_point("init_case")
workflow.add_edge("init_case", "intelligence_gathering")
workflow.add_edge("intelligence_gathering", "clue_analysis")
workflow.add_conditional_edges("clue_analysis", route_after_clue)
workflow.add_edge("suspect_profiling", "human_review")
workflow.add_conditional_edges("human_review", route_after_review)
workflow.add_edge("report_generation", END)

app = workflow.compile()
```

---

## 四、智能体设计

### 4.1 情报收集 Agent (Intelligence Agent)

| 项 | 内容 |
|---|---|
| **角色** | 情报分析员 |
| **目标** | 围绕案件与已知嫌疑人，从多源数据中检索、整合结构化情报 |
| **输入** | case_id、known_suspects、human_feedback（可选） |
| **输出** | List[IntelligenceCard] |

**工具 (Tools)**：
- `search_dossier(case_id)`：检索案件卷宗
- `query_person(person_id)`：查询人员户籍/前科信息
- `query_communication(person_id)`：查询通讯记录
- `query_finance(person_id)`：查询银行流水
- `search_knowledge_base(query)`：向量库语义检索
- `web_search(query)`：外部公开信息检索（演示用 mock）

**Prompt 模板**：
```
你是一名资深情报分析员，负责案件【{case_id}】的情报收集工作。

案件简述：{case_brief}
已知嫌疑人：{known_suspects}
{human_feedback_section}

请围绕案件类型（{case_type}）的核心要素，使用可用工具完成：
1. 调取已知嫌疑人的户籍、前科、通讯、银行流水信息
2. 识别需要进一步关注的关键人物与关系
3. 将每条情报整理为标准化情报卡片

输出要求：
- 每条情报包含：来源、时间、地点、涉及人员、内容摘要、可信度（高/中/低）
- 标注信息缺失项，供后续补充
- 如发现新的关联人员，加入 known_suspects 列表
```

### 4.2 线索分析 Agent (Clue Agent)

| 项 | 内容 |
|---|---|
| **角色** | 线索分析专家 |
| **目标** | 对情报进行关联分析、时空碰撞、异常识别，输出可追溯线索 |
| **输入** | intelligence_cards、case_brief |
| **输出** | List[Clue] |

**工具 (Tools)**：
- `build_relation_graph(cards)`：构建关系图谱
- `spatiotemporal_collision(cards)`：时空轨迹碰撞
- `detect_anomaly(cards)`：异常行为识别（如频繁大额转账、深夜通讯聚集）
- `score_clue(clue)`：线索可信度评分

**Prompt 模板**：
```
你是一名线索分析专家，负责对以下情报进行深度分析。

案件简述：{case_brief}
情报卡片集合：{intelligence_cards}

请执行：
1. 构建人员关系图谱，识别核心节点与异常聚集子群
2. 进行时空碰撞分析，发现多人共现的时间/地点
3. 识别异常行为模式（资金、通讯、出行）
4. 对每条线索给出可信度评分（0-100）与推理依据

输出要求：
- 每条线索包含：线索描述、关联证据ID、可信度评分、推理过程
- 按可信度降序排列
- 明确标注信息不足、需要补充情报的情况（设置 needs_more_intel）
```

### 4.3 嫌疑人画像 Agent (Profile Agent)

| 项 | 内容 |
|---|---|
| **角色** | 犯罪画像分析师 |
| **目标** | 为每位嫌疑人构建多维度画像与关系网络 |
| **输入** | known_suspects、intelligence_cards、clues |
| **输出** | List[Profile] |

**工具 (Tools)**：
- `aggregate_person_data(person_id, cards)`：聚合个人数据
- `build_behavior_pattern(person_id)`：行为模式建模
- `draw_relation_graph(person_id)`：个人关系图谱
- `assess_risk(person_id)`：风险评估

**Prompt 模板**：
```
你是一名犯罪画像分析师，为以下嫌疑人构建画像。

嫌疑人列表：{suspects}
可用情报：{intelligence_cards}
可用线索：{clues}

请为每位嫌疑人输出：
1. 基础画像：身份信息、前科、社会关系
2. 行为模式：通讯规律、资金往来特征、活动轨迹
3. 关系网络：核心关联人、组织角色定位（如组织者/执行者/资金中转）
4. 风险评估：嫌疑等级（高/中/低）及依据

输出要求：
- 关系网络以结构化数据呈现，便于前端渲染图谱
- 嫌疑等级需有明确推理链
- 区分"已证实"与"推测"内容
```

### 4.4 报告生成 Agent (Report Agent)

| 项 | 内容 |
|---|---|
| **角色** | 侦查报告撰写员 |
| **目标** | 汇总全流程产出，生成结构化侦查报告 |
| **输入** | case_brief、intelligence_cards、clues、profiles、human_feedback |
| **输出** | InvestigationReport |

**工具 (Tools)**：
- `format_evidence_chain(clues, cards)`：梳理证据链
- `generate_timeline(cards)`：生成案件时间线
- `render_markdown(template, data)`：渲染报告

**Prompt 模板**：
```
你是侦查报告撰写员，请基于以下材料生成标准化侦查报告。

案件简述：{case_brief}
情报卡片：{intelligence_cards}
线索清单：{clues}
嫌疑人画像：{profiles}
办案人员反馈：{human_feedback}

报告结构要求：
1. 案件概述
2. 侦查过程（按时间线）
3. 关键线索与证据链（每条线索附证据ID与可信度）
4. 嫌疑人画像摘要（含嫌疑等级与依据）
5. 侦查建议（下一步取证方向、重点监控对象）
6. 证据清单附录

输出要求：
- 客观陈述，区分事实与推测
- 每条结论标注证据来源
- 使用规范的侦查术语
```

---

## 五、数据结构设计

### 5.1 核心数据模型 (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Optional
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
    occurred_at: Optional[datetime]
    location: Optional[str]
    involved_persons: List[str]
    content: str                   # 情报内容摘要
    credibility: Credibility
    missing_info: List[str] = Field(default_factory=list)

class Clue(BaseModel):
    clue_id: str
    case_id: str
    description: str
    related_card_ids: List[str]    # 关联情报
    related_persons: List[str]
    score: int = Field(ge=0, le=100)
    reasoning: str                 # 推理过程
    needs_more_intel: bool = False

class Profile(BaseModel):
    person_id: str
    case_id: str
    basic_info: dict
    behavior_pattern: dict
    relation_network: dict         # 节点 + 边，供图谱渲染
    risk_level: str                # high/medium/low
    risk_reasoning: str

class InvestigationReport(BaseModel):
    report_id: str
    case_id: str
    overview: str
    timeline: List[dict]
    key_clues: List[dict]
    suspect_profiles: List[dict]
    recommendations: List[str]
    evidence_appendix: List[dict]
    generated_at: datetime
```

### 5.2 关系图谱数据格式

```python
# 供前端（如 Streamlit + pyvis / echarts）渲染
relation_network = {
    "nodes": [
        {"id": "P001", "label": "张三", "group": "suspect", "size": 30},
        {"id": "P002", "label": "李四", "group": "associate", "size": 20},
    ],
    "edges": [
        {"source": "P001", "target": "P002", "relation": "通讯频繁", "weight": 8},
        {"source": "P001", "target": "P002", "relation": "资金往来", "weight": 5},
    ],
}
```

---

## 六、共享记忆层设计

### 6.1 向量库 (ChromaDB)

- **Collection 划分**：
  - `dossiers`：案件卷宗（按案件分块）
  - `person_profiles`：人员历史画像
  - `case_knowledge`：案件类型相关知识（贩毒常见模式、走私链条特征等）
- **Embedding**：bge-small-zh（轻量中文）或 LLM 自带 embedding
- **检索方式**：语义检索 + 元数据过滤（case_id、person_id、time）

### 6.2 知识图谱 (NetworkX)

- **节点**：人员、地点、机构、案件、物品
- **边**：通讯、资金、同行、亲属、同案
- **用途**：关系推理、子群发现、中心性分析、路径查找
- **持久化**：GraphML 文件

### 6.3 案件状态库

- SQLite 存储案件流程状态、Agent 产出、人工反馈
- 便于断点续跑与流程审计

---

## 七、演示场景设计

### 7.1 虚拟案件：跨境贩毒网络侦查

**案件简述**：
2026 年 3 月，A 市公安局禁毒支队在例行检查中查获一批冰毒（约 2kg），缴获物品中有一部未登记手机。经初步侦查，锁定嫌疑人"张三"（P001），疑似本地分销节点。需进一步摸清上下游链条。

**模拟数据集**（存放于 `data/cases/CASE-2026-001/`）：

| 数据类型 | 文件 | 说明 |
|---|---|---|
| 卷宗 | `dossier.json` | 案件基础信息、查获经过 |
| 人员 | `persons.json` | 张三及关联人员 8-10 人 |
| 通讯 | `communications.json` | 通话/聊天记录（含异常聚集时段） |
| 银行 | `finance.json` | 银行流水（含分散转入、集中转出） |
| 轨迹 | `trajectories.json` | 出行轨迹（含边境往返） |
| 前科 | `criminal_records.json` | 部分人员前科信息 |

**关键线索设计**（供验证 Agent 分析能力）：
1. 张三与"李四"（P002）深夜通讯频繁，李四有边境往返轨迹
2. 张三账户每月固定日期收到多笔小额转入，随后单笔大额转出至"王五"（P003）
3. 王五与一名前科人员"赵六"（P004）共同出现在多个地点
4. 存在一名未成年人"小七"（P005）被卷入分销环节（未成年人犯罪要素）

### 7.2 预期演示效果

1. 用户在界面录入案件 ID，启动侦查流程
2. 可视化展示 LangGraph 状态流转（当前节点、进度）
3. 各 Agent 产出实时展示（情报卡片、线索列表、画像、关系图谱）
4. 人工审核节点：用户可标记"需补充情报"并给出反馈，触发回溯
5. 最终输出可下载的侦查报告（Markdown / PDF）
6. 关系图谱以交互式图谱呈现

---

## 八、实施计划

### 8.1 阶段划分

| 阶段 | 内容 | 产出 |
|---|---|---|
| **P0：骨架** | 项目结构、依赖、配置、State 定义 | 可运行的空框架 |
| **P1：数据** | 虚拟案件数据集 + 数据加载工具 | CASE-2026-001 可加载 |
| **P2：Agent** | 4 个 Agent + 工具函数实现 | 单 Agent 可独立运行 |
| **P3：编排** | LangGraph 状态机串联 + 条件路由 | 全链路可跑通 |
| **P4：UI** | Streamlit 界面 + 图谱可视化 | 可演示 |
| **P5：优化** | Prompt 调优、异常处理、迭代上限 | 演示稳定 |

### 8.2 验收标准

- 输入 CASE-2026-001，5 分钟内输出完整侦查报告
- 报告含 ≥3 条可追溯线索、≥2 个嫌疑人画像
- 关系图谱正确呈现核心人物关系
- 人工干预节点可触发情报回溯（至少 1 轮）
- 流程无死循环、无未捕获异常

---

## 九、风险与约束

| 风险 | 影响 | 应对 |
|---|---|---|
| LLM 输出不稳定 | Agent 产出格式不一致 | Pydantic 校验 + 重试机制 + 结构化输出（function calling） |
| 情报回溯死循环 | 流程无法终止 | iteration 上限（默认 3 轮）+ 强制进入下一阶段 |
| 模拟数据偏简单 | 演示效果受限 | 设计 4 类线索覆盖各 Agent 能力 |
| 知识图谱渲染 | UI 复杂度高 | 用 pyvis/echarts 简单图谱，不做复杂交互 |
| 案件类型差异大 | 4 类案件要素不同 | 按案件类型加载不同 Prompt 模板与知识库 |

---

## 十、后续演进方向（非本期）

- 接入真实数据源（公安内网、户籍、通讯系统）
- 私有化部署 LLM（保障数据安全）
- 知识图谱迁移至 Neo4j，支持大规模图查询
- 增加更多 Agent（如物证分析、电子取证、法律适用）
- 多案件关联分析、串并案能力
- 审计与可解释性：全流程推理链留痕

---

*本文档为演示原型阶段设计，将随开发迭代持续更新。*
