"""报告生成 Agent

汇总情报收集、线索分析、嫌疑人画像各环节产出，生成结构化侦查报告，
并渲染为可下载的 Markdown 字符串。报告涵盖案件概述、侦查过程时间线、
关键线索与证据链、嫌疑人画像摘要、侦查建议、证据清单附录六个章节。
"""
import logging
from datetime import datetime
from typing import Any, Dict, List

from case_assistant.state import CaseState
from case_assistant.tools.report_tools import (
    format_evidence_chain,
    generate_timeline,
    render_markdown,
)

logger = logging.getLogger(__name__)


class ReportAgent:
    """报告生成 Agent - 汇总全流程产出，生成结构化侦查报告"""

    def run(self, state: CaseState) -> dict:
        """执行报告生成

        - 调用 format_evidence_chain 梳理证据链
        - 调用 generate_timeline 生成时间线
        - 组装 InvestigationReport
        - 调用 render_markdown 生成可下载的 Markdown 报告
        - 返回 {report: {...}, current_phase: "report_generation"}
        """
        case_id = state.get("case_id", "")
        case_brief = state.get("case_brief", "")
        intelligence_cards = state.get("intelligence_cards") or []
        clues = state.get("clues") or []
        profiles = state.get("profiles") or []
        human_feedback = state.get("human_feedback") or ""
        errors: List[str] = list(state.get("errors") or [])

        try:
            # 梳理证据链
            evidence_chains = format_evidence_chain(clues, intelligence_cards)

            # 生成时间线
            timeline = generate_timeline(intelligence_cards, case_id)

            # 组装关键线索（带证据链信息）
            key_clues = self._build_key_clues(clues, evidence_chains)

            # 组装嫌疑人画像摘要
            suspect_profiles = self._build_suspect_profiles(profiles)

            # 生成侦查建议
            recommendations = self._build_recommendations(
                clues=clues,
                profiles=profiles,
                evidence_chains=evidence_chains,
                human_feedback=human_feedback,
            )

            # 案件概述
            overview = self._build_overview(
                case_brief=case_brief,
                profiles=profiles,
                clues=clues,
                human_feedback=human_feedback,
            )

            generated_at = datetime.now().isoformat()
            report_id = f"RPT-{case_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            report: Dict[str, Any] = {
                "report_id": report_id,
                "case_id": case_id,
                "overview": overview,
                "timeline": timeline,
                "key_clues": key_clues,
                "suspect_profiles": suspect_profiles,
                "recommendations": recommendations,
                "evidence_appendix": evidence_chains,
                "generated_at": generated_at,
            }

            # 渲染 Markdown
            try:
                report["markdown"] = render_markdown(report)
            except Exception as e:
                msg = f"ReportAgent: Markdown 渲染失败 - {e}"
                logger.exception(msg)
                errors.append(msg)
                report["markdown"] = ""

            return {
                "report": report,
                "current_phase": "report_generation",
                "errors": errors,
            }
        except Exception as e:
            msg = f"ReportAgent: 报告生成失败 - {e}"
            logger.exception(msg)
            errors.append(msg)
            return {
                "report": {
                    "report_id": f"RPT-{case_id}-ERROR",
                    "case_id": case_id,
                    "overview": "报告生成失败，请检查 errors 字段。",
                    "timeline": [],
                    "key_clues": [],
                    "suspect_profiles": [],
                    "recommendations": [],
                    "evidence_appendix": [],
                    "generated_at": datetime.now().isoformat(),
                    "markdown": "",
                },
                "current_phase": "report_generation",
                "errors": errors,
            }

    def _build_overview(self, case_brief: str, profiles: list,
                        clues: list, human_feedback: str) -> str:
        """构建案件概述。"""
        parts: List[str] = []
        if case_brief:
            parts.append(case_brief)
        else:
            parts.append("（案件简述缺失）")

        # 画像与线索数量摘要
        parts.append(
            f"本次侦查共完成嫌疑人画像 {len(profiles)} 份，梳理线索 {len(clues)} 条。"
        )

        # 高风险嫌疑人提示
        high_risk = [
            p for p in profiles
            if isinstance(p, dict) and p.get("risk_level") == "high"
        ]
        if high_risk:
            names = []
            for p in high_risk:
                basic = p.get("basic_info") or {}
                name = basic.get("name", p.get("person_id", ""))
                names.append(name)
            parts.append(
                f"其中高风险嫌疑人 {len(high_risk)} 名：{', '.join(names)}，"
                f"建议作为重点侦查对象。"
            )

        # 人工反馈响应
        if human_feedback:
            parts.append(
                f"已纳入办案人员反馈意见进行报告调整：{human_feedback}"
            )

        return "".join(parts)

    def _build_key_clues(self, clues: list,
                         evidence_chains: list) -> List[Dict[str, Any]]:
        """组装关键线索列表，合并证据链信息。"""
        # 以 clue_id 为键建立证据链索引
        chain_index = {
            ec.get("clue_id"): ec
            for ec in evidence_chains
            if isinstance(ec, dict) and ec.get("clue_id")
        }

        key_clues: List[Dict[str, Any]] = []
        for clue in clues:
            if not isinstance(clue, dict):
                continue
            clue_id = clue.get("clue_id")
            chain = chain_index.get(clue_id, {})
            merged = dict(clue)
            # 合并证据链的可信度与来源信息
            if chain:
                merged["credibility"] = chain.get("credibility", "")
                merged["sources"] = chain.get("sources", [])
            key_clues.append(merged)

        # 按评分降序排列
        key_clues.sort(key=lambda x: x.get("score", 0), reverse=True)
        return key_clues

    def _build_suspect_profiles(self, profiles: list) -> List[Dict[str, Any]]:
        """组装嫌疑人画像摘要。"""
        suspect_profiles: List[Dict[str, Any]] = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            suspect_profiles.append(dict(profile))
        # 高风险优先
        risk_order = {"high": 0, "medium": 1, "low": 2}
        suspect_profiles.sort(
            key=lambda x: risk_order.get(x.get("risk_level", "low"), 3)
        )
        return suspect_profiles

    def _build_recommendations(self, clues: list, profiles: list,
                               evidence_chains: list,
                               human_feedback: str) -> List[str]:
        """生成侦查建议，基于线索评分、画像风险等级与人工反馈。"""
        recommendations: List[str] = []

        # 1. 高风险嫌疑人重点监控建议
        high_risk = [
            p for p in profiles
            if isinstance(p, dict) and p.get("risk_level") == "high"
        ]
        medium_risk = [
            p for p in profiles
            if isinstance(p, dict) and p.get("risk_level") == "medium"
        ]
        if high_risk:
            names = []
            for p in high_risk:
                basic = p.get("basic_info") or {}
                names.append(basic.get("name", p.get("person_id", "")))
            recommendations.append(
                f"对高风险嫌疑人（{', '.join(names)}）实施重点监控，"
                f"优先调取其通讯、资金、轨迹的实时数据。"
            )
        if medium_risk:
            names = []
            for p in medium_risk:
                basic = p.get("basic_info") or {}
                names.append(basic.get("name", p.get("person_id", "")))
            recommendations.append(
                f"对中风险嫌疑人（{', '.join(names)}）保持关注，"
                f"适时开展外围取证。"
            )

        # 2. 高分线索取证方向
        high_score_clues = [
            c for c in clues
            if isinstance(c, dict) and c.get("score", 0) >= 70
        ]
        if high_score_clues:
            top = high_score_clues[0]
            recommendations.append(
                f"围绕高分线索「{top.get('description', '')}」开展深入取证，"
                f"进一步固定证据链。"
            )

        # 3. 需补充情报的线索
        needs_more = [
            c for c in clues
            if isinstance(c, dict) and c.get("needs_more_intel")
        ]
        if needs_more:
            recommendations.append(
                f"共有 {len(needs_more)} 条线索标注信息不足，"
                f"建议开展补充侦查，重点补充相关人员的通讯与资金数据。"
            )

        # 4. 关系网络扩展建议
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            relation = profile.get("relation_network") or {}
            nodes = relation.get("nodes") or []
            if len(nodes) >= 3:
                basic = profile.get("basic_info") or {}
                name = basic.get("name", profile.get("person_id", ""))
                recommendations.append(
                    f"围绕嫌疑人 {name} 的关系网络（{len(nodes)} 个关联节点）"
                    f"开展串并案分析，挖掘潜在同伙。"
                )
                break  # 仅给一条关系网络建议

        # 5. 未成年人保护提示（如涉及）
        minor_keywords = ["未成年", "小孩", "少年", "学生"]
        minor_mentioned = False
        for item in list(clues) + list(profiles):
            if not isinstance(item, dict):
                continue
            text = str(item)
            if any(kw in text for kw in minor_keywords):
                minor_mentioned = True
                break
        # 也检查画像中是否有未成年人
        if not minor_mentioned:
            for p in profiles:
                if not isinstance(p, dict):
                    continue
                basic = p.get("basic_info") or {}
                age = basic.get("age")
                if isinstance(age, (int, float)) and age < 18:
                    minor_mentioned = True
                    break
        if minor_mentioned:
            recommendations.append(
                "本案涉及未成年人，应依法启动未成年人保护程序，"
                "指定合适成年人到场，并联系未成年人保护组织介入。"
            )

        # 6. 人工反馈响应建议
        if human_feedback:
            recommendations.append(
                f"已根据办案人员反馈「{human_feedback}」调整侦查方向，"
                f"建议据此开展专项核查。"
            )

        # 兜底建议
        if not recommendations:
            recommendations.append(
                "继续跟踪现有线索，定期复核嫌疑人动态，"
                "视情报补充情况适时调整侦查策略。"
            )

        return recommendations
