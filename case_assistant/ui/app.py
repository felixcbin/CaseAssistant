"""CaseAssistant 案件侦查助手 - Streamlit UI

提供案件录入、流程可视化、Agent 产出展示、人工审核、报告下载等功能。
"""
import streamlit as st
from typing import Dict, Any, List

from case_assistant.data import loader
from case_assistant.orchestration.workflow import (
    run_case_to_review,
    resume_after_review,
    run_case,
)
from case_assistant.memory.knowledge_graph import get_knowledge_graph

# 页面配置
st.set_page_config(
    page_title="CaseAssistant 案件侦查助手",
    page_icon="🔍",
    layout="wide",
)


# ==================== 辅助函数 ====================

def render_graph_html(graph_data: Dict[str, Any], height: int = 500) -> str:
    """将关系图谱数据渲染为 pyvis 交互式 HTML"""
    try:
        from pyvis.network import Network
        import tempfile
        import os

        net = Network(height=f"{height}px", width="100%", directed=True)
        net.toggle_physics(True)

        # 颜色映射
        color_map = {
            "suspect": "#ff4444",
            "associate": "#ff9900",
            "witness": "#4488cc",
            "person": "#66aa66",
            "unknown": "#999999",
        }

        nodes = graph_data.get("nodes", []) or []
        edges = graph_data.get("edges", []) or []

        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", node.get("name", node_id))
            group = node.get("group", node.get("node_type", "unknown"))
            size = node.get("size", 20)
            color = color_map.get(group, "#999999")
            net.add_node(node_id, label=label, size=size, color=color, title=str(node))

        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "related")
            weight = edge.get("weight", 1.0)
            net.add_edge(source, target, title=relation, value=weight, label=relation)

        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        )
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tmp_file.name)
        return html
    except Exception as e:
        return f"<p>图谱渲染失败: {e}</p>"


def show_intelligence_cards(cards: List[Dict[str, Any]]):
    """展示情报卡片"""
    if not cards:
        st.info("暂无情报卡片")
        return

    st.caption(f"共 {len(cards)} 张情报卡片")
    for card in cards:
        card_id = card.get("card_id", "")
        source_type = card.get("source_type", "")
        content = card.get("content", "")
        credibility = card.get("credibility", "")
        persons = card.get("involved_persons", []) or []

        cred_color = {"high": "green", "medium": "orange", "low": "red"}.get(
            credibility, "gray"
        )

        with st.expander(f"[{card_id}] {source_type} - {content[:60]}..."):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.write(f"**内容**: {content}")
            with cols[1]:
                st.markdown(
                    f"**可信度**: :{cred_color}[{credibility}]"
                )
            with cols[2]:
                st.write(f"**涉及人员**: {', '.join(persons)}")

            if card.get("occurred_at"):
                st.write(f"**时间**: {card.get('occurred_at')}")
            if card.get("location"):
                st.write(f"**地点**: {card.get('location')}")
            if card.get("source"):
                st.write(f"**来源**: {card.get('source')}")
            if card.get("missing_info"):
                st.warning(f"**信息缺失**: {', '.join(card['missing_info'])}")


def show_clues(clues: List[Dict[str, Any]]):
    """展示线索列表"""
    if not clues:
        st.info("暂无线索")
        return

    st.caption(f"共 {len(clues)} 条线索（按可信度降序）")
    for clue in clues:
        clue_id = clue.get("clue_id", "")
        description = clue.get("description", "")
        score = clue.get("score", 0)
        reasoning = clue.get("reasoning", "")
        persons = clue.get("related_persons", []) or []
        needs_more = clue.get("needs_more_intel", False)

        score_color = (
            "green" if score >= 70 else "orange" if score >= 40 else "red"
        )

        with st.expander(f"[{clue_id}] {description[:60]}... (评分: {score})"):
            st.write(f"**描述**: {description}")
            st.markdown(f"**可信度评分**: :{score_color}[{score}/100]")
            st.write(f"**推理过程**: {reasoning}")
            if persons:
                st.write(f"**关联人员**: {', '.join(persons)}")
            if needs_more:
                st.warning("此线索标注信息不足，建议补充情报")


def show_profiles(profiles: List[Dict[str, Any]]):
    """展示嫌疑人画像"""
    if not profiles:
        st.info("暂无嫌疑人画像")
        return

    st.caption(f"共 {len(profiles)} 份画像")

    risk_colors = {"high": "red", "medium": "orange", "low": "green"}
    risk_labels = {"high": "高风险", "medium": "中风险", "low": "低风险"}

    for profile in profiles:
        person_id = profile.get("person_id", "")
        basic_info = profile.get("basic_info", {}) or {}
        name = basic_info.get("name", person_id)
        risk_level = profile.get("risk_level", "low")
        risk_score = profile.get("risk_score", 0)
        risk_reasoning = profile.get("risk_reasoning", "")
        risk_factors = profile.get("risk_factors", []) or []

        color = risk_colors.get(risk_level, "gray")
        label = risk_labels.get(risk_level, risk_level)

        with st.expander(f"{name} ({person_id}) - :{color}[{label}] ({risk_score}分)"):
            # 基础信息
            st.subheader("基础信息")
            info_cols = st.columns(3)
            with info_cols[0]:
                st.write(f"**姓名**: {name}")
                st.write(f"**性别**: {basic_info.get('gender', '')}")
                st.write(f"**年龄**: {basic_info.get('age', '')}")
            with info_cols[1]:
                st.write(f"**身份证**: {basic_info.get('id_card', '')}")
                st.write(f"**职业**: {basic_info.get('occupation', '')}")
                st.write(f"**电话**: {basic_info.get('phone', '')}")
            with info_cols[2]:
                st.write(f"**地址**: {basic_info.get('address', '')}")
                st.write(f"**状态**: {basic_info.get('status', '')}")

            if basic_info.get("remark"):
                st.info(f"**备注**: {basic_info.get('remark')}")

            # 行为模式
            behavior = profile.get("behavior_pattern", {}) or {}
            if behavior:
                st.subheader("行为模式")
                comm = behavior.get("communication", {}) or {}
                if comm:
                    st.write("**通讯规律**:")
                    st.write(
                        f"  通话时段分布: {comm.get('hour_distribution', {})}\n"
                        f"  频繁联系人: {comm.get('frequent_contacts', [])}\n"
                        f"  深夜通话比例: {comm.get('late_night_ratio', 0):.1%}"
                    )
                finance = behavior.get("finance", {}) or {}
                if finance:
                    st.write("**资金往来**:")
                    st.write(
                        f"  转入总额: {finance.get('total_inflow', 0)}\n"
                        f"  转出总额: {finance.get('total_outflow', 0)}\n"
                        f"  大额交易: {finance.get('large_transactions', 0)} 笔"
                    )
                trajectory = behavior.get("trajectory", {}) or {}
                if trajectory:
                    st.write("**活动轨迹**:")
                    st.write(
                        f"  频繁出入地点: {trajectory.get('frequent_locations', [])}\n"
                        f"  出行次数: {trajectory.get('travel_count', 0)}"
                    )

            # 风险评估
            st.subheader("风险评估")
            st.markdown(f"**嫌疑等级**: :{color}[{label}] (评分: {risk_score})")
            st.write(f"**评估依据**: {risk_reasoning}")
            if risk_factors:
                st.write("**风险因素**:")
                for factor in risk_factors:
                    st.write(f"  - {factor}")


def show_report(report: Dict[str, Any]):
    """展示侦查报告"""
    if not report:
        st.info("暂无报告")
        return

    markdown = report.get("markdown", "")
    if markdown:
        st.markdown(markdown)
    else:
        # 降级展示结构化数据
        st.subheader("案件概述")
        st.write(report.get("overview", ""))

        st.subheader("时间线")
        for event in report.get("timeline", []):
            st.write(
                f"- {event.get('time', '')} | {event.get('event', '')}"
            )

        st.subheader("关键线索")
        for clue in report.get("key_clues", []):
            st.write(f"- [{clue.get('score', 0)}分] {clue.get('description', '')}")

        st.subheader("嫌疑人画像摘要")
        for profile in report.get("suspect_profiles", []):
            basic = profile.get("basic_info", {}) or {}
            st.write(
                f"- {basic.get('name', profile.get('person_id', ''))} "
                f"({profile.get('risk_level', '')})"
            )

        st.subheader("侦查建议")
        for rec in report.get("recommendations", []):
            st.write(f"- {rec}")

    # 下载按钮
    if markdown:
        st.download_button(
            label="下载报告 (Markdown)",
            data=markdown.encode("utf-8"),
            file_name=f"{report.get('case_id', 'case')}_report.md",
            mime="text/markdown",
        )


def show_workflow_progress(state: Dict[str, Any]):
    """展示工作流进度"""
    phases = [
        ("init_case", "初始化案件"),
        ("intelligence_gathering", "情报收集"),
        ("clue_analysis", "线索分析"),
        ("suspect_profiling", "嫌疑人画像"),
        ("human_review", "人工审核"),
        ("report_generation", "报告生成"),
    ]

    current = state.get("current_phase", "")
    iteration = state.get("iteration", 0)

    st.write("### 侦查流程进度")
    if iteration > 0:
        st.caption(f"第 {iteration + 1} 轮迭代（情报回溯）")

    cols = st.columns(len(phases))
    for i, (phase_id, phase_label) in enumerate(phases):
        with cols[i]:
            if phase_id == current:
                st.markdown(f"**:{phase_label}**")
                st.success("当前")
            elif phases.index((phase_id, phase_label)) < [
                p[0] for p in phases
            ].index(current) if current in [p[0] for p in phases] else 0:
                st.write(phase_label)
                st.write("完成")
            else:
                st.write(phase_label)
                st.write("待执行")


# ==================== 主页面 ====================

def main():
    st.title("CaseAssistant 案件侦查助手")
    st.caption("多智能体协作案件侦查系统 - 演示原型")

    # 侧边栏
    with st.sidebar:
        st.header("案件选择")
        cases = loader.list_cases()
        if not cases:
            st.warning("未找到可用案件数据")
            case_id = st.text_input("手动输入案件ID")
        else:
            case_id = st.selectbox("选择案件", cases)

        st.divider()
        st.header("流程控制")

        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("启动侦查", type="primary")
        with col2:
            if st.session_state.get("workflow_state"):
                reset_button = st.button("重置")
            else:
                reset_button = False

        if reset_button:
            st.session_state.pop("workflow_state", None)
            st.session_state.pop("review_done", None)
            st.rerun()

    # 初始化 session_state
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "review_done" not in st.session_state:
        st.session_state.review_done = False

    # 启动侦查
    if run_button and case_id:
        with st.spinner("正在执行侦查流程（情报收集 -> 线索分析 -> 嫌疑人画像）..."):
            state = run_case_to_review(case_id)
            st.session_state.workflow_state = state
            st.session_state.review_done = False
        st.rerun()

    state = st.session_state.workflow_state

    if state is None:
        st.info("请在左侧选择案件并点击「启动侦查」开始")

        # 展示项目说明
        st.divider()
        st.header("系统说明")
        st.write("""
        CaseAssistant 是一个多智能体案件侦查辅助系统，包含 4 个协作 Agent：

        1. **情报收集 Agent** - 从多源数据中检索、整合结构化情报
        2. **线索分析 Agent** - 关联分析、时空碰撞、异常识别
        3. **嫌疑人画像 Agent** - 多维度画像与关系网络构建
        4. **报告生成 Agent** - 汇总产出，生成结构化侦查报告

        系统基于 LangGraph 状态机编排，支持人工审核与情报回溯。
        """)
        return

    # 展示进度
    show_workflow_progress(state)

    # 展示错误
    errors = state.get("errors", []) or []
    if errors:
        with st.expander(f"异常记录 ({len(errors)} 条)", expanded=False):
            for err in errors:
                st.error(err)

    # Tab 展示各 Agent 产出
    if not st.session_state.review_done:
        # 人工审核前的展示
        tab_intel, tab_clues, tab_profiles, tab_graph, tab_review = st.tabs([
            "情报卡片",
            "线索分析",
            "嫌疑人画像",
            "关系图谱",
            "人工审核",
        ])

        with tab_intel:
            show_intelligence_cards(state.get("intelligence_cards", []))

        with tab_clues:
            show_clues(state.get("clues", []))

        with tab_profiles:
            show_profiles(state.get("profiles", []))

        with tab_graph:
            # 展示全局关系图谱
            kg = get_knowledge_graph()
            graph_data = kg.get_graph_data()
            if graph_data.get("nodes"):
                st.subheader("全局关系图谱")
                html = render_graph_html(graph_data)
                st.components.v1.html(html, height=550)
            else:
                # 降级：展示首个画像的关系网络
                profiles = state.get("profiles", [])
                if profiles:
                    rn = profiles[0].get("relation_network", {})
                    if rn.get("nodes"):
                        st.subheader(f"{profiles[0].get('person_id', '')} 的关系网络")
                        html = render_graph_html(rn)
                        st.components.v1.html(html, height=550)
                    else:
                        st.info("暂无关系图谱数据")
                else:
                    st.info("暂无关系图谱数据")

            # 各嫌疑人关系图谱
            profiles = state.get("profiles", [])
            if profiles:
                st.divider()
                st.subheader("各嫌疑人关系网络")
                for profile in profiles:
                    rn = profile.get("relation_network", {})
                    if rn.get("nodes"):
                        person_id = profile.get("person_id", "")
                        basic = profile.get("basic_info", {}) or {}
                        name = basic.get("name", person_id)
                        st.write(f"**{name} ({person_id})**")
                        html = render_graph_html(rn, height=400)
                        st.components.v1.html(html, height=420)

        with tab_review:
            st.subheader("人工审核")
            st.write("""
            请审核以上情报卡片、线索和画像结果。
            如需补充情报，请在下方输入反馈意见并选择「需要补充情报」。
            如审核通过，直接点击「生成报告」。
            """)

            feedback = st.text_area(
                "审核反馈（可选）",
                placeholder="例如：需要重点调查张三与李四的资金往来...",
                height=100,
            )

            col1, col2 = st.columns(2)
            with col1:
                need_more = st.checkbox("需要补充情报", value=False)

            with col2:
                generate_button = st.button(
                    "生成报告",
                    type="primary",
                    key="generate_report",
                )

            if generate_button:
                with st.spinner("正在生成侦查报告..."):
                    final_state = resume_after_review(
                        state,
                        human_feedback=feedback if feedback.strip() else None,
                        needs_more_intel=need_more,
                    )
                    st.session_state.workflow_state = final_state
                    st.session_state.review_done = True
                st.rerun()

    else:
        # 报告展示
        tab_report, tab_intel, tab_clues, tab_profiles, tab_graph = st.tabs([
            "侦查报告",
            "情报卡片",
            "线索分析",
            "嫌疑人画像",
            "关系图谱",
        ])

        with tab_report:
            show_report(state.get("report", {}))

        with tab_intel:
            show_intelligence_cards(state.get("intelligence_cards", []))

        with tab_clues:
            show_clues(state.get("clues", []))

        with tab_profiles:
            show_profiles(state.get("profiles", []))

        with tab_graph:
            kg = get_knowledge_graph()
            graph_data = kg.get_graph_data()
            if graph_data.get("nodes"):
                html = render_graph_html(graph_data)
                st.components.v1.html(html, height=550)
            else:
                profiles = state.get("profiles", [])
                if profiles:
                    rn = profiles[0].get("relation_network", {})
                    if rn.get("nodes"):
                        html = render_graph_html(rn)
                        st.components.v1.html(html, height=550)
                st.info("暂无关系图谱数据")


if __name__ == "__main__":
    main()
