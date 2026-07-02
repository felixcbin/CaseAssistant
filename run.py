#!/usr/bin/env python3
"""CaseAssistant 启动脚本"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_streamlit():
    """启动 Streamlit UI"""
    os.system(
        "streamlit run case_assistant/ui/app.py --server.port 8501 --server.headless true"
    )


def run_cli():
    """命令行模式：直接运行案件侦查并输出报告"""
    from case_assistant.orchestration.workflow import run_case

    # 解析参数：python run.py --cli [CASE_ID]
    case_id = sys.argv[2] if len(sys.argv) > 2 else "CASE-2026-001"
    print("正在运行案件侦查: {}".format(case_id))

    state = run_case(case_id)

    print("\n" + "=" * 60)
    print("侦查完成")
    print("=" * 60)

    report = state.get("report", {})
    if report:
        markdown = report.get("markdown", "")
        if markdown:
            # 输出到文件
            output_file = "{}_report.md".format(case_id)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown)
            print("报告已保存至: {}".format(output_file))
        else:
            print("报告生成失败，请检查 errors 字段")
    else:
        print("未生成报告")

    # 输出摘要
    cards = state.get("intelligence_cards", [])
    clues = state.get("clues", [])
    profiles = state.get("profiles", [])
    errors = state.get("errors", [])

    print("\n摘要:")
    print("  情报卡片: {} 张".format(len(cards)))
    print("  线索: {} 条".format(len(clues)))
    print("  嫌疑人画像: {} 份".format(len(profiles)))
    if errors:
        print("  异常: {} 条".format(len(errors)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_streamlit()
