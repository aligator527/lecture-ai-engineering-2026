"""第3回 宿題（実装）のテスト：make_quiz、学習ログのツール、承認（Human-in-the-Loop）の流れ。

LLM も API キーも使わない。保存先は pytest の一時フォルダに差し替え、本物の learning_log/ は汚さない。
"""

import json

import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approval


@pytest.fixture(autouse=True)
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", path)
    return path


def _inputs(*answers):
    """input() の代わり。決めておいた答えを順に返す。"""
    it = iter(answers)
    return lambda prompt="": next(it)


def _returns(result, tool_name):
    return [
        p.content for m in result.all_messages() for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == tool_name
    ]


# --- make_quiz ---

def test_make_quiz_returns_question_and_answer_of_the_lecture():
    quiz = json.loads(tools.make_quiz(2))
    lecture2 = {e["term"]: e["definition"] for e in tools.load_glossary() if e["lecture"] == 2}
    term = quiz["question"].removeprefix("「").removesuffix("」の意味は？")
    assert lecture2[term] == quiz["answer"]


def test_make_quiz_reports_lecture_without_terms():
    message = tools.make_quiz(0)
    assert "ありません" in message and "第2回" in message


# --- 学習ログのツール ---

def test_save_list_and_read_learning_log(log_dir):
    message = tools.save_learning_log("LoRAのまとめ", "低ランク行列だけを学習する。")
    assert "保存しました" in message

    logs = json.loads(tools.list_learning_logs())
    assert [log["title"] for log in logs] == ["LoRAのまとめ"]
    body = tools.read_learning_log(logs[0]["filename"])
    assert "低ランク行列だけを学習する。" in body


def test_save_learning_log_stays_inside_folder(log_dir):
    tools.save_learning_log("../../etc/passwd", "x")
    saved = list(log_dir.iterdir())
    assert len(saved) == 1 and saved[0].parent == log_dir
    assert ".." not in saved[0].name and "/" not in saved[0].name


def test_save_learning_log_rejects_empty(log_dir):
    assert "保存はしていません" in tools.save_learning_log(" ", "本文")
    assert not log_dir.exists()


def test_save_learning_log_does_not_overwrite(log_dir):
    tools.save_learning_log("同じ題名", "1回目")
    tools.save_learning_log("同じ題名", "2回目")
    bodies = sorted(p.read_text(encoding="utf-8") for p in log_dir.iterdir())
    assert len(bodies) == 2 and "1回目" in bodies[0] + bodies[1] and "2回目" in bodies[0] + bodies[1]


def test_read_learning_log_cannot_escape_folder(log_dir, tmp_path):
    (tmp_path / "secret.md").write_text("secret", encoding="utf-8")
    assert "見つかりませんでした" in tools.read_learning_log("../secret.md")


def test_list_learning_logs_when_empty():
    assert "まだありません" in tools.list_learning_logs()


# --- 承認の流れ（TestModel が save_learning_log を呼ぶ） ---

def _run(mode="ask", input_fn=None):
    model = TestModel(call_tools=["save_learning_log"])
    return run_with_approval(agent, "学習ログに保存して", model, approval_mode=mode, input_fn=input_fn)


def test_approved_save_writes_file(log_dir):
    result = _run(input_fn=_inputs("y"))
    assert "保存しました" in _returns(result, "save_learning_log")[0]
    assert len(list(log_dir.iterdir())) == 1
    assert isinstance(result.output, str)


def test_denied_save_writes_nothing(log_dir):
    result = _run(input_fn=_inputs("n"))
    assert "拒否" in _returns(result, "save_learning_log")[0]
    assert not log_dir.exists()


def test_edited_save_uses_user_arguments(log_dir):
    # e で編集 → title を書き換え、content は Enter で元のまま
    result = _run(input_fn=_inputs("e", "直した題名", ""))
    assert "保存しました" in _returns(result, "save_learning_log")[0]
    (saved,) = log_dir.iterdir()
    assert saved.read_text(encoding="utf-8").startswith("# 直した題名")


@pytest.mark.parametrize(("mode", "saved"), [("yes", 1), ("no", 0)])
def test_auto_approval_modes_do_not_ask(log_dir, mode, saved):
    def never_called(prompt=""):
        raise AssertionError("--yes / --no のときは利用者に聞かない")

    _run(mode=mode, input_fn=never_called)
    count = len(list(log_dir.iterdir())) if log_dir.exists() else 0
    assert count == saved


def test_usage_counts_requests_before_and_after_approval():
    result = _run(mode="yes")
    assert result.usage.requests >= 2
