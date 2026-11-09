"""第3回: Pydantic AI で作る学習支援エージェント。

使い方:
    uv run python -m app.agent "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
    uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"   # ツール呼び出しを表示
    uv run python -m app.agent --chat --trace                   # 対話モード（複数ターン、exit で終了）
    uv run python -m app.agent --yes "今日の学びを学習ログに保存して"  # 承認を自動で許可（--no で自動拒否）
    uv run python -m app.agent --mcp --trace "LoRAとは？"        # 第5回: ツールを MCP サーバーから受け取る
    uv run python -m app.agent --mcp --list-tools               # 第5回: モデルに見えるツール一覧（APIキー不要）

エージェント = モデル + ループ + ツール。
モデルは tools.py の関数を「ツール」として呼び出し、その結果を見て回答を作る。
第5回では、同じツールを MCP サーバー（app/mcp_server.py）経由で受け取る。
"""

import argparse
import inspect
import json
from pathlib import Path

from collections.abc import Callable

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.exceptions import ModelHTTPError
from fastmcp import Client
from pydantic_ai.mcp import MCPToolset, StdioTransport
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.google import GoogleProvider

from app import tools
from app.config import MODEL, api_error_message, get_api_key

INSTRUCTIONS = """\
あなたはAIエンジニアリング講座（AIE2026）の学習支援エージェントです。
- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。
- 回の内容を聞かれたら get_lecture_summary を使う。
- 確認問題やクイズを求められたら make_quiz を使い、問題だけを出して答えはすぐに明かさない。
- 学習の記録を頼まれたら、文章で確認を取らずに save_learning_log を呼ぶ（保存前の承認はシステムが利用者に確認する）。
- 過去の学習ログを聞かれたら list_learning_logs で一覧を見て、read_learning_log で中身を読む。
- 用語集に見つからないことは、推測で断定せず「用語集にはない」と伝える。
- 回答の最後に、関係する回（第N回）を示す。
- 宿題の答えそのものは書かず、考え方のヒントで導く。
"""

# モデルは実行時に渡す（テストではAPIを呼ばない TestModel に差し替えるため）
# 承認待ちのツールがあると、結果は str ではなく DeferredToolRequests になる
agent = Agent(instructions=INSTRUCTIONS, output_type=[str, DeferredToolRequests])
agent.tool_plain(tools.search_glossary)
agent.tool_plain(tools.get_lecture_summary)
agent.tool_plain(tools.make_quiz)
# ファイル保存は取り消しにくいので、実行前に利用者の承認を挟む（Human-in-the-Loop）
agent.tool_plain(tools.save_learning_log, requires_approval=True)
# 読むだけのツールは取り消す必要がないので、承認なしで使わせる
agent.tool_plain(tools.list_learning_logs)
agent.tool_plain(tools.read_learning_log)

# 第5回: 使う MCP サーバーの設定ファイル。Claude Desktop や Cursor と同じ "mcpServers" 形式
MCP_CONFIG = Path(__file__).resolve().parent / "mcp_servers.json"


def load_mcp_servers(config_path: Path = MCP_CONFIG) -> list[MCPToolset]:
    """設定ファイル（mcpServers 形式）を読み、サーバーごとに接続（MCPToolset）を作る。

    - "command" があれば stdio：サーバーをサブプロセスとして起動する
    - "url" があれば Streamable HTTP：起動済みのサーバーに接続する
    ツール名には「サーバー名_」を付ける（例: glossary_search_glossary）。複数のサーバーに
    同じ名前のツールがあっても区別できるようにするため。

    Pydantic AI には同じことをする load_mcp_toolsets() もある。ここで自分で組み立てているのは、
    古い SDK で作られたサーバー（fetch など）につなぐとき、接続方式の自動判定（"auto"）が
    警告を大量に出すため。"legacy" に固定すると新旧どちらのサーバーにも静かにつながる。
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))["mcpServers"]
    toolsets = []
    for name, spec in config.items():
        if "url" in spec:
            transport = spec["url"]
        else:
            transport = StdioTransport(spec["command"], spec.get("args", []), env=spec.get("env"))
        toolsets.append(MCPToolset(Client(transport, mode="legacy")).prefixed(name))
    return toolsets


def build_mcp_agent(config_path: Path = MCP_CONFIG) -> Agent:
    """第5回: 関数を直接登録する代わりに、MCP サーバーが公開するツールを使うエージェント。INSTRUCTIONS は同じ。"""
    return Agent(instructions=INSTRUCTIONS, toolsets=load_mcp_servers(config_path))


def build_model() -> GoogleModel:
    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key()))


def print_tools(target: Agent) -> None:
    """モデルに見えているツールの一覧（名前・説明文・引数）を表示する。LLM は呼ばない。

    TestModel（テストで使う偽モデル）にツール一覧を渡し、そこから読み出す。
    MCP サーバーのツールも、第3回の関数ツールも、モデルから見れば同じ形をしている。
    """
    model = TestModel(call_tools=[])
    target.run_sync("ツール一覧", model=model)
    tools_seen = model.last_model_request_parameters.function_tools
    print(f"モデルに見えているツール: {len(tools_seen)} 個")
    for tool in tools_seen:
        print(f"\n[{tool.name}]")
        print("  " + inspect.cleandoc(tool.description or "").replace("\n", "\n  "))
        params = tool.parameters_json_schema.get("properties", {})
        print(f"  引数: {json.dumps(params, ensure_ascii=False)}")


def print_trace(result, start: int = 0) -> None:
    """どのツールが、どの引数で呼ばれ、何を返したかを表示する。

    start: 対話モードで、このターンより前の履歴（all_messages()[:start]）を表示しないための位置。
    """
    print("\n--- trace ---")
    for message in result.all_messages()[start:]:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"[tool return] {part.tool_name} -> {str(part.content)[:120]}…")
    print(f"[usage] {result.usage}")


APPROVAL_MODES = ("ask", "yes", "no")


def _input_multiline(prompt: str, input_fn: Callable[[str], str]) -> str:
    """「.」だけの行が来るまで複数行を読む。"""
    print(prompt)
    lines = []
    while (line := input_fn("")) != ".":
        lines.append(line)
    return "\n".join(lines)


def _edit_args(args: dict, input_fn: Callable[[str], str]) -> dict:
    """引数を1つずつ編集する。何も入力しなければ元の値のまま。"""
    edited = dict(args)
    for key, value in args.items():
        if not isinstance(value, str):
            continue
        if "\n" in value:
            new = _input_multiline(f"  {key} の新しい値（「.」だけの行で終了、すぐ「.」で元のまま）:", input_fn)
        else:
            new = input_fn(f"  {key} の新しい値（Enter か「.」で元のまま）: ")
        if new.strip() and new.strip() != ".":
            edited[key] = new
    return edited


def ask_approvals(
    requests: DeferredToolRequests,
    mode: str = "ask",
    input_fn: Callable[[str], str] = input,
) -> DeferredToolResults:
    """承認待ちのツール呼び出しごとに、実行してよいかを決める。

    mode: "ask" は利用者に聞く、"yes" はすべて許可、"no" はすべて拒否（非対話・CI 用）。
    "ask" では y（許可）/ N（拒否）/ e（引数を編集してから許可）を選べる。
    """
    results = DeferredToolResults()
    for call in requests.approvals:
        args = call.args_as_dict()
        print(f"\n[approval] {call.tool_name} を次の引数で実行しようとしています:")
        for key, value in args.items():
            print(f"  {key}: {value}")
        if mode == "ask":
            try:
                answer = input_fn("実行してよいですか？ [y/N/e(編集)]: ").strip().lower()
            except EOFError:
                answer = ""
        else:
            answer = "y" if mode == "yes" else "n"
            print(f"[approval] --{mode} のため自動で{'許可' if mode == 'yes' else '拒否'}しました")

        if answer in ("y", "yes"):
            results.approvals[call.tool_call_id] = True
        elif answer in ("e", "edit"):
            try:
                edited = _edit_args(args, input_fn)
            except EOFError:
                results.approvals[call.tool_call_id] = ToolDenied("編集が中断されたため、実行しませんでした。")
                continue
            # モデルが作った引数を、利用者が直した引数に差し替えて実行する
            results.approvals[call.tool_call_id] = ToolApproved(override_args=edited)
        else:
            results.approvals[call.tool_call_id] = ToolDenied("利用者が実行を拒否しました。")
    return results


def run_with_approval(
    target: Agent,
    question: str,
    model,
    message_history=None,
    approval_mode: str = "ask",
    input_fn: Callable[[str], str] = input,
):
    """承認待ちで止まったら確認し、その結果と会話履歴を渡して再実行する。"""
    result = target.run_sync(question, model=model, message_history=message_history)
    while isinstance(result.output, DeferredToolRequests):
        result = target.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=ask_approvals(result.output, approval_mode, input_fn),
            model=model,
            usage=result.usage,  # 1回目の呼び出し回数も合算して [usage] に出す
        )
    return result


def chat(target: Agent, model, trace: bool, approval_mode: str) -> None:
    """対話モード。会話履歴を引き継ぎながら、exit が入力されるまで質問を受け付ける。"""
    print("対話モードです。終わるときは exit と入力してください。")
    history = None
    while True:
        try:
            question = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        start = len(history or [])
        try:
            result = run_with_approval(target, question, model, history, approval_mode)
        except ModelHTTPError as e:
            print(api_error_message(e.status_code, str(e.body)[:200]))
            continue  # 履歴はそのままにして、もう一度質問できるようにする
        print(f"\nエージェント> {result.output}")
        if trace:
            print_trace(result, start)
        history = result.all_messages()


def main() -> None:
    parser = argparse.ArgumentParser(description="学習支援エージェントに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--trace", action="store_true", help="ツール呼び出しを表示する")
    parser.add_argument("--chat", action="store_true", help="対話モード（会話を続けて質問できる）")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--yes", action="store_const", const="yes", dest="approval", help="承認が必要なツールをすべて自動で許可する")
    approval.add_argument("--no", action="store_const", const="no", dest="approval", help="承認が必要なツールをすべて自動で拒否する")
    parser.add_argument("--mcp", action="store_true", help="第5回: ツールを MCP サーバー（app/mcp_servers.json）から受け取る")
    parser.add_argument("--list-tools", action="store_true", help="モデルに見えるツール一覧を表示して終了する（APIキー不要）")
    args = parser.parse_args()
    approval_mode = args.approval or "ask"

    target = build_mcp_agent() if args.mcp else agent
    if args.list_tools:
        print_tools(target)
        return
    if not args.question and not args.chat:
        parser.error("質問を指定してください（または --chat で対話モード）")

    print(f"[model] {MODEL}")
    if args.mcp:
        print("[tools] MCP サーバー（app/mcp_servers.json）")
    model = build_model()
    if args.chat:
        chat(target, model, args.trace, approval_mode)
        return
    try:
        result = run_with_approval(target, args.question, model, approval_mode=approval_mode)
    except ModelHTTPError as e:
        print(api_error_message(e.status_code, str(e.body)[:200]))
        raise SystemExit(1)
    print(result.output)
    if args.trace:
        print_trace(result)


if __name__ == "__main__":
    main()
