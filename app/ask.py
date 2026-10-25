"""第1回: 素のLLMに質問するスクリプト（宿題「実装」: 会話履歴を保って続けて質問できる）。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話モード。履歴を保ったまま続けて質問できる
    uv run python -m app.ask --chat "LoRAとは何ですか？"   # 最初の質問を指定して会話を始める

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

会話モードでは、毎回それまでの履歴をすべて送り直す。入力トークン数がターンごとに
増えていくことを、表示されるトークン数で確かめよう。
"""

import argparse
import time

from google import genai
from google.genai import chats, errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

PREVIEW_CHARS = 30
HELP_TEXT = """\
コマンド:
  /help          このヘルプを表示
  /history       履歴の一覧を表示（各ターンを短く）
  /history ID    指定したターンの質問と回答を全文表示（例: /history 2）
  /reset         履歴を消去
  /exit          終了（/quit や Ctrl+D でも終了できます）"""


class Usage:
    """会話全体のトークン数と応答時間を積算する。"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0

    def record(self, response, seconds: float) -> None:
        meta = response.usage_metadata
        input_tokens = (meta.prompt_token_count or 0) if meta else 0
        output_tokens = (meta.candidates_token_count or 0) if meta else 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.seconds += seconds
        print(
            f"[トークン] 入力 {input_tokens} / 出力 {output_tokens}"
            f"（累計 入力 {self.input_tokens} / 出力 {self.output_tokens}）"
            f" [応答時間] {seconds:.2f}秒"
        )


def ask(question: str) -> str:
    """履歴なしで1回だけ質問する。"""
    client = genai.Client(api_key=get_api_key())
    start = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    Usage().record(response, time.perf_counter() - start)
    return response.text


def send(chat: chats.Chat, question: str, usage: Usage) -> str:
    """会話の続きとして質問する。履歴は chat が保持し、毎回まとめて送る。"""
    start = time.perf_counter()
    response = chat.send_message(question)
    usage.record(response, time.perf_counter() - start)
    return response.text


def content_text(content) -> str:
    return "".join(part.text or "" for part in content.parts or [])


def preview(text: str) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= PREVIEW_CHARS else one_line[:PREVIEW_CHARS] + "…"


def turns(chat: chats.Chat) -> list[tuple[str, str]]:
    """履歴を（質問, 回答）の組のリストにする。ID はこのリストの1始まりの番号。"""
    history = chat.get_history()
    return [(content_text(q), content_text(a)) for q, a in zip(history[0::2], history[1::2])]


def print_history(chat: chats.Chat, turn_id: str | None) -> None:
    pairs = turns(chat)
    if not pairs:
        print("（履歴はまだありません）")
        return
    if turn_id is None:
        for i, (question, answer) in enumerate(pairs, start=1):
            print(f"[{i}] あなた: {preview(question)} / モデル: {preview(answer)}")
        print("全文を見るには /history ID")
        return
    if not turn_id.isdigit() or not 1 <= int(turn_id) <= len(pairs):
        print(f"ID は 1〜{len(pairs)} の数字で指定してください。")
        return
    question, answer = pairs[int(turn_id) - 1]
    print(f"[{turn_id}] あなた:\n{question}\n\nモデル:\n{answer}")


def chat_loop(first_question: str | None) -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    usage = Usage()
    print("会話を始めます。/help でコマンドの一覧を表示します。\n")

    question = first_question
    while True:
        if question is None:
            try:
                question = input("質問> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        else:
            print(f"質問> {question}")

        if question.startswith("/"):
            command, _, arg = question.partition(" ")
            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                print(HELP_TEXT)
            elif command == "/history":
                print_history(chat, arg.strip() or None)
            elif command == "/reset":
                chat = client.chats.create(model=MODEL)
                print("履歴を消去しました。")
            else:
                print(f"不明なコマンドです: {command}（/help で一覧を表示）")
        elif question.lower() in ("exit", "quit"):
            break
        elif question:
            try:
                print(f"\n{send(chat, question, usage)}\n")
            except errors.APIError as e:
                # 失敗したターンは履歴に残らないので、待ってから同じ質問をし直せる
                print(api_error_message(e.code, e.message))
        question = None

    print(
        f"終了します。累計トークン 入力 {usage.input_tokens} / 出力 {usage.output_tokens}、"
        f"応答時間の合計 {usage.seconds:.2f}秒"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat_loop(args.question)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
