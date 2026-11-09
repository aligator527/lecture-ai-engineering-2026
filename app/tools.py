"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
import re
from datetime import datetime
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語集から確認問題を1問作り、問題と答えを返す。

    「第2回の確認問題を出して」「第3回の復習クイズがほしい」のように、
    確認問題・クイズ・復習問題を求められたときに使う。
    返り値の answer は採点用なので、出題時には question だけを見せ、答えはすぐに明かさないこと。
    """
    candidates = [e for e in load_glossary() if e["lecture"] == lecture]
    if not candidates:
        available = sorted({e["lecture"] for e in load_glossary()})
        return (
            f"第{lecture}回の用語は用語集にありません。"
            f"問題を作れる回: {', '.join(f'第{n}回' for n in available)}"
        )
    entry = random.choice(candidates)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]},
        ensure_ascii=False,
    )


def save_learning_log(title: str, content: str) -> str:
    """学習の記録を learning_log/ フォルダに Markdown ファイルとして保存する。

    利用者が「今日学んだことを記録して」「学習ログに保存して」のように、
    学習内容の保存を明示的に頼んだときだけ使う。
    title には記録の短い題名（例: "LoRAとQLoRAの違い"）、content には保存する本文を入れる。
    """
    if not title.strip() or not content.strip():
        return "title と content の両方を指定してください。保存はしていません。"
    now = datetime.now()
    # ファイル名に使えない文字や、フォルダの外を指す「/」「..」を取り除く
    safe_title = re.sub(r'[\\/:*?"<>|\s.]+', "_", title.strip()).strip("_")[:50] or "log"
    LEARNING_LOG_DIR.mkdir(exist_ok=True)
    path = LEARNING_LOG_DIR / f"{now:%Y%m%d-%H%M%S}-{safe_title}.md"
    n = 2
    while path.exists():  # 同じ秒に同じ題名で保存しても、前の記録を上書きしない
        path = LEARNING_LOG_DIR / f"{now:%Y%m%d-%H%M%S}-{safe_title}-{n}.md"
        n += 1
    path.write_text(
        f"# {title.strip()}\n\n- 日時: {now:%Y-%m-%d %H:%M}\n\n{content.strip()}\n",
        encoding="utf-8",
    )
    return f"学習ログを保存しました: {path.relative_to(LEARNING_LOG_DIR.parent)}"


def list_learning_logs() -> str:
    """learning_log/ に保存済みの学習ログの一覧（ファイル名と題名）を、新しい順に返す。

    「これまでの学習ログを見せて」「前に何を記録した？」のように、
    過去の記録を振り返りたいと言われたときに使う。中身は read_learning_log で読む。
    """
    files = sorted(LEARNING_LOG_DIR.glob("*.md"), reverse=True) if LEARNING_LOG_DIR.is_dir() else []
    if not files:
        return "保存済みの学習ログはまだありません。"
    logs = []
    for path in files:
        first_line = path.read_text(encoding="utf-8").splitlines()[:1]
        title = first_line[0].lstrip("# ").strip() if first_line else path.stem
        logs.append({"filename": path.name, "title": title})
    return json.dumps(logs, ensure_ascii=False)


def read_learning_log(filename: str) -> str:
    """保存済みの学習ログを1つ読み、本文を返す。

    filename には list_learning_logs が返したファイル名をそのまま入れる
    （例: "20261109-051836-LoRAとQLoRAの違い.md"）。
    """
    path = LEARNING_LOG_DIR / Path(filename).name  # フォルダの外は読ませない
    if path.suffix != ".md" or not path.is_file():
        return f"学習ログ「{filename}」は見つかりませんでした。list_learning_logs でファイル名を確認してください。"
    return path.read_text(encoding="utf-8")
