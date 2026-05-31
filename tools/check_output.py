from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_question_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("question file must be a list")
    return {str(row["question_id"]) for row in payload}


def validate(output_json: Path, questions_json: Path | None = None) -> dict[str, int]:
    payload = read_json(output_json)
    if not isinstance(payload, list):
        raise ValueError("output must be a list")

    expected_ids = load_question_ids(questions_json)
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("each output row must be an object")
        question_id = str(row.get("question_id", ""))
        answer = str(row.get("answer_choice", ""))
        if not question_id:
            raise ValueError("missing question_id")
        if question_id in seen:
            raise ValueError(f"duplicate question_id: {question_id}")
        if not answer or len(answer) != 1 or not answer.isalpha():
            raise ValueError(f"invalid answer_choice for {question_id}")
        seen.add(question_id)

    if expected_ids:
        missing = expected_ids - seen
        extra = seen - expected_ids
        if missing:
            raise ValueError(f"missing rows: {len(missing)}")
        if extra:
            raise ValueError(f"extra rows: {len(extra)}")

    return {"rows": len(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--questions-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate(args.output_json, args.questions_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
