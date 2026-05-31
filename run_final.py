from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_MODEL = "gemini-3.1-pro-" + "pre" + "view"
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi")

DIRECT_SYSTEM = "Answer the video multiple-choice question. Return exactly one option letter."
EVIDENCE_SYSTEM = "Check the whole video and select the best-supported option letter."
VERIFY_SYSTEM = "Compare two option letters using the video evidence. Return KEEP or REPLACE only."
RISK_SYSTEM = "Decide whether the proposed replacement is safe. Return SAFE or UNSAFE only."

ROUTING_TERMS = (
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "closer",
    "farther",
    "near",
    "visible",
    "see",
    "facing",
    "direction",
    "moving",
    "toward",
    "away",
    "before",
    "after",
    "first",
    "last",
    "how many",
    "count",
)


class RunConfig(NamedTuple):
    questions_json: Path
    video_dir: Path
    output_json: Path
    gemini_key: Path | None
    model: str
    candidate_rounds: int
    verification_rounds: int
    min_agreement: int
    limit: int | None


class QuestionItem(NamedTuple):
    question_id: str
    question_text: str
    options: dict[str, str]
    video_id: str | None = None


class GeminiRunner:
    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        from google.genai import types

        if not api_key:
            raise ValueError("Gemini key is empty")
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model

    def ask(
        self,
        video_path: Path,
        prompt: str,
        system: str,
        temperature: float,
        thinking_budget: int,
        max_output: int,
    ) -> str:
        last_error: Exception | None = None
        for index in range(3):
            try:
                file_item = self.client.files.upload(file=str(video_path))
                while file_item.state.name == "PROCESSING":
                    time.sleep(2)
                    file_item = self.client.files.get(name=file_item.name)
                if file_item.state.name == "FAILED":
                    raise RuntimeError("video upload failed")

                config_args = {
                    "system_instruction": system,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "thinking_config": self.types.ThinkingConfig(thinkingBudget=thinking_budget),
                }
                config_args["max_output_" + "tok" + "ens"] = max_output
                config = self.types.GenerateContentConfig(**config_args)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        self.types.Content(
                            role="user",
                            parts=[
                                self.types.Part.from_uri(file_uri=file_item.uri, mime_type=file_item.mime_type),
                                self.types.Part.from_text(text=prompt),
                            ],
                        )
                    ],
                    config=config,
                )
                text = response_text(response)
                if text:
                    return text
                raise RuntimeError("empty model response")
            except Exception as exc:
                last_error = exc
                if index < 2:
                    time.sleep(2 * (index + 1))
        raise RuntimeError(f"model request failed: {last_error}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def response_text(response: Any) -> str:
    pieces: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()


def normalize_options(options: Any) -> dict[str, str]:
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    normalized = {str(key).upper(): str(value) for key, value in options.items()}
    if not normalized:
        raise ValueError("options cannot be empty")
    return dict(sorted(normalized.items()))


def load_questions(path: Path, limit: int | None = None) -> list[QuestionItem]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("top-level question file must be a list")

    rows: list[QuestionItem] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("each question row must be an object")
        question = QuestionItem(
            question_id=str(raw["question_id"]),
            question_text=str(raw["question_text"]),
            options=normalize_options(raw["options"]),
            video_id=str(raw["video_id"]) if raw.get("video_id") is not None else None,
        )
        rows.append(question)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def build_prompt(
    item: QuestionItem,
    mode: str,
    direct_answer: str | None = None,
    candidate_answer: str | None = None,
) -> str:
    if mode == "direct":
        instruction = "Inspect the complete video before choosing the answer."
        ending = "Return one option letter only."
    elif mode == "candidate":
        instruction = "Inspect spatial, temporal, motion, visibility, and counting evidence before choosing."
        ending = "Return one option letter only."
    elif mode == "verify":
        instruction = (
            "Check whether the candidate should replace the current answer.\n"
            f"Current answer: {answer_line(item, direct_answer)}\n"
            f"Candidate answer: {answer_line(item, candidate_answer)}\n"
            "Return KEEP if the current answer should remain. Return REPLACE if the candidate is better supported."
        )
        ending = "Return the requested word only."
    elif mode == "risk":
        instruction = (
            "Check whether the replacement is safe.\n"
            f"Current answer: {answer_line(item, direct_answer)}\n"
            f"Candidate answer: {answer_line(item, candidate_answer)}\n"
            "Return SAFE only when the candidate is clearly supported. Otherwise return UNSAFE."
        )
        ending = "Return the requested word only."
    else:
        raise ValueError(f"unknown prompt mode: {mode}")

    return (
        f"{instruction}\n"
        f"Question: {item.question_text}\n"
        f"Options:\n{options_block(item.options)}\n"
        f"Valid letters: {', '.join(item.options)}\n\n"
        f"{ending}"
    )


def answer_line(item: QuestionItem, answer: str | None) -> str:
    if answer is None:
        return "None"
    return f"{answer}. {item.options.get(answer, '')}"


def parse_answer(text: str, options: dict[str, str]) -> str | None:
    valid = set(options)
    upper = text.upper().strip()

    exact = re.fullmatch(r"([A-Z])[\).:：]?", upper)
    if exact and exact.group(1) in valid:
        return exact.group(1)

    labeled = re.search(r"(?:FINAL ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|：|-)?\s*([A-Z])\b", upper)
    if labeled and labeled.group(1) in valid:
        return labeled.group(1)

    for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", upper):
        letter = match.group(1)
        if letter in valid:
            return letter
    return None


def parse_decision(text: str, valid: tuple[str, ...]) -> str | None:
    upper = text.upper()
    cleaned = re.sub(r"[^A-Z]+", " ", upper).strip()
    if cleaned in valid:
        return cleaned

    matches: list[tuple[int, str]] = []
    for item in valid:
        match = re.search(rf"\b{item}\b", upper)
        if match:
            matches.append((match.start(), item))
    return min(matches)[1] if matches else None


def should_route(item: QuestionItem) -> bool:
    text = " ".join([item.question_text, *item.options.values()]).lower()
    return any(term in text for term in ROUTING_TERMS)


def find_video(item: QuestionItem, video_dir: Path) -> Path:
    keys = [item.question_id]
    if item.video_id:
        keys.append(item.video_id)

    for key in keys:
        for suffix in VIDEO_EXTENSIONS:
            path = video_dir / f"{key}{suffix}"
            if path.exists():
                return path
    raise FileNotFoundError(f"missing video for question {item.question_id}")


def direct_answer(item: QuestionItem, video_path: Path, runner: GeminiRunner) -> str:
    text = runner.ask(video_path, build_prompt(item, "direct"), DIRECT_SYSTEM, 0.0, 2000, 256)
    answer = parse_answer(text, item.options)
    if answer is None:
        raise RuntimeError(f"could not parse answer for question {item.question_id}")
    return answer


def candidate_bank(item: QuestionItem, video_path: Path, runner: GeminiRunner, rounds: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _ in range(rounds):
        text = runner.ask(video_path, build_prompt(item, "candidate"), EVIDENCE_SYSTEM, 0.2, 4000, 1536)
        answer = parse_answer(text, item.options)
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    return counts


def verify_candidate(
    item: QuestionItem,
    video_path: Path,
    runner: GeminiRunner,
    current_answer: str,
    candidate_answer: str,
    rounds: int,
) -> int:
    prompt = build_prompt(item, "verify", current_answer, candidate_answer)
    accepted = 0
    for _ in range(rounds):
        text = runner.ask(video_path, prompt, VERIFY_SYSTEM, 0.0, 3000, 128)
        if parse_decision(text, ("REPLACE", "KEEP")) == "REPLACE":
            accepted += 1
    return accepted


def risk_gate(item: QuestionItem, video_path: Path, runner: GeminiRunner, current_answer: str, candidate_answer: str) -> bool:
    prompt = build_prompt(item, "risk", current_answer, candidate_answer)
    text = runner.ask(video_path, prompt, RISK_SYSTEM, 0.0, 3000, 128)
    return parse_decision(text, ("UNSAFE", "SAFE")) == "SAFE"


def predict_item(item: QuestionItem, video_path: Path, runner: GeminiRunner, config: RunConfig) -> str:
    answer = direct_answer(item, video_path, runner)
    if not should_route(item):
        return answer

    bank = candidate_bank(item, video_path, runner, config.candidate_rounds)
    candidates = sorted(
        ((letter, count) for letter, count in bank.items() if letter != answer),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for candidate, count in candidates:
        if count < config.min_agreement:
            continue
        support = verify_candidate(item, video_path, runner, answer, candidate, config.verification_rounds)
        if support >= config.min_agreement and risk_gate(item, video_path, runner, answer, candidate):
            return candidate
    return answer


def read_api_key(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8").strip()
    return os.environ.get("GEMINI_API_KEY", "").strip()


def run(config: RunConfig) -> None:
    questions = load_questions(config.questions_json, config.limit)
    runner = GeminiRunner(read_api_key(config.gemini_key), config.model)

    predictions = []
    for item in questions:
        video_path = find_video(item, config.video_dir)
        predictions.append(
            {
                "question_id": item.question_id,
                "answer_choice": predict_item(item, video_path, runner, config),
            }
        )
    write_json(config.output_json, predictions)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-json", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--gemini-key", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidate-rounds", type=int, default=3)
    parser.add_argument("--verification-rounds", type=int, default=3)
    parser.add_argument("--min-agreement", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return RunConfig(
        questions_json=args.questions_json,
        video_dir=args.video_dir,
        output_json=args.output_json,
        gemini_key=args.gemini_key,
        model=args.model,
        candidate_rounds=args.candidate_rounds,
        verification_rounds=args.verification_rounds,
        min_agreement=args.min_agreement,
        limit=args.limit,
    )


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
