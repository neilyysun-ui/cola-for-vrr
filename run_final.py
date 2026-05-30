from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


MODEL = "gemini-3.1-pro-" + "pre" + "view"
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi")
DIRECT_SYSTEM = "Answer video multiple-choice questions. Return exactly one option letter."
CANDIDATE_SYSTEM = "Generate one candidate answer after checking spatial, temporal, depth, motion, visibility, and counting evidence. Return exactly one option letter."
VERIFY_SYSTEM = "Compare the direct answer and candidate answer against the video. Return KEEP or REPLACE only."
RISK_SYSTEM = "Decide whether replacing the direct answer is safe. Return SAFE or UNSAFE only."
HARD_TERMS = (
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


class GeminiRunner:
    def __init__(self, key_path: Path) -> None:
        from google import genai
        from google.genai import types

        api_key = key_path.read_text(encoding="utf-8").strip()
        if not api_key:
            raise ValueError("empty Gemini key file")
        self.client = genai.Client(api_key=api_key)
        self.types = types

    def ask(self, video_path: Path, prompt: str, system: str, temperature: float, thinking_budget: int, max_output: int) -> str:
        last_error: Exception | None = None
        for i in range(3):
            try:
                item = self.client.files.upload(file=str(video_path))
                while item.state.name == "PROCESSING":
                    time.sleep(2)
                    item = self.client.files.get(name=item.name)
                if item.state.name == "FAILED":
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
                    model=MODEL,
                    contents=[
                        self.types.Content(
                            role="user",
                            parts=[
                                self.types.Part.from_uri(file_uri=item.uri, mime_type=item.mime_type),
                                self.types.Part.from_text(text=prompt),
                            ],
                        )
                    ],
                    config=config,
                )
                text = response_text(response)
                if not text:
                    raise RuntimeError("empty model response")
                return text
            except Exception as exc:
                last_error = exc
                if i < 2:
                    time.sleep(2 * (i + 1))
        raise RuntimeError(f"model request failed: {last_error}")


def response_text(response: Any) -> str:
    pieces: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                pieces.append(text)
    return "\n".join(pieces).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def options_text(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {options[letter]}" for letter in sorted(options))


def build_prompt(row: dict[str, Any], mode: str, direct_answer: str | None = None, candidate_answer: str | None = None) -> str:
    options = row["options"]
    if mode == "direct":
        instruction = "Inspect the whole clip before choosing."
    elif mode == "candidate":
        instruction = "Inspect the whole clip carefully and propose the best candidate answer."
    elif mode == "verify":
        instruction = (
            "Check whether the candidate answer should replace the direct answer.\n"
            f"Direct answer: {direct_answer}. {options[direct_answer or '']}\n"
            f"Candidate answer: {candidate_answer}. {options[candidate_answer or '']}\n"
            "Return KEEP if the direct answer should remain. Return REPLACE if the candidate is better supported."
        )
    elif mode == "risk":
        instruction = (
            "Check whether the replacement is safe.\n"
            f"Direct answer: {direct_answer}. {options[direct_answer or '']}\n"
            f"Candidate answer: {candidate_answer}. {options[candidate_answer or '']}\n"
            "Return SAFE only if the candidate is clearly supported and the reference object, time segment, and visual relation are not ambiguous. Otherwise return UNSAFE."
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    ending = "Return one option letter only." if mode in ("direct", "candidate") else "Return the requested word only."
    return (
        f"{instruction}\n"
        f"Question: {row['question_text']}\n"
        f"Options:\n{options_text(options)}\n"
        f"Valid letters: {', '.join(sorted(options))}\n\n"
        f"{ending}"
    )


def parse_answer(text: str, options: dict[str, str]) -> str | None:
    valid = set(options)
    upper = text.upper().strip()
    exact = re.fullmatch(r"([A-Z])[\).:：]?", upper)
    if exact and exact.group(1) in valid:
        return exact.group(1)
    match = re.search(r"(?:FINAL ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|：|-)?\s*([A-Z])\b", upper)
    if match and match.group(1) in valid:
        return match.group(1)
    for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", upper):
        if match.group(1) in valid:
            return match.group(1)
    return None


def parse_gate(text: str, valid: tuple[str, ...]) -> str | None:
    upper = text.upper()
    cleaned = re.sub(r"[^A-Z]+", " ", upper).strip()
    if cleaned in valid:
        return cleaned
    matches: list[tuple[int, str]] = []
    for item in valid:
        match = re.search(rf"\b{item}\b", upper)
        if match:
            matches.append((match.start(), item))
    if matches:
        return min(matches)[1]
    return None


def is_hard(row: dict[str, Any]) -> bool:
    text = " ".join([row["question_text"], *row["options"].values()]).lower()
    return any(term in text for term in HARD_TERMS)


def find_video(row: dict[str, Any], video_dir: Path) -> Path:
    question_id = str(row["question_id"])
    for suffix in VIDEO_EXTENSIONS:
        path = video_dir / f"{question_id}{suffix}"
        if path.exists():
            return path
    video_id = str(row.get("video_id", ""))
    for suffix in VIDEO_EXTENSIONS:
        path = video_dir / f"{video_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing video for question {question_id}")


def build_candidate_bank(row: dict[str, Any], video_path: Path, runner: GeminiRunner, rounds: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _ in range(rounds):
        text = runner.ask(video_path, build_prompt(row, "candidate"), CANDIDATE_SYSTEM, 0.2, 4000, 1536)
        answer = parse_answer(text, row["options"])
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    return counts


def verify_candidate(row: dict[str, Any], video_path: Path, runner: GeminiRunner, direct_answer: str, candidate_answer: str, rounds: int) -> int:
    accepted = 0
    prompt = build_prompt(row, "verify", direct_answer, candidate_answer)
    for _ in range(rounds):
        text = runner.ask(video_path, prompt, VERIFY_SYSTEM, 0.0, 3000, 128)
        decision = parse_gate(text, ("REPLACE", "KEEP"))
        if decision == "REPLACE":
            accepted += 1
    return accepted


def risk_gate(row: dict[str, Any], video_path: Path, runner: GeminiRunner, direct_answer: str, candidate_answer: str) -> bool:
    text = runner.ask(video_path, build_prompt(row, "risk", direct_answer, candidate_answer), RISK_SYSTEM, 0.0, 3000, 128)
    return parse_gate(text, ("UNSAFE", "SAFE")) == "SAFE"


def predict(row: dict[str, Any], video_path: Path, runner: GeminiRunner, candidate_rounds: int, verification_rounds: int, min_agreement: int) -> str:
    direct = runner.ask(video_path, build_prompt(row, "direct"), DIRECT_SYSTEM, 0.0, 2000, 256)
    answer = parse_answer(direct, row["options"])
    if answer is None:
        raise RuntimeError(f"could not parse answer for question {row['question_id']}")
    if is_hard(row):
        bank = build_candidate_bank(row, video_path, runner, candidate_rounds)
        candidates = sorted((item for item in bank.items() if item[0] != answer), key=lambda item: item[1], reverse=True)
        for candidate, count in candidates:
            if count < min_agreement:
                continue
            accepted = verify_candidate(row, video_path, runner, answer, candidate, verification_rounds)
            if accepted >= min_agreement and risk_gate(row, video_path, runner, answer, candidate):
                answer = candidate
                break
    return answer


def run(questions_json: Path, video_dir: Path, key_path: Path, output_json: Path, candidate_rounds: int, verification_rounds: int, min_agreement: int) -> None:
    rows = read_json(questions_json)
    runner = GeminiRunner(key_path)
    predictions = []
    for row in rows:
        video_path = find_video(row, video_dir)
        predictions.append({"question_id": row["question_id"], "answer_choice": predict(row, video_path, runner, candidate_rounds, verification_rounds, min_agreement)})
    write_json(output_json, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-json", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--gemini-key", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--candidate-rounds", type=int, default=3)
    parser.add_argument("--verification-rounds", type=int, default=3)
    parser.add_argument("--min-agreement", type=int, default=2)
    args = parser.parse_args()
    run(args.questions_json, args.video_dir, args.gemini_key, args.output_json, args.candidate_rounds, args.verification_rounds, args.min_agreement)


if __name__ == "__main__":
    main()
