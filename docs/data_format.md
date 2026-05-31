# Data Format

## Questions

`--questions-json` should contain a JSON list. Each item uses:

```json
{
  "question_id": "example_0001",
  "video_id": "example_0001",
  "question_text": "What happens first?",
  "options": {
    "A": "The person opens the door.",
    "B": "The person picks up the box.",
    "C": "The person turns off the light."
  }
}
```

`video_id` is optional. Videos are resolved from `--video-dir` by `question_id` first and `video_id` second.

## Videos

Supported video suffixes:

```text
.mp4 .webm .mov .mkv .avi
```

## Output

The output JSON is a list:

```json
[
  {
    "question_id": "example_0001",
    "answer_choice": "A"
  }
]
```
