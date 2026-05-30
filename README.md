# cola-for-vrr

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run_final.py \
  --questions-json test_qa.json \
  --video-dir videos \
  --gemini-key gemini_key.txt \
  --output-json predictions.json \
  --candidate-rounds 3 \
  --verification-rounds 3 \
  --min-agreement 2
```

Input JSON is a list of questions with `question_id`, `question_text`, and `options`.
Videos are resolved from `video-dir` by `question_id` first and `video_id` second.
The flow is direct answer, hard-case routing, candidate bank construction, repeated candidate verification, risk gate, and final JSON.
For routed hard cases, a candidate replaces the direct answer only when it reaches the agreement threshold and passes the risk gate.
The script writes only the JSON specified by `--output-json`.
