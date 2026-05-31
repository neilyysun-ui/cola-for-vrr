# cola-for-vrr

Compact inference package for VRR video question answering. The repository keeps the runnable prediction pipeline and lightweight utilities, while local data, keys, generated predictions, and caches stay outside version control.

## Repository Structure

```text
.
├── run_final.py                  # End-to-end inference runner
├── requirements.txt              # Pip dependencies
├── environment.yml               # Conda environment
├── .env.example                  # Example runtime variables
├── configs/
│   └── run_full_inference.sh     # Shell recipe
├── docs/
│   ├── data_format.md            # Input and output schema
│   └── method_overview.md        # Short method description
├── examples/
│   └── questions.example.json
└── tools/
    └── check_output.py           # Output validator
```

## Setup

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate vrr-final
```

Set the Gemini key in the environment:

```bash
export GEMINI_API_KEY=your_key_here
```

You can also pass a private key file with `--gemini-key`.

## Run

```bash
python run_final.py \
  --questions-json data/test_qa.json \
  --video-dir data/videos \
  --output-json outputs/predictions.json
```

The same command can be launched through the shell recipe:

```bash
QUESTIONS_JSON=data/test_qa.json \
VIDEO_DIR=data/videos \
OUTPUT_JSON=outputs/predictions.json \
bash configs/run_full_inference.sh
```

Useful options:

```text
--model
--candidate-rounds
--verification-rounds
--min-agreement
--limit
```

## Validate

```bash
python tools/check_output.py \
  --output-json outputs/predictions.json \
  --questions-json data/test_qa.json
```

The runner writes a JSON list with `question_id` and `answer_choice`.
