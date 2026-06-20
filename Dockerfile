# Linux x86-64 image so the competition engine (libcg.so) loads.
# On Apple Silicon (M-series), build & run with --platform linux/amd64
# (Docker Desktop emulates x86-64 via Rosetta 2 — works, a bit slower).
FROM --platform=linux/amd64 python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Before building, place the Kaggle engine at ./cg and the generated tables
# (capability_table.json, attack_table.json) in the project root — they're
# git-ignored, so mount or copy them in. Default: run a few self-play games.
CMD ["python", "run_game.py", "--quiet", "--games", "5", "--log", "logs/eval.jsonl"]
