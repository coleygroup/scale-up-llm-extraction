#!/bin/bash
#SBATCH -J 120b_refine
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=192G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --mail-user=xiaoqis@mit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --chdir=/home/xiaoqis/projects/vllm_deploy
#SBATCH --requeue
#SBATCH --open-mode=append

mkdir -p logs

set -euo pipefail

module load gcc/12.2.0 apptainer/1.1.9
source .venv/bin/activate

# params
PORT=8013
MODEL=openai/gpt-oss-120b
export PORT MODEL

export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"

# 1)start vllm server
echo "Starting vLLM server..." >&2
apptainer exec --nv --cleanenv \
  --bind /home/xiaoqis/orcd/pool/hf:/hf \
  --env HF_HOME=/hf \
  containers/vllm.sif \
  python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --tensor-parallel-size 2 \
  --port "${PORT}" \
  > logs/vllm_server_120b.log 2>&1 &
SERVER_PID=$!

###### option 2 wait until ready ######
cleanup() {
  echo "Stopping server..."
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo -n "Waiting for server to be ready"
for i in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo " — ready."
    break
  fi
  echo -n "."
  sleep 2
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo; echo "Server died early; tail follows:"; tail -n 100 logs/vllm_server_120b.log; exit 1
  fi
  if [ $i -eq 600 ]; then
    echo; echo "Timed out waiting for server."; exit 1
  fi
done

# 3)run evaluation script
echo "Running evaluation script..."
# python figure4_new/llm_evaluation_helper_v3.py \
#   --input results/gptoss120b_v2/results.jsonl \
#   --model "${MODEL}" \
#   --max-workers 20

python figure4_new/llm_evaluation_helper_v3.py \
  --input results/gptoss120b_v2/results.jsonl \
  --exclude-jsonl scaleup_all_120b_results_rerun.jsonl \
  --output-prefix scaleup_condition_changes_20251219_003700_rerun_v4 \
  --model "${MODEL}" \
  --max-workers 5