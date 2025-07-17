# Setup

On a GPU node, install the dependencies:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm gradio requests
```

# Open House Demo Workflow

1. Launch vLLM server (GPU node)

```shell
sbatch run_vllm_server.sh
```

3. Launch the submission dashboard (CPU node). Convert the Gradio public URL into QR code and paste it into the slides.

```shell
python dashboard.py 0 submission
```

4. Launch eval job (CPU node)

```shell
python evaluate_pairs.py --question_id 0 --vllm-url VLLM_SERVER_ADDRESS
```

5. Launch leaderboard dashboard (CPU node)

```shell
python dashboard.py 0 leaderboard
```
