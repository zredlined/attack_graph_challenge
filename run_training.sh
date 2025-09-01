#!/bin/bash
# Run script for GRPO Attack Graph Challenge training

# Setup environment
echo "Setting up environment..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
if ! python -c "import transformers" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Optional: Generate dataset (if you want to create a new one)
# Uncomment the lines below to generate a fresh dataset
# echo "Generating attack graph dataset..."
# python scripts/dataset_generation/generate_attack_dataset.py \
#   --difficulty simple \
#   --num_samples 50000 \
#   --upload_to_hf \
#   --hf_username meowterspace45

# Default to 8 GPUs setup (7 for training, 1 for vLLM) - adjust as needed
NUM_GPUS=${1:-8}
NUM_PROCESSES=$((NUM_GPUS - 1))

echo "Using $NUM_GPUS GPUs: $NUM_PROCESSES for training, 1 for vLLM generation"
echo "Training on dataset: meowterspace45/attack-graph-challenge"

# Run the attack graph GRPO training
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --config_file configs/accelerate/deepspeed_zero3.yaml \
    scripts/run_r1_grpo.py \
    --config configs/training/grpo-attack-graph.yaml

echo "Training complete! Check runs/ directory for outputs."