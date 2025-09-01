#!/bin/bash
# Run script for GRPO Attack Graph Challenge training

# Setup environment
echo "Setting up environment..."
export HF_HUB_ENABLE_HF_TRANSFER=1

# Check for virtual environment (commented out by default)
#if [ ! -d "venv" ]; then
#    echo "Creating virtual environment..."
#    python -m venv venv
#fi
#source venv/bin/activate

# Install dependencies if needed (commented out by default)
#if ! python -c "import transformers" &> /dev/null; then
#    echo "Installing dependencies..."
#    pip install -r requirements.txt
#fi

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

# Create logs directory
mkdir -p logs

# Get timestamp for log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Logging all output to: $LOG_FILE"
echo "You can monitor progress with: tail -f $LOG_FILE"

# Run the attack graph GRPO training with full logging
{
    echo "=== Training started at $(date) ==="
    echo "Configuration: $NUM_GPUS GPUs, $NUM_PROCESSES training processes"
    echo "Dataset: meowterspace45/attack-graph-challenge"
    echo "Log file: $LOG_FILE"
    echo "=================================="
    echo
    
    accelerate launch \
        --num_processes $NUM_PROCESSES \
        --config_file configs/accelerate/deepspeed_zero3.yaml \
        scripts/run_r1_grpo.py \
        --config configs/training/grpo-attack-graph.yaml
    
    echo
    echo "=== Training completed at $(date) ==="
    echo "Check runs/ directory for model outputs"
    echo "Check logs/ directory for training logs"
    
} 2>&1 | tee "$LOG_FILE"

echo "Training complete! Check runs/ directory for outputs."
echo "Full log saved to: $LOG_FILE"