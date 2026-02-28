#!/bin/bash
# Resume training from checkpoint
# Run this from breast-cancer-detection directory

SESSION_NAME="breast-cancer-training"

echo "========================================="
echo "Resuming Training"
echo "========================================="

# Find latest checkpoint
checkpoint=$(ls -t output/model_*.pth 2>/dev/null | head -1)

if [ -z "$checkpoint" ]; then
    echo "ERROR: No checkpoint found in output/"
    echo "Available files:"
    ls -la output/
    exit 1
fi

echo "Found checkpoint: $checkpoint"

# Kill existing session if running
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Killing existing session..."
    tmux kill-session -t $SESSION_NAME
fi

# Start new session with resume
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "python detectron.py -c train --weights $checkpoint --resume 2>&1 | tee -a training.log" Enter

echo ""
echo "Training resumed!"
echo "Attach: tmux attach -t $SESSION_NAME"
