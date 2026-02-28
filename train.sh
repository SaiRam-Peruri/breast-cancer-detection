#!/bin/bash
# Start training with tmux for persistence
# Run this from breast-cancer-detection directory

SESSION_NAME="breast-cancer-training"

echo "========================================="
echo "Starting Training Session"
echo "========================================="

# Check if already running
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session already exists!"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Or kill it: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to tmux session
tmux send-keys -t $SESSION_NAME "echo 'Starting training...'" Enter
tmux send-keys -t $SESSION_NAME "python detectron.py -c train 2>&1 | tee training.log" Enter

echo "Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach: tmux attach -t $SESSION_NAME"
echo "  Detach: Ctrl+B, then D"
echo "  View logs: tail -f training.log"
echo "  Check GPU: watch -n 1 nvidia-smi"
echo "  Check disk: df -h"
echo ""
echo "To attach now, run: tmux attach -t $SESSION_NAME"
