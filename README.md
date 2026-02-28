# Breast Cancer Detection - Optimized for AWS EC2

Optimized setup for training on full CBIS-DDSM dataset (152GB) with 200-300GB storage.

## 📁 Repository Structure (Cleaned)

```
breast-cancer-detection/
├── detectron.py              # Main training script (optimized for A10G 24GB)
├── convert_dataset.py        # Dataset conversion (CBIS-DDSM only)
├── classification_model.py   # XAI/heatmap model
├── coco_to_classification.py # Convert COCO to classification format
├── xai.py                    # Explainable AI visualizations
├── visualizer.py             # Dataset visualization
├── utils.py                  # Helper functions
├── filters.py                # Image preprocessing
├── filter_config.json        # Filter configuration
├── radiomics_extractor.py    # Feature extraction
├── requirements.txt          # Python dependencies
├── setup.sh                  # EC2 setup script
├── download_data.sh          # Download dataset from Kaggle
├── train.sh                  # Start training with tmux
├── resume.sh                 # Resume training from checkpoint
└── .gitignore               # Git ignore rules
```

## 🚀 Quick Start (AWS EC2)

### Step 1: Launch EC2 Instance

**Recommended:** `g5.2xlarge` (A10G GPU, 24GB VRAM)
- **AMI:** Ubuntu 22.04 LTS
- **Storage:** 300GB SSD (gp3)
- **Cost:** ~$1.20/hour

### Step 2: Connect and Setup

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@YOUR-EC2-IP

# Run setup
./setup.sh

# Clone this repository
git clone https://github.com/YOUR_USERNAME/breast-cancer-detection.git
cd breast-cancer-detection

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 3: Download Dataset

```bash
# Upload your Kaggle API key first
# From your local machine:
# scp -i your-key.pem kaggle.json ubuntu@YOUR-EC2-IP:~/.kaggle/

# Download CBIS-DDSM
./download_data.sh
```

### Step 4: Convert Dataset

```bash
# Convert DICOM/JPEG to training format (takes 2-4 hours)
python convert_dataset.py

# Check output
ls -lh images/ train.json val.json test.json
```

### Step 5: Start Training

```bash
# Start training in tmux session
./train.sh

# Or manually:
tmux new -s training
python detectron.py -c train 2>&1 | tee training.log
```

## 📊 Training Configuration

**Optimized for:**
- GPU: A10G (24GB VRAM)
- Batch size: 8
- Workers: 4
- Checkpoint: Every 1000 iterations

**Expected time:** 5-7 days for 150 epochs

## 🔍 Monitoring Training

```bash
# Attach to training session
tmux attach -t breast-cancer-training

# Check GPU usage (in another terminal)
watch -n 1 nvidia-smi

# Check logs
tail -f training.log

# Check disk space
df -h
```

## 🔄 Resume Training

If training crashes or you stop it:

```bash
./resume.sh
```

Or manually:
```bash
tmux attach -t breast-cancer-training
# Find latest checkpoint
ls -lt output/model_*.pth | head -1
# Resume
python detectron.py -c train --weights output/model_XXXXX.pth --resume
```

## 📥 Download Trained Model

```bash
# From your local machine
scp -i your-key.pem ubuntu@YOUR-EC2-IP:~/workspace/breast-cancer-detection/output/model_final.pth ./
```

## 💾 Storage Management (200-300GB)

**Space breakdown:**
- Raw dataset: ~11GB (subset) or ~152GB (full)
- Converted images: ~50GB
- Training checkpoints: ~10GB
- Total: ~70-220GB

**If running low on space:**
```bash
# Delete raw JPEGs after conversion (keep DICOM)
rm -rf datasets/CBIS-DDSM/jpeg/

# Delete old checkpoints (keep best 3)
cd output && ls -t model_*.pth | tail -n +4 | xargs rm
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)
```python
# In detectron.py, reduce:
batch_size = 4  # From 8
num_workers = 2  # From 4
```

### Disk Full
```bash
# Check what's using space
du -sh *
# Clean up
rm -rf datasets/CBIS-DDSM/jpeg/  # If conversion done
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi -l 1
# Should be >90%. If not, increase num_workers in detectron.py
```

## 📈 Expected Results

After 150 epochs on full CBIS-DDSM:
- mAP (mass detection): ~0.75-0.85
- Training time: 5-7 days
- Final model: `output/model_final.pth`

## 📧 Support

For issues, check:
1. `training.log` for errors
2. `nvidia-smi` for GPU issues
3. `df -h` for disk space

## 📝 Notes

- **Only CBIS-DDSM** is configured (edit `convert_dataset.py` to add others)
- **Only 'mass'** class is enabled (edit `chosen_classes` for calcification)
- Training saves checkpoints every 1000 iterations
- Use `tmux` to keep training running after disconnect
