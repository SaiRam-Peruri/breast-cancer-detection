# Breast Cancer Detection - Optimized for AWS EC2

Optimized setup for training on **official TCIA CBIS-DDSM dataset** (163.51GB, 1,566 studies) with **300GB server storage**.

## 📊 Dataset: CBIS-DDSM (Official TCIA Version)

- **Source:** The Cancer Imaging Archive (TCIA)
- **Size:** 163.51GB (raw DICOM)
- **Studies:** 1,566 mammography studies
- **Format:** DICOM (converted to JPEG for training)
- **Annotations:** Mass and calcification bounding boxes
- **Classes:** Benign, Malignant, Normal
- **Manifest:** [CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia](https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia)

## 💾 Storage Breakdown (300GB Server for 163GB Dataset)

| Component | Size | Notes |
|-----------|------|-------|
| **CBIS-DDSM Dataset (raw DICOM)** | ~163GB | Downloaded from TCIA |
| **Converted images/** | ~40GB | JPEG conversion from DICOM |
| **Training checkpoints** | ~10GB | model_*.pth files in output/ |
| **Ubuntu OS + Dependencies** | ~20GB | CUDA, Python, packages |
| **Temp files, logs, buffer** | ~67GB | Safety margin |
| **TOTAL** | **~300GB** | ✅ Fits in 300GB server storage |

**Minimum recommended: 300GB | Ideal: 350GB for safety**

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

### Step 3: Download Dataset (TCIA - 163GB)

The official CBIS-DDSM dataset must be downloaded from TCIA using NBIA Data Retriever.

#### Option A: Download with NBIA Data Retriever (Recommended)

**On your LOCAL machine (Windows/Mac/Linux):**
1. Download NBIA Data Retriever: https://wiki.cancerimagingarchive.net/x/2QKPBQ
2. Download manifest file: https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia
3. Open NBIA Data Retriever → File → Open Manifest → Select .tcia file
4. Set download folder to a local directory
5. Wait for download to complete (4-8 hours depending on connection)

**Transfer to EC2:**
```bash
# From your local machine, upload the DICOM files:
scp -r -i your-key.pem /path/to/CBIS-DDSM/dicom ubuntu@YOUR-EC2-IP:~/workspace/breast-cancer-detection/datasets/CBIS-DDSM/
```

#### Option B: Direct Download on EC2 (Advanced)

```bash
# Check available download options
./download_data.sh

# Download CSV metadata (required for annotations)
./download_csv.sh
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

## 💾 Storage Management (300GB Server for 163GB Dataset)

### Space Breakdown for TCIA CBIS-DDSM:
```
163GB  - Raw CBIS-DDSM DICOM files (downloaded from TCIA)
 40GB  - Converted images/ (JPEG generated from DICOM)
 10GB  - Training checkpoints (output/)
 20GB  - Ubuntu OS + CUDA + Python packages
 10GB  - Temp files, logs, git repo
------
243GB  - TOTAL USED
 57GB  - FREE BUFFER (for safety)
------
300GB  - SERVER STORAGE
```

### Storage Management Commands:

**Check disk space:**
```bash
df -h
```

**If running low on space (<50GB free):**

1. **Delete raw DICOM files after conversion** (saves ~163GB!):
```bash
# ⚠️ WARNING: Only do this AFTER verifying conversion worked!
# And ONLY if you don't need the original DICOMs

# Test that images/ folder is complete
ls images/train/ | wc -l  # Should be ~10,000+

# If conversion successful, delete DICOMs to free 163GB
rm -rf datasets/CBIS-DDSM/dicom/

# Keep only CSV files for annotations
cd datasets/CBIS-DDSM && ls -lh
```

2. **Delete old checkpoints** (keep last 3):
```bash
cd output
ls -t model_*.pth | tail -n +4 | xargs rm -f
ls -lh  # Verify space freed
```

3. **Clean temp files:**
```bash
rm -rf /tmp/*
rm -rf ~/.cache/pip
```

### Emergency: Extreme Low Space

If you only have **200GB storage**, do this BEFORE training:

```bash
# 1. Download dataset
cd datasets/CBIS-DDSM/jpeg

# 2. Convert in batches (process 50GB at a time)
# Move first 50GB to images/, train/, val/, test/
# Then delete those JPEGs
# Repeat for next batch

# 3. Or use streaming mode (modify convert_dataset.py)
# Process one folder at a time
```

**Recommendation:** Use **300GB storage** for safety with 152GB dataset.

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
