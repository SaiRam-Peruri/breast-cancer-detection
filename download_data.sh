#!/bin/bash
# Download CBIS-DDSM dataset from Kaggle
# Run this from breast-cancer-detection directory

set -e

echo "========================================="
echo "Downloading CBIS-DDSM Dataset"
echo "========================================="

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: kaggle.json not found!"
    echo "Please upload your Kaggle API key:"
    echo "  mkdir -p ~/.kaggle"
    echo "  # Copy kaggle.json to ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Install Kaggle API
pip install -q kaggle

# Create directories
mkdir -p datasets/CBIS-DDSM/csv
mkdir -p datasets/CBIS-DDSM/jpeg

echo ""
echo "Downloading CBIS-DDSM dataset (~11GB subset or ~152GB full)..."
echo "This will take 1-3 hours depending on your connection."
echo ""

# Download from Kaggle
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p datasets/CBIS-DDSM --unzip

# Organize files
echo "Organizing files..."
find datasets/CBIS-DDSM -name "*.csv" -exec mv {} datasets/CBIS-DDSM/csv/ \;
find datasets/CBIS-DDSM -name "*.jpg" -exec mv {} datasets/CBIS-DDSM/jpeg/ \; 2>/dev/null || true

# Check size
echo ""
echo "Download complete!"
du -sh datasets/CBIS-DDSM/
echo ""
echo "CSV files:"
ls -lh datasets/CBIS-DDSM/csv/
echo ""
echo "JPEG files count:"
ls datasets/CBIS-DDSM/jpeg/ | wc -l
