#!/bin/bash
# Download CSV metadata files from Kaggle (needed for TCIA DICOM files)
# These CSVs contain the annotations for the CBIS-DDSM dataset

set -e

echo "========================================="
echo "Downloading CSV Metadata from Kaggle"
echo "========================================="

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: kaggle.json not found!"
    echo ""
    echo "To get your Kaggle API key:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Download kaggle.json"
    echo "  4. Upload to this server:"
    echo "     scp -i your-key.pem kaggle.json ubuntu@YOUR-IP:~/.kaggle/"
    echo "  5. Set permissions: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Install Kaggle API
pip install -q kaggle

# Create directories
mkdir -p datasets/CBIS-DDSM/csv

echo ""
echo "Downloading CSV metadata files..."
echo "These contain the annotations for bounding boxes and labels"
echo ""

# Download from Kaggle (only CSV files, not images)
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p /tmp/cbis_csv --unzip

# Extract only CSV files
echo "Extracting CSV files..."
find /tmp/cbis_csv -name "*.csv" -exec mv {} datasets/CBIS-DDSM/csv/ \;

# Cleanup
rm -rf /tmp/cbis_csv

echo ""
echo "✓ CSV files downloaded successfully!"
echo ""
ls -lh datasets/CBIS-DDSM/csv/
echo ""
echo "Now you can run: python convert_dataset.py"
