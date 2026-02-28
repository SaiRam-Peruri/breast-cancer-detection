#!/bin/bash
# Download CBIS-DDSM dataset from TCIA (The Cancer Imaging Archive)
# Official dataset: 163.51GB, 1,566 studies
# Run this from breast-cancer-detection directory

set -e

echo "========================================="
echo "Downloading CBIS-DDSM from TCIA"
echo "Official Dataset: 163.51GB, 1,566 studies"
echo "========================================="

# Create directories
mkdir -p datasets/CBIS-DDSM/dicom
mkdir -p datasets/CBIS-DDSM/csv

echo ""
echo "CBIS-DDSM must be downloaded from TCIA using one of these methods:"
echo ""
echo "METHOD 1: NBIA Data Retriever (Recommended)"
echo "  1. Download NBIA Data Retriever from:"
echo "     https://wiki.cancerimagingarchive.net/x/2QKPBQ"
echo "  2. Download the manifest file (.tcia):"
echo "     https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"
echo "  3. Open NBIA Data Retriever and load the .tcia file"
echo "  4. Set download location to: $(pwd)/datasets/CBIS-DDSM/dicom/"
echo "  5. Start download (will take 4-8 hours)"
echo ""
echo "METHOD 2: wget (Command Line)"
echo "  Run the following commands:"
echo ""
echo "  # Download the manifest"
echo "  wget https://www.cancerimagingarchive.net/wp-content/uploads/CBIS-DDSM-All-doiJNLP-zzWs5zfZ.tcia"
echo ""
echo "  # Use TCIA downloader or extract URLs and wget"
echo "  # Note: TCIA requires special authentication"
echo ""
echo "METHOD 3: Kaggle (Smaller Subset - NOT recommended for full training)"
echo "  If you want the Kaggle subset (~11GB) instead:"
echo "  ./download_kaggle.sh"
echo ""
echo "========================================="
echo "After Download Complete:"
echo "========================================="
echo "1. Verify DICOM files:"
echo "   find datasets/CBIS-DDSM/dicom -name '*.dcm' | wc -l"
echo "   # Should show ~10,000+ DICOM files"
echo ""
echo "2. Download CSV metadata from Kaggle:"
echo "   ./download_csv.sh"
echo ""
echo "3. Then run conversion:"
echo "   python convert_dataset.py"
echo ""

# Check if dicom folder already has data
if [ -d "datasets/CBIS-DDSM/dicom" ]; then
    dicom_count=$(find datasets/CBIS-DDSM/dicom -name "*.dcm" 2>/dev/null | wc -l)
    if [ "$dicom_count" -gt 0 ]; then
        echo "✓ Found $dicom_count DICOM files already downloaded!"
        echo "  Location: datasets/CBIS-DDSM/dicom/"
        echo ""
        echo "Next step: Download CSV files with ./download_csv.sh"
    else
        echo "⚠ DICOM folder exists but no .dcm files found"
        echo "  Please download the dataset using methods above"
    fi
else
    echo "⚠ No DICOM folder found. Please download dataset first."
fi
