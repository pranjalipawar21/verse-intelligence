#!/bin/bash
echo "============================================================"
echo "  Verse Intelligence — Setup Script"
echo "  Supports: English, Hindi, Marathi + European Languages"
echo "============================================================"
echo

echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

echo
echo "[3/5] Installing all dependencies..."
pip install -r requirements.txt

echo
echo "[4/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo
echo "[5/5] Setup complete!"
echo
echo "============================================================"
echo " OPTIONAL: Pre-download large Indic models (recommended)"
echo " Hindi  → mBART-50  (~2.3 GB)"
echo " Marathi → NLLB-200 (~2.4 GB)"
echo "============================================================"
echo
read -p "Download Indic models now? (y/n): " DOWNLOAD_INDIC

if [ "$DOWNLOAD_INDIC" = "y" ] || [ "$DOWNLOAD_INDIC" = "Y" ]; then
    echo
    echo "Downloading mBART-50 for Hindi..."
    python -c "
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
print('Downloading mBART-50...')
MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
print('mBART-50 DONE!')
"
    echo
    echo "Downloading NLLB-200 for Marathi..."
    python -c "
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
print('Downloading NLLB-200...')
NllbTokenizerFast.from_pretrained('facebook/nllb-200-distilled-600M')
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
print('NLLB-200 DONE!')
"
    echo "All Indic models ready!"
fi

echo
echo "============================================================"
echo " TO RUN THE APP:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo " Then open: http://localhost:5000"
echo "============================================================"
