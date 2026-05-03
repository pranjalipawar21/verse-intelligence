@echo off
echo ============================================================
echo   Verse Intelligence — Setup Script
echo   Supports: English, Hindi, Marathi + European Languages
echo ============================================================
echo.

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 ( echo ERROR: Python not found. Install from python.org & pause & exit )
call venv\Scripts\activate

echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [3/5] Installing all dependencies...
pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed & pause & exit )

echo.
echo [4/5] Downloading spaCy English model...
python -m spacy download en_core_web_sm

echo.
echo [5/5] Setup complete!
echo.
echo ============================================================
echo  OPTIONAL: Pre-download large Indic models (recommended)
echo  Hindi uses mBART-50 (~2.3 GB)
echo  Marathi uses NLLB-200 (~2.4 GB)
echo ============================================================
echo.
set /p DOWNLOAD_INDIC="Download Indic models now? (y/n): "
if /i "%DOWNLOAD_INDIC%"=="y" (
    echo.
    echo Downloading mBART-50 for Hindi... (this may take 5-10 min)
    python -c "from transformers import MBartForConditionalGeneration, MBart50TokenizerFast; print('Downloading mBART-50...'); MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); print('mBART-50 DONE!')"
    echo.
    echo Downloading NLLB-200 for Marathi... (this may take 5-10 min)
    python -c "from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast; print('Downloading NLLB-200...'); NllbTokenizerFast.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); print('NLLB-200 DONE!')"
    echo.
    echo All Indic models downloaded!
)

echo.
echo ============================================================
echo  TO RUN THE APP:
echo    venv\Scripts\activate
echo    python app.py
echo  Then open: http://localhost:5000
echo ============================================================
pause
