# Verse Intelligence — Context-Aware Neural Poetry Translation System

> Multi-stage NLP pipeline: MarianMT · BiLSTM+Attention · Sentence-BERT · LLM Refinement · RAG

---

## Architecture Overview

```
Input Poem
    │
    ▼
┌─────────────────────────────────────┐
│  1. Preprocessing (NLTK + spaCy)    │
│     Tokenisation · POS · NER · Dep  │
└──────────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  2. RAG Module (optional)        │
    │     FAISS · Sentence-BERT        │
    │     Retrieves stylistically      │
    │     similar poems for context    │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  3. NMT Translation              │
    │     MarianMT (Hugging Face)      │
    │     +Custom BiLSTM+Attention     │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  4. LLM Enhancement (optional)   │
    │     GPT-4o / Claude              │
    │     Preserves: tone · rhythm ·   │
    │     metaphor · cultural nuance   │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  5. Embedding (Sentence-BERT)    │
    │     all-MiniLM-L6-v2             │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  6. Evaluation Framework         │
    │     BLEU · BERTScore             │
    │     Cosine Similarity            │
    │     Semantic Drift               │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │  7. Sentiment & Emotion          │
    │     VADER · Transformer model    │
    │     Sentiment Drift Score        │
    └─────────────────────────────────┘
                   │
                   ▼
          Flask API + Web UI
```

---

## Project Structure

```
poetry_nmt/
├── app.py                        ← Flask application entry point
├── requirements.txt              ← All Python dependencies
├── train_bilstm.py               ← BiLSTM+Attention training script
│
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py           ← Main pipeline controller
│   ├── preprocessor.py           ← NLTK + spaCy preprocessing
│   ├── translator.py             ← MarianMT neural translation
│   ├── embedder.py               ← Sentence-BERT embeddings
│   ├── evaluator.py              ← BLEU, BERTScore, cosine, drift
│   ├── sentiment.py              ← VADER + emotion classification
│   ├── llm_enhancer.py           ← OpenAI / Anthropic refinement
│   └── rag.py                    ← FAISS retrieval-augmented generation
│
├── models/
│   └── bilstm_attention.py       ← Custom BiLSTM+Attention Seq2Seq
│
├── static/
│   ├── css/style.css             ← Dark editorial UI styles
│   └── js/app.js                 ← Frontend JavaScript
│
└── templates/
    └── index.html                ← Main web interface
```

---

## Quick Start

### Step 1 — Create Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ PyTorch note: The requirements install the CPU version.
> For GPU support, visit https://pytorch.org/get-started/locally/ and install with CUDA.

### Step 3 — Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Run the application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Supported Language Pairs

### 🇮🇳 Indic Languages

| Pair              | Engine       | Model                                   | Size   |
|-------------------|--------------|-----------------------------------------|--------|
| EN → Hindi (हिन्दी) | mBART-50    | facebook/mbart-large-50-many-to-many-mmt | ~2.3 GB |
| EN → Marathi (मराठी)| NLLB-200   | facebook/nllb-200-distilled-600M         | ~2.4 GB |
| Hindi → English   | mBART-50     | facebook/mbart-large-50-many-to-many-mmt | ~2.3 GB |
| Marathi → English | NLLB-200     | facebook/nllb-200-distilled-600M         | ~2.4 GB |
| Hindi → Marathi   | NLLB-200     | facebook/nllb-200-distilled-600M         | ~2.4 GB |
| Marathi → Hindi   | NLLB-200     | facebook/nllb-200-distilled-600M         | ~2.4 GB |

> ⚠️ Indic models are large. They download **once** and are cached in `~/.cache/huggingface/`.
> First translation may take **2–5 minutes**. Subsequent translations are fast.

### 🌍 European Languages

| Pair    | Engine    | Model                          | Size    |
|---------|-----------|--------------------------------|---------|
| EN → FR | MarianMT  | Helsinki-NLP/opus-mt-en-fr     | ~300 MB |
| EN → DE | MarianMT  | Helsinki-NLP/opus-mt-en-de     | ~300 MB |
| EN → ES | MarianMT  | Helsinki-NLP/opus-mt-en-es     | ~300 MB |
| EN → IT | MarianMT  | Helsinki-NLP/opus-mt-en-it     | ~300 MB |
| EN → RO | MarianMT  | Helsinki-NLP/opus-mt-en-ro     | ~300 MB |
| EN → ZH | MarianMT  | Helsinki-NLP/opus-mt-en-zh     | ~300 MB |
| FR → EN | MarianMT  | Helsinki-NLP/opus-mt-fr-en     | ~300 MB |
| DE → EN | MarianMT  | Helsinki-NLP/opus-mt-de-en     | ~300 MB |
| ES → EN | MarianMT  | Helsinki-NLP/opus-mt-es-en     | ~300 MB |
| IT → EN | MarianMT  | Helsinki-NLP/opus-mt-it-en     | ~300 MB |

European models download automatically on first use.

---

## Features

### Translation Modes

| Mode | Description |
|------|-------------|
| **Literal** | Raw MarianMT neural translation |
| **Context-Aware** | MarianMT → LLM refinement (requires API key) |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Cosine Similarity** | Embedding-space alignment (0–1, higher = better) |
| **BLEU Score** | n-gram lexical overlap |
| **BERTScore F1** | Contextual token-level similarity |
| **Semantic Drift** | L2 distance between embedding vectors |
| **Sentiment Drift** | Change in emotional compound score |

### LLM Enhancement

Provide an API key for either:
- **OpenAI** (`sk-...`) — uses `gpt-4o`
- **Anthropic** (`sk-ant-...`) — uses `claude-opus-4-5`

The LLM prompt instructs the model to preserve:
- Emotional tone and mood
- Poetic rhythm and line breaks
- Metaphorical/symbolic meaning
- Cultural idioms (adapted for target language)

### RAG (Retrieval-Augmented Generation)

When enabled, retrieves stylistically similar poems from a FAISS vector index
and injects them as style context into the LLM enhancement prompt.

Comes pre-loaded with poems by Frost, E.E. Cummings, Dylan Thomas, Angelou, and Poe.
**To add your own corpus:** edit `pipeline/rag.py → SAMPLE_POEMS`.

---

## Training the BiLSTM+Attention Model

The custom Seq2Seq model lives in `models/bilstm_attention.py`.
To train it on the toy dataset:

```bash
python train_bilstm.py
```

This trains a character-level model and saves a checkpoint to
`models/bilstm_ckpt.pt`.

**For a real model:**
1. Replace `TOY_PAIRS` in `train_bilstm.py` with a proper parallel corpus
   (e.g., Tatoeba, WMT En-Fr, OPUS).
2. Switch from character-level to word-level or BPE tokenisation.
3. Integrate the trained model into `pipeline/translator.py` as an alternative option.

---

## API Reference

### `POST /api/translate`

**Request body:**
```json
{
  "poem": "Two roads diverged in a yellow wood...",
  "src_lang": "en",
  "tgt_lang": "fr",
  "mode": "context_aware",
  "llm_api_key": "sk-...",
  "use_rag": false
}
```

**Response:**
```json
{
  "preprocessing": { "tokens": [...], "pos_tags": [...], "named_entities": [...] },
  "raw_translation": "Deux chemins divergeaient dans un bois jaune...",
  "enhanced_translation": "Deux routes bifurquaient dans un bois d'automne...",
  "metrics": {
    "cosine_similarity": 0.87,
    "bleu_score": 0.34,
    "bert_score": { "precision": 0.91, "recall": 0.89, "f1": 0.90 },
    "semantic_drift": 1.23
  },
  "sentiment": {
    "original": { "label": "positive", "compound": 0.12, "emotions": {...} },
    "translated": { "label": "positive", "compound": 0.09, "emotions": {...} },
    "drift": { "compound_drift": 0.03, "preserved": true }
  },
  "embeddings": { "original_shape": [384], "translated_shape": [384] }
}
```

### `GET /api/languages`
Returns list of supported language pairs.

### `GET /api/health`
Returns `{"status": "ok"}`.

---

## Resume Description

**Context-Aware Neural Machine Translation and Poetic Semantics Evaluation Pipeline**
`Python · NLTK · spaCy · Hugging Face Transformers · Sentence-BERT · BiLSTM+Attention · Flask · LLM Pipeline · RAG`

- Engineered a multi-stage NMT pipeline integrating NLTK/spaCy preprocessing, MarianMT transformer translation, and a custom Bi-directional LSTM with Bahdanau attention mechanism to preserve semantic intent and stylistic nuance in poetic text.
- Implemented cross-lingual embedding representations using Sentence-BERT (`all-MiniLM-L6-v2`) for cosine similarity, semantic drift analysis, and embedding-space alignment across source and translated text.
- Designed a comprehensive evaluation framework: BLEU Score, BERTScore (P/R/F1), Cosine Similarity, Semantic Drift, and VADER-based Sentiment Drift analysis for quantifying meaning preservation beyond lexical overlap.
- Integrated an LLM-based contextual refinement layer (GPT-4o / Claude) to enhance translation quality by preserving emotional tone, metaphorical structure, and poetic rhythm.
- Built a hybrid Retrieval-Augmented Generation (RAG) module using FAISS for style-aware poem retrieval, injecting stylistic context into the LLM enhancement prompt.
- Deployed a full-stack interactive system (Flask + Vanilla JS) with real-time metric visualization, sentiment comparison charts, and preprocessing stage inspection.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: spacy` | Run `pip install spacy` then `python -m spacy download en_core_web_sm` |
| Hindi/Marathi translation frozen | mBART/NLLB download ~2.4 GB on first use — wait 3–5 min, check terminal |
| `sentencepiece` not found | Run `pip install sentencepiece protobuf` |
| `MBart50TokenizerFast` error | Run `pip install transformers --upgrade` (need ≥4.40) |
| Hindi text shows as boxes `□□□` | Browser needs Devanagari font — Chrome/Edge handle this automatically |
| MarianMT model slow to load | First run downloads ~300 MB — cached after. Subsequent loads are instant. |
| `CUDA out of memory` | Models run on CPU by default. For GPU add `device_map="auto"` in translator.py |
| LLM enhancement returns raw translation | Check API key is correct and has credits |
| `faiss` install fails on Windows | Use `pip install faiss-cpu` (not `faiss-gpu`) |
| Port 5000 in use | Edit `port=5000` in `app.py` to any free port e.g. 8080 |
| `accelerate` warning on load | Run `pip install accelerate` — optional but speeds up large model loading |

### Pre-downloading Indic models (recommended)

Run this once before using the app to avoid waiting during translation:

```powershell
# Pre-download mBART-50 for Hindi
python -c "from transformers import MBartForConditionalGeneration, MBart50TokenizerFast; MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt'); print('mBART-50 ready!')"

# Pre-download NLLB-200 for Marathi
python -c "from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast; NllbTokenizerFast.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); print('NLLB-200 ready!')"
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask 3.x + Flask-CORS |
| NLP Preprocessing | NLTK 3.8 + spaCy 3.7 |
| Neural Translation | Hugging Face MarianMT |
| Custom Seq2Seq | PyTorch BiLSTM + Bahdanau Attention |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Evaluation | bert-score, NLTK BLEU |
| Sentiment | VADER + cardiffnlp/twitter-roberta-base-emotion |
| RAG | FAISS (CPU) |
| LLM APIs | OpenAI GPT-4o / Anthropic Claude |
| Frontend | HTML5 + CSS3 + Vanilla JS + Chart.js |
