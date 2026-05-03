"""
train_bilstm.py
Training script for the custom BiLSTM + Attention Seq2Seq model.
Uses a tiny character-level toy dataset for demonstration.
Replace with a real parallel corpus (e.g., Tatoeba, WMT) for production use.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.bilstm_attention import BiLSTMSeq2Seq

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Tiny Toy Dataset (replace with real parallel corpus) ──────────────────────

TOY_PAIRS = [
    ("hello world",      "bonjour monde"),
    ("the sky is blue",  "le ciel est bleu"),
    ("i love poetry",    "j'aime la poesie"),
    ("beautiful flower", "belle fleur"),
    ("deep dark night",  "nuit profonde sombre"),
    ("river flows gently","la riviere coule doucement"),
    ("silent mountain",  "montagne silencieuse"),
    ("ancient wisdom",   "sagesse ancienne"),
] * 64   # repeat to create enough batches for demo


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, add_sos=False, add_eos=True):
    tokens = []
    if add_sos:
        tokens.append(vocab["<SOS>"])
    tokens += [vocab.get(ch, vocab["<UNK>"]) for ch in text]
    if add_eos:
        tokens.append(vocab["<EOS>"])
    return tokens


class PoetryPairDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=40):
        self.data      = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt, tgt_txt = self.data[idx]
        src = encode(src_txt[:self.max_len], self.src_vocab, add_sos=False, add_eos=True)
        tgt = encode(tgt_txt[:self.max_len], self.tgt_vocab, add_sos=True,  add_eos=True)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs, tgts


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(epochs: int = 20, batch_size: int = 16, lr: float = 1e-3,
          save_path: str = "models/bilstm_ckpt.pt"):

    src_texts  = [p[0] for p in TOY_PAIRS]
    tgt_texts  = [p[1] for p in TOY_PAIRS]
    src_vocab  = build_vocab(src_texts)
    tgt_vocab  = build_vocab(tgt_texts)

    logger.info(f"src vocab size: {len(src_vocab)} | tgt vocab size: {len(tgt_vocab)}")

    dataset    = PoetryPairDataset(TOY_PAIRS, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = BiLSTMSeq2Seq(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        embed_dim=64,
        hidden_dim=128,
        n_layers=1,
        dropout=0.0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            output = model(src, tgt, teacher_forcing_ratio=0.5)
            # output: (B, tgt_len, vocab) — shift to align with targets
            output = output[:, 1:, :].contiguous()
            tgt    = tgt[:, 1:].contiguous()

            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch:>3}/{epochs} | Loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state":  model.state_dict(),
        "src_vocab":    src_vocab,
        "tgt_vocab":    tgt_vocab,
        "config": {
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "embed_dim": 64,
            "hidden_dim": 128,
        },
    }, save_path)
    logger.info(f"Checkpoint saved to: {save_path}")
    return model, src_vocab, tgt_vocab


if __name__ == "__main__":
    train(epochs=20)
