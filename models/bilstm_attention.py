"""
models/bilstm_attention.py
Custom Bi-directional LSTM + Attention Seq2Seq model for experimental translation.
This is a character-level model for demonstration; swap the vocab for word-level as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── Attention Mechanism ───────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention."""

    def __init__(self, enc_hidden: int, dec_hidden: int):
        super().__init__()
        self.attn    = nn.Linear(enc_hidden * 2 + dec_hidden, dec_hidden)
        self.v       = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, hidden: torch.Tensor,
                encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden*2)
        src_len = encoder_outputs.size(1)
        hidden  = hidden.unsqueeze(1).repeat(1, src_len, 1)          # (B, L, D)
        energy  = torch.tanh(self.attn(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))                                                             # (B, L, D)
        attention = self.v(energy).squeeze(2)                         # (B, L)
        weights   = F.softmax(attention, dim=1)                       # (B, L)
        context   = torch.bmm(weights.unsqueeze(1),
                               encoder_outputs).squeeze(1)            # (B, D*2)
        return context, weights


# ── Encoder ──────────────────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=n_layers,
                                 batch_first=True,
                                 bidirectional=True,
                                 dropout=dropout if n_layers > 1 else 0)
        self.dropout   = nn.Dropout(dropout)
        self.fc_h      = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c      = nn.Linear(hidden_dim * 2, hidden_dim)
        self.n_layers  = n_layers

    def forward(self, src: torch.Tensor):
        embedded = self.dropout(self.embedding(src))                   # (B, L, E)
        outputs, (hidden, cell) = self.lstm(embedded)

        # Merge bidirectional layers: take top forward + backward hidden
        hidden = torch.tanh(self.fc_h(
            torch.cat((hidden[-2], hidden[-1]), dim=1)
        ))                                                             # (B, H)
        cell   = torch.tanh(self.fc_c(
            torch.cat((cell[-2], cell[-1]), dim=1)
        ))
        return outputs, hidden, cell


# ── Decoder ──────────────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 enc_hidden: int, dropout: float = 0.3):
        super().__init__()
        self.attention  = BahdanauAttention(enc_hidden, hidden_dim)
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim + enc_hidden * 2, hidden_dim,
                                  batch_first=True)
        self.fc_out     = nn.Linear(hidden_dim + enc_hidden * 2 + embed_dim,
                                    vocab_size)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, tgt_token: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor):
        # tgt_token: (B,)
        tgt_token = tgt_token.unsqueeze(1)                            # (B, 1)
        embedded  = self.dropout(self.embedding(tgt_token))           # (B, 1, E)

        context, attn_weights = self.attention(hidden, encoder_outputs)
        context = context.unsqueeze(1)                                 # (B, 1, H*2)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input,
                                           (hidden.unsqueeze(0),
                                            cell.unsqueeze(0)))
        hidden = hidden.squeeze(0)
        cell   = cell.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output.squeeze(1), context.squeeze(1),
                        embedded.squeeze(1)), dim=1)
        )
        return prediction, hidden, cell, attn_weights


# ── Full Seq2Seq Model ────────────────────────────────────────────────────────

class BiLSTMSeq2Seq(nn.Module):
    """BiLSTM Encoder + Attention Decoder Seq2Seq."""

    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = BiLSTMEncoder(src_vocab, embed_dim, hidden_dim,
                                     n_layers, dropout)
        self.decoder = AttentionDecoder(tgt_vocab, embed_dim, hidden_dim,
                                        hidden_dim, dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len    = tgt.size(1)
        tgt_vocab  = self.decoder.fc_out.out_features

        outputs        = torch.zeros(batch_size, tgt_len, tgt_vocab).to(src.device)
        enc_out, h, c  = self.encoder(src)

        dec_input = tgt[:, 0]                  # <SOS> token
        for t in range(1, tgt_len):
            output, h, c, _ = self.decoder(dec_input, h, c, enc_out)
            outputs[:, t]   = output
            top1 = output.argmax(1)
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t]
            else:
                dec_input = top1
        return outputs


# ── Convenience builder ───────────────────────────────────────────────────────

def build_model(src_vocab_size: int = 5000, tgt_vocab_size: int = 5000,
                embed_dim: int = 128, hidden_dim: int = 256) -> BiLSTMSeq2Seq:
    model = BiLSTMSeq2Seq(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"BiLSTM+Attention model: {total_params:,} trainable parameters")
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_model()
    src   = torch.randint(1, 5000, (2, 20))   # batch=2, src_len=20
    tgt   = torch.randint(1, 5000, (2, 15))   # batch=2, tgt_len=15
    out   = model(src, tgt)
    print(f"Output shape: {out.shape}")        # (2, 15, 5000)
    print("BiLSTM+Attention Seq2Seq: OK")
