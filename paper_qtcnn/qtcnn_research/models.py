"""Models under comparison: the hybrid QTCNN and its classical counterparts.

Every sequence model here shares the same encoder interface and head width, so
differences in the results tables come from the block being ablated rather than
from incidental capacity differences. Parameter counts are reported alongside
every result for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from qtcnn_research.quantum import (
    CircuitConfig,
    QuantumMode,
    build_quantum_block,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
class TemporalConvEncoder(nn.Module):
    """Dilated causal 1D CNN reducing ``[B, T, F]`` to ``[B, out_dim]``.

    Causal padding matters: a symmetric ``padding=1`` convolution, as in the
    legacy v2 encoder, mixes each position with the step that follows it. Within
    a window that only spans past data the label is still unaffected, but the
    receptive field is then not the one the architecture claims, so the causal
    form is used here.
    """

    def __init__(self, in_dim: int, out_dim: int = 8, hidden: int = 64, kernel: int = 3,
                 n_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.kernel = kernel
        layers: List[nn.Module] = []
        d_prev = in_dim
        self.pads: List[int] = []
        for b in range(n_blocks):
            dilation = 2**b
            layers.append(nn.Conv1d(d_prev, hidden, kernel_size=kernel, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            self.pads.append((kernel - 1) * dilation)
            d_prev = hidden
        self.convs = nn.ModuleList(layers)
        self.n_blocks = n_blocks
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, out_dim), nn.Tanh()
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # [B, F, T]
        pad_idx = 0
        for i, layer in enumerate(self.convs):
            if isinstance(layer, nn.Conv1d):
                h = nn.functional.pad(h, (self.pads[pad_idx], 0))  # left-pad only
                pad_idx += 1
            h = layer(h)
        h = h[:, :, -1]  # last time step of a causal stack
        # Tanh keeps the encoder output inside the angle range the circuit expects.
        return self.head(h) * np.pi


class LSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 8, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, out_dim), nn.Tanh())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]) * np.pi


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 8, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, dropout: float = 0.1, seq_len: int = 20):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, out_dim), nn.Tanh())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1]) * np.pi


ENCODERS = {"tcn": TemporalConvEncoder, "lstm": LSTMEncoder, "transformer": TransformerEncoder}


# ----------------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------------
@dataclass
class QTCNNConfig:
    n_features: int = 21
    seq_len: int = 20
    encoder: str = "tcn"
    encoder_hidden: int = 64
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    quantum_mode: QuantumMode = QuantumMode.TRAINED
    head_hidden: Sequence[int] = (64, 32)
    dropout: float = 0.1
    # When False the head sees only the quantum score, which isolates how much
    # signal the circuit itself carries; the default lets it see both.
    head_sees_encoder: bool = True
    seed: int = 0


class QTCNN(nn.Module):
    """Temporal encoder -> quantum convolution -> classification head.

    The ablation switches live in the config rather than in separate classes so
    that every condition trains through exactly the same code path.
    """

    def __init__(self, cfg: QTCNNConfig):
        super().__init__()
        self.cfg = cfg
        enc_cls = ENCODERS[cfg.encoder]
        kwargs = dict(in_dim=cfg.n_features, out_dim=cfg.circuit.n_qubits)
        if cfg.encoder == "transformer":
            kwargs["seq_len"] = cfg.seq_len
        elif cfg.encoder == "tcn":
            kwargs["hidden"] = cfg.encoder_hidden
        else:
            kwargs["hidden"] = cfg.encoder_hidden
        self.encoder = enc_cls(**kwargs)

        self.quantum = build_quantum_block(cfg.quantum_mode, cfg.circuit, seed=cfg.seed)
        self.has_quantum = self.quantum is not None

        in_dim = (cfg.circuit.n_qubits if cfg.head_sees_encoder else 0) + (1 if self.has_quantum else 0)
        if in_dim == 0:
            raise ValueError("model has no inputs: enable the quantum block or the encoder path")

        layers: List[nn.Module] = []
        d_prev = in_dim
        for width in cfg.head_hidden:
            layers += [nn.Linear(d_prev, width), nn.GELU(), nn.Dropout(cfg.dropout)]
            d_prev = width
        layers.append(nn.Linear(d_prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        parts = []
        if self.has_quantum:
            parts.append(self.quantum(enc).unsqueeze(1))
        if self.cfg.head_sees_encoder:
            parts.append(enc)
        feats = torch.cat(parts, dim=1)
        return self.head(feats).squeeze(1)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {
            "encoder": count_parameters(self.encoder),
            "quantum": count_parameters(self.quantum) if self.has_quantum else 0,
            "head": count_parameters(self.head),
            "total": count_parameters(self),
        }


# ----------------------------------------------------------------------------
# Purely classical sequence baselines
# ----------------------------------------------------------------------------
class ClassicalSequenceModel(nn.Module):
    """Encoder plus MLP head, with no quantum path at all.

    Two distinct roles are filled by this class, and they must not be confused.
    At the hybrid's own width it is architecturally identical to the
    ``qtcnn_noquantum`` ablation condition and would duplicate it. The
    ``tcn`` baseline therefore uses a wider, deeper configuration: a control
    condition answers "does the circuit add anything at matched capacity", while
    a baseline answers "does any of this beat a classical model somebody would
    actually deploy". Reporting a deliberately narrowed classical model as the
    baseline is the failure mode this separation avoids.
    """

    def __init__(self, n_features: int, seq_len: int = 20, encoder: str = "tcn",
                 hidden: int = 64, embed_dim: int = 8, head_hidden: Sequence[int] = (64, 32),
                 dropout: float = 0.1, n_blocks: Optional[int] = None):
        super().__init__()
        enc_cls = ENCODERS[encoder]
        kwargs = dict(in_dim=n_features, out_dim=embed_dim)
        if encoder == "transformer":
            kwargs["seq_len"] = seq_len
        else:
            kwargs["hidden"] = hidden
        if encoder == "tcn" and n_blocks is not None:
            kwargs["n_blocks"] = n_blocks
        self.encoder = enc_cls(**kwargs)
        layers: List[nn.Module] = []
        d_prev = embed_dim
        for width in head_hidden:
            layers += [nn.Linear(d_prev, width), nn.GELU(), nn.Dropout(dropout)]
            d_prev = width
        layers.append(nn.Linear(d_prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


class MLPBaseline(nn.Module):
    """Flattened-window MLP: the weakest sequence-aware reference point."""

    def __init__(self, n_features: int, seq_len: int = 20,
                 hidden: Sequence[int] = (256, 128, 64), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d_prev = n_features * seq_len
        for width in hidden:
            layers += [nn.Linear(d_prev, width), nn.GELU(), nn.Dropout(dropout)]
            d_prev = width
        layers.append(nn.Linear(d_prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(1)


def build_model(name: str, n_features: int, seq_len: int = 20, seed: int = 0, **kwargs) -> nn.Module:
    """Registry used by the experiment runner.

    Names starting with ``qtcnn_`` select a quantum ablation condition; the rest
    are classical references.
    """
    torch.manual_seed(seed)

    if name.startswith("qtcnn"):
        mode = {
            "qtcnn": QuantumMode.TRAINED,
            "qtcnn_trained": QuantumMode.TRAINED,
            "qtcnn_frozen": QuantumMode.FROZEN,
            "qtcnn_surrogate": QuantumMode.SURROGATE,
            "qtcnn_noquantum": QuantumMode.NONE,
            "qtcnn_quantumonly": QuantumMode.TRAINED,
        }[name]
        circuit = CircuitConfig(
            n_qubits=kwargs.pop("n_qubits", 8),
            n_layers=kwargs.pop("n_layers", 3),
            embedding=kwargs.pop("embedding", CircuitConfig().embedding),
            shared_params=kwargs.pop("shared_params", True),
            ansatz=kwargs.pop("ansatz", CircuitConfig().ansatz),
        )
        cfg = QTCNNConfig(
            n_features=n_features,
            seq_len=seq_len,
            encoder=kwargs.pop("encoder", "tcn"),
            circuit=circuit,
            quantum_mode=mode,
            head_sees_encoder=(name != "qtcnn_quantumonly"),
            seed=seed,
            **kwargs,
        )
        return QTCNN(cfg)

    if name in ("tcn", "lstm", "transformer"):
        if name == "tcn":
            # Sized as a baseline, not as a matched control: the matched control
            # is qtcnn_noquantum, which this would otherwise be identical to.
            kwargs.setdefault("hidden", 128)
            kwargs.setdefault("embed_dim", 32)
            kwargs.setdefault("n_blocks", 3)
            kwargs.setdefault("head_hidden", (128, 64))
        return ClassicalSequenceModel(n_features, seq_len=seq_len, encoder=name, **kwargs)
    if name == "mlp":
        return MLPBaseline(n_features, seq_len=seq_len, **kwargs)

    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = [
    "qtcnn_trained",
    "qtcnn_frozen",
    "qtcnn_surrogate",
    "qtcnn_noquantum",
    "qtcnn_quantumonly",
    "tcn",
    "lstm",
    "transformer",
    "mlp",
]
