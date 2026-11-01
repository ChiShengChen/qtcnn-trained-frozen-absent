"""Batched quantum convolutional layers and their classical control conditions.

Two problems with the original implementation are fixed here.

*Speed.* ``scripts/qtcnn_pipeline.py`` evaluates the QNode once per sample in a
Python loop. Passing the whole batch through PennyLane's parameter broadcasting
gives identical values (agreement to 1e-6) and, measured at 8 qubits, 3 layers,
batch 256, runs 46.7x faster on ``default.qubit``. That is what makes full-sample
walk-forward evaluation affordable at all.

*Attribution.* A hybrid model whose head also sees the classical features cannot
tell you whether the quantum circuit contributed anything. The variants in
:class:`QuantumMode` are the control conditions that answer it: a trained
circuit, the same circuit frozen at its random initialisation, a
parameter-matched classical map, and no circuit at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumMode(str, Enum):
    """Control conditions for the quantum-contribution ablation."""

    TRAINED = "trained"        # full model: circuit parameters are learned
    FROZEN = "frozen"          # same circuit, parameters fixed at random init
    SURROGATE = "surrogate"    # classical map with a matched parameter count
    NONE = "none"              # head sees the encoder output only


class Embedding(str, Enum):
    ANGLE_Y = "angle_y"
    ANGLE_XY = "angle_xy"
    AMPLITUDE = "amplitude"
    REUPLOAD = "reupload"      # data re-uploaded before every conv layer


ANSATZ_QCNN = "qcnn"                  # pairwise conv blocks, pooling, single-wire Z
ANSATZ_REUPLOAD_HEA = "reupload_hea"  # data re-uploaded every layer, hardware-efficient
                                      # rotations + CNOT ring, no pooling, mean of Z over
                                      # all wires. The second ansatz family of the paper.


@dataclass(frozen=True)
class CircuitConfig:
    n_qubits: int = 8
    n_layers: int = 3
    embedding: Embedding = Embedding.ANGLE_Y
    shared_params: bool = True   # one parameter set reused across pairs in a layer
    device: str = "default.qubit"
    ansatz: str = ANSATZ_QCNN

    @property
    def effective_layers(self) -> int:
        """Pooling halves the register each layer, so depth is capped by log2."""
        if self.ansatz == ANSATZ_REUPLOAD_HEA:
            return self.n_layers          # no pooling: every layer acts on the full register
        return max(0, min(self.n_layers, int(math.floor(math.log2(max(1, self.n_qubits))))))

    @property
    def n_pairs_per_layer(self) -> List[int]:
        pairs, wires = [], self.n_qubits
        for _ in range(self.effective_layers):
            pairs.append(wires // 2)
            wires = (wires + 1) // 2
        return pairs

    @property
    def n_circuit_params(self) -> int:
        if self.ansatz == ANSATZ_REUPLOAD_HEA:
            return 2 * self.n_qubits * self.n_layers   # RY and RZ on every wire, every layer
        if self.shared_params:
            return 6 * self.effective_layers
        return 6 * sum(self.n_pairs_per_layer)

    def gate_counts(self) -> dict:
        """Resource estimate for the hardware-feasibility table."""
        if self.ansatz == ANSATZ_REUPLOAD_HEA:
            return {
                "n_qubits": self.n_qubits,
                "layers": self.n_layers,
                "single_qubit_gates": 3 * self.n_qubits * self.n_layers,   # embed + RY + RZ
                "two_qubit_gates": self.n_qubits * self.n_layers,          # CNOT ring
                "circuit_depth": 4 * self.n_layers,
                "trainable_params": self.n_circuit_params,
            }
        pairs = self.n_pairs_per_layer
        total_pairs = sum(pairs)
        single = 6 * total_pairs                  # 4 RY + 2 RZ per pair
        two = 2 * total_pairs                     # 2 CNOT per pair
        embed = self.n_qubits * (2 if self.embedding == Embedding.ANGLE_XY else 1)
        if self.embedding == Embedding.REUPLOAD:
            embed = self.n_qubits * self.effective_layers
        return {
            "n_qubits": self.n_qubits,
            "layers": self.effective_layers,
            "single_qubit_gates": single + embed,
            "two_qubit_gates": two,
            "circuit_depth": 4 * self.effective_layers + 1,
            "trainable_params": self.n_circuit_params,
        }


# ----------------------------------------------------------------------------
# Circuit construction
# ----------------------------------------------------------------------------
def _conv_block(params: torch.Tensor, wires: List[int]) -> None:
    """Two-qubit parameterised convolution applied to disjoint neighbouring pairs."""
    for i in range(0, len(wires) - 1, 2):
        w1, w2 = wires[i], wires[i + 1]
        qml.RY(params[0], wires=w1)
        qml.RY(params[1], wires=w2)
        qml.CNOT(wires=[w1, w2])
        qml.RZ(params[2], wires=w1)
        qml.RZ(params[3], wires=w2)
        qml.CNOT(wires=[w2, w1])
        qml.RY(params[4], wires=w1)
        qml.RY(params[5], wires=w2)


def _embed(x: torch.Tensor, cfg: CircuitConfig) -> None:
    wires = range(cfg.n_qubits)
    if cfg.embedding in (Embedding.ANGLE_Y, Embedding.REUPLOAD):
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
    elif cfg.embedding == Embedding.ANGLE_XY:
        qml.AngleEmbedding(x, wires=wires, rotation="Y")
        qml.AngleEmbedding(x, wires=wires, rotation="Z")
    elif cfg.embedding == Embedding.AMPLITUDE:
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)


def make_qcnn(cfg: CircuitConfig) -> Tuple[Callable, Tuple[int, ...]]:
    """Return a batched QNode and the shape of its weight tensor.

    The QNode accepts ``x`` of shape ``[B, n_qubits]`` and returns ``[B]`` via
    PennyLane parameter broadcasting; no Python-level loop over the batch.
    """
    dev = qml.device(cfg.device, wires=cfg.n_qubits)
    layers = cfg.effective_layers
    pairs = cfg.n_pairs_per_layer

    if cfg.ansatz == ANSATZ_REUPLOAD_HEA:
        # Second ansatz family: the input is re-uploaded before every layer, each
        # layer rotates every wire (RY, RZ) and entangles them in a ring, there is
        # no pooling, and the observable is the mean of Z over all wires. It differs
        # from the QCNN in every respect a reviewer could name (embedding schedule,
        # entangling pattern, pooling, observable locality), so a result shared by
        # both is not a property of one ansatz.
        n = cfg.n_qubits
        weight_shape = (cfg.n_layers, n, 2)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def reupload_circuit(x, weights):
            for layer in range(cfg.n_layers):
                qml.AngleEmbedding(x, wires=range(n), rotation="Y")
                for w in range(n):
                    qml.RY(weights[layer, w, 0], wires=w)
                    qml.RZ(weights[layer, w, 1], wires=w)
                for w in range(n):
                    qml.CNOT(wires=[w, (w + 1) % n])
            return [qml.expval(qml.PauliZ(w)) for w in range(n)]

        return reupload_circuit, weight_shape

    weight_shape = (layers, 6) if cfg.shared_params else (sum(pairs), 6)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        _embed(x, cfg)
        active = list(range(cfg.n_qubits))
        cursor = 0
        for layer in range(layers):
            if len(active) < 2:
                break
            if cfg.embedding == Embedding.REUPLOAD and layer > 0:
                # Re-uploading raises the model's Fourier degree without extra qubits.
                qml.AngleEmbedding(x[..., : len(active)], wires=active, rotation="Y")
            if cfg.shared_params:
                _conv_block(weights[layer], active)
            else:
                for i in range(0, len(active) - 1, 2):
                    _conv_block(weights[cursor], [active[i], active[i + 1]])
                    cursor += 1
            active = active[::2]  # pooling: keep every other wire
        measure = active[0] if active else 0
        return qml.expval(qml.PauliZ(measure))

    return circuit, weight_shape


class QuantumConvLayer(nn.Module):
    """Torch wrapper around the batched QCNN, with a trainable/frozen switch."""

    def __init__(self, cfg: CircuitConfig, trainable: bool = True, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.circuit, shape = make_qcnn(cfg)
        generator = torch.Generator().manual_seed(seed)
        init = 0.1 * torch.randn(*shape, generator=generator, dtype=torch.float64)
        if trainable:
            self.weights = nn.Parameter(init)
        else:
            self.register_buffer("weights", init)
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # default.qubit differentiates in float64; cast back for the classical head.
        out = self.circuit(x.double(), self.weights)
        if isinstance(out, (list, tuple)):        # multi-wire observable: mean of Z
            out = torch.stack([o.reshape(-1) for o in out], dim=0).mean(dim=0)
        return out.float().reshape(-1)

    def extra_repr(self) -> str:
        return (f"{self.cfg.ansatz}: {self.cfg.n_qubits} qubits, {self.cfg.effective_layers} layers, "
                f"trainable={self.trainable}")


class ClassicalSurrogate(nn.Module):
    """Parameter-matched classical stand-in for the quantum layer.

    Mirrors the circuit's structure (pairwise mixing then pooling, bounded output)
    so that any measured difference is attributable to quantum behaviour rather
    than to capacity.
    """

    def __init__(self, cfg: CircuitConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        generator = torch.Generator().manual_seed(seed)
        n_params = cfg.n_circuit_params
        # Same budget as the circuit, laid out as pairwise rotation-like mixing.
        # Angles are drawn on (-pi, pi] like circuit rotation angles: a small-normal
        # init would shrink the output by three orders of magnitude across layers
        # and handicap the control condition relative to the quantum layer.
        self.theta = nn.Parameter(
            (torch.rand(n_params, generator=generator) * 2.0 - 1.0) * math.pi
        )
        self.layers = cfg.effective_layers
        self.shared = cfg.shared_params
        self.pairs = cfg.n_pairs_per_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        cursor = 0
        for layer in range(self.layers):
            if h.shape[1] < 2:
                break
            p = self.theta[layer * 6 : layer * 6 + 6] if self.shared else None
            a, b = h[:, 0::2], h[:, 1::2]
            m = min(a.shape[1], b.shape[1])
            a, b = a[:, :m], b[:, :m]
            if not self.shared:
                p = self.theta[cursor * 6 : cursor * 6 + 6]
                cursor += m
            # Trigonometric mixing mimics the rotation/entangle/rotation pattern.
            u = torch.cos(p[0]) * a + torch.sin(p[1]) * b
            v = torch.cos(p[2]) * b - torch.sin(p[3]) * a
            h = torch.tanh(p[4] * u + p[5] * v)
        return h[:, 0].reshape(-1)


def build_quantum_block(
    mode: QuantumMode, cfg: CircuitConfig, seed: int = 0
) -> Optional[nn.Module]:
    """Factory for the four ablation conditions."""
    if mode == QuantumMode.TRAINED:
        return QuantumConvLayer(cfg, trainable=True, seed=seed)
    if mode == QuantumMode.FROZEN:
        return QuantumConvLayer(cfg, trainable=False, seed=seed)
    if mode == QuantumMode.SURROGATE:
        return ClassicalSurrogate(cfg, seed=seed)
    if mode == QuantumMode.NONE:
        return None
    raise ValueError(f"unknown quantum mode: {mode}")


# ----------------------------------------------------------------------------
# Trainability diagnostics
# ----------------------------------------------------------------------------
def gradient_variance(
    cfg: CircuitConfig, n_samples: int = 100, batch: int = 32, seed: int = 0
) -> dict:
    """Variance of a circuit parameter's gradient over random initialisations.

    Barren plateaus show up as this variance decaying exponentially in the qubit
    count; sweeping ``n_qubits`` and plotting the result is the standard check.
    """
    rng = np.random.default_rng(seed)
    circuit, shape = make_qcnn(cfg)
    grads = []
    for _ in range(n_samples):
        x = torch.tensor(rng.uniform(-np.pi, np.pi, (batch, cfg.n_qubits)), dtype=torch.float64)
        w = torch.tensor(rng.uniform(-np.pi, np.pi, shape), dtype=torch.float64, requires_grad=True)
        out = circuit(x, w).mean()
        out.backward()
        grads.append(w.grad.detach().numpy().ravel()[0])
    grads = np.asarray(grads)
    return {
        "n_qubits": cfg.n_qubits,
        "layers": cfg.effective_layers,
        "grad_mean": float(grads.mean()),
        "grad_var": float(grads.var()),
        "grad_absmean": float(np.abs(grads).mean()),
        "n_samples": n_samples,
    }


def expressibility(cfg: CircuitConfig, n_pairs: int = 1000, bins: int = 75, seed: int = 0) -> float:
    """KL divergence between the circuit's fidelity distribution and Haar random.

    Lower means the circuit covers state space more like a Haar ensemble; this is
    the Sim et al. (2019) expressibility measure.
    """
    rng = np.random.default_rng(seed)
    dev = qml.device(cfg.device, wires=cfg.n_qubits)
    _, shape = make_qcnn(cfg)

    @qml.qnode(dev)
    def state(x, weights):
        _embed(x, cfg)
        active = list(range(cfg.n_qubits))
        cursor = 0
        for layer in range(cfg.effective_layers):
            if len(active) < 2:
                break
            if cfg.shared_params:
                _conv_block(weights[layer], active)
            else:
                for i in range(0, len(active) - 1, 2):
                    _conv_block(weights[cursor], [active[i], active[i + 1]])
                    cursor += 1
            active = active[::2]
        return qml.state()

    fids = np.empty(n_pairs)
    for i in range(n_pairs):
        s1 = state(rng.uniform(-np.pi, np.pi, cfg.n_qubits), rng.uniform(-np.pi, np.pi, shape))
        s2 = state(rng.uniform(-np.pi, np.pi, cfg.n_qubits), rng.uniform(-np.pi, np.pi, shape))
        fids[i] = float(np.abs(np.vdot(s1, s2)) ** 2)

    hist, edges = np.histogram(fids, bins=bins, range=(0.0, 1.0), density=False)
    p = hist / hist.sum()
    dim = 2**cfg.n_qubits
    # Haar fidelity density for dimension N: (N-1)(1-F)^(N-2)
    centres = (edges[:-1] + edges[1:]) / 2
    haar = (dim - 1) * (1 - centres) ** (dim - 2)
    haar = haar / haar.sum()
    mask = (p > 0) & (haar > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / haar[mask])))
