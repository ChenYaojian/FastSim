"""
Verify that FastSim and cuStateVec produce the same final state (up to bit ordering).

FastSim uses big-endian state indexing (qubit 0 = MSB of state index).
cuStateVec uses little-endian (qubit 0 = LSB). We convert FastSim state to cuStateVec
order then compare by fidelity and element-wise tolerance.
"""

import numpy as np


def bit_reverse_index(i: int, n_bits: int) -> int:
    """Reverse the lower n_bits of i. Used to convert between big- and little-endian."""
    out = 0
    for j in range(n_bits):
        out = (out << 1) | ((i >> j) & 1)
    return out


def _bit_reverse_indices(n_qubits: int) -> np.ndarray:
    """Vectorized: rev[i] = bit_reverse(i, n_qubits) for i in 0..2**n_qubits-1."""
    n = 2**n_qubits
    rev = np.empty(n, dtype=np.intp)
    for i in range(n):
        rev[i] = bit_reverse_index(i, n_qubits)
    return rev


def fastsim_state_to_cusv_order(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Convert FastSim state vector (big-endian index) to cuStateVec order (little-endian).
    state: 1D array of length 2**n_qubits.
    """
    rev = _bit_reverse_indices(n_qubits)
    return state[rev].copy()


def verify_consistency(
    state_fastsim: np.ndarray,
    state_cusv: np.ndarray,
    n_qubits: int,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict:
    """
    Compare two state vectors after converting FastSim to cuStateVec bit order.

    Returns:
        dict with keys: fidelity (|⟨ψ1|ψ2⟩|²), max_abs_diff, passed (bool), message.
    """
    if state_fastsim.shape != state_cusv.shape:
        return {
            "fidelity": 0.0,
            "max_abs_diff": float("inf"),
            "passed": False,
            "message": f"Shape mismatch: FastSim {state_fastsim.shape} vs cuStateVec {state_cusv.shape}",
        }
    rev = _bit_reverse_indices(n_qubits)
    fastsim_cusv_order = state_fastsim[rev].copy()   # FastSim (big-endian) -> cuStateVec (little-endian)
    overlap = np.vdot(fastsim_cusv_order, state_cusv)
    fidelity = float(np.abs(overlap) ** 2)
    diff = np.abs(fastsim_cusv_order - state_cusv)
    max_abs_diff = float(np.max(diff))
    passed = max_abs_diff <= (atol + rtol * np.max(np.abs(state_cusv)))
    msg = (
        f"fidelity={fidelity:.10f}, max_abs_diff={max_abs_diff:.2e}"
        + (" (PASS)" if passed else " (FAIL)")
    )
    return {
        "fidelity": fidelity,
        "max_abs_diff": max_abs_diff,
        "passed": passed,
        "message": msg,
    }
