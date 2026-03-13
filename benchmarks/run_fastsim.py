"""
FastSim runner for state-vector benchmark: run a circuit spec on GPU with
sync and optional memory measurement. Returns list of wall-clock times (seconds)
and optionally peak GPU memory in bytes.
"""

import os
import sys
import time

# Ensure project root is on path so we can import fastsim and load config
_bench_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_bench_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

import torch

from circuit_generator import load_circuit_spec
from fastsim.circuit import Circuit, load_gates_from_config

CONFIG_PATH = os.path.join(_project_root, "configs", "gates_config.json")


def _num_qubits_from_spec(spec):
    """Infer number of qubits from circuit spec."""
    max_q = -1
    for g in spec:
        for q in g["qubits"]:
            if q > max_q:
                max_q = q
    return max_q + 1


def _build_circuit_from_spec(spec, device):
    """Build a FastSim Circuit from spec on the given device."""
    load_gates_from_config(CONFIG_PATH)
    num_qubits = _num_qubits_from_spec(spec)
    circuit = Circuit(num_qubits, device=device)

    for gate in spec:
        gate_name = gate["gate_name"]
        qubits = gate["qubits"]
        params = gate.get("parameters", None)
        if params is not None:
            params = torch.tensor(params, dtype=torch.float32, device=device)
        circuit.add_gate(gate_name, qubits, params)

    return circuit


def run_fastsim(
    circuit_spec,
    device=None,
    warmup=5,
    repeat=20,
    measure_memory=False,
):
    """
    Run FastSim on the given circuit spec; return times and optional memory.

    Args:
        circuit_spec: List of gate dicts (from circuit_generator or load_circuit_spec).
        device: torch.device; default cuda if available else cpu.
        warmup: Number of warmup forward passes.
        repeat: Number of timed forward passes.
        measure_memory: If True, reset peak stats and return peak allocated bytes.

    Returns:
        times: List[float] of wall-clock seconds per forward pass.
        mem_bytes: Optional[int], peak GPU memory in bytes (if measure_memory and CUDA).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    circuit = _build_circuit_from_spec(circuit_spec, device)
    num_qubits = circuit.num_qubits
    dim = 2**num_qubits

    # Initial state |0⟩ on same device (avoid CPU->GPU transfer in timed region)
    state = torch.zeros(dim, dtype=torch.complex64, device=device)
    state[0] = 1.0
    state = state.unsqueeze(0)  # [1, 2^n]

    if measure_memory and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        _ = circuit(state)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed runs
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = circuit(state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mem_bytes = None
    if measure_memory and device.type == "cuda":
        mem_bytes = torch.cuda.max_memory_allocated(device)

    return times, mem_bytes


def run_fastsim_get_final_state(circuit_spec, device=None):
    """
    Run the circuit once and return the final state vector as a numpy array (complex64).
    For use in verification against cuStateVec.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circuit = _build_circuit_from_spec(circuit_spec, device)
    num_qubits = circuit.num_qubits
    dim = 2**num_qubits
    state = torch.zeros(dim, dtype=torch.complex64, device=device)
    state[0] = 1.0
    state = state.unsqueeze(0)
    out = circuit(state)
    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
    return out[0].detach().cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="Path to circuit spec JSON (or '-' for stdin)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--memory", action="store_true", help="Report peak GPU memory")
    args = parser.parse_args()

    if args.spec == "-":
        import json
        spec = json.load(sys.stdin)
    else:
        spec = load_circuit_spec(args.spec)

    device = torch.device(args.device)
    times, mem = run_fastsim(
        spec,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
        measure_memory=args.memory,
    )

    import numpy as np
    times_arr = np.array(times)
    print(f"FastSim: mean={times_arr.mean():.6f}s std={times_arr.std():.6f}s (n={len(times)})")
    if mem is not None:
        print(f"Peak GPU memory: {mem / 2**20:.2f} MiB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
