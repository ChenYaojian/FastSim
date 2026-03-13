"""
Main entry for FastSim vs cuStateVec state-vector benchmark.

Usage:
  python benchmarks/bench_statevector_vs_custatevec.py --num-qubits 12 --num-gates 500
  python benchmarks/bench_statevector_vs_custatevec.py --spec path/to/circuit_spec.json

Generates a deterministic circuit (or loads from JSON), runs FastSim and optionally
cuStateVec on the same workload, and outputs mean ± std time and optional GPU memory.
"""

import argparse
import json
import os
import sys

_bench_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_bench_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

from circuit_generator import (
    generate_circuit_spec,
    generate_cover_all_gates_spec,
    load_circuit_spec,
    save_circuit_spec,
    _default_config_path,
)
from run_fastsim import run_fastsim, run_fastsim_get_final_state
from run_custatevec import run_custatevec, run_custatevec_get_final_state, CUSTATEVEC_AVAILABLE
from verify_consistency import verify_consistency


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FastSim vs cuStateVec on same circuit workload."
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=12,
        help="Number of qubits (used when generating circuit)",
    )
    parser.add_argument(
        "--num-gates",
        type=int,
        default=500,
        help="Number of gates (used when generating circuit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for circuit generation",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Path to circuit spec JSON (if set, overrides num-qubits/num-gates/seed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to gates_config.json; if set, generated circuit uses all gates from config (default: project configs/gates_config.json)",
    )
    parser.add_argument(
        "--gate-set",
        type=str,
        nargs="*",
        default=None,
        help="Restrict gate set for generation (e.g. H X CNOT); overrides --config when generating",
    )
    parser.add_argument(
        "--cover-all-gates",
        action="store_true",
        help="Generate a circuit with exactly one application of each gate in config (ensures every gate type is tested); uses --config path, --num-qubits (min 3), --seed",
    )
    parser.add_argument(
        "--save-spec",
        type=str,
        default=None,
        help="If set, save generated spec to this path",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup runs per backend",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Number of timed runs per backend",
    )
    parser.add_argument(
        "--no-custatevec",
        action="store_true",
        help="Skip cuStateVec (e.g. when not installed)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Report peak GPU memory (FastSim only for now)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results to this JSON file",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device for FastSim (default: cuda)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify FastSim and cuStateVec final states agree (bit-order aligned)",
    )
    parser.add_argument(
        "--verify-max-qubits",
        type=int,
        default=20,
        help="Only run verification when num_qubits <= this (default: 20)",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for state comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for state comparison (default: 1e-4)",
    )
    args = parser.parse_args()

    def _num_qubits_from_spec(s):
        return max(q for g in s for q in g["qubits"]) + 1

    if args.cover_all_gates:
        config_path = args.config or _default_config_path()
        if not os.path.isfile(config_path):
            raise SystemExit(f"Cover-all-gates requires config file; not found: {config_path}")
        num_qubits = max(args.num_qubits, 3)  # TOFFOLI needs 3 qubits
        spec = generate_cover_all_gates_spec(config_path, num_qubits, args.seed)
        num_qubits = _num_qubits_from_spec(spec)
        num_gates = len(spec)
        gate_types = list(dict.fromkeys(g["gate_name"] for g in spec))
        print(f"Cover-all-gates: {num_gates} gates ({len(gate_types)} types), {num_qubits} qubits (seed={args.seed})")
        print(f"  Gate types tested: {', '.join(gate_types)}")
        if args.save_spec:
            save_circuit_spec(spec, args.save_spec)
            print(f"  Saved spec to {args.save_spec}")
    elif args.spec:
        spec = load_circuit_spec(args.spec)
        num_qubits = _num_qubits_from_spec(spec)
        num_gates = len(spec)
        print(f"Loaded spec: {num_qubits} qubits, {num_gates} gates from {args.spec}")
    else:
        config_path = args.config or _default_config_path()
        if not os.path.isfile(config_path):
            config_path = None
        spec = generate_circuit_spec(
            args.num_qubits,
            args.num_gates,
            args.seed,
            gate_set=args.gate_set,
            config_path=config_path,
        )
        num_qubits = _num_qubits_from_spec(spec)  # actual qubits used by spec
        num_gates = args.num_gates
        print(f"Generated circuit: {num_qubits} qubits, {num_gates} gates (seed={args.seed})")
        if args.save_spec:
            save_circuit_spec(spec, args.save_spec)
            print(f"Saved spec to {args.save_spec}")

    device = __import__("torch").device(args.device)
    run_cusv = not args.no_custatevec and CUSTATEVEC_AVAILABLE

    results = {}

    # FastSim
    print("\n--- FastSim ---")
    times_fastsim, mem_fastsim = run_fastsim(
        spec,
        device=device,
        warmup=args.warmup,
        repeat=args.repeat,
        measure_memory=args.memory,
    )
    import numpy as np
    t = np.array(times_fastsim)
    results["fastsim"] = {
        "mean_s": float(t.mean()),
        "std_s": float(t.std()),
        "n": len(times_fastsim),
        "peak_memory_bytes": int(mem_fastsim) if mem_fastsim is not None else None,
    }
    print(f"  mean = {results['fastsim']['mean_s']:.6f} s  std = {results['fastsim']['std_s']:.6f} s")

    # cuStateVec
    if run_cusv:
        print("\n--- cuStateVec ---")
        times_cusv, mem_cusv = run_custatevec(
            spec,
            warmup=args.warmup,
            repeat=args.repeat,
            measure_memory=args.memory,
        )
        if times_cusv is not None:
            t2 = np.array(times_cusv)
            results["custatevec"] = {
                "mean_s": float(t2.mean()),
                "std_s": float(t2.std()),
                "n": len(times_cusv),
                "peak_memory_bytes": int(mem_cusv) if mem_cusv is not None else None,
            }
            print(f"  mean = {results['custatevec']['mean_s']:.6f} s  std = {results['custatevec']['std_s']:.6f} s")
        else:
            results["custatevec"] = None
            print("  (run failed)")
    else:
        if not args.no_custatevec and not CUSTATEVEC_AVAILABLE:
            print("\n--- cuStateVec --- (skipped: cuQuantum/CuPy not available)")
        results["custatevec"] = None

    # Optional: verify both backends produce the same final state
    if args.verify and run_cusv and num_qubits <= args.verify_max_qubits:
        print("\n--- Result consistency verification ---")
        try:
            state_fastsim = run_fastsim_get_final_state(spec, device)
            state_cusv = run_custatevec_get_final_state(spec)
            if state_cusv is None:
                print("  Could not get cuStateVec final state, skipping verification")
                results["verify"] = {"skipped": "custatevec state unavailable"}
            else:
                ver = verify_consistency(
                    state_fastsim,
                    state_cusv,
                    num_qubits,
                    atol=args.verify_atol,
                    rtol=args.verify_rtol,
                )
                results["verify"] = ver
                print(f"  {ver['message']}")
                if not ver["passed"]:
                    print("  Note: States still differ after bit-order alignment.")
                    if num_gates > 100 or num_qubits > 10:
                        print("  For large circuits, FAIL is often due to complex64 accumulation. Try --num-gates 20 (or smaller) to confirm agreement.")
        except Exception as e:
            print(f"  Verification error: {e}")
            results["verify"] = {"error": str(e)}
    elif args.verify and (not run_cusv or num_qubits > args.verify_max_qubits):
        msg = "Verification skipped: " + (
            "cuStateVec not run" if not run_cusv else f"num_qubits={num_qubits} > --verify-max-qubits={args.verify_max_qubits}"
        )
        print("\n--- Result consistency verification ---")
        print(f"  {msg}")
        results["verify"] = {"skipped": msg}

    results["config"] = {
        "num_qubits": num_qubits,
        "num_gates": num_gates,
        "seed": args.seed if not args.spec else None,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "device": args.device,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
