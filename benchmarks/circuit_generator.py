"""
Reproducible circuit generator for FastSim vs cuStateVec benchmarking.

Generates a deterministic circuit (gate types + qubit indices, and params for
parametric gates) from num_qubits, num_gates, seed, and gate_set.
Output format matches Circuit.from_json() gate sequence: list of
{"gate_name": str, "qubits": list, "parameters": optional}.

When config_path is provided, gate set and param counts are loaded from
gates_config.json so all gates defined there are supported.
"""

import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple

# Fallback when not loading from config: small non-parametric set + param lists
DEFAULT_GATE_SET = ["H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"]
PARAMETRIC_1Q = ["RX", "RY", "RZ"]
PARAMETRIC_2Q = ["ZZ"]


def _default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "configs", "gates_config.json")


def load_gate_info_from_config(
    config_path: str,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Load gate names, param count per gate, and num_qubits per gate from gates_config.json.

    Returns:
        gate_names: list of all gate names
        param_count_by_name: gate_name -> number of parameters (0 for non-parametric)
        num_qubits_by_name: gate_name -> number of qubits the gate acts on
    """
    with open(config_path) as f:
        config = json.load(f)
    gate_names = []
    param_count_by_name = {}
    num_qubits_by_name = {}
    for g in config["gates"]:
        name = g["name"]
        gate_names.append(name)
        num_qubits_by_name[name] = g["num_qubits"]
        if g.get("is_parametric"):
            param_count_by_name[name] = len(g.get("param_names", []))
        else:
            param_count_by_name[name] = 0
    return gate_names, param_count_by_name, num_qubits_by_name


def _gate_num_qubits(name: str, num_qubits_by_name: Optional[Dict[str, int]] = None) -> int:
    """Return number of qubits the gate acts on."""
    if num_qubits_by_name and name in num_qubits_by_name:
        return num_qubits_by_name[name]
    if name in ("CNOT", "CZ", "SWAP", "CY", "CS", "CSWAP", "FREDKIN", "ISWAP"):
        return 2
    if name in ("TOFFOLI",):
        return 3
    if name in ("HI", "HI_XX", "HI_YY", "HI_ZZ", "ZZ"):
        return 2
    return 1


def generate_circuit_spec(
    num_qubits: int,
    num_gates: int,
    seed: int,
    gate_set: Optional[List[str]] = None,
    config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a deterministic circuit specification.

    Args:
        num_qubits: Number of qubits.
        num_gates: Number of gates to generate.
        seed: Random seed for reproducibility.
        gate_set: List of gate names. If None and config_path is set, uses all gates from config.
        config_path: Path to gates_config.json. If set, gate set and param counts are loaded
                     from config (so all gates in config are supported).

    Returns:
        List of gate dicts: {"gate_name": str, "qubits": list, "parameters": optional}.
    """
    rng = random.Random(seed)
    two_pi = 2 * 3.14159265359

    param_count_by_name: Dict[str, int] = {}
    num_qubits_by_name: Optional[Dict[str, int]] = None

    if config_path and os.path.isfile(config_path):
        _names, param_count_by_name, num_qubits_by_name = load_gate_info_from_config(config_path)
        if gate_set is None:
            gate_set = _names
    if gate_set is None:
        gate_set = DEFAULT_GATE_SET

    spec = []
    for _ in range(num_gates):
        name = rng.choice(gate_set)
        nq = _gate_num_qubits(name, num_qubits_by_name)
        if nq == 1:
            qubits = [rng.randint(0, num_qubits - 1)]
        else:
            qubits = rng.sample(range(num_qubits), nq)

        gate_entry: Dict[str, Any] = {"gate_name": name, "qubits": qubits}
        nparams = param_count_by_name.get(name)
        if nparams is None:
            # Fallback when not from config
            if name in PARAMETRIC_1Q:
                nparams = 1
            elif name in PARAMETRIC_2Q:
                nparams = 2
            else:
                nparams = 0
        if nparams > 0:
            gate_entry["parameters"] = [rng.uniform(0, two_pi) for _ in range(nparams)]
        spec.append(gate_entry)

    return spec


def generate_cover_all_gates_spec(
    config_path: str,
    num_qubits: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a circuit spec that contains **at least one application of each gate**
    defined in gates_config.json. Used to verify that every gate type is tested
    (e.g. FastSim vs cuStateVec consistency for all config gates).

    Args:
        config_path: Path to gates_config.json.
        num_qubits: Number of qubits (must be >= max gate num_qubits in config, e.g. 3 for TOFFOLI).
        seed: Random seed for qubit indices and parameters.

    Returns:
        List of gate dicts; order is one gate per config gate type (then optionally
        a second pass for 2nd application per gate, etc.). Gate names appear in
        config order.
    """
    gate_names, param_count_by_name, num_qubits_by_name = load_gate_info_from_config(
        config_path
    )
    rng = random.Random(seed)
    two_pi = 2 * 3.14159265359
    spec = []
    for name in gate_names:
        nq = num_qubits_by_name[name]
        if num_qubits < nq:
            raise ValueError(
                f"num_qubits={num_qubits} is less than gate {name} num_qubits={nq}"
            )
        if nq == 1:
            qubits = [rng.randint(0, num_qubits - 1)]
        else:
            qubits = rng.sample(range(num_qubits), nq)
        gate_entry: Dict[str, Any] = {"gate_name": name, "qubits": qubits}
        nparams = param_count_by_name.get(name, 0)
        if nparams > 0:
            gate_entry["parameters"] = [rng.uniform(0, two_pi) for _ in range(nparams)]
        spec.append(gate_entry)
    return spec


def save_circuit_spec(spec: List[Dict[str, Any]], path: str) -> None:
    """Save circuit spec to JSON file."""
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def load_circuit_spec(path: str) -> List[Dict[str, Any]]:
    """Load circuit spec from JSON file."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--num-gates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--gate-set", type=str, nargs="*", default=None)
    parser.add_argument("--config", type=str, nargs="?", const="", default=None, metavar="PATH", help="use gates from config: no arg = default config path, or give path to gates_config.json")
    args = parser.parse_args()
    config_path = None
    if args.config is not None:
        config_path = args.config if args.config else _default_config_path()
    if config_path and not os.path.isfile(config_path):
        config_path = None
    gate_set = args.gate_set if args.gate_set else None
    spec = generate_circuit_spec(args.num_qubits, args.num_gates, args.seed, gate_set, config_path)
    if args.output:
        save_circuit_spec(spec, args.output)
        print(f"Saved {len(spec)} gates to {args.output}")
    else:
        print(json.dumps(spec, indent=2))
