"""
cuStateVec runner for state-vector benchmark: run the same circuit spec on GPU
using cuQuantum cuStateVec, with sync and optional memory measurement.

Gate matrices are either taken from the built-in set or loaded from gates_config.json,
so all gates defined in config are supported (including parametric: RX, RY, RZ, ZZ, U, HI, HI_XX, etc.).

Requires: cuQuantum Python (e.g. pip install cuquantum-cu12), CuPy, NVIDIA GPU.
If cuQuantum or CuPy is not available, run_custatevec() returns (None, None) and
the main benchmark script can skip cuStateVec.
"""

import json
import os
import sys
import time

_bench_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_bench_dir, ".."))
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

# Optional imports: support cuQuantum 26.x (bindings) and older (custatevec)
try:
    import cupy as cp
    try:
        from cuquantum.bindings import custatevec as cusv
        import cuquantum
        cudaTypes = cuquantum.cudaDataType
    except ImportError:
        import cuquantum.custatevec as cusv
        from cuquantum import cudaTypes
    CUSTATEVEC_AVAILABLE = True
except ImportError:
    CUSTATEVEC_AVAILABLE = False
    cp = None
    cusv = None
    cudaTypes = None

import numpy as np

# Built-in gate matrices (complex64, row-major). Kept for speed when not using config.
_GATE_MATRICES = {
    "H": np.array([[0.70710678, 0.70710678], [0.70710678, -0.70710678]], dtype=np.complex64),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
    "CNOT": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64
    ),
    "CZ": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex64
    ),
    "SWAP": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64
    ),
}

# Gates config (lazy-loaded from gates_config.json) for full gate set support
_CONFIG_GATES = None
_CONFIG_PATH_DEFAULT = os.path.join(_bench_dir, "..", "configs", "gates_config.json")


def _load_gates_config(config_path=None):
    global _CONFIG_GATES
    if _CONFIG_GATES is not None:
        return
    path = config_path or _CONFIG_PATH_DEFAULT
    if not os.path.isfile(path):
        _CONFIG_GATES = []
        return
    with open(path) as f:
        data = json.load(f)
    _CONFIG_GATES = data.get("gates", [])


def _parse_complex_matrix(matrix_list):
    """Parse matrix from config (strings like 'j', '-1j') into np.array complex64."""
    def parse_cell(x):
        if isinstance(x, str):
            if x == "j":
                return 1j
            if x == "-j":
                return -1j
            if x.endswith("j"):
                return complex(x)
            return float(x)
        if isinstance(x, (int, float)):
            return complex(x)
        return x
    rows = [[parse_cell(x) for x in row] for row in matrix_list]
    return np.array(rows, dtype=np.complex64)


def _eval_matrix_func(func_str, param_names, params):
    """Evaluate matrix_func from config with given params; return np.array complex64."""
    namespace = {
        "cos": np.cos,
        "sin": np.sin,
        "exp": np.exp,
        "pi": np.pi,
        "j": 1j,
        "1j": 1j,
        "np": np,
    }
    func_str = func_str.strip()
    if not func_str.startswith("lambda"):
        raise ValueError("matrix_func must be a lambda expression")
    fn = eval(func_str, namespace)
    if len(params) != len(param_names):
        raise ValueError(f"Gate expects {len(param_names)} params, got {len(params)}")
    args = [float(p) for p in params]
    result = fn(*args)
    return np.array(result, dtype=np.complex64)


def _matrix_from_config(gate_name, params, use_cusv_layout):
    """Build gate matrix from gates_config.json (after _load_gates_config)."""
    gate_info = None
    for g in _CONFIG_GATES:
        if g["name"] == gate_name:
            gate_info = g
            break
    if gate_info is None:
        return None
    n_qubits = gate_info["num_qubits"]
    if gate_info.get("is_parametric"):
        if not params:
            raise ValueError(f"Parametric gate {gate_name} requires parameters")
        raw = _eval_matrix_func(
            gate_info["matrix_func"],
            gate_info.get("param_names", []),
            params,
        )
    else:
        raw = _parse_complex_matrix(gate_info["matrix"])
    if use_cusv_layout and n_qubits >= 2:
        raw = _matrix_fastsim_to_cusv_layout(raw, n_qubits)
    return raw


def _matrix_fastsim_to_cusv_layout(matrix: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Convert gate matrix from FastSim convention to cuStateVec convention.
    FastSim: row/col index = first_qubit * 2^(n-1) + ... (MSB first).
    cuStateVec: row/col index = first_target * 2^0 + ... (LSB first).
    So new[i, j] = old[bit_reverse(i, n), bit_reverse(j, n)].
    """
    n = matrix.shape[0]
    if n == 1:
        return matrix.copy()
    n_bits = int(np.log2(n))
    rev = [0] * n
    for i in range(n):
        r = 0
        for j in range(n_bits):
            r = (r << 1) | ((i >> j) & 1)
        rev[i] = r
    return matrix[np.ix_(rev, rev)].copy()


def _get_matrix_for_gate(gate_name: str, params=None, use_cusv_layout: bool = True):
    """Return 2x2, 4x4 or 8x8 complex64 matrix for the gate. If use_cusv_layout, convert to cuStateVec layout (LSB = first target)."""
    if gate_name in _GATE_MATRICES:
        raw = _GATE_MATRICES[gate_name].copy()
        n = int(np.log2(raw.shape[0]))
        if use_cusv_layout and n >= 2:
            return _matrix_fastsim_to_cusv_layout(raw, n)
        return raw
    _load_gates_config()
    from_config = _matrix_from_config(gate_name, params or [], use_cusv_layout)
    if from_config is not None:
        return from_config
    raise ValueError(f"Unsupported gate for cuStateVec: {gate_name}")


def _num_qubits_from_spec(spec):
    max_q = -1
    for g in spec:
        for q in g["qubits"]:
            if q > max_q:
                max_q = q
    return max_q + 1


def run_custatevec(
    circuit_spec,
    warmup=5,
    repeat=20,
    measure_memory=False,
):
    """
    Run cuStateVec on the given circuit spec; return times and optional memory.

    Args:
        circuit_spec: List of gate dicts (gate_name, qubits, optional parameters).
        warmup: Number of warmup full-circuit passes.
        repeat: Number of timed passes.
        measure_memory: If True, report peak GPU memory (if available).

    Returns:
        times: List[float] of wall-clock seconds per full circuit, or None if unavailable.
        mem_bytes: Optional[int], peak GPU memory in bytes, or None.
    """
    if not CUSTATEVEC_AVAILABLE:
        return None, None

    n_qubits = _num_qubits_from_spec(circuit_spec)
    dim = 2**n_qubits

    # State vector on GPU (CuPy), initial state |0⟩
    state_vector = cp.zeros(dim, dtype=cp.complex64)
    state_vector[0] = 1.0

    handle = cusv.create()
    try:
        # Workspace size for apply_matrix (max over all gates in spec)
        workspace_size = 0
        for gate in circuit_spec:
            name = gate["gate_name"]
            n_targets = len(gate["qubits"])
            matrix = _get_matrix_for_gate(name, gate.get("parameters"))
            layout = getattr(cusv.MatrixLayout, "ROW_MAJOR", None) or cusv.MatrixLayout.ROW
            compute_type = getattr(cusv.ComputeType, "COMPUTE_32F", None) or cudaTypes.CUDA_C_32F
            size = cusv.apply_matrix_get_workspace_size(
                handle,
                cudaTypes.CUDA_C_32F,
                n_qubits,
                matrix.ctypes.data,
                cudaTypes.CUDA_C_32F,
                layout,
                0,  # adjoint
                n_targets,
                0,  # n_controls
                compute_type,
            )
            workspace_size = max(workspace_size, size)
        workspace = cp.cuda.alloc(workspace_size) if workspace_size > 0 else None
        workspace_ptr = workspace.ptr if workspace is not None else 0

        def run_circuit():
            state_vector_cp = cp.zeros(dim, dtype=cp.complex64)
            state_vector_cp[0] = 1.0
            sv_ptr = state_vector_cp.data.ptr
            for gate in circuit_spec:
                name = gate["gate_name"]
                qubits = gate["qubits"]
                matrix = _get_matrix_for_gate(name, gate.get("parameters"))
                n_targets = len(qubits)
                # targets: qubit indices (cuStateVec uses little-endian; we apply to same indices)
                layout = getattr(cusv.MatrixLayout, "ROW_MAJOR", None) or cusv.MatrixLayout.ROW
                compute_type = getattr(cusv.ComputeType, "COMPUTE_32F", None) or cudaTypes.CUDA_C_32F
                cusv.apply_matrix(
                    handle,
                    sv_ptr,
                    cudaTypes.CUDA_C_32F,
                    n_qubits,
                    matrix.ctypes.data,
                    cudaTypes.CUDA_C_32F,
                    layout,
                    0,  # adjoint
                    qubits,
                    n_targets,
                    [],  # controls
                    [],
                    0,  # n_controls
                    compute_type,
                    workspace_ptr,
                    workspace_size,
                )
            return state_vector_cp

        if measure_memory:
            cp.cuda.Stream.null.synchronize()
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            # Peak stat may not be in CuPy; we skip or use driver API. For simplicity we skip.
            mem_bytes = None
        else:
            mem_bytes = None

        # Warmup
        for _ in range(warmup):
            run_circuit()
        cp.cuda.Stream.null.synchronize()

        # Timed runs
        times = []
        for _ in range(repeat):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            run_circuit()
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        return times, mem_bytes
    finally:
        cusv.destroy(handle)


def run_custatevec_get_final_state(circuit_spec):
    """
    Run the circuit once and return the final state vector as a numpy array (complex64).
    Returns None if cuStateVec is not available. For use in verification against FastSim.
    """
    if not CUSTATEVEC_AVAILABLE:
        return None
    n_qubits = _num_qubits_from_spec(circuit_spec)
    dim = 2**n_qubits
    handle = cusv.create()
    try:
        workspace_size = 0
        for gate in circuit_spec:
            name = gate["gate_name"]
            n_targets = len(gate["qubits"])
            matrix = _get_matrix_for_gate(name, gate.get("parameters"))
            layout = getattr(cusv.MatrixLayout, "ROW_MAJOR", None) or cusv.MatrixLayout.ROW
            compute_type = getattr(cusv.ComputeType, "COMPUTE_32F", None) or cudaTypes.CUDA_C_32F
            size = cusv.apply_matrix_get_workspace_size(
                handle,
                cudaTypes.CUDA_C_32F,
                n_qubits,
                matrix.ctypes.data,
                cudaTypes.CUDA_C_32F,
                layout,
                0,
                n_targets,
                0,
                compute_type,
            )
            workspace_size = max(workspace_size, size)
        workspace = cp.cuda.alloc(workspace_size) if workspace_size > 0 else None
        workspace_ptr = workspace.ptr if workspace is not None else 0

        state_vector = cp.zeros(dim, dtype=cp.complex64)
        state_vector[0] = 1.0
        sv_ptr = state_vector.data.ptr
        for gate in circuit_spec:
            name = gate["gate_name"]
            qubits = gate["qubits"]
            matrix = _get_matrix_for_gate(name, gate.get("parameters"))
            n_targets = len(qubits)
            layout = getattr(cusv.MatrixLayout, "ROW_MAJOR", None) or cusv.MatrixLayout.ROW
            compute_type = getattr(cusv.ComputeType, "COMPUTE_32F", None) or cudaTypes.CUDA_C_32F
            cusv.apply_matrix(
                handle,
                sv_ptr,
                cudaTypes.CUDA_C_32F,
                n_qubits,
                matrix.ctypes.data,
                cudaTypes.CUDA_C_32F,
                layout,
                0,
                qubits,
                n_targets,
                [],
                [],
                0,
                compute_type,
                workspace_ptr,
                workspace_size,
            )
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(state_vector)
    finally:
        cusv.destroy(handle)


def main():
    import argparse
    from circuit_generator import load_circuit_spec

    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="Path to circuit spec JSON")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--memory", action="store_true")
    args = parser.parse_args()

    if not CUSTATEVEC_AVAILABLE:
        print("cuStateVec not available (install cuQuantum and CuPy with CUDA)", file=sys.stderr)
        return 1

    spec = load_circuit_spec(args.spec)
    times, mem = run_custatevec(
        spec,
        warmup=args.warmup,
        repeat=args.repeat,
        measure_memory=args.memory,
    )

    if times is None:
        return 1
    times_arr = np.array(times)
    print(f"cuStateVec: mean={times_arr.mean():.6f}s std={times_arr.std():.6f}s (n={len(times)})")
    if mem is not None:
        print(f"Peak GPU memory: {mem / 2**20:.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
