from qulacs.quantum_operator import create_quantum_operator_from_openfermion_file
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.observable import create_observable_from_openfermion_text
from qulacs import QuantumState
from format.gen_of_afm import gen_afm

list_n = [2, 3, 4]
with open("results.txt", "w") as file:
    file.write("System Size\tEigenvalue\n")  # Write the header
    for n in list_n:
        of_text = gen_afm(n)
        operator = create_observable_from_openfermion_text(of_text)
        n_qubit = operator.get_qubit_count()

        state = QuantumState(n_qubit)
        state.set_Haar_random_state(0)
        value = operator.solve_ground_state_eigenvalue_by_lanczos_method(
            state, 50)

        # Write the system size and eigenvalue to the file
        file.write(f"{n_qubit}\t{value}\n")
