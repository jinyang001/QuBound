import numpy as np
import pandas as pd
from qiskit import Aer, transpile, assemble
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qiskit.dagcircuit import DAGCircuit
from qiskit.visualization import plot_histogram
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, execute, Aer
from qiskit.converters import circuit_to_dag
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.exceptions import BackendPropertyError
from qiskit_aer.noise import NoiseModel
from datetime import datetime, timedelta
from typing import List
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qiskit.compiler import transpile
from datetime import datetime, timedelta
import time

# Initialize an empty mapping from gate types to integers for encoding
gate_type_to_int = {}
sim_times = []
num_qubits_after_transpile_list = []
date_list=[]
N_STATES = 4
CARE_STATES = ["000", "001", "010", "011","100","101","110","111"]  # 3 qubit circuit
# CARE_STATES = ["0100", "1000", "0000", "1100","0111","1011","0011","1111"]  # 4 qubit circuit
# CARE_STATES = ["00000", "00001", "00010", "10001","00111","10011","01100","01101"]  # 5 qubit circuit
provider = IBMProvider()
backend = provider.get_backend('ibm_hanoi')

# Define your Quantum Circuit here
# def create_quantum_circuit():
#     qc = QuantumCircuit(4)
#     qc.rx(1.2, 0)
#     qc.ry(0.7, 1)
#     qc.h(2)
#     qc.cx(0, 1)
#     qc.h(3)
#     qc.cx(2, 3)
#     qc.measure_all()
#     return qc
# def create_quantum_circuit():
#     qc = QuantumCircuit(5)
#
#     # Initial operations
#     qc.h([0, 1, 2])
#     qc.rx(1.3, 3)
#     qc.ry(0.9, 4)
#
#     # Entanglement operations
#     qc.cx(0, 1)
#     qc.cz(2, 3)
#     qc.cy(4, 0)
#
#     # Additional gates
#     qc.rz(0.5, 2)
#     qc.swap(3, 4)
#
#     # Entangle more
#     qc.ccx(0, 1, 2)
#
#     # Some single qubit operations
#     qc.rx(1.1, 0)
#     qc.ry(1.1, 4)
#
#     # End with measurements
#     qc.measure_all()
#
#     return qc
def create_quantum_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


from qiskit_experiments.library import StandardRB
from qiskit import transpile


def create_rb_circuit(num_qubits=2, rb_pattern=[[0, 1]], lengths=[1], num_samples=1, seed=10):
    """
    Create a Randomized Benchmarking circuit.

    Parameters:
    - num_qubits (int): Number of qubits.
    - rb_pattern (list): RB pattern.
    - lengths (list): Lengths of the RB sequences.
    - num_samples (int): Number of samples for each sequence length.
    - seed (int): Random seed.

    Returns:
    - list[QuantumCircuit]: A list of RB circuits.
    """

    # Generate RB circuits (standard)
    rb_exp = StandardRB(
        physical_qubits=list(range(num_qubits)),
        lengths=lengths,
        num_samples=num_samples,
        seed=seed
    )
    rb_circs = rb_exp.circuits()

    return rb_circs


# Example usage
rb_circuits = create_rb_circuit(num_qubits=3, rb_pattern=[0,1,2])
print(rb_circuits[0])  # Print the first RB circuit


# For circuit information collection:
def gather_circuit_info(qc):
    num_qubits = qc.num_qubits
    num_gates = len(qc.data)
    num_params = sum(len(gate[0].params) for gate in qc.data)
    circuit_depth = qc.depth()

    return num_qubits, num_gates, num_params, circuit_depth

# Generate the dataset
qc = create_quantum_circuit()
# qc=rb_circuits[0]
qc.draw('mpl')
plt.show()
# ss
num_qubits = qc.num_qubits

num_qubits_info, num_gates_info, num_params_info, circuit_depth_info = gather_circuit_info(qc)

def add_gate_to_dictionary(gate_name):
    """
    Dynamically adds gate types to gate_type_to_int if not already present.
    """
    if gate_name not in gate_type_to_int:
        gate_type_to_int[gate_name] = len(gate_type_to_int)


def estimate_MACs(transpiled_qc):
    n_qubits = transpiled_qc.num_qubits
    total_MACs = 0
    for gate_data in transpiled_qc.data:
        gate = gate_data[0]
        qubits_involved = gate_data[1:]
        print(gate_data[1:][0])
        print(len(gate_data[1:][0]))
        # print(gate_data[1:])
        # print(len(gate_data[1:]))
        if len(qubits_involved) == 1:  # Single qubit gates
            total_MACs += 2 ** n_qubits
        elif len(qubits_involved) == 2:  # Two qubit gates like CNOT
            total_MACs += 4 ** n_qubits

    return total_MACs


def qc_to_vector(qc: QuantumCircuit, properties) -> List[List[List[float]]]:
    transpiled_qc = transpile(qc, backend=backend,backend_properties=properties, optimization_level=0)
    # transpiled_qc.draw('mpl')
    # plt.show()
    # print(transpiled_qc.depth())
    # sss
    num_qubits = transpiled_qc.num_qubits
    num_qubits_after_transpile_list.append(num_qubits)
    stages = []

    # Update our gate dictionary to include any new gate types
    dag: DAGCircuit = circuit_to_dag(transpiled_qc)
    for gate in dag.topological_op_nodes():
        add_gate_to_dictionary(gate.name)

    # Function to create a blank stage
    def blank_stage():
        return [[0 for _ in range(5)] for _ in range(num_qubits)]

    # Iterate through layers of the DAG
    for layer in dag.layers():
        stage = blank_stage()
        for node in layer['graph'].op_nodes():
            gate = node
            if gate.name == 'barrier':
                continue

            # Get qubit indices and gate type
            qubit_indices = [transpiled_qc.find_bit(qubit).index for qubit in gate.qargs]
            gate_type = gate_type_to_int[gate.name]
            gate_param = gate.op.params[0] if gate.op.params else 0  # Extract the gate parameter or use 0 if none

            # Set the gate info for each involved qubit in the stage
            for i, qubit_index in enumerate(qubit_indices):
                t1, t2 = properties.t1(qubit_index), properties.t2(qubit_index)
                stage[qubit_index][0], stage[qubit_index][1], stage[qubit_index][2], stage[qubit_index][
                    3] = gate_type, gate_param, t1, t2

                # Include gate error if applicable
                if gate.name != 'reset':
                    try:
                        # gate_error = 0
                        if gate.name == 'measure':
                            stage[qubit_index][4] = properties.readout_error(qubit_index)
                            # readout_error = properties.readout_error(qubit_index)
                            # gate_error = readout_error  # Add the readout error to the gate error
                        else:
                            gate_error = properties.gate_error(gate.name, qubit_indices)
                            stage[qubit_index][4] = gate_error
                        # stage[qubit_index][4] = gate_error  # Update the gate error
                    except BackendPropertyError:
                        pass  # Default value is already 0 if no gate error

        stages.append(stage)

    return stages

def preprocess_input(qc_vector):
    processed_vector = []
    for stage in qc_vector:
        stage_array = np.array(stage)
        processed_vector.append(stage_array.tolist())
    return processed_vector


def check_inconsistencies(X_data):
    # Store the length of the first input_vector as the reference length
    reference_length = 1

    # Iterate through X_data and compare each input_vector's length to the reference
    for idx, input_vector in enumerate(X_data):
        if len(input_vector) != reference_length:
            print(f"Inconsistency found at index {idx}. Length: {len(input_vector)}")
            print(input_vector)
            print("=" * 50)  # Print a separator for readability



from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def pad_data_sequences(data):
    return pad_sequence([torch.tensor(sample) for sample in data], batch_first=True, padding_value=0)

def generate_dataset(num_samples):
    # global CARE_STATES
    qc = create_quantum_circuit()
    # qc=rb_circuits[0]
    X_data = []
    y_data = []
    hours_list = [6, 14, 22]  # Times during which runs are made
    initial_date = datetime(day=15, month=8, year=2023)  # Starting date without hour

    sample_count = 0  # counter to keep track of the number of samples collected
    while sample_count < num_samples:
        for hour in hours_list:  # Loop over desired hours
            if sample_count >= num_samples:  # Break out of inner loop if we've collected enough samples
                break

            t = initial_date.replace(hour=hour)  # set the hour for the current iteration
            # print(f"Processing data for datetime: {t}")
            try:
                properties = backend.properties(datetime=t)
                date_list.append(t)
                qc_vector = qc_to_vector(qc, properties)

                input_vector = preprocess_input(qc_vector)

                # Perform a noisy simulation
                start_time = time.time()
                noise_model = NoiseModel.from_backend_properties(properties)
                # print(noise_model)

                # Get coupling map from backend
                coupling_map = backend.configuration().coupling_map
                # Get basis gates from noise model
                basis_gates = noise_model.basis_gates
                simulator = Aer.get_backend('qasm_simulator')
                result = execute(qc, simulator,coupling_map=coupling_map,
                         basis_gates=basis_gates, noise_model=noise_model,shots=10000,optimization_level=0).result()
                counts = result.get_counts(qc)
                # plot_histogram(counts)
                # plt.show()
                # sss
                end_time = time.time()
                sim_time = end_time - start_time
                sim_times.append(sim_time)
                # Create a target array for all possible states and normalize to create probabilities
                num_possible_states = 2 ** num_qubits
                all_states_counts = [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(num_possible_states)]
                total_counts = sum(all_states_counts)
                all_states_probabilities = [count / total_counts for count in all_states_counts]

                # # Pair up states with their counts
                # state_count_pairs = list(
                #     zip([bin(i)[2:].zfill(num_qubits) for i in range(num_possible_states)], all_states_counts))
                #
                # # Sort state-count pairs based on counts in descending order
                # sorted_state_count_pairs = sorted(state_count_pairs, key=lambda x: x[1], reverse=True)
                #
                # # Print sorted states and their counts
                # for state, count in sorted_state_count_pairs:
                #     print(f"State: {state} - Count: {count/total_counts}")
                # sss

                # # Sort states by their probabilities and take the top N_STATES
                # sorted_probs_indexes = np.argsort(all_states_probabilities)[::-1]
                # top_probs = [all_states_probabilities[i] for i in sorted_probs_indexes[:N_STATES]]

                target_probs = [all_states_probabilities[int(state, 2)] for state in CARE_STATES]

                X_data.append(input_vector)
                y_data.append(target_probs)
                sample_count += 1  # Increment the sample count
            except Exception as e:
                print(f"Error at sample {sample_count}: {e}")
        initial_date -= timedelta(days=1)
    # Call the function
    X_data_padded = pad_data_sequences(X_data)
    # check_inconsistencies(X_data_padded)
    return np.array(X_data_padded), np.array(y_data)


if __name__ == "__main__":
    X, Y = generate_dataset(10)

    # Flatten X
    num_samples, _, _, _ = X.shape
    X_flattened = X.reshape(num_samples, -1)  # This flattens the inner dimensions of X

    # Create a DataFrame
    df_X = pd.DataFrame(X_flattened)
    df_X['sim_times'] = sim_times
    df_X['num_qubits_info'] = num_qubits_info
    df_X['num_gates_info'] = num_gates_info
    df_X['num_params_info'] = num_params_info
    df_X['circuit_depth_info'] = circuit_depth_info
    df_X['num_qubits_after_transpile'] = num_qubits_after_transpile_list
    df_X['date'] = date_list
    # For Y, since it's a list of lists, it can be directly converted to a DataFrame
    df_Y = pd.DataFrame(Y, columns=[f'state_{state}' for state in CARE_STATES])
    # df_Y = pd.DataFrame(Y, columns=CARE_STATES)
    # df_Y = pd.DataFrame(Y)

    # Combining X and Y DataFrames side by side
    df_combined = pd.concat([df_X, df_Y], axis=1)
    # Sorting the combined DataFrame by date in ascending order
    df_combined = df_combined.sort_values(by='date')

    # Save to CSV
    df_combined.to_csv('datasets/test.csv', index=False)


    print("Data generation complete. Dataset saved to 'dataset.csv'.")
