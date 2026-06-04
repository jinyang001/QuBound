import numpy as np
import pandas as pd
from qiskit import Aer, transpile, assemble
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers.models import BackendProperties
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
import supermarq
import json
from qiskit.providers.fake_provider import FakeMumbai
from qiskit.providers.fake_provider import FakeKolkata

# Initialize an empty mapping from gate types to integers for encoding
gate_type_to_int = {}
sim_times = []
num_qubits_after_transpile_list = []
date_list = []

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CARE_STATES = [format(i, '04b') for i in range(2 ** 4)]

print(CARE_STATES)

backend = FakeMumbai()


# 4 qubit GHZ
def create_quantum_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    return qc


from qiskit import transpile


# Example usage


# For circuit information collection:
def gather_circuit_info(qc):
    num_qubits = qc.num_qubits
    num_gates = len(qc.data)
    num_params = sum(len(gate[0].params) for gate in qc.data)
    circuit_depth = qc.depth()

    return num_qubits, num_gates, num_params, circuit_depth


# Generate the dataset
qc = create_quantum_circuit()

num_qubits = qc.num_qubits
qc.draw('mpl')
plt.show()
np.random.seed(None)
# ss
num_qubits_info, num_gates_info, num_params_info, circuit_depth_info = gather_circuit_info(qc)


def add_gate_to_dictionary(gate_name):
    """
    Dynamically adds gate types to gate_type_to_int if not already present.
    """
    if gate_name not in gate_type_to_int:
        gate_type_to_int[gate_name] = len(gate_type_to_int) + 1


def qc_to_vector(qc: QuantumCircuit, properties) -> List[List[List[float]]]:
    coupling_map = backend.configuration().coupling_map
    noise_model = NoiseModel.from_backend_properties(properties)
    basis_gates = noise_model.basis_gates

    transpiled_qc = transpile(qc, coupling_map=coupling_map, basis_gates=basis_gates, backend_properties=properties,
                              optimization_level=0,
                              initial_layout=[0, 1, 4, 7],
                              routing_method='basic')

    # Determine active qubits
    dag: DAGCircuit = circuit_to_dag(transpiled_qc)
    active_qubits = set()
    for gate in dag.topological_op_nodes():
        if gate.name != 'barrier':  # Ignore barriers
            for qubit in gate.qargs:
                active_qubits.add(transpiled_qc.find_bit(qubit).index)
    active_qubits = sorted(list(active_qubits))

    # Update our gate dictionary to include any new gate types
    for gate in dag.topological_op_nodes():
        add_gate_to_dictionary(gate.name)

    def blank_stage():
        # Initialize with Identity gates represented by the number 10
        return [[0, 0, 0, 0, 0] for _ in active_qubits]

    stages = []

    # Iterate through layers of the DAG
    for layer in dag.layers():
        stage = blank_stage()
        for node in layer['graph'].op_nodes():
            gate = node
            # Get qubit indices and gate type
            qubit_indices = [transpiled_qc.find_bit(qubit).index for qubit in gate.qargs]
            gate_type = gate_type_to_int[gate.name]
            gate_param = gate.op.params[0] if gate.op.params else 0  # Extract the gate parameter or use 0 if none

            # Set the gate info for each involved qubit in the stage
            for i, qubit_index in enumerate(qubit_indices):
                if qubit_index in active_qubits:  # Ensure we only consider active qubits
                    t1, t2 = properties.t1(qubit_index), properties.t2(qubit_index)
                    qubit_pos = active_qubits.index(qubit_index)  # Find the position in the active qubits list
                    stage[qubit_pos][0], stage[qubit_pos][1], stage[qubit_pos][2], stage[qubit_pos][
                        3] = gate_type, gate_param, t1, t2

                    if gate.name == 'barrier':
                        stage[qubit_pos][0], stage[qubit_pos][1], stage[qubit_pos][2], stage[qubit_pos][
                            3] = gate_type, gate_param, 0, 0
                    # Include gate error if applicable
                    if gate.name != 'reset':
                        try:
                            if gate.name == 'measure':
                                stage[qubit_pos][4] = properties.readout_error(qubit_index)
                            else:
                                gate_error = properties.gate_error(gate.name, qubit_indices)
                                stage[qubit_pos][4] = gate_error
                        except BackendPropertyError:
                            pass
        stages.append(stage)

    return stages


def preprocess_input(qc_vector):
    processed_vector = []
    for stage in qc_vector:
        stage_array = np.array(stage)
        processed_vector.append(stage_array.tolist())
    return processed_vector


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def pad_data_sequences(data):
    return pad_sequence([torch.tensor(sample) for sample in data], batch_first=True, padding_value=0)


def generate_dataset(num_samples):
    qc = create_quantum_circuit()

    X_data = []
    y_data = []
    hours_list = [0, 6, 12, 18]  # Times during which runs are made
    initial_date = datetime(day=15, month=10, year=2023)  # Starting date without hour

    with open('properties/ibmq_mumbai_properties.json', 'r') as f:
        stored_properties = json.load(f)

    sample_count = 0  # counter to keep track of the number of samples collected
    while sample_count < num_samples:
        if (sample_count) % 100 == 0:
            print(f"Processed {sample_count} samples...")
        for hour in hours_list:  # Loop over desired hours
            if sample_count >= num_samples:  # Break out of inner loop if we've collected enough samples
                break
            t = initial_date.replace(hour=hour)  # set the hour for the current iteration
            timestamp_str = t.strftime('%Y-%m-%d %H:%M:%S')
            properties_data = stored_properties.get(timestamp_str, None)
            if properties_data:
                for key, value in properties_data.items():
                    if isinstance(value, str):
                        try:
                            properties_data[key] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                properties = BackendProperties.from_dict(properties_data)

            try:
                properties = properties

                qc_vector = qc_to_vector(qc, properties)

                input_vector = preprocess_input(qc_vector)

                # Perform a noisy simulation
                start_time = time.time()
                print("start")
                noise_model = NoiseModel.from_backend_properties(properties)

                # Get coupling map from backend
                coupling_map = backend.configuration().coupling_map
                # Get basis gates from noise model
                basis_gates = noise_model.basis_gates
                simulator = Aer.get_backend('qasm_simulator')

                result = execute(qc, simulator, coupling_map=coupling_map,
                                 basis_gates=basis_gates, noise_model=noise_model, shots=1000,
                                 optimization_level=0,
                                 initial_layout=[0, 1, 4, 7],
                                 routing_method='basic').result()
                counts = result.get_counts(qc)

                end_time = time.time()
                sim_time = end_time - start_time
                print("end")
                # Create a target array for all possible states and normalize to create probabilities
                num_possible_states = 2 ** num_qubits
                all_states_counts = [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(num_possible_states)]
                total_counts = sum(all_states_counts)
                all_states_probabilities = [count / total_counts for count in all_states_counts]
                target_probs = [all_states_probabilities[int(state, 2)] for state in CARE_STATES]
                X_data.append(input_vector)
                y_data.append(target_probs)


                sim_times.append(sim_time)
                date_list.append(t)
                sample_count += 1  # Increment the sample count
            except Exception as e:
                print(f"Error at sample {sample_count}: {e}")
                continue
        initial_date -= timedelta(days=1)
    # Call the function
    X_data_padded = pad_data_sequences(X_data)
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
    df_X['num_qubits_after_transpile'] = 4
    df_X['date'] = date_list
    # For Y, since it's a list of lists, it can be directly converted to a DataFrame
    df_Y = pd.DataFrame(Y, columns=[f'state_{state}' for state in CARE_STATES])

    # Combining X and Y DataFrames side by side
    df_combined = pd.concat([df_X, df_Y], axis=1)
    # Sorting the combined DataFrame by date in ascending order
    df_combined = df_combined.sort_values(by='date')

    # Save to CSV
    df_combined.to_csv('datasets/ibmq_mumbai/GHZ4_ibmq_mumbai_test.csv', index=False)

    print("Data generation complete. Dataset saved to 'dataset.csv'.")
