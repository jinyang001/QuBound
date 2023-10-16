from qiskit import IBMQ, transpile, Aer, assemble
from qiskit.providers import JobStatus
from qiskit.providers.ibmq import least_busy
import time
import pandas as pd
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
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
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

# Load your IBMQ account
IBMQ.load_account()

# Create the quantum circuit
def create_quantum_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc

qc = create_quantum_circuit()
# qc = QuantumCircuit(3)
# qc.h(0)
# qc.cx(0, 1)
# qc.cx(1, 2)
# qc.measure_all()
#
# qc1 = QuantumCircuit(3)
# qc1.h(0)
# qc1.cx(0, 1)
# qc1.cx(1, 2)
# qc1.measure_all()

# Use ibmq_qasm_simulator for testing
# provider = IBMQ.get_provider(hub='ibm-q-ornl', group='ornl', project='csc517')

provider = IBMProvider(instance='ibm-q-lanl/lanl/quantum-optimiza')
# backend = provider.get_backend('ibmq_qasm_simulator')
backend = provider.get_backend('ibm_perth')
# a=backend.configuration().max_experiments
# print(a)
# ss
# Create a list of the same circuit repeated 80 times
# circuits = [qc, qc1]
circuits = [qc for _ in range(1)]
# Execute the list of circuits in a single job
print("Executing 80 circuits...")
# service = QiskitRuntimeService()
# backend = service.get_backend("ibm_nairobi")
# options = Options()
# options.optimization_level = 0
#
# sampler = Sampler(backend, options=options)
# job = sampler.run(circuits,shots=8192)
# result = job.result()
# print(result)
# ss
# # Transpile circuits for the backend
# transpiled_circuits = transpile(circuits, backend=backend, optimization_level=0)
# qobj = assemble(transpiled_circuits, backend=backend, shots=10000)
# job = backend.run(qobj)
# sss
job = execute(circuits, backend, optimization_level=0, shots=10000)
#
# ss
# Monitor the job status
while job.status() not in [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]:
    print(f"Status @ {time.strftime('%Y-%m-%d %H:%M:%S')}: {job.status().name},"
          f" status={job.status()}")
    time.sleep(60)  # check every 10 seconds

# After executing the job, retrieve the results
results = job.result()
print(results)
sss
quasi_dists = results.quasi_dists

# Prepare a data structure for the desired CSV format
num_qubits = 3
possible_states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
data_for_csv = {state: [] for state in possible_states}

for q_dist in quasi_dists:
    for state in possible_states:
        # Convert numeric state key to binary format (e.g., 0 to '000')
        state_key = int(state, 2)
        prob = q_dist.get(state_key, 0)  # Get the probability, default to 0 if not present
        data_for_csv[state].append(prob)

df = pd.DataFrame(data_for_csv)

# Save to CSV
df.to_csv('ibmq_results.csv', index=False)

print("Results saved to ibmq_results.csv!")
