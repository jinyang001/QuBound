import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit, execute, Aer
import time
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
# Constants
num_epochs = 200
batch_size = 64
CARE_STATES = ["000", "001", "010", "011","100","101","110","111"]
# CARE_STATES = ["000", "111"]
# CARE_STATES = ["000"]
# CARE_STATES = ["0100", "0000"]
N_STATES = len(CARE_STATES)

# Initialize an empty mapping from gate types to integers for encoding
gate_type_to_int = {}

# RMSE
def compute_rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())

# Modified KL Divergence to handle batches
def compute_kl_divergence_batch(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    kl_divs = np.sum(p * np.log(p / q), axis=1)  # Compute along the second axis
    return np.mean(kl_divs)  # Average the results

# Modified Total Variation Distance to handle batches
def compute_tvd_batch(p, q):
    tvds = 0.5 * np.sum(np.abs(p - q), axis=1)  # Compute along the second axis
    return np.mean(tvds)  # Average the results
def plot_time_series(df, states):
    # Assuming 'date' is a column in df representing the date of each sample.
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(30, 6))

    for state in states:
        state_col_name = f"state_{state}"
        plt.plot(df['date'], df[state_col_name], label=state)

    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.title('State Probabilities Over Time')
    plt.legend()
    plt.show()

# Parameters to control sizes
num_train_samples = 1600
num_test_samples = 80

# 1. Load Data from CSV
data = pd.read_csv('datasets/ibm_auckland.csv')

# Removing outliers
Q1 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.25)
Q3 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[[f'state_{state}' for state in CARE_STATES]] < (Q1 - 1.5 * IQR)) |
              (data[[f'state_{state}' for state in CARE_STATES]] > (Q3 + 1.5 * IQR))).any(axis=1)]

plot_time_series(data, CARE_STATES)
# sss
# # Assuming the 'sx' gate error is in the 3rd column, and you have a date or timestamp column (say the first column).
# date_col = 'date'  # assuming the date or timestamp is in the first column
# sx_gate_error_col_index = 139  # replace 2 with the appropriate index if different
#
# # Calculating the IQR for the 'sx' gate error column
# Q1_sx = data.iloc[:, sx_gate_error_col_index].quantile(0.25)
# Q3_sx = data.iloc[:, sx_gate_error_col_index].quantile(0.75)
# IQR_sx = Q3_sx - Q1_sx
#
# # Filtering out the outliers
# filtered_data = data[~((data.iloc[:, sx_gate_error_col_index] < (Q1_sx - 1.5 * IQR_sx)) |
#                        (data.iloc[:, sx_gate_error_col_index] > (Q3_sx + 1.5 * IQR_sx)))]
#
# # Plotting the sx gate error over time without outliers
# plt.figure(figsize=(30, 6))
# # plt.plot(filtered_data.iloc[:, 951], filtered_data.iloc[:, sx_gate_error_col_index], marker='o', linestyle='-', color='blue')
# plt.scatter(filtered_data.iloc[:, 951], filtered_data.iloc[:, sx_gate_error_col_index], color='blue')
# plt.title('sx Gate Error Over Time (Outliers Removed)')
# plt.xlabel('Date')
# plt.ylabel('sx Gate Error')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if they're dates
# plt.tight_layout()
# plt.grid(True)
# plt.show()
#
# # Assuming the 'sx' gate error is in the 3rd column, and you have a date or timestamp column (say the first column).
# date_col = 'date'  # assuming the date or timestamp is in the first column
# cnot_gate_error_col_index = 409  # replace 2 with the appropriate index if different
#
# # Calculating the IQR for the 'sx' gate error column
# Q1_sx = data.iloc[:, cnot_gate_error_col_index].quantile(0.25)
# Q3_sx = data.iloc[:, cnot_gate_error_col_index].quantile(0.75)
# IQR_sx = Q3_sx - Q1_sx
#
# # Filtering out the outliers
# filtered_data = data[~((data.iloc[:, cnot_gate_error_col_index] < (Q1_sx - 1.5 * IQR_sx)) |
#                        (data.iloc[:, cnot_gate_error_col_index] > (Q3_sx + 1.5 * IQR_sx)))]
#
# # Plotting the sx gate error over time without outliers
# plt.figure(figsize=(30, 6))
# # plt.plot(filtered_data.iloc[:, 951], filtered_data.iloc[:, sx_gate_error_col_index], marker='o', linestyle='-', color='blue')
# plt.scatter(filtered_data.iloc[:, 951], filtered_data.iloc[:, cnot_gate_error_col_index], color='red')
# plt.title('CNOT Gate Error Over Time (Outliers Removed)')
# plt.xlabel('Date')
# plt.ylabel('CNOT Gate Error')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if they're dates
# plt.tight_layout()
# plt.grid(True)
# plt.show()

# sss
# Set a random seed for reproducibility
RANDOM_SEED = 42

# Shuffle the data using the set random seed
# data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
# Load metadata
num_qubits_info = data['num_qubits_info'].iloc[0]
num_gates_info = data['num_gates_info'].iloc[0]
num_params_info = data['num_params_info'].iloc[0]
circuit_depth_info = data['circuit_depth_info'].iloc[0]
num_qubits_after_transpile = data['num_qubits_after_transpile'].iloc[0]
sim_times = data['sim_times'].values

# Assuming the last 2^num_qubits columns store y_data and the column just before that is 'sim_times'
metadata_cols = 7  # Adjust this if you have more metadata columns
X_data_all = data.iloc[:, :- (8 + metadata_cols)].values
care_state_cols = [f'state_{state}' for state in CARE_STATES]
y_data_all = data[care_state_cols].values

# 2. Directly slice data to get test samples
X_test = X_data_all[-num_test_samples:]
y_test = y_data_all[-num_test_samples:]
sim_times_test = sim_times[-num_test_samples:]

# 3. From the remaining data, slice again to get the training samples
X_train = X_data_all[-(num_test_samples + num_train_samples):-num_test_samples]
y_train_original = y_data_all[-(num_test_samples + num_train_samples):-num_test_samples]

result = sm.tsa.seasonal_decompose(y_train_original, model='additive', period=7)

y_train = result.trend
seasonal_train = result.seasonal
residual_train = result.resid

sim_times_train = sim_times[-(num_test_samples + num_train_samples):-num_test_samples]
# Find non-NaN indices
# valid_indices = ~np.isnan(y_train)
valid_indices = ~np.isnan(y_train).any(axis=1)
# print(valid_indices)


# Filter X_train, y_train and sim_times_train using valid indices
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]
seasonal_train = seasonal_train[valid_indices]
residual_train = residual_train[valid_indices]
# X_train = X_train[valid_indices]
# seasonal_train=seasonal_train[~np.isnan(seasonal_train)]
# residual_train=residual_train[~np.isnan(residual_train)]
# y_train = y_train[valid_indices]
# print(y_train)


# # original curve
# plt.figure(figsize=(30, 6))
# plt.plot(y_train_original, label='Original', color='red')
# plt.title('Original Curve')
#
# # trend curve
# plt.figure(figsize=(30, 6))
# plt.plot(y_train, label='Trend', color='orange')
# plt.title('Trend Curve')
# plt.show()
#
# # residual curve
# plt.figure(figsize=(30, 6))
# plt.plot(residual_train, label='Trend', color='yellow')
# plt.title('residual Curve')
# plt.show()
#
# # Assuming CARE_STATES is a list of state names, e.g., ["000", "001", "010", ...]
# for idx, state in enumerate(CARE_STATES):
#     # Create a new figure for the state
#     plt.figure(figsize=(30, 18))
#
#     # Subplot 1: Original curve
#     plt.subplot(3, 1, 1)
#     plt.plot(y_train_original[:, idx], label='Original', color='red')
#     plt.title(f'Original Curve for State {state}')
#     plt.legend()
#
#     # Subplot 2: Trend curve
#     plt.subplot(3, 1, 2)
#     plt.plot(y_train[:, idx], label='Trend', color='orange')
#     plt.title(f'Trend Curve for State {state}')
#     plt.legend()
#
#     # Subplot 3: Residual curve
#     plt.subplot(3, 1, 3)
#     plt.plot(residual_train[:, idx], label='Residual', color='yellow')
#     plt.title(f'Residual Curve for State {state}')
#     plt.legend()
#
#     plt.tight_layout()  # Adjust layout to prevent overlap
#     plt.show()  # Display the figure
#
# sss
# Generate random indices for shuffling
shuffle_indices = np.arange(X_train.shape[0])
np.random.seed(RANDOM_SEED)  # For reproducibility
np.random.shuffle(shuffle_indices)

# Shuffle X_train and y_train using these indices
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# Reshape X_data
print(X_train.shape[0])
X_train = X_train.reshape(X_train.shape[0], -1, num_qubits_after_transpile, 5)
X_test = X_test.reshape(num_test_samples, -1, num_qubits_after_transpile, 5)

# Prepare PyTorch Datasets for training and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# Data Loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ss
# Define the RNN model with output units
class EnhancedRNNModel(nn.Module):
    def __init__(self, num_qubits):
        super(EnhancedRNNModel, self).__init__()

        self.hidden_size = 64  # Increased hidden units
        self.num_layers = 1  # Number of LSTM layers
        # Bi-directional set to False for now. When you're ready, you can set it to True
        self.rnn = nn.LSTM(input_size=num_qubits_after_transpile * 5,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           # dropout=0.2,
                           batch_first=True,
                           bidirectional=False)

        self.fc = nn.Linear(self.hidden_size, N_STATES)
        # self.fc2 = nn.Linear(32, N_STATES)

    def forward(self, x):
        # Reshape x to flatten the last two dimensions
        x = x.view(x.size(0), x.size(1), -1)  # New shape: (batch_size, num_stages, num_qubits * 5)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        # out = F.relu(out)
        # out = self.fc2(out)
        return out


# Initialize the model, loss function, and optimizer

model = EnhancedRNNModel(num_qubits_info)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=0.004)

# Training loop
train_losses = []
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    train_losses.append(loss.item())

print('Training finished.')

# Evaluate a subset of training data
train_predicted_results = []
train_actual_results = []
model.eval()


with torch.no_grad():
    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        train_predicted_results.extend(outputs.numpy())
        train_actual_results.extend(y_batch.numpy())
        break  # Only evaluate one batch for inspection
train_predicted_results = np.array(train_predicted_results)
train_actual_results = np.array(train_actual_results)


# Test the model
model.eval()
test_loss = 0
predicted_results = []
actual_results = []
prediction_times = []
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        start_time = time.time()
        outputs = model(X_test_batch)
        end_time = time.time()
        avg_prediction_time_per_sample = (end_time - start_time) / len(X_test_batch)
        prediction_times.extend([avg_prediction_time_per_sample] * len(X_test_batch))
        loss = criterion(outputs, y_test_batch)
        test_loss += loss.item()
        predicted_results.extend(outputs.numpy())
        actual_results.extend(y_test_batch.numpy())

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Convert to numpy arrays for easier handling
predicted_results = np.array(predicted_results)
actual_results = np.array(actual_results)

# Replicating the baseline for the entire test batch
baseline_sample = y_data_all[0]
all_states_probabilities_baseline_batch = np.tile(baseline_sample, (len(actual_results), 1))
rmse = compute_rmse(predicted_results, actual_results)
baseline_rmse = compute_rmse(all_states_probabilities_baseline_batch, actual_results)
improvement_rmse = (baseline_rmse - rmse) / baseline_rmse * 100

kl_div = compute_kl_divergence_batch(actual_results, predicted_results)
baseline_kl_div = compute_kl_divergence_batch(actual_results, all_states_probabilities_baseline_batch)
improvement_kl = (baseline_kl_div - kl_div) / baseline_kl_div * 100

tvd = compute_tvd_batch(actual_results, predicted_results)
baseline_tvd = compute_tvd_batch(actual_results, all_states_probabilities_baseline_batch)
improvement_tvd = (baseline_tvd - tvd) / baseline_tvd * 100


average_baseline_latency = np.mean(sim_times_test)
average_our_method_latency = np.mean(prediction_times)
improvement_latency = (average_baseline_latency - average_our_method_latency) / average_baseline_latency * 100

def format_number(num):
    """Formats number in scientific notation if it's too small, otherwise in normal float format."""
    return "{:.4e}".format(num) if num < 1e-4 else "{:.4f}".format(num)

# Print headers (column names) first. You can run this only once.
print("Metric\tBaseline\tOur Method\t% Improvement")

# Values
print(f"Qubits\t{num_qubits_info}\t-\t-")
print(f"Gates\t{num_gates_info}\t-\t-")
print(f"Parameters\t{num_params_info}\t-\t-")
print(f"Circuit Depth\t{circuit_depth_info}\t-\t-")

print(f"RMSE\t{baseline_rmse:.4f}\t{rmse:.4f}\t{improvement_rmse:.2f}%")
print(f"KL Divergence\t{baseline_kl_div:.4f}\t{kl_div:.4f}\t{improvement_kl:.2f}%")
print(f"Total Variation Distance\t{baseline_tvd:.4f}\t{tvd:.4f}\t{improvement_tvd:.2f}%")
print(f"Average Time (s)\t{average_baseline_latency:.4f}\t{format_number(average_our_method_latency)}\t{improvement_latency:.2f}%")

# Plotting the training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the actual vs predicted results for training data
plt.figure(figsize=(12, 6))
plt.plot(train_predicted_results, label='Train Predicted', color='green')
plt.plot(train_actual_results, label='Train Actual', color='blue')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.title('Training Data: Actual vs Predicted Results')
plt.show()

# predicted_results
# predicted_upper_bound  = predicted_results + seasonal_upper_bound + residual_upper_bound
# predicted_lower_bound  = predicted_results + seasonal_lower_bound + residual_lower_bound


# predicted_upper_bound1  = predicted_results + seasonal_upper_bound1 + residual_upper_bound1
# predicted_lower_bound1  = predicted_results + seasonal_lower_bound1 + residual_lower_bound1

# # Plotting the actual vs predicted results for test data
# plt.figure(figsize=(12, 6))
# plt.plot(predicted_results, label='Test Predicted', color='orange')
# plt.plot(actual_results, label='Test Actual', color='blue')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Test Data: Actual vs Predicted Results')
# plt.show()
#
# # Visualization
# plt.figure(figsize=(12, 6))
# plt.plot(predicted_upper_bound, label='Upper Bound', color='red')
# plt.plot(predicted_lower_bound, label='Lower Bound', color='green')
# plt.plot(predicted_results, label='Test Predicted', color='orange')
# plt.plot(actual_results, label='Actual', color='blue')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Test Data with Predicted Bounds')
# plt.show()
#
# # Visualization
# plt.figure(figsize=(12, 6))
# plt.plot(predicted_upper_bound1, label='Upper Bound', color='red')
# plt.plot(predicted_lower_bound1, label='Lower Bound', color='green')
# plt.plot(predicted_results, label='Test Predicted', color='orange')
# plt.plot(actual_results, label='Actual', color='blue')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Test Data with Predicted Bounds')
# plt.show()
###############
def create_quantum_circuit():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


def simulate_noiseless(qc: QuantumCircuit, num_shots: int = 10000):
    """Simulates a given quantum circuit without any noise."""
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=num_shots).result()
    counts = result.get_counts(qc)

    num_qubits = qc.num_qubits
    num_possible_states = 2 ** num_qubits
    all_states_counts = [counts.get(bin(i)[2:].zfill(num_qubits), 0) for i in range(num_possible_states)]
    total_counts = sum(all_states_counts)
    all_states_probabilities = [count / total_counts for count in all_states_counts]
    target_probs = [all_states_probabilities[int(state, 2)] for state in CARE_STATES]

    return target_probs




# x_values = [0.90, 0.975, 0.995]
x_values = [0.995]
for x in x_values:
    z_score = norm.ppf(x)

    residual_upper_bounds = []
    residual_lower_bounds = []
    for i in range(residual_train.shape[1]):
        residual_state = residual_train[:, i]
        residual_mean = np.nanmean(residual_state)
        residual_std = np.nanstd(residual_state)

        residual_ci = z_score * residual_std

        residual_upper_bounds.append(residual_mean + residual_ci)
        residual_lower_bounds.append(residual_mean - residual_ci)

    # Setting up a grid of plots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))

    for i, ax in enumerate(axes.ravel()):
        sns.kdeplot(residual_train[:, i], ax=ax, shade=True)

        # Plotting upper and lower bounds for the residual
        ax.axvline(x=residual_upper_bounds[i], color='r', linestyle='--', label='Residual Upper Bound')
        ax.axvline(x=residual_lower_bounds[i], color='g', linestyle='--', label='Residual Lower Bound')

        ax.set_title(f'Residuals for CARE_STATE {CARE_STATES[i]} with Bounds (Confidence: {(1-2*(1-x))*100}%)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Setting up a grid of plots
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))

    for i, ax in enumerate(axes.ravel()):
        ax.plot(residual_train[:, i], label='Residual', color='yellow')

        # Plotting upper and lower bounds for the residual
        ax.axhline(y=residual_upper_bounds[i], color='r', linestyle='--', label='Residual Upper Bound')
        ax.axhline(y=residual_lower_bounds[i], color='g', linestyle='--', label='Residual Lower Bound')

        ax.set_title(f'Residuals for CARE_STATE {CARE_STATES[i]} with Bounds (Confidence: {(1-2*(1-x))*100}%)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Convert the lists to numpy arrays for vectorized operations
    residual_upper_bounds = np.array(residual_upper_bounds)
    residual_lower_bounds = np.array(residual_lower_bounds)

    # Adjust the upper and lower bound computation
    predicted_upper_bound = predicted_results  + residual_upper_bounds
    predicted_lower_bound = predicted_results  + residual_lower_bounds

    states_to_plot = ["000", "001", "010", "011","100","101","110","111"]
    indices_to_plot = [CARE_STATES.index(state) for state in states_to_plot]

    train_predicted_selected = train_predicted_results[:, indices_to_plot]
    train_actual_selected = train_actual_results[:, indices_to_plot]

    predicted_selected = predicted_results[:, indices_to_plot]
    actual_selected = actual_results[:, indices_to_plot]
    predicted_upper_bound_selected = predicted_upper_bound[:, indices_to_plot]
    predicted_lower_bound_selected = predicted_lower_bound[:, indices_to_plot]


    # Number of states to plot
    num_states = len(states_to_plot)

    # Create subplots for the train data
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), states_to_plot)):
        ax.plot(train_predicted_selected[:, idx], label=f'Train Predicted {state}', color='green')
        ax.plot(train_actual_selected[:, idx], label=f'Train Actual {state}', color='blue')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Train Data: Actual vs Predicted for {state}')
    plt.tight_layout()
    plt.show()

    # Create subplots for the test data
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), states_to_plot)):
        ax.plot(predicted_upper_bound_selected[:, idx], linestyle='--', label='Predicted Upper Bound', color='red')
        ax.plot(predicted_lower_bound_selected[:, idx], linestyle='--', label='Predicted Lower Bound', color='green')
        # ax.plot(predicted_selected[:, idx], label=f'Test Predicted {state}', color='orange')
        ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Test Data: Actual vs Predicted for {state} with Bounds (Confidence: {(1-2*(1-x))*100}%)')
    plt.tight_layout()
    plt.show()

    # #####plot idea simulation + bounds
    # qc = create_quantum_circuit()
    # # Now, let's gather the noiseless results
    # noiseless_simulation_results = [simulate_noiseless(qc) for _ in range(80)]
    # noiseless_simulation_results = np.array(noiseless_simulation_results)  # Convert to numpy array for easy arithmetic
    #
    # # Assuming residual_upper_bounds and residual_lower_bounds are numpy arrays
    # noiseless_upper_bound = noiseless_simulation_results + residual_upper_bounds[:80]
    # noiseless_lower_bound = noiseless_simulation_results + residual_lower_bounds[:80]
    #
    # # Now, you can plot these values as previously mentioned
    # fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    # for idx, (ax, state) in enumerate(zip(axes.ravel(), CARE_STATES)):
    #     # ax.plot(noiseless_simulation_results[:, idx], label=f'Noiseless {state}', color='purple')
    #     ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
    #     ax.plot(noiseless_upper_bound[:, idx], linestyle='--', label='Upper Bound', color='red')
    #     ax.plot(noiseless_lower_bound[:, idx], linestyle='--', label='Lower Bound', color='green')
    #
    #     ax.set_xlabel('Sample')
    #     ax.set_ylabel('Value')
    #     ax.legend()
    #     ax.set_title(f'Noiseless Simulation Plus Bounds Results for {state}')
    #     ax.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    # Theoretical probabilities for the states: [000, 001, 010, 011, 100, 101, 110, 111]
    perfect_probs = np.array([0.5 if state == '000' or state == '111' else 0 for state in CARE_STATES])
    # perfect_probs = np.array([1 if state == '000' else 0 for state in CARE_STATES])
    # Create the upper and lower bounds using the residuals you've calculated
    perfect_upper_bound = perfect_probs + residual_upper_bounds[:80]
    perfect_lower_bound = perfect_probs + residual_lower_bounds[:80]

    # Now, you can plot these values as previously mentioned
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    for idx, (ax, state) in enumerate(zip(axes.ravel(), CARE_STATES)):
        ax.plot(actual_selected[:, idx], label=f'Test Actual {state}', color='blue')
        ax.plot([perfect_upper_bound[idx]] * 80, linestyle='--', label='Upper Bound', color='red')
        ax.plot([perfect_lower_bound[idx]] * 80, linestyle='--', label='Lower Bound', color='green')

        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title(f'Noiseless Simulation Plus Bounds Results for {state}')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
