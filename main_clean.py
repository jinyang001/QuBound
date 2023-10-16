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
N_STATES = len(CARE_STATES)

# Initialize an empty mapping from gate types to integers for encoding
gate_type_to_int = {}

# Parameters to control sizes
num_train_samples = 1650
num_test_samples = 80

# 1. Load Data from CSV
data = pd.read_csv('datasets/ibm_auckland.csv')

# Removing outliers
Q1 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.25)
Q3 = data[[f'state_{state}' for state in CARE_STATES]].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[[f'state_{state}' for state in CARE_STATES]] < (Q1 - 1.5 * IQR)) |
              (data[[f'state_{state}' for state in CARE_STATES]] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Set a random seed for reproducibility
RANDOM_SEED = 42

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

valid_indices = ~np.isnan(y_train).any(axis=1)

# Filter X_train, y_train and sim_times_train using valid indices
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]
seasonal_train = seasonal_train[valid_indices]
residual_train = residual_train[valid_indices]

# Generate random indices for shuffling
shuffle_indices = np.arange(X_train.shape[0])
np.random.seed(RANDOM_SEED)  # For reproducibility
np.random.shuffle(shuffle_indices)

# Define the size of the validation set
num_val_samples = 50

# Shuffle X_train and y_train using these indices
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# Split out validation set from training set
X_val = X_train[:num_val_samples]
y_val = y_train[:num_val_samples]

X_train = X_train[num_val_samples:]
y_train = y_train[num_val_samples:]

# Reshape X_data
X_train = X_train.reshape(X_train.shape[0], -1, num_qubits_after_transpile, 5)
X_val = X_val.reshape(num_val_samples, -1, num_qubits_after_transpile, 5)
X_test = X_test.reshape(num_test_samples, -1, num_qubits_after_transpile, 5)

# Prepare PyTorch Datasets for training, validation, and testing
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# Data Loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

# Evaluate the validation dataset
train_predicted_results = []
train_actual_results = []
model.eval()


with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        train_predicted_results.extend(outputs.numpy())
        train_actual_results.extend(y_batch.numpy())
        # break  # Only evaluate one batch for inspection
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


def calculate_bcr_bdp(actual, lower_bound, upper_bound):
    # Number of samples where actual values are within the bounds
    within_bounds = np.sum((lower_bound <= actual) & (actual <= upper_bound))

    # Calculate BCR
    bcr = (within_bounds / len(actual)) * 100

    # Calculate BDP
    deviation = np.where(actual < lower_bound, lower_bound - actual,
                         np.where(actual > upper_bound, actual - upper_bound, 0))

    # Normalize the BDP
    max_possible_deviation = np.max(actual) - np.min(actual)
    normalized_bdp = np.sum(deviation) / (len(actual) * max_possible_deviation) * 100

    return bcr, normalized_bdp


def calculate_bcr_bdp_for_each_state(predicted_upper, predicted_lower, actual_values):
    statewise_bcr = []
    statewise_bdp = []

    for idx in range(predicted_upper.shape[1]):  # Loop over states
        bcr, bdp = calculate_bcr_bdp(actual_values[:, idx], predicted_lower[:, idx], predicted_upper[:, idx])
        statewise_bcr.append(bcr)
        statewise_bdp.append(bdp)

    return statewise_bcr, statewise_bdp

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

    # Using the function with your data
    statewise_bcr, statewise_bdp = calculate_bcr_bdp_for_each_state(predicted_upper_bound_selected,
                                                                    predicted_lower_bound_selected, actual_selected)

    # Printing the results
    for state, bcr, bdp in zip(states_to_plot, statewise_bcr, statewise_bdp):
        print(f"For state {state}:")
        print(f"Bound-Compliance Rate (BCR): {bcr:.2f}%")
        print(f"Normalized Bound-Deviation Penalty (BDP): {bdp:.2f}%")
        print("----------")

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
