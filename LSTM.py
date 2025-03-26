import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset (replace with actual file path if using CSV)
file_path = "Accelerated Failure Time_arXiv_scrape.csv"
df = pd.read_csv(file_path)

# Drop any missing values
df = df.dropna()

# Identify time series columns
time_column = df.columns[0]  # Assume first column is the time index (year)
feature_columns = df.columns[1:]  # Other columns are publication counts

# Convert data to numerical format
df[time_column] = df[time_column].astype(int)
df[feature_columns] = df[feature_columns].astype(float)

# Normalize features using MinMaxScaler
scalers = {col: MinMaxScaler() for col in feature_columns}
for col in feature_columns:
    df[col] = scalers[col].fit_transform(df[[col]])

# Convert dataframe to numpy array
data = df[feature_columns].values
time_steps = 5  # Lookback window for LSTM

# Function to create sequences for LSTM
def create_sequences(data, time_steps):
    sequences, targets = [], []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        targets.append(data[i + time_steps])  # Predict next step
    return np.array(sequences), np.array(targets)

# Create sequences for training
X, y = create_sequences(data, time_steps)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader
batch_size = 16
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last time step output

# Model hyperparameters
input_size = len(feature_columns)
hidden_size = 64
num_layers = 2
output_size = len(feature_columns)

# Initialize model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "lstm_publications_model.pth")

print("Training complete! Model saved as 'lstm_publications_model.pth'.")
