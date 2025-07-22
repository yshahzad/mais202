import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Load and prepare data
file_path = "Data/Survival Analysis/Survival Analysis_arXiv_scrape.csv"
df = pd.read_csv(file_path)
df[df.columns[2]] = pd.to_datetime(df[df.columns[2]])

# Group by month and count publications
monthly_counts = df.groupby(df[df.columns[2]].dt.to_period("M")).size().reset_index(name="count")
monthly_counts["date"] = monthly_counts[df.columns[2]].dt.to_timestamp()
monthly_counts = monthly_counts[["date", "count"]].sort_values("date")

# Normalize
scaler = MinMaxScaler()
monthly_counts["count_normalized"] = scaler.fit_transform(monthly_counts[["count"]])

# Sequence builder
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

values = monthly_counts["count_normalized"].values
time_steps = 12
X, y = create_sequences(values, time_steps)

X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 50
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Forecast next 60 months
model.eval()
future_preds = []
last_seq = torch.tensor(values[-time_steps:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

with torch.no_grad():
    for _ in range(60):
        next_val = model(last_seq)
        future_preds.append(next_val.item())
        last_seq = torch.cat((last_seq[:, 1:, :], next_val.unsqueeze(1)), dim=1)

# Inverse transform to get actual predicted publication counts
predicted_counts = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Create date index for forecast
last_date = monthly_counts["date"].iloc[-1]
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=60, freq='MS')

# Plot
plt.figure(figsize=(14, 6))
plt.plot(monthly_counts["date"], monthly_counts["count"], label="Historical")
plt.plot(future_dates, predicted_counts, label="Forecast (Next 5 Years)", linestyle="--")
plt.title("Forecast of Monthly Publications Using LSTM")
plt.xlabel("Date")
plt.ylabel("Publication Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
