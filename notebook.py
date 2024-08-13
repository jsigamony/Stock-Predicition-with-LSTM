import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler

# Load and display data
df = pd.read_csv('AMZN.csv')
df.head(10)

def plot_graph(x: torch.tensor):
    plt.figure(figsize=(20, 8))
    plt.plot(x, label="Predicted Closing")
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

data = df['Close'].values
data_tensor = torch.from_numpy(data).type(torch.float).unsqueeze(dim=1)
print(data_tensor.shape)
plot_graph(data_tensor)

# Split data into train and test
split = int(len(data) * 0.8)
train_data = data_tensor[:split]
test_data = data_tensor[split:]
print(train_data[:5])
print(test_data[:5])
print(train_data.shape, test_data.shape)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Plot closing values
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_data)
ax.set_title("Closing Values")
ax.set_xlabel("Day")
ax.set_ylabel("Closing Price")
plt.show()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.fit_transform(test_data)

def sequencing(scaled_tensor: torch.Tensor, seq_len: int):
    X, y = [], []
    for i in range(len(scaled_tensor) - seq_len):
        X.append(scaled_tensor[i:i+seq_len])
        y.append(scaled_tensor[i+seq_len])
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()

X_train, y_train = sequencing(scaled_train_data, 30)
X_test, y_test = sequencing(scaled_test_data, 30)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        h0, c0 = h0.to(device), c0.to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Initialize model
input_dim = 1
hidden_dim = 8
num_layers = 1
output_dim = 1
num_epochs = 500

stock_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
stock_model.to(device)

# Define loss and optimizer
loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(stock_model.parameters(), lr=0.001)

# Move data to device
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Make untrained predictions
stock_model.eval()
with torch.inference_mode():
    y_pred_values = stock_model(X_train)

plt.figure(figsize=(10, 6))
plt.plot(y_pred_values.cpu().detach().numpy(), color="blue", label="Predicted train prices")
plt.plot(y_train.cpu(), color="red", label="Actual train prices")
plt.title("Train predictions")
plt.legend()
plt.show()

# Training and Testing Loop
epochs = 1000

for epoch in range(epochs):
    stock_model.train()
    train_output = stock_model(X_train)
    loss = loss_fn(train_output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    test_output = stock_model(X_test)
    test_loss = loss_fn(test_output, y_test)
    optimizer.zero_grad()
    test_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'epoch: {epoch} | loss: {loss:.4f} | test loss: {test_loss:.4f}')

# Make predictions
stock_model.eval()
with torch.inference_mode():
    test_preds = stock_model(X_test)

actuals = scaler.inverse_transform(y_test.cpu().detach().numpy())
preds = scaler.inverse_transform(test_preds.cpu().detach().numpy())

plt.figure(figsize=(10, 6))
plt.plot(actuals, color="blue", label="Actual prices")
plt.plot(preds, color="red", label="Predicted prices")
plt.title("Amazon Stock Predictions")
plt.legend()
plt.show()

# Make predictions using the model
stock_model.eval()
with torch.inference_mode():
    predicted_prices = stock_model(y_train)

actuals = pd.DataFrame(scaler.inverse_transform(y_test.detach().numpy()))
print(actuals.head())
preds = pd.DataFrame(scaler.inverse_transform(predicted_prices.detach().numpy()))
print(preds.head())

plt.figure(figsize=(20, 8))
plt.plot(y_test, label="Actual Closing")
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
