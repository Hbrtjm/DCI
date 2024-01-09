import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_request import StockDatabaseManager

def train_model(stock_name):
    # Example DataFrame loading
    db_manager = StockDatabaseManager("stock_data.db")
    try:
        print("Trying to get the datafile name")
        data_file = db_manager.load_database()[stock_name]['stock_filename'] 
        print(data_file)
    except Exception as e:
        print(e)
        try:
            db_manager.fetch_and_save_stock_data(stock_name)
            data_file = db_manager.load_database()[stock_name]['stock_filename']
        except:
            #Handle the "can't find the given stock data eroor"
            return

    file_content = pd.read_csv(data_file)
    df = file_content['Low']
    # Normalize your data
    # Assuming 'df' is your DataFrame
    normalized_df = (df - df.mean()) / df.std()

    # Convert DataFrame to PyTorch tensors
    features = normalized_df.values
    targets = df.values

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Create datasets
    train_data = TensorDataset(features_tensor, targets_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    class SimpleNet(torch.nn.Module):
        def __init__(self, input_size):
            super(SimpleNet, self).__init__()
            self.linear = torch.nn.Linear(input_size, 1)

        def forward(self, x):
            return self.linear(x)

    # Initialize the model
    model = SimpleNet(input_size=features_tensor.shape[1])
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

train_model('BTC')