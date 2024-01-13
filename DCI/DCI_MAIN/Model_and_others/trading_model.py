import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

try:
    from .data_request import StockDatabaseManager
except:
    from data_request import StockDatabaseManager

class SimpleNet(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNet, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
    def forward(self, x):
        return self.linear(x)

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
        except Exception as e:
            print(f"An exception has occured: {e}")
            #Handle the "can't find the given stock data eroor"
            return
    column_name = 'Low'
    try:
        file_content = pd.read_csv(data_file)
        if column_name not in file_content.columns:
            raise ValueError("Column {column_name} not found in data")

        df = file_content[column_name]
        normalized_df = (df - df.mean()) / df.std()

        features = normalized_df.values
        targets = df.values

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        train_data = TensorDataset(features_tensor, targets_tensor)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    except Exception as e:
        print(f"Error in data processing: {e}")
        return

    # Initialize the model
    model = SimpleNet(input_size=features_tensor.shape[1])
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
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
    
    # Saving the results to a file 
    model_save_path = f"{stock_name}_model_weights.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model was saved into {model_save_path}")

# Input_size should be adjusted to the batch size

def recall_model(stock_name,input_size=1):
    
    from pathlib import Path
    import os
    # Setting BASE_DIR because of the import in the other module
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = os.path.join(BASE_DIR,f"Model_and_others/{stock_name}_model_weights.pth")

    # Redifine the model again
    model = SimpleNet(input_size=input_size)
    
    # Load the saved weights 
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

# In views we just use predict(stock_name) 

def predict(stock_name):
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    print("Getting the model")
    model = recall_model(stock_name,1)
    print("Getting current data")
    current_data = get_current_data(stock_name,1)
    print("Current data problem")
    with torch.no_grad():
        data_tensor = torch.tensor(current_data,dtype=torch.float32).unsqueeze(1)
        logits = model(data_tensor)
        probabilities = sigmoid(logits)  # Apply sigmoid if not in the model
        probabilities = probabilities.numpy()
        
        # Assuming 0.5 as the threshold for buy/sell recommendation
        recommendations = ["Buy" if prob > 0.5 else "Sell" for prob in probabilities]

        return list(zip(probabilities, recommendations))

# Gets current stocks data

def get_current_data(stock_name, batch_size):
    # Calculate start and end times with a custom delay for historical data
    delay = 50
    end_time = datetime.now() - timedelta(days=delay)
    start_time = end_time - timedelta(days=delay+1)

    # Convert times to string format for yfinance
    start_str = start_time.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')

    # Fetch data from yfinance
    data = yf.download(stock_name, start=start_str, end=end_str, interval='1h')
    print(data)
    # Batch data
    batched_data = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]

    return batched_data

# batch_size = 1
stock_name = "BTC"
predict(stock_name)
# train_model('BTC')