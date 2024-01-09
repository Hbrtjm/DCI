import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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
    
    model_path = f"{stock_name}_model_weights.pth"

    # Redifine the model again
    model = SimpleNet(input_size=input_size)
    
    # Load the saved weights 
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

# In views we use predict(recall_model(stock_name,batch),currentdata(stock_name,batch)) 

def predict(model, data):
    with torch.no_grad():
        data_tensor = torch.tensor(data,dtype=torch.float32).unsqueeze(1)
        predictions = model(data_tensor)
        return predictions.numpy()

stock_name = 'BTC'

def get_current_data(stock_name,batch_size):
    return [130321.32]

predict(recall_model(stock_name,1),get_current_data(stock_name,1))
# train_model('BTC')