import yfinance as yf
import pickle
import os

class StockDatabaseManager:
    def __init__(self, db_file):
        self.db_file = db_file
        self.data = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as file:
                return pickle.load(file)
        return {}

    def save_database(self):
        with open(self.db_file, 'wb') as file:
            pickle.dump(self.data, file)

    def fetch_and_save_stock_data(self, stock_name):
        # Check if data already exists
        if stock_name in self.data:
            print(f"Data for {stock_name} already exists. Filename: {self.data[stock_name]['stock_filename']}")
            return
        try:    
            # Fetch historical stock data
            stock = yf.Ticker(stock_name)
            hist_data = stock.history(period="max")

            # Save data to a CSV file
            file_name = f'historical_{stock_name}.csv'
            hist_data.to_csv(file_name)
            print(f'Data saved to {file_name}')

            # Update database, the entries may not be deleted
            record_id = len(self.data) + 1
            self.data[stock_name] = {
                'ID': record_id,
                'stock_name': stock_name,
                'stock_filename': file_name
            }
            self.save_database()
        except HTTPError as e:
            if e.status == 404:
                return ""

# db_manager = StockDatabaseManager('stock_data.db')
# stock_name = input("Enter the stock ticker symbol: ")
# db_manager.fetch_and_save_stock_data(stock_name)