
import pandas as pd

def investigate_data():
    try:
        data_file = 'data/market_features.parquet'
        df = pd.read_parquet(data_file)
        
        if 'close' in df.columns:
            print("--- Close Column Description ---")
            print(df['close'].describe())
        else:
            print("Close column not found in the data.")
            
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    investigate_data()
