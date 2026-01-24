
import pandas as pd
import pandas_ta as ta

def investigate_atr():
    try:
        data_file = 'data/market_features.parquet'
        df = pd.read_parquet(data_file)
        
        if 'ATR_14' not in df.columns:
            print("ATR_14 column not found, calculating it now...")
            df['ATR_14'] = ta.atr(
                high=df['high'], 
                low=df['low'],
                close=df['close'], 
                length=14
            )
            df.dropna(subset=['ATR_14'], inplace=True)
        
        print("\n--- ATR_14 Column Description ---")
        print(df['ATR_14'].describe())
            
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    investigate_atr()
