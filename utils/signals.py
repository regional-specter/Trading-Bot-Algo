
import pandas as pd

def generate_signals(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Generates trading signals based on market regimes.

    Args:
        df_input (pd.DataFrame): DataFrame containing market data and a 'regime' column.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'signal' column.
    """
    if 'regime' not in df_input.columns:
        raise ValueError("Input DataFrame must contain a 'regime' column.")

    df = df_input.copy()
    signals = pd.Series(index=df.index, dtype=str, name='signal').fillna('Neutral')

    # --- Helper features for signal confirmation ---
    df['rsi_5_prev'] = df['rsi_5'].shift(1) 

    # --- Entry Signal Logic ---
    # Case 1: Uptrend Impulse with positive volume momentum
    signals[(df['regime'] == 'Uptrend - Impulse') & (df['volume_zscore'] > 0.0)] = 'Enter Long'
    
    # Case 2: Accumulation with significant volume
    signals[(df['regime'] == 'Ranging - Accumulation') & (df['volume_zscore'] > 0.5)] = 'Enter Long'
    
    # Case 3: Uptrend Pullback with RSI recovery
    signals[
        (df['regime'] == 'Uptrend - Pullback') & 
        (df['rsi_5'] > df['rsi_5_prev']) &    
        (df['rsi_5'] < 50) &                   
        (df['rsi_14'] > 60)                    
    ] = 'Enter Long'
    
    # Case 4: Ranging with Uptrend Bias and neutral-to-positive volume
    signals[(df['regime'] == 'Ranging - Uptrend Bias') & (df['volume_zscore'] > 0)] = 'Enter Long'

    # --- Exit Signal Logic ---
    # Exit on clear, strong reversals or high volatility
    signals[df['regime'] == 'Downtrend - Impulse'] = 'Exit Long'
    signals[df['regime'] == 'Volatile - Choppy'] = 'Exit Long'

    # Neutral signals for all other cases
    signals[~signals.isin(['Enter Long', 'Exit Long'])] = 'Neutral' 
    
    df['signal'] = signals
    
    # Drop helper feature
    df.drop(columns=['rsi_5_prev'], inplace=True) 
    
    return df

if __name__ == '__main__':
    # Example usage:
    # This block will only run when the script is executed directly
    try:
        # We need the regime data first. Let's use the function from our new regime script.
        from regime import classify_regimes_with_kmeans

        # Assuming the script is run from the root of the project
        data_file = 'data/market_features.parquet'
        df = pd.read_parquet(data_file)
        
        # 1. Classify regimes
        df_with_regimes = classify_regimes_with_kmeans(df)
        
        # 2. Generate signals
        df_with_signals = generate_signals(df_with_regimes)

        print("Successfully generated signals:")
        print(df_with_signals['signal'].value_counts())
        
        print("\nDataFrame head with new 'signal' column:")
        print(df_with_signals[['datetime', 'regime', 'signal']].head())

    except FileNotFoundError:
        print(f"Error: The data file was not found. Make sure you are running this script from the project root")
    except Exception as e:
        print(f"An error occurred: {e}")
