import pandas as pd

def generate_signals(df_input: pd.DataFrame, signal_params: dict = None) -> pd.DataFrame:
    """
    Generates trading signals based on market regimes and configurable parameters.

    Args:
        df_input (pd.DataFrame): DataFrame containing market data and a 'regime' column.
        signal_params (dict, optional): A dictionary of parameters to tweak signal logic. 
                                        If None, default parameters are used.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'signal' column.
    """
    if 'regime' not in df_input.columns:
        raise ValueError("Input DataFrame must contain a 'regime' column.")

    if signal_params is None:
        signal_params = {
            'uptrend_impulse_vol_z_threshold': 0.0,
            'accumulation_vol_z_threshold': 0.5,
            'pullback_rsi_5_lt': 50,
            'pullback_rsi_14_gt': 60,
            'ranging_uptrend_vol_z_threshold': 0.0
        }

    df = df_input.copy()
    signals = pd.Series(index=df.index, dtype=str, name='signal').fillna('Neutral')

    df['rsi_5_prev'] = df['rsi_5'].shift(1) 

    # --- Entry Signal Logic ---
    signals[
        (df['regime'] == 'Uptrend - Impulse') & 
        (df['volume_zscore'] > signal_params['uptrend_impulse_vol_z_threshold'])
    ] = 'Enter Long'
    
    signals[
        (df['regime'] == 'Ranging - Accumulation') & 
        (df['volume_zscore'] > signal_params['accumulation_vol_z_threshold'])
    ] = 'Enter Long'
    
    signals[
        (df['regime'] == 'Uptrend - Pullback') & 
        (df['rsi_5'] > df['rsi_5_prev']) &    
        (df['rsi_5'] < signal_params['pullback_rsi_5_lt']) &                   
        (df['rsi_14'] > signal_params['pullback_rsi_14_gt'])                    
    ] = 'Enter Long'
    
    signals[
        (df['regime'] == 'Ranging - Uptrend Bias') & 
        (df['volume_zscore'] > signal_params['ranging_uptrend_vol_z_threshold'])
    ] = 'Enter Long'

    # --- Exit Signal Logic ---
    signals[df['regime'] == 'Downtrend - Impulse'] = 'Exit Long'
    signals[df['regime'] == 'Volatile - Choppy'] = 'Exit Long'

    signals[~signals.isin(['Enter Long', 'Exit Long'])] = 'Neutral' 
    
    df['signal'] = signals
    
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
