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
            'ranging_uptrend_vol_z_threshold': 0.0,
            'uptrend_impulse_confidence': 0.8,
            'accumulation_confidence': 0.7,
            'pullback_confidence': 0.6,
            'ranging_uptrend_confidence': 0.5
        }

    df = df_input.copy()
    signals = pd.Series(index=df.index, dtype=str, name='signal').fillna('Neutral')
    confidence = pd.Series(index=df.index, dtype=float, name='signal_confidence').fillna(0.0)

    df['rsi_5_prev'] = df['rsi_5'].shift(1) 

    # --- Entry Signal Logic ---
    uptrend_impulse_mask = (df['regime'] == 'Uptrend - Impulse') & (df['volume_zscore'] > signal_params['uptrend_impulse_vol_z_threshold'])
    signals[uptrend_impulse_mask] = 'Enter Long'
    confidence[uptrend_impulse_mask] = signal_params['uptrend_impulse_confidence']
    
    accumulation_mask = (df['regime'] == 'Ranging - Accumulation') & (df['volume_zscore'] > signal_params['accumulation_vol_z_threshold'])
    signals[accumulation_mask] = 'Enter Long'
    confidence[accumulation_mask] = signal_params['accumulation_confidence']

    pullback_mask = (df['regime'] == 'Uptrend - Pullback') & (df['rsi_5'] > df['rsi_5_prev']) & (df['rsi_5'] < signal_params['pullback_rsi_5_lt']) & (df['rsi_14'] > signal_params['pullback_rsi_14_gt'])
    signals[pullback_mask] = 'Enter Long'
    confidence[pullback_mask] = signal_params['pullback_confidence']
    
    ranging_uptrend_mask = (df['regime'] == 'Ranging - Uptrend Bias') & (df['volume_zscore'] > signal_params['ranging_uptrend_vol_z_threshold'])
    signals[ranging_uptrend_mask] = 'Enter Long'
    confidence[ranging_uptrend_mask] = signal_params['ranging_uptrend_confidence']

    # --- Exit Signal Logic ---
    signals[df['regime'] == 'Downtrend - Impulse'] = 'Exit Long'
    signals[df['regime'] == 'Volatile - Choppy'] = 'Exit Long'

    signals[~signals.isin(['Enter Long', 'Exit Long'])] = 'Neutral' 
    
    df['signal'] = signals
    df['signal_confidence'] = confidence
    
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
