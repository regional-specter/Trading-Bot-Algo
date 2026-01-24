import pandas_ta as ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# --- regime.py content ---
def classify_regimes_with_kmeans(market_data_df: pd.DataFrame, k: int = 7, cluster_to_regime_map: dict = None) -> pd.DataFrame:
    """
    Applies K-Means clustering to a DataFrame to determine market regimes.

    Args:
        market_data_df (pd.DataFrame): DataFrame containing the market features.
        k (int): The number of clusters (regimes) to find.
        cluster_to_regime_map (dict, optional): A dictionary to map cluster labels to regime names.
                                                If not provided, a default map will be used.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'regime' column.
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

    features_for_clustering = [
        'rsi_5',
        'rsi_14',
        'trend_strength',
        'rolling_volatility',
        'volume_zscore',
        'rolling_zscore'
    ]

    # Ensure all required columns are present
    missing_cols = [col for col in features_for_clustering if col not in market_data_df.columns]
    if missing_cols:
        # Instead of raising an error, fill with NaNs and return without clustering if critical data is missing
        print(f"Warning: Missing required columns for clustering: {missing_cols}. Filling 'regime' with 'Unknown'.")
        market_data_df['regime'] = 'Unknown'
        return market_data_df


    clustering_df = market_data_df[features_for_clustering].copy()
    clustering_df_cleaned = clustering_df.dropna()
    original_indices = clustering_df_cleaned.index

    if clustering_df_cleaned.empty:
        # Return with an empty regime column if no data to cluster
        market_data_df['regime'] = 'Unknown'
        return market_data_df

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df_cleaned)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(scaled_features)

    cluster_labels_series = pd.Series(clusters, index=original_indices, name='cluster_label')

    # Merge cluster labels back
    if 'cluster_label' in market_data_df.columns:
        market_data_df = market_data_df.drop(columns=['cluster_label'])

    market_data_df = market_data_df.merge(
        cluster_labels_series,
        left_index=True,
        right_index=True,
        how='left'
    )
    market_data_df['cluster_label'] = market_data_df['cluster_label'].fillna(-1).astype(int)

    # Interpret Clusters to map to regime names
    cluster_means = market_data_df[market_data_df['cluster_label'] != -1].groupby(
        'cluster_label'
    )[features_for_clustering].mean()

    # Use the provided map, or the default if not provided.
    if cluster_to_regime_map is None:
        cluster_to_regime_map = {
            0: 'Ranging - Uptrend Bias',
            1: 'Uptrend - Impulse',
            2: 'Ranging - Downtrend Bias',
            3: 'Downtrend - Impulse',
            4: 'Uptrend - Overbought',
            5: 'Ranging - Accumulation',
            6: 'Volatile - Choppy',
            -1: 'Unknown'
        }

    market_data_df['regime'] = market_data_df['cluster_label'].map(cluster_to_regime_map).fillna('Unknown')

    # Drop the intermediate cluster label column
    market_data_df = market_data_df.drop(columns=['cluster_label'])

    return market_data_df


# --- signals.py content ---
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

    # Ensure 'rsi_5' exists before trying to shift it
    if 'rsi_5' in df.columns:
        df['rsi_5_prev'] = df['rsi_5'].shift(1)
    else:
        df['rsi_5_prev'] = np.nan # Or handle missing RSI_5 differently if appropriate

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

    if 'rsi_5_prev' in df.columns:
        df.drop(columns=['rsi_5_prev'], inplace=True)

    return df

# --- simulation.py run_backtest content ---
def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_per_trade_percentage: float = 0.01,
    max_investment_per_trade_percentage: float = 0.25,
    atr_multiplier: float = 2.0,
    take_profit_atr_multiplier: float = 3.0
) -> dict:
    """
    Runs a backtest simulation with dynamic, capped position sizing and dynamic take profit.

    Args:
        df (pd.DataFrame): DataFrame with necessary columns.
        initial_capital (float): Starting capital.
        risk_per_trade_percentage (float): Portfolio percentage to risk per trade.
        max_investment_per_trade_percentage (float): Max portfolio percentage for a single trade.
        atr_multiplier (float): ATR multiplier for stop loss.
        take_profit_atr_multiplier (float): ATR multiplier for take profit.

    Returns:
        dict: A dictionary containing performance metrics and the trades log.
    """


    required_cols = ['close', 'signal', 'signal_confidence', 'ATR_14', 'regime', 'datetime']
    if any(col not in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    # --- State Tracking & Logging ---
    df['position_shares'] = 0.0
    df['cash'] = initial_capital
    df['portfolio_value'] = initial_capital
    trades_log = []

    shares_in_position = 0.0
    buy_price_per_share = 0.0
    highest_price_since_buy = 0.0
    entry_atr = 0.0

    # --- Simulation Loop ---
    for i in range(1, len(df)):
        # Carry over portfolio values from the previous day
        df.loc[i, 'cash'] = df.loc[i-1, 'cash']
        df.loc[i, 'position_shares'] = df.loc[i-1, 'position_shares']
        df.loc[i, 'portfolio_value'] = df.loc[i-1, 'portfolio_value']

        current_close_price = df.loc[i, 'close']
        signal = df.loc[i, 'signal']
        confidence = df.loc[i, 'signal_confidence']
        current_atr = df.loc[i, 'ATR_14'] if pd.notna(df.loc[i, 'ATR_14']) else 0.0

        # --- LOGIC WHEN IN A POSITION ---
        if shares_in_position > 0:
            highest_price_since_buy = max(highest_price_since_buy, current_close_price)
            trailing_stop_price = highest_price_since_buy - (current_atr * atr_multiplier)
            take_profit_price = buy_price_per_share + (entry_atr * take_profit_atr_multiplier)

            is_take_profit_hit = current_close_price >= take_profit_price
            is_trailing_stop_hit = current_close_price <= trailing_stop_price
            is_signal_exit = signal == 'Exit Long'

            if is_take_profit_hit or is_trailing_stop_hit or is_signal_exit:
                exit_type = 'SELL (TP)' if is_take_profit_hit else ('SELL (Trail SL)' if is_trailing_stop_hit else 'SELL (Signal)')

                exit_value = current_close_price * shares_in_position
                pnl_dollars = (current_close_price - buy_price_per_share) * shares_in_position

                df.loc[i, 'cash'] += exit_value
                df.loc[i, 'position_shares'] = 0

                trades_log.append({
                    'Type': exit_type,
                    'Entry Price': buy_price_per_share,
                    'Exit Date': df.loc[i, 'datetime'],
                    'Exit Price': current_close_price,
                    'Quantity': shares_in_position,
                    'P/L ($)': pnl_dollars,
                    'Signal': signal if exit_type == 'SELL (Signal)' else exit_type,
                    'Regime': df.loc[i, 'regime']
                })
                shares_in_position, buy_price_per_share, highest_price_since_buy, entry_atr = 0, 0, 0, 0

        # --- LOGIC WHEN NOT IN A POSITION ---
        elif shares_in_position == 0 and signal == 'Enter Long':

            stop_loss_distance_per_share = current_atr * atr_multiplier
            if stop_loss_distance_per_share > 0:

                current_portfolio_value = df.loc[i, 'portfolio_value']

                # 1. Calculate position size based on risk, scaled by confidence
                dollar_amount_to_risk = current_portfolio_value * risk_per_trade_percentage * confidence
                num_shares_based_on_risk = np.floor(dollar_amount_to_risk / stop_loss_distance_per_share)

                # 2. Calculate position size based on max investment
                max_investment_dollars = current_portfolio_value * max_investment_per_trade_percentage
                num_shares_based_on_max_investment = np.floor(max_investment_dollars / current_close_price)

                # 3. Use the smaller of the two position sizes
                num_shares_to_buy = min(num_shares_based_on_risk, num_shares_based_on_max_investment)

                trade_cost = num_shares_to_buy * current_close_price

                if num_shares_to_buy > 0 and df.loc[i, 'cash'] >= trade_cost:
                    df.loc[i, 'position_shares'] = num_shares_to_buy
                    df.loc[i, 'cash'] -= trade_cost

                    buy_price_per_share = current_close_price
                    shares_in_position = num_shares_to_buy
                    highest_price_since_buy = current_close_price
                    entry_atr = current_atr

                    trades_log.append({
                        'Type': 'BUY',
                        'Entry Date': df.loc[i, 'datetime'],
                        'Entry Price': current_close_price,
                        'Quantity': num_shares_to_buy
                    })

        # Update portfolio value at the end of the day
        df.loc[i, 'portfolio_value'] = df.loc[i, 'cash'] + (df.loc[i, 'position_shares'] * current_close_price)


    # --- Process and Finalize Trades Log ---
    processed_trades_log = []
    open_trade = {}
    for trade_event in trades_log:
        if trade_event['Type'] == 'BUY':
            open_trade = trade_event.copy()
        elif 'SELL' in trade_event['Type'] and open_trade:
            sell_trade = {**open_trade, **trade_event}

            pnl_dollars = (sell_trade['Exit Price'] - sell_trade['Entry Price']) * sell_trade['Quantity']
            total_cost = sell_trade['Entry Price'] * sell_trade['Quantity']
            pnl_percent = (pnl_dollars / total_cost) * 100 if total_cost > 0 else 0

            sell_trade['P/L ($)'] = pnl_dollars
            sell_trade['P/L (%)'] = pnl_percent

            processed_trades_log.append(sell_trade)
            open_trade = {}

    trades_df_final = pd.DataFrame(processed_trades_log)

    # --- Performance Metrics ---
    final_portfolio_value = df['portfolio_value'].iloc[-1]
    total_return_pct = ((final_portfolio_value / initial_capital) - 1) * 100
    buy_hold_return_pct = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

    rolling_max = df['portfolio_value'].cummax()
    daily_drawdown = df['portfolio_value'] / rolling_max - 1.0
    max_drawdown_pct = daily_drawdown.min() * 100

    return {
        "final_portfolio_value": final_portfolio_value,
        "total_return_pct": total_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "trades_log": trades_df_final
    }
