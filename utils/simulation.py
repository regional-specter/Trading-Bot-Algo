
import pandas as pd
import numpy as np

def run_backtest(
    df: pd.DataFrame, 
    initial_capital: float = 100000.0, 
    invest_dollar_amount_per_trade: float = 5000.0, 
    atr_multiplier: float = 2.0, 
    take_profit_percentage: float = 0.0075
) -> dict:
    """
    Runs a backtest simulation on a DataFrame with trading signals.

    Args:
        df (pd.DataFrame): DataFrame with 'close', 'signal', 'ATR_14' columns.
        initial_capital (float): The starting capital for the backtest.
        invest_dollar_amount_per_trade (float): Fixed dollar amount to invest in each trade.
        atr_multiplier (float): Multiplier for ATR to set the trailing stop loss.
        take_profit_percentage (float): Percentage gain at which to exit a trade.

    Returns:
        dict: A dictionary containing performance metrics and the trades log.
    """
    
    required_cols = ['close', 'signal', 'ATR_14', 'regime', 'datetime']
    if any(col not in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing one of the required columns: {required_cols}")

    # --- State Tracking & Logging ---
    df['position_shares'] = 0
    df['cash'] = initial_capital
    df['portfolio_value'] = initial_capital
    trades_log = [] 

    shares_in_position = 0
    buy_price_per_share = 0.0
    highest_price_since_buy = 0.0

    # --- Simulation Loop ---
    for i in range(1, len(df)):
        df.loc[i, 'cash'] = df.loc[i-1, 'cash']
        df.loc[i, 'position_shares'] = df.loc[i-1, 'position_shares']

        current_close_price = df.loc[i, 'close']
        signal = df.loc[i, 'signal']
        current_atr = df.loc[i, 'ATR_14'] if pd.notna(df.loc[i, 'ATR_14']) else 0.0

        # --- LOGIC WHEN IN A POSITION ---
        if shares_in_position > 0:
            highest_price_since_buy = max(highest_price_since_buy, current_close_price)
            trailing_stop_price = highest_price_since_buy - (current_atr * atr_multiplier)
            take_profit_price = buy_price_per_share * (1 + take_profit_percentage)

            is_take_profit_hit = current_close_price >= take_profit_price
            is_trailing_stop_hit = current_close_price <= trailing_stop_price
            is_signal_exit = signal == 'Exit Long'

            if is_take_profit_hit or is_trailing_stop_hit or is_signal_exit:
                exit_type = 'SELL (TP)' if is_take_profit_hit else ('SELL (Trail SL)' if is_trailing_stop_hit else 'SELL (Signal)')
                
                pnl_dollars = (current_close_price - buy_price_per_share) * shares_in_position
                df.loc[i, 'cash'] += current_close_price * shares_in_position
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
                shares_in_position, buy_price_per_share, highest_price_since_buy = 0, 0, 0

        # --- LOGIC WHEN NOT IN A POSITION ---
        elif shares_in_position == 0 and signal == 'Enter Long':
            num_shares_to_buy = np.floor(invest_dollar_amount_per_trade / current_close_price)
            if num_shares_to_buy > 0 and df.loc[i, 'cash'] >= num_shares_to_buy * current_close_price:
                df.loc[i, 'position_shares'] = num_shares_to_buy
                df.loc[i, 'cash'] -= num_shares_to_buy * current_close_price

                buy_price_per_share = current_close_price 
                shares_in_position = num_shares_to_buy
                highest_price_since_buy = current_close_price

                trades_log.append({
                    'Type': 'BUY', 
                    'Entry Date': df.loc[i, 'datetime'],
                    'Entry Price': current_close_price, 
                    'Quantity': num_shares_to_buy
                })

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

if __name__ == '__main__':
    from regime import classify_regimes_with_kmeans
    from signals import generate_signals
    import pandas_ta as ta

    try:
        data_file = 'data/market_features.parquet'
        base_df = pd.read_parquet(data_file)
        
        # --- Prepare Data ---
        print("1. Classifying regimes...")
        df_with_regimes = classify_regimes_with_kmeans(base_df)
        
        print("2. Generating signals...")
        df_with_signals = generate_signals(df_with_regimes)

        # Ensure ATR is calculated, as it's needed for the backtest
        if 'ATR_14' not in df_with_signals.columns:
            print("Calculating ATR_14...")
            df_with_signals['ATR_14'] = ta.atr(
                high=df_with_signals['high'],
                low=df_with_signals['low'],
                close=df_with_signals['close'],
                length=14
            )
        df_with_signals.dropna(subset=['ATR_14'], inplace=True)
        df_with_signals.reset_index(drop=True, inplace=True)


        # --- Run Simulations with Different Parameters ---
        print("\n--- Starting Simulations ---")
        
        simulation_params = [
            {'atr_multiplier': 1.5, 'take_profit_percentage': 0.005},
            {'atr_multiplier': 2.0, 'take_profit_percentage': 0.0075},
            {'atr_multiplier': 2.5, 'take_profit_percentage': 0.01},
            {'atr_multiplier': 3.5, 'take_profit_percentage': 0.000085},
            {'atr_multiplier': 4.5, 'take_profit_percentage': 0.000095}
        ]
        
        for i, params in enumerate(simulation_params):
            print(f"\n--- Running Simulation #{i+1} ---")
            print(f"Parameters: ATR Multiplier={params['atr_multiplier']}, Take Profit={params['take_profit_percentage']:.2%}")
            
            results = run_backtest(
                df=df_with_signals.copy(), # Use a copy to avoid state issues
                atr_multiplier=params['atr_multiplier'],
                take_profit_percentage=params['take_profit_percentage']
            )
            
            print("  Performance Metrics:")
            print(f"    Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
            print(f"    Total Strategy Return: {results['total_return_pct']:.2f}%")
            print(f"    Max Drawdown:          {results['max_drawdown_pct']:.2f}%")
            print(f"    Number of Trades:      {len(results['trades_log'])}")
        
        print(f"\nReference Buy & Hold Return: {results['buy_hold_return_pct']:.2f}%")

    except FileNotFoundError:
        print(f"Error: Data file not found. Ensure you are in the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")
