import pandas_ta as ta
import pandas as pd
import numpy as np
import optuna
from regime import classify_regimes_with_kmeans
from signals import generate_signals  

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

if __name__ == '__main__':
    # --- Data Preparation ---
    try:
        data_file = 'data/market_features.parquet'
        base_df = pd.read_parquet(data_file)
        
        # --- Prepare Data ---
        print("1. Classifying regimes...")
        
        df_with_regimes = classify_regimes_with_kmeans(base_df)
        
        # We will now generate signals inside the optimization loop
        # print("2. Generating signals...")
        # df_with_signals = generate_signals(df_with_regimes)

        # Ensure ATR is calculated
        if 'ATR_14' not in df_with_regimes.columns:
            df_with_regimes['ATR_14'] = ta.atr(
                high=df_with_regimes['high'], 
                low=df_with_regimes['low'],
                close=df_with_regimes['close'], 
                length=14
            )
            
        df_with_regimes.dropna(subset=['ATR_14'], inplace=True)
        df_with_regimes.reset_index(drop=True, inplace=True)

        # --- Optuna Optimization ---
        def objective(trial):
            """
            This is the function Optuna will try to maximize.
            It runs a backtest with a set of parameters and returns a performance score.
            """
            # 1. Define the search space for your parameters
            trade_params = {
                'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 5.0),
                'take_profit_atr_multiplier': trial.suggest_float('take_profit_atr_multiplier', 1.0, 10.0),
                'risk_per_trade_percentage': trial.suggest_float('risk_per_trade_percentage', 0.005, 0.05, log=True),
                'max_investment_per_trade_percentage': trial.suggest_float('max_investment_per_trade_percentage', 0.1, 0.5)
            }
            
            signal_params = {
                'uptrend_impulse_vol_z_threshold': trial.suggest_float('uptrend_impulse_vol_z_threshold', -1.0, 2.0),
                'accumulation_vol_z_threshold': trial.suggest_float('accumulation_vol_z_threshold', 0.0, 3.0),
                'pullback_rsi_5_lt': trial.suggest_int('pullback_rsi_5_lt', 40, 60),
                'pullback_rsi_14_gt': trial.suggest_int('pullback_rsi_14_gt', 50, 70),
                'ranging_uptrend_vol_z_threshold': trial.suggest_float('ranging_uptrend_vol_z_threshold', -1.0, 2.0),
                'uptrend_impulse_confidence': trial.suggest_float('uptrend_impulse_confidence', 0.5, 1.0),
                'accumulation_confidence': trial.suggest_float('accumulation_confidence', 0.5, 1.0),
                'pullback_confidence': trial.suggest_float('pullback_confidence', 0.5, 1.0),
                'ranging_uptrend_confidence': trial.suggest_float('ranging_uptrend_confidence', 0.5, 1.0)
            }

            # 2. Generate signals with the suggested parameters
            df_with_signals = generate_signals(df_with_regimes.copy(), signal_params=signal_params)
            
            # 3. Run the backtest with the suggested parameters
            results = run_backtest(
                df=df_with_signals,
                atr_multiplier=trade_params['atr_multiplier'],
                take_profit_atr_multiplier=trade_params['take_profit_atr_multiplier'],
                risk_per_trade_percentage=trade_params['risk_per_trade_percentage'],
                max_investment_per_trade_percentage=trade_params['max_investment_per_trade_percentage']
            )
            
            # 4. Return the value to be maximized (your objective)
            profit = results['total_return_pct']
            drawdown = abs(results['max_drawdown_pct'])
            num_trades = len(results['trades_log'])

            # Penalize strategies that don't trade
            if num_trades < 10:
                return -1000.0

            if drawdown < 0.1: 
                drawdown = 0.1
            
            # Score should reward profit and number of trades, but penalize drawdown
            score = (profit / drawdown) * (1 + num_trades / 100.0) # Simple scaling factor
            
            if profit <= 0:
                return float(profit)
                
            return float(score)

        # --- Create and run the optimization study ---
        print("\n--- Starting Bayesian Optimization with Optuna (200 Trials) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200, show_progress_bar=True)

        # --- Print the results ---
        print("\n--- Optimization Complete ---")
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        best_trial = study.best_trial

        print(f"  Value (Optimized Score): {best_trial.value:.4f}")
        print("  Best Parameters Found:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value:.4f}")

        # --- Run and display the final backtest with the best parameters ---
        print("\n--- Running Final Backtest with Best Parameters ---")
        best_params = best_trial.params
        
        final_signal_params = {
            'uptrend_impulse_vol_z_threshold': best_params.get('uptrend_impulse_vol_z_threshold'),
            'accumulation_vol_z_threshold': best_params.get('accumulation_vol_z_threshold'),
            'pullback_rsi_5_lt': best_params.get('pullback_rsi_5_lt'),
            'pullback_rsi_14_gt': best_params.get('pullback_rsi_14_gt'),
            'ranging_uptrend_vol_z_threshold': best_params.get('ranging_uptrend_vol_z_threshold'),
            'uptrend_impulse_confidence': best_params.get('uptrend_impulse_confidence'),
            'accumulation_confidence': best_params.get('accumulation_confidence'),
            'pullback_confidence': best_params.get('pullback_confidence'),
            'ranging_uptrend_confidence': best_params.get('ranging_uptrend_confidence')
        }
        
        final_df_with_signals = generate_signals(df_with_regimes.copy(), signal_params=final_signal_params)
        
        print("\n--- Signal Counts with Best Parameters ---")
        print(final_df_with_signals['signal'].value_counts())
        
        final_results = run_backtest(
            df=final_df_with_signals,
            atr_multiplier=best_params.get('atr_multiplier'),
            take_profit_atr_multiplier=best_params.get('take_profit_atr_multiplier'),
            risk_per_trade_percentage=best_params.get('risk_per_trade_percentage'),
            max_investment_per_trade_percentage=best_params.get('max_investment_per_trade_percentage')
        )
        
        print("\n  --- Final Performance Metrics ---")
        print(f"    Final Portfolio Value: ${final_results['final_portfolio_value']:,.2f}")
        print(f"    Total Strategy Return: {final_results['total_return_pct']:.2f}%")
        print(f"    Max Drawdown:          {final_results['max_drawdown_pct']:.2f}%")
        print(f"    Number of Trades:      {len(final_results['trades_log'])}")
        print(f"\nReference Buy & Hold Return: {final_results['buy_hold_return_pct']:.2f}%")

    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file}'. Ensure you are in the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")
