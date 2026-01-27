import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import datetime, timedelta
from collections import deque
import random

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Imports for the new trading core logic
import pandas as pd
import pandas_ta as ta
import numpy as np
from utils.trader_core import classify_regimes_with_kmeans, generate_signals, run_backtest # Using the new module

console = Console()

def make_layout() -> Layout:
    """Defines the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(Layout(name="side", ratio=1), Layout(name="body", ratio=2))
    layout["side"].split(
        Layout(name="market_view", ratio=2),  # Combined market view
        Layout(name="signals_view", ratio=1)
    )
    layout["body"].split_row(Layout(name="trades_table", ratio=1))

    return layout

def generate_stacked_bar_chart(positive_change: float, negative_change: float, max_abs_change: float, bar_length: int = 50, bar_height: int = 2) -> Panel:
    """
    Generates a stacked bar chart panel with separate bars for positive and negative values.
    Bars pivot from the left.
    """
    if max_abs_change == 0:
        max_abs_change = 0.001 # Avoid division by zero

    panel_content = Text()

    # Positive bar (Green)
    scaled_positive = int((positive_change / max_abs_change) * bar_length)
    green_bar_line = Text("█" * scaled_positive, style="green") + Text(" " * (bar_length - scaled_positive), style="dim")
    for _ in range(bar_height):
        panel_content.append(green_bar_line)
        # Only add newline if not the last line of the bar and it's not the end of the entire panel content for this bar
        if _ < bar_height - 1:
            panel_content.append("\n")

    # Add separator/space between positive and negative bars, only if there are bars
    if bar_height > 0:
        panel_content.append("\n")
    
    # Negative bar (Red)
    scaled_negative = int((abs(negative_change) / max_abs_change) * bar_length)
    red_bar_line = Text("█" * scaled_negative, style="red") + Text(" " * (bar_length - scaled_negative), style="dim")
    for _ in range(bar_height):
        panel_content.append(red_bar_line)
        # Only add newline if not the last line of the bar
        if _ < bar_height - 1:
            panel_content.append("\n")


    title = f"Price Movement (Up: {positive_change:+.4f}, Down: {negative_change:+.4f})"
    return Panel(panel_content, title=title, border_style="blue")


def generate_signals_panel(
    current_signal: str = "Neutral",
    confidence: float = 0.0,
    regime: str = "Unknown",
    current_close: float = 0.0,
    current_atr: float = 0.0,
    rsi_5: float = 0.0,
    rsi_14: float = 0.0,
    volume_zscore: float = 0.0,
    trend_strength: float = 0.0,
    portfolio_value: float = 0.0
):
    """Generates a signals panel with real-time signal, confidence, regime, and key indicators."""
    signals_content = Text()
    signals_content.append(f"Price: {current_close:.2f}\n", style="bold white")
    signals_content.append(f"Portfolio: ${portfolio_value:,.2f}\n", style="bold green")
    signals_content.append(f"Regime: {regime}\n", style="bold yellow")
    signals_content.append(f"Signal: {current_signal}\n", style="bold blue")
    signals_content.append(f"Confidence: {confidence:.2f}\n", style="bold green")
    signals_content.append(f"ATR (14): {current_atr:.4f}\n", style="white")
    signals_content.append(f"RSI (5): {rsi_5:.2f} | RSI (14): {rsi_14:.2f}\n", style="white")
    signals_content.append(f"Volume Z: {volume_zscore:.2f}\n", style="white")
    signals_content.append(f"Trend Strength: {trend_strength:.2f}\n", style="white")
    return Panel(signals_content, title="Market Analysis", border_style="blue")


def generate_trades_table(trades_log: list):
    """Generates a table of trades."""
    table = Table(title="Trade Log")
    table.add_column("Datetime", justify="center", style="cyan", no_wrap=True)
    table.add_column("Type", justify="center", style="magenta", no_wrap=True)
    table.add_column("Price", justify="right", style="green", no_wrap=True)
    table.add_column("Qty", justify="right", style="yellow", no_wrap=True)
    table.add_column("P/L ($)", justify="right", style="red", no_wrap=True)
    table.add_column("P/L (%)", justify="right", style="red", no_wrap=True)


    if not trades_log:
        table.add_row("N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    else:
        # Displaying last few trades, limit for TUI readability
        for trade in trades_log[-10:]:
            trade_type = trade.get('Type', 'N/A')
            price = trade.get('Exit Price', trade.get('Entry Price', 'N/A'))
            qty = trade.get('Quantity', 'N/A')
            pnl_dollar = trade.get('P/L ($)', 'N/A')
            pnl_percent = trade.get('P/L (%)', 'N/A')
            trade_date = trade.get('Exit Date', trade.get('Entry Date', datetime.now())).strftime("%H:%M:%S")

            table.add_row(
                str(trade_date),
                str(trade_type),
                f"{price:.2f}" if isinstance(price, (int, float)) else str(price),
                f"{qty:.2f}" if isinstance(qty, (int, float)) else str(qty),
                f"{pnl_dollar:.2f}" if isinstance(pnl_dollar, (int, float)) else str(pnl_dollar),
                f"{pnl_percent:.2f}" if isinstance(pnl_percent, (int, float)) else str(pnl_percent)
            )

    return Panel(table, title="Trade Log", border_style="blue")


if __name__ == "__main__":
    # --- Data Loading and Preprocessing ---
    DATA_FILE = 'data/market_features.parquet'
    try:
        full_df = pd.read_parquet(DATA_FILE)
        full_df['datetime'] = pd.to_datetime(full_df['datetime'])
        full_df.set_index('datetime', inplace=True)
        full_df.sort_index(inplace=True)

        # Ensure ATR is calculated upfront for the whole dataset
        if 'ATR_14' not in full_df.columns:
            full_df['ATR_14'] = ta.atr(
                high=full_df['high'],
                low=full_df['low'],
                close=full_df['close'],
                length=14
            )
        full_df.dropna(subset=['ATR_14'], inplace=True)

        console.print(f"Loaded {len(full_df)} rows from {DATA_FILE}")

    except FileNotFoundError:
        console.print(f"[bold red]Error: Data file not found at '{DATA_FILE}'. Ensure you are in the project root and the data is available.[/bold red]")
        exit()
    except Exception as e:
        console.print(f"[bold red]An error occurred during data loading or preprocessing: {e}[/bold red]")
        exit()

    # --- Simulation State ---
    # Using default parameters for signals and backtest for now, will allow customization later.
    SIGNAL_PARAMS = {
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

    TRADE_PARAMS = {
        'initial_capital': 100000.0,
        'risk_per_trade_percentage': 0.01,
        'max_investment_per_trade_percentage': 0.25,
        'atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0
    }

def make_layout() -> Layout:
    """Defines the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(name="footer", size=3),
    )

    # Use ratios to dynamically size side and body
    layout["main"].split_row(Layout(name="side", ratio=3), Layout(name="body", ratio=5))
    layout["side"].split(
        Layout(name="market_view", ratio=2),  # Combined market view
        Layout(name="signals_view", ratio=3)
    )
    layout["body"].split_row(Layout(name="trades_table", ratio=1))

    return layout

def generate_stacked_bar_chart(positive_change: float, negative_change: float, max_abs_change: float, bar_length: int = 50, bar_height: int = 2) -> Panel:
    """
    Generates a stacked bar chart panel with separate bars for positive and negative values.
    Bars pivot from the left.
    """
    if max_abs_change == 0:
        max_abs_change = 0.001 # Avoid division by zero

    panel_content = Text()

    # Positive bar (Green)
    scaled_positive = int((positive_change / max_abs_change) * bar_length)
    green_bar = Text("█" * scaled_positive, style="green") + Text(" " * (bar_length - scaled_positive), style="dim")
    for _ in range(bar_height):
        panel_content.append(green_bar)
    panel_content.append("\n") # Newline between positive and negative bars

    # Negative bar (Red)
    scaled_negative = int((abs(negative_change) / max_abs_change) * bar_length)
    red_bar = Text("█" * scaled_negative, style="red") + Text(" " * (bar_length - scaled_negative), style="dim")
    for _ in range(bar_height):
        panel_content.append(red_bar)

    title = f"Price Movement (Up: {positive_change:+.4f}, Down: {negative_change:+.4f})"
    return Panel(panel_content, title=title, border_style="blue")


def generate_signals_panel(
    current_signal: str = "Neutral",
    confidence: float = 0.0,
    regime: str = "Unknown",
    current_close: float = 0.0,
    current_atr: float = 0.0,
    rsi_5: float = 0.0,
    rsi_14: float = 0.0,
    volume_zscore: float = 0.0,
    trend_strength: float = 0.0,
    portfolio_value: float = 0.0
):
    """Generates a signals panel with real-time signal, confidence, regime, and key indicators."""
    signals_content = Text()
    signals_content.append(f"Price: {current_close:.2f}\n", style="bold white")
    signals_content.append(f"Portfolio: ${portfolio_value:,.2f}\n", style="bold green")
    signals_content.append(f"Regime: {regime}\n", style="bold yellow")
    signals_content.append(f"Signal: {current_signal}\n", style="bold blue")
    signals_content.append(f"Confidence: {confidence:.2f}\n", style="bold green")
    signals_content.append(f"ATR (14): {current_atr:.4f}\n", style="white")
    signals_content.append(f"RSI (5): {rsi_5:.2f} | RSI (14): {rsi_14:.2f}\n", style="white")
    signals_content.append(f"Volume Z: {volume_zscore:.2f}\n", style="white")
    signals_content.append(f"Trend Strength: {trend_strength:.2f}\n", style="white")
    return Panel(signals_content, title="Market Analysis", border_style="blue")


def generate_trades_table(trades_log: list):
    """Generates a table of trades."""
    table = Table(title="Trade Log")
    table.add_column("Datetime", justify="center", style="cyan", no_wrap=True)
    table.add_column("Type", justify="center", style="magenta", no_wrap=True)
    table.add_column("Price", justify="right", style="green", no_wrap=True)
    table.add_column("Qty", justify="right", style="yellow", no_wrap=True)
    table.add_column("P/L ($)", justify="right", style="red", no_wrap=True)
    table.add_column("P/L (%)", justify="right", style="red", no_wrap=True)


    if not trades_log:
        table.add_row("N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    else:
        # Displaying last few trades, limit for TUI readability
        for trade in trades_log[-10:]:
            trade_type = trade.get('Type', 'N/A')
            price = trade.get('Exit Price', trade.get('Entry Price', 'N/A'))
            qty = trade.get('Quantity', 'N/A')
            pnl_dollar = trade.get('P/L ($)', 'N/A')
            pnl_percent = trade.get('P/L (%)', 'N/A')
            trade_date = trade.get('Exit Date', trade.get('Entry Date', datetime.now())).strftime("%H:%M:%S")

            table.add_row(
                str(trade_date),
                str(trade_type),
                f"{price:.2f}" if isinstance(price, (int, float)) else str(price),
                f"{qty:.2f}" if isinstance(qty, (int, float)) else str(qty),
                f"{pnl_dollar:.2f}" if isinstance(pnl_dollar, (int, float)) else str(pnl_dollar),
                f"{pnl_percent:.2f}" if isinstance(pnl_percent, (int, float)) else str(pnl_percent)
            )

    return Panel(table, title="Trade Log", border_style="blue")


if __name__ == "__main__":
    # --- Data Loading and Preprocessing ---
    DATA_FILE = 'data/market_features.parquet'
    try:
        full_df = pd.read_parquet(DATA_FILE)
        full_df['datetime'] = pd.to_datetime(full_df['datetime'])
        full_df.set_index('datetime', inplace=True)
        full_df.sort_index(inplace=True)

        # Ensure ATR is calculated upfront for the whole dataset
        if 'ATR_14' not in full_df.columns:
            full_df['ATR_14'] = ta.atr(
                high=full_df['high'],
                low=full_df['low'],
                close=full_df['close'],
                length=14
            )
        full_df.dropna(subset=['ATR_14'], inplace=True)

        console.print(f"Loaded {len(full_df)} rows from {DATA_FILE}")

    except FileNotFoundError:
        console.print(f"[bold red]Error: Data file not found at '{DATA_FILE}'. Ensure you are in the project root and the data is available.[/bold red]")
        exit()
    except Exception as e:
        console.print(f"[bold red]An error occurred during data loading or preprocessing: {e}[/bold red]")
        exit()

    # --- Simulation State ---
    # Using default parameters for signals and backtest for now, will allow customization later.
    SIGNAL_PARAMS = {
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

    TRADE_PARAMS = {
        'initial_capital': 100000.0,
        'risk_per_trade_percentage': 0.01,
        'max_investment_per_trade_percentage': 0.25,
        'atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0
    }

    # Initialize simulation variables
    sim_cash = TRADE_PARAMS['initial_capital']
    sim_shares_in_position = 0.0
    sim_buy_price_per_share = 0.0
    sim_highest_price_since_buy = 0.0
    sim_entry_atr = 0.0
    sim_trades_log = []
    sim_portfolio_value = TRADE_PARAMS['initial_capital']

    # Store recent absolute price changes for gauge scaling
    recent_abs_price_changes = deque(maxlen=50)

    layout = make_layout()

    header_panel = Panel("Trading Bot Dashboard", style="bold white on black")
    layout['header'].update(header_panel)

    # Initialize with empty data
    initial_chart_max_value = full_df['ATR_14'].mean() * 3 if not full_df['ATR_14'].empty else 0.1
    layout['market_view'].update(generate_stacked_bar_chart(0.0, 0.0, max_abs_change=initial_chart_max_value))
    layout['signals_view'].update(generate_signals_panel())
    layout['body'].update(generate_trades_table(sim_trades_log))

    DEBUG_MODE = True # Set to True to enable debug prints

    with Live(layout, screen=True, refresh_per_second=4) as live: # refresh 4 times per second
        try:
            # Iterate through the pre-loaded DataFrame
            for i in range(len(full_df)):
                current_tick_df = full_df.iloc[i:i+1].copy() # Process one row at a time
                if current_tick_df.empty:
                    continue

                current_datetime = current_tick_df.index[0]
                current_close_price = current_tick_df['close'].iloc[0]
                current_high_price = current_tick_df['high'].iloc[0]
                current_low_price = current_tick_df['low'].iloc[0]
                current_atr = current_tick_df['ATR_14'].iloc[0]

                # --- Layer 1 & 2: Get Regime and Signal for current tick ---
                window_size = 30
                start_idx = max(0, i - window_size + 1)
                processing_df = full_df.iloc[start_idx : i + 1].copy()

                signal_to_display = "Warmup"
                confidence_to_display = 0.0
                regime_to_display = "Warmup"
                rsi_5_to_display = 0.0
                rsi_14_to_display = 0.0
                volume_zscore_to_display = 0.0
                trend_strength_to_display = 0.0

                if len(processing_df) >= max(14, window_size): # Ensure enough data for indicators
                    # Recalculate features for the processing window if they aren't directly available
                    # This is a simplification; in a real-time system, features would be incrementally updated.
                    # For now, we fetch from the pre-calculated full_df for display.
                    # Only apply clustering/signals if there's enough data
                    temp_df_for_regime = processing_df.copy()
                    
                    # Ensure RSI values are present for signals
                    if 'rsi_5' not in temp_df_for_regime.columns:
                        temp_df_for_regime['rsi_5'] = ta.rsi(temp_df_for_regime['close'], length=5)
                    if 'rsi_14' not in temp_df_for_regime.columns:
                        temp_df_for_regime['rsi_14'] = ta.rsi(temp_df_for_regime['close'], length=14)
                    if 'volume_zscore' not in temp_df_for_regime.columns:
                        # Assuming 'volume' column exists
                        if 'volume' in temp_df_for_regime.columns:
                             temp_df_for_regime['volume_zscore'] = (temp_df_for_regime['volume'] - temp_df_for_regime['volume'].rolling(window=window_size).mean()) / temp_df_for_regime['volume'].rolling(window=window_size).std()
                        else:
                            temp_df_for_regime['volume_zscore'] = 0.0 # Default if no volume
                    if 'trend_strength' not in temp_df_for_regime.columns:
                        # Simple proxy for trend strength, can be ADX or similar
                        temp_df_for_regime['trend_strength'] = abs(temp_df_for_regime['close'].diff(window_size).fillna(0))


                    df_with_regimes = classify_regimes_with_kmeans(temp_df_for_regime.dropna().copy())
                    if not df_with_regimes.empty:
                        last_regime_row = df_with_regimes.iloc[-1]
                        regime_to_display = last_regime_row['regime']

                        df_with_signals = generate_signals(df_with_regimes.copy(), signal_params=SIGNAL_PARAMS)
                        last_signal_row = df_with_signals.iloc[-1]
                        signal_to_display = last_signal_row['signal']
                        confidence_to_display = last_signal_row['signal_confidence']

                        # Extract additional info for signals panel
                        rsi_5_to_display = last_regime_row.get('rsi_5', 0.0)
                        rsi_14_to_display = last_regime_row.get('rsi_14', 0.0)
                        volume_zscore_to_display = last_regime_row.get('volume_zscore', 0.0)
                        trend_strength_to_display = last_regime_row.get('trend_strength', 0.0)


                if DEBUG_MODE:
                    console.print(f"[{current_datetime.strftime('%H:%M:%S')}] Regime: {regime_to_display:<20}, Signal: {signal_to_display:<15}, Conf: {confidence_to_display:.2f}, Pos: {sim_shares_in_position:.2f}")


                # --- Layer 3 & 4: Decision & Risk (simplified for real-time display) ---
                new_trade_made = False

                if sim_shares_in_position > 0:
                    sim_highest_price_since_buy = max(sim_highest_price_since_buy, current_high_price)
                    trailing_stop_price = sim_highest_price_since_buy - (sim_entry_atr * TRADE_PARAMS['atr_multiplier'])
                    take_profit_price = sim_buy_price_per_share + (sim_entry_atr * TRADE_PARAMS['take_profit_atr_multiplier'])

                    is_take_profit_hit = current_close_price >= take_profit_price
                    is_trailing_stop_hit = current_close_price <= trailing_stop_price
                    is_signal_exit = signal_to_display == 'Exit Long'

                    if is_take_profit_hit or is_trailing_stop_hit or is_signal_exit:
                        exit_type = 'SELL (TP)' if is_take_profit_hit else ('SELL (Trail SL)' if is_trailing_stop_hit else 'SELL (Signal)')

                        exit_value = current_close_price * sim_shares_in_position
                        pnl_dollars = (current_close_price - sim_buy_price_per_share) * sim_shares_in_position
                        total_cost = sim_buy_price_per_share * sim_shares_in_position
                        pnl_percent = (pnl_dollars / total_cost) * 100 if total_cost > 0 else 0

                        sim_cash += exit_value
                        sim_portfolio_value = sim_cash

                        # Find the corresponding BUY trade to update
                        matching_buy_index = -1
                        for trade_idx, trade in enumerate(sim_trades_log):
                            if trade['Type'] == 'BUY' and 'Exit Date' not in trade:
                                matching_buy_index = trade_idx
                                break

                        if matching_buy_index != -1:
                            sim_trades_log[matching_buy_index].update({
                                'Exit Date': current_datetime,
                                'Exit Price': current_close_price,
                                'P/L ($)': pnl_dollars,
                                'P/L (%)': pnl_percent,
                                'Signal': signal_to_display if exit_type == 'SELL (Signal)' else exit_type,
                                'Regime': regime_to_display
                            })
                        else:
                            sim_trades_log.append({
                                'Type': exit_type,
                                'Entry Date': "N/A", # No matching buy found
                                'Entry Price': sim_buy_price_per_share,
                                'Exit Date': current_datetime,
                                'Exit Price': current_close_price,
                                'Quantity': sim_shares_in_position,
                                'P/L ($)': pnl_dollars,
                                'P/L (%)': pnl_percent,
                                'Signal': signal_to_display if exit_type == 'SELL (Signal)' else exit_type,
                                'Regime': regime_to_display
                            })

                        if DEBUG_MODE:
                            console.print(f"[{current_datetime.strftime('%H:%M:%S')}] --- SELL ({exit_type}) --- Qty: {sim_shares_in_position:.2f}, Price: {current_close_price:.2f}, PnL: {pnl_dollars:+.2f}, Port: {sim_portfolio_value:,.2f}")
                        
                        sim_shares_in_position, sim_buy_price_per_share, sim_highest_price_since_buy, sim_entry_atr = 0, 0, 0, 0
                        new_trade_made = True

                elif sim_shares_in_position == 0 and signal_to_display == 'Enter Long':
                    stop_loss_distance_per_share = current_atr * TRADE_PARAMS['atr_multiplier']
                    if stop_loss_distance_per_share > 0 and confidence_to_display > 0:
                        dollar_amount_to_risk = sim_portfolio_value * TRADE_PARAMS['risk_per_trade_percentage'] * confidence_to_display
                        num_shares_based_on_risk = np.floor(dollar_amount_to_risk / stop_loss_distance_per_share)

                        max_investment_dollars = sim_portfolio_value * TRADE_PARAMS['max_investment_per_trade_percentage']
                        num_shares_based_on_max_investment = np.floor(max_investment_dollars / current_close_price)

                        num_shares_to_buy = min(num_shares_based_on_risk, num_shares_based_on_max_investment)
                        trade_cost = num_shares_to_buy * current_close_price

                        if num_shares_to_buy > 0 and sim_cash >= trade_cost:
                            sim_shares_in_position = num_shares_to_buy
                            sim_cash -= trade_cost
                            sim_buy_price_per_share = current_close_price
                            sim_highest_price_since_buy = current_high_price
                            sim_entry_atr = current_atr

                            sim_trades_log.append({
                                'Type': 'BUY',
                                'Entry Date': current_datetime,
                                'Entry Price': current_close_price,
                                'Quantity': num_shares_to_buy,
                                'Signal': signal_to_display,
                                'Regime': regime_to_display
                            })
                            if DEBUG_MODE:
                                console.print(f"[{current_datetime.strftime('%H:%M:%S')}] *** BUY *** Qty: {num_shares_to_buy:.2f}, Price: {current_close_price:.2f}, Port: {sim_portfolio_value:,.2f}, Conf: {confidence_to_display:.2f}")
                            new_trade_made = True

                sim_portfolio_value = sim_cash + (sim_shares_in_position * current_close_price)

                # --- Calculate price change for chart ---
                current_price_change = 0.0
                if i > 0:
                    current_price_change = current_close_price - full_df['close'].iloc[i-1]
                
                positive_change_for_chart = max(0.0, current_price_change)
                negative_change_for_chart = min(0.0, current_price_change)
                
                # Keep track of recent absolute price changes to scale the bars
                if i > 0: # Only append if there was a previous point to calculate change from
                    recent_abs_price_changes.append(abs(current_price_change))
                
                # Determine max_abs_change for scaling dynamically
                chart_max_value = max(
                    current_atr * 3, # Example: 3 times current ATR
                    max(recent_abs_price_changes) if recent_abs_price_changes else 0.001 # Max of recent changes
                )
                if chart_max_value == 0: chart_max_value = 0.001 # Prevent div by zero


                # --- Update Panels ---
                layout['market_view'].update(generate_stacked_bar_chart(positive_change_for_chart, negative_change_for_chart, chart_max_value))
                layout['signals_view'].update(generate_signals_panel(
                    signal_to_display,
                    confidence_to_display,
                    regime_to_display,
                    current_close=current_close_price,
                    current_atr=current_atr,
                    rsi_5=rsi_5_to_display,
                    rsi_14=rsi_14_to_display,
                    volume_zscore=volume_zscore_to_display,
                    trend_strength=trend_strength_to_display,
                    portfolio_value=sim_portfolio_value
                ))
                layout['body'].update(generate_trades_table(sim_trades_log))

                time.sleep(0.25)

        except KeyboardInterrupt:
            pass