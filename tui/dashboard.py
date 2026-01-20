
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

console = Console()

def make_layout() -> Layout:
    """Defines the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2))
    layout["side"].split(Layout(name="market_view", ratio=2), Layout(name="signals_view", ratio=1))
    return layout

def generate_market_chart(data):
    """Generates a market chart panel."""
    chart = Text()
    for price in data:
        if price > 0:
            chart.append("█", style="green")
        else:
            chart.append("█", style="red")
    
    # To make it 2 chars tall
    chart_tall = Text("\n").join([chart, chart])

    return Panel(chart_tall, title="Market Price", border_style="blue")

def generate_signals_panel():
    """Generates a signals panel."""
    signals_content = "Market is currently trending upwards.\n"
    signals_content += "Signal: Strong Buy\n"
    signals_content += "Confidence: 85%"
    return Panel(signals_content, title="Market Analysis", border_style="blue")

def generate_trades_table():
    """Generates a table of trades."""
    table = Table(title="Trades")
    table.add_column("Datetime", justify="center", style="cyan")
    table.add_column("Side", justify="center", style="magenta")
    table.add_column("Price", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("PnL", justify="right", style="red")

    now = datetime.now()
    # Dummy data
    table.add_row(now.strftime("%Y-%m-%d %H:%M:%S"), "BUY", "42000.12", "0.1", "+10.50")
    table.add_row((now - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"), "SELL", "42010.50", "0.1", "+0.93")
    table.add_row((now - timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S"), "BUY", "41990.80", "0.2", "-20.00")
    
    return Panel(table, title="Trade Log", border_style="blue")

if __name__ == "__main__":
    market_data = deque(maxlen=50)
    for _ in range(50):
        market_data.append(random.choice([-1, 1]))

    layout = make_layout()

    # Create panels
    header_panel = Panel("Trading Bot Dashboard", style="bold white on black")
    
    layout['header'].update(header_panel)
    layout['market_view'].update(generate_market_chart(market_data))
    layout['signals_view'].update(generate_signals_panel())
    layout['body'].update(generate_trades_table())


    with Live(layout, redirect_stderr=False) as live:
        try:
            while True:
                time.sleep(1)
                market_data.append(random.choice([-1, 1]))
                layout['market_view'].update(generate_market_chart(market_data))
                live.refresh()
        except KeyboardInterrupt:
            pass

