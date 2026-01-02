from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, BarColumn
from rich.text import Text
import time

from data_collection import load_data, sample_time_window

console = Console()

REFRESH_SECONDS = 0.5
WINDOW_SIZE = 5


def make_layout():
    layout = Layout()

    layout.split_column(
        Layout(name="top", size=4),     # much shorter
        Layout(name="bottom")
    )

    layout["top"].split_row(
        Layout(name="snapshot", ratio=2),
        Layout(name="bars", ratio=5)
    )

    layout["bottom"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    return layout


def snapshot_panel(candle, equity, position):
    text = Text(
        f"{candle.name}\n"
        f"Price: {float(candle['Close']):.2f}\n"
        f"Equity: ${equity:.2f} | Pos: {position}"
    )

    return Panel(text, border_style="cyan", title="Market")


def make_exposure_progress():
    progress = Progress(
        BarColumn(bar_width=30),
        expand=True
    )

    long_task = progress.add_task("[green]LONG", total=100)
    short_task = progress.add_task("[red]SHORT", total=100)

    return progress, long_task, short_task

def exposure_panel(progress, height=3):
    bars = "\n".join(str(progress) for _ in range(height))
    return Panel(bars, title="Exposure")

def left_panel(equity):
    return Panel(
        f"Equity: {equity:.2f}\nPnL: 0.00\nDD: 0%",
        title="State"
    )


def right_panel():
    return Panel(
        "Action: HOLD\nConfidence: 0.61",
        title="Decision"
    )


def run():
    df = load_data("AAPL")
    window = sample_time_window(df, WINDOW_SIZE)

    equity = 10_000
    position = "FLAT"
    position_size = 0  # -100 → 100

    layout = make_layout()

    progress, long_task, short_task = make_exposure_progress()

    with Live(
        Panel(layout, title="PAPER TRADING SIMULATION"),
        refresh_per_second=10,
        screen=False
    ):
        for _, candle in window.iterrows():

            # fake exposure movement for demo
            position_size = (position_size + 20) % 100

            progress.update(long_task, completed=position_size)
            progress.update(short_task, completed=0)

            layout["snapshot"].update(
                snapshot_panel(candle, equity, position)
            )

            layout["bars"].update(
                exposure_panel(progress, height=3)
           )

            layout["left"].update(left_panel(equity))
            layout["right"].update(right_panel())

            time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    run()
