## **Decision Engine (ML Model)**

- Financial time series does not obey stable distributions. in the sense that the bot must assume that statistical properties (mean, variance, correlations) shift across the window and any model that assumes IID data will quietly fail. This is related to data **non-stationarity**

- Indicators that look predictive in one window may be useless or harmful in the others. Evaluations must be walk-forward, and never random splits. This is related to **causality vs correlation**

- We are not optimising for a prediction error but we are optimising for capital growth under constraints (our budget, personal portfolio and more). This shifts the loss function from MSE to PnL-aware objectives (returns, drawdowns, risk-adjusted rewards and more). This is related to **decision-centric learning**

- Actions affect future states through capital, positive sizing, and exposure. This immediately disqualified pure supervised learning as a complete solution

    - **Layer 1 | Market State & Context Layer :** It processes raw price data into context such as trends, volatility, momentum strength and behaviours. This is where rolling statistics, volatility measures, regime classifiers or lightweight time-series models work. It describes the environment to the rest of the system

    - **Layer 2 | Signal & Edge Extraction Layer :** It works with outputs from Layer 1 and produces signals such as Directional bias (bullish / bearish / neutral), Confidence and expected short-term payoff. This is where supervised ML fits best. These models don’t decide trades but estimate advantage. Predicts and estimates whether a trade might be profitable and how strong that belief is

    - **Layer 3 | Decision & Policy Layer :** This layer takes Market context (Layer 1), Signal strength (Layer 2), current position, remaining capital, past outcomes and more to produce a market decision, position size and the timing preference. This is where **reinforcement learning** or rule-constrained policies belong

    - **Layer 4 | Risk, Positioning & Trade Management Layer :** This wraps every decision with hard rules such as position sizing, stop-loss placement, take-profit logic, max drawdown limits and more. This layer keeps the trader alive long enough to learn

## Further Todo :

Comprehensive Review: Possible Areas for Improvement Across All Layers

  Layer 1 | Market State & Context Layer (Regime Classification)
   * Current State: Multi-timescale RSI and trend_strength hybrid model.
   * Purpose: To accurately label the market environment (Uptrend - Pullback,
     Ranging, etc.). If this is inaccurate, all downstream layers will be flawed.
   * Areas for Improvement:
       1. Threshold Tuning (Crucial & Ongoing): Your current RSI (e.g.,
          RSI_PRIMARY_UP=55/45, RSI14_STRONG_BULLISH=60, RSI5_PULLBACK_MIN=30) and
          TREND_STRENGTH_THRESHOLD values are examples. These are your primary
          levers.
           * Action: Experiment with these thresholds to make regime classifications
             more precise. Use the ADX validation (and the numerical metrics) to
             guide you. For example, if Uptrend - Pullback periods often lead to
             losses, perhaps RSI5_PULLBACK_MIN is too low, allowing deeper
             pullbacks.
       2. Inclusion of Volatility Filter: Your model classifies Volatile periods,
          but doesn't actively avoid them yet. High volatility often means
          choppiness.
           * Action: Consider incorporating rolling_volatility into the
             classify_regime function to ensure an 'Uptrend - Pullback' isn't
             labeled during excessively volatile times, or create a 'Highly Volatile
             - Unpredictable' regime to avoid.
       3. Volume Confirmation: Price action confirmed by volume is generally more
          reliable.
           * Action: Integrate volume_zscore. For an 'Uptrend - Impulse', ideally
             volume_zscore should be positive. For an 'Uptrend - Pullback',
             volume_zscore should ideally be neutral or slightly negative (pullbacks
             on low volume are healthy).
       4. Regime Persistence: A "flickering" regime (switching rapidly) can generate
          too many noisy signals.
           * Action: Add logic to ensure a regime must persist for a minimum number
             of periods before a switch is confirmed.

  Layer 2 | Signal & Edge Extraction Layer (Signal Generation)
   * Current State: Selective long-only strategy: Bullish on strong Uptrend -
     Pullback (with rsi_14 > 60, rsi_5 > 30); Bearish on Ranging or any Downtrend.
   * Purpose: To define high-probability entry and exit points. This is where your
     actual "edge" is found.
   * Areas for Improvement:
       1. Refine Entry Triggers (Beyond Pullback):
           * Action: Is "buy the dip" the only strategy? What about a "breakout"
             strategy (e.g., 'Bullish' on Uptrend - Impulse after a clear Ranging
             period)?
           * Confirmation of Pullback End: Instead of buying during the pullback,
             buy after the pullback shows a clear sign of ending (e.g., rsi_5
             crosses above 50, or rsi_5 turns up from its recent low within the
             Uptrend - Pullback regime).
       2. Refine Exit Triggers (Timeliness):
           * Action: The current exit is on a regime change. Is this fast enough?
             Consider a more aggressive exit if the Uptrend - Impulse becomes weak
             (e.g., rsi_5 drops below 50).
           * Take-Profit Targets: Implement logic to take profits when a certain
             percentage gain is achieved (e.g., exit if price is +2% from entry).
       3. Signal Strength/Confidence: Can you assign a confidence level to signals?
           * Action: For example, a Bullish signal where rsi_14 is 70+ and
             volume_zscore is high might be a "High Conviction Buy."

  Layer 3 | Decision & Policy Layer (Trading Rules)
   * Current State: Simple: if Bullish and no position, buy; if Bearish and in
     position, sell.
   * Purpose: Translates signals into concrete actions, considering current position
     and capital.
   * Areas for Improvement:
       1. Position Sizing (Revisit `INVEST_DOLLAR_AMOUNT_PER_TRADE`): You set it to
          $5000.
           * Action: Is this optimal? What if you invest a percentage of your
             current capital instead of a fixed dollar amount? (e.g., risk 1% of
             equity per trade, calculate shares based on stop loss distance).
       2. Scaling In/Out:
           * Action: Instead of one large buy, can you scale into a position (e.g.,
             buy 50% now, 50% if it dips further but stays in Uptrend - Pullback)?

  Layer 4 | Risk, Positioning & Trade Management Layer (Execution & Protection)
   * Current State: Basic stop-loss (0.5%), fixed quantity position sizing.
   * Purpose: To protect capital, manage exposure, and ensure sustainability. This
     layer prevents catastrophic losses.
   * Areas for Improvement:
       1. Stop-Loss Tuning:
           * Action: The STOP_LOSS_PERCENTAGE = 0.005 (0.5%) is crucial. Is it too
             tight (getting stopped out too often before the move happens)? Is it
             too wide (allowing too much loss)? Experiment with 0.0075 (0.75%), 0.01
             (1%), etc.
           * Volatility-Based Stop: Instead of a fixed percentage, use a stop loss
             based on Average True Range (ATR). ATR_STOP_LOSS = ATR * N (e.g., 2 *
             ATR).
           * Trailing Stop-Loss: Move the stop loss up as the price moves in your
             favor, locking in profits.
       2. Take-Profit Orders:
           * Action: Implement a target. If price moves +X% from entry,
             automatically sell to lock in profits. This prevents winning trades
             from turning into losing trades.
       3. Transaction Costs & Slippage:
           * Action: Your current backtest assumes perfect execution at the close
             price with no fees. Add a small percentage for commission and slippage
             (e.g., 0.001 or 0.1% per trade) for a more realistic performance
             assessment.