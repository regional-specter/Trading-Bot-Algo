"""
The model does not necessarily predict anything, yet it uses a timeseries model and 
compresses time into structure so the market trend can be understood by the rest of 
the layers in the bot

In this context, a trend is persistent directional bias after removing noise

That means:
- Direction (up / down / flat)
- Strength (weak → strong)
- Stability (smooth → choppy)

Possible Models :

1) Kalman Filter (Trend + Noise Decomposition)

Separates observed price into:
- Hidden trend state
- Measurement noise
Produces a smooth, adaptive trend line that reacts faster than MAs without overreacting

2) State-Space Models

Formalize price as:
- Latent trend
- Optional drift
- Optional mean reversion
Useful when you want explicit regime probabilities

3) AR / ARIMA-style Models (Constrained)
Used only to estimate directional persistence, not forecasts

4) Tiny RNN / 1D-CNN (Optional)
Only if heavily regularized. Outputs trend confidence, not price targets

There are constraints to this model:

- Must never see future data
- Must not optimize for PnL
- Must not emit trade signals
"""