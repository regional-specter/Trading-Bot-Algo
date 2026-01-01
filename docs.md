## **Decision Engine (ML Model)**

- Financial time series does not obey stable distributions. in the sense that the bot must assume that statistical properties (mean, variance, correlations) shift across the window and any model that assumes IID data will quietly fail. This is related to data **non-stationarity**

- Indicators that look predictive in one window may be useless or harmful in the others. Evaluations must be walk-forward, and never random splits. This is related to **causality vs correlation**

- We are not optimising for a prediction error but we are optimising for capital growth under constraints (our budget, personal portfolio and more). This shifts the loss function from MSE to PnL-aware objectives (returns, drawdowns, risk-adjusted rewards and more). This is related to **decision-centric learning**

- Actions affect future states through capital, positive sizing, and exposure. This immediately disqualified pure supervised learning as a complete solution

    - **Layer 1 | Market State & Context Layer :** It processes raw price data into context such as trends, volatility, momentum strength and behaviours. This is where rolling statistics, volatility measures, regime classifiers or lightweight time-series models work. It describes the environment to the rest of the system

    - **Layer 2 | Signal & Edge Extraction Layer :** It works with outputs from Layer 1 and produces signals such as Directional bias (bullish / bearish / neutral), Confidence and expected short-term payoff. This is where supervised ML fits best. These models don’t decide trades but estimate advantage. Predicts and estimates whether a trade might be profitable and how strong that belief is

    - **Layer 3 | Decision & Policy Layer :** This layer takes Market context (Layer 1), Signal strength (Layer 2), current position, remaining capital, past outcomes and more to produce a market decision, position size and the timing preference. This is where **reinforcement learning** or rule-constrained policies belong

    - **Layer 4 | Risk, Positioning & Trade Management Layer :** This wraps every decision with hard rules such as position sizing, stop-loss placement, take-profit logic, max drawdown limits and more. This layer keeps the trader alive long enough to learn