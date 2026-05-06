# config_trabalho.py - USE ISSO AGORA!
from core.optimizer_engine import OptimizationConfig

config = OptimizationConfig(
    fast_indicators=['SMA', 'EMA', 'HMA', 'WMA'],
    slow_indicators=['SMA', 'EMA', 'HMA', 'WMA'],
    fast_min=5, fast_max=30, fast_step=2,
    slow_min=20, slow_max=60, slow_step=2,
    timeframes=[3, 5, 10],  # Use timeframes MAIORES
    min_trades=30,
    min_sharpe=-999,  # Ignora Sharpe por enquanto
    min_profit_factor=1.0,  # Apenas > 1.0
    max_dd=20.0,
    slippage=0.5,
    cost=0.2,
    batch_size=2000,
    n_jobs=8
)
