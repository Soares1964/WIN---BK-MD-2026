# config_ajustada.py - USE ISSO AGORA!
from core.optimizer_engine import OptimizationConfig

config = OptimizationConfig(
    # Indicadores
    fast_indicators=['SMA', 'EMA', 'HMA', 'WMA'],
    slow_indicators=['SMA', 'EMA', 'HMA', 'WMA'],
    
    # Períodos (mais lentos para reduzir volatilidade)
    fast_min=10,        # Aumentado
    fast_max=30,
    fast_step=2,
    
    slow_min=30,        # Aumentado
    slow_max=60,
    slow_step=2,
    
    # Timeframes MAIORES (menos ruído)
    timeframes=[3, 5, 10],
    
    # FILTROS REALISTAS para este mercado volátil
    min_trades=30,
    min_sharpe=-999,     # Ignora Sharpe (por enquanto)
    min_profit_factor=1.05,  # Só exige lucro
    max_dd=20.0,         # Aceita drawdown maior
    
    # Custos
    slippage=0.5,
    cost=0.2,
    
    # Performance
    batch_size=2000,
    n_jobs=8
)