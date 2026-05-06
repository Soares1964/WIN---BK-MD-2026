# ma_lab/config_producao.py
from core.optimizer_engine import OptimizationConfig

def get_config_producao():
    """
    Configuração final para produção baseada nos testes reais
    """
    config = OptimizationConfig(
        # =========================================================
        # INDICADORES VALIDADOS
        # =========================================================
        fast_indicators=['EMA', 'HMA'],     # Os melhores
        slow_indicators=['SMA'],            # SMA venceu como lenta
        
        # =========================================================
        # PERÍODOS OTIMIZADOS
        # =========================================================
        fast_min=10,
        fast_max=30,
        fast_step=2,
        
        slow_min=25,
        slow_max=35,        # Foco na região dos melhores
        slow_step=2,
        
        # =========================================================
        # TIMEFRAME ÚNICO (O MELHOR)
        # =========================================================
        timeframes=[5],
        
        # =========================================================
        # FILTROS DE QUALIDADE AJUSTADOS
        # =========================================================
        min_trades=100,                     # Aumentado para consistência
        min_sharpe=-999,                     # Ainda ignorando Sharpe
        min_profit_factor=1.10,               # Só os realmente bons
        max_dd=8.0,                          # Drawdown controlado
        
        # =========================================================
        # CUSTOS REALISTAS
        # =========================================================
        slippage=0.5,
        cost=0.2,
        
        # =========================================================
        # PERFORMANCE
        # =========================================================
        batch_size=5000,
        n_jobs=8,
        use_cache=True,
        debug=False
    )
    return config

def get_sistemas_recomendados():
    """
    Retorna lista dos melhores sistemas encontrados
    """
    sistemas = [
        {
            'nome': 'CAMPEÃO',
            'fast': 'EMA',
            'fast_period': 26,
            'slow': 'SMA',
            'slow_period': 29,
            'timeframe': 5,
            'retorno': 30.5,
            'pf': 1.13,
            'dd': 6.0
        },
        {
            'nome': 'SEGURO',
            'fast': 'HMA',
            'fast_period': 10,
            'slow': 'SMA',
            'slow_period': 29,
            'timeframe': 5,
            'retorno': 27.5,
            'pf': 1.12,
            'dd': 3.8
        },
        {
            'nome': 'CONSISTENTE',
            'fast': 'EMA',
            'fast_period': 18,
            'slow': 'SMA',
            'slow_period': 25,
            'timeframe': 5,
            'retorno': 28.9,
            'pf': 1.13,
            'dd': 6.6
        }
    ]
    return sistemas