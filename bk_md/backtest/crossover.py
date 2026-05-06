import numpy as np
import pandas as pd


class CrossoverBacktest:
    """Backtest de crossover com sinal 1/-1/0 (flat)"""

    def run(self, price, fast_ma, slow_ma):
        """
        Executa backtest baseado em cruzamento.
        
        Sinal: 1 (long) quando fast > slow, -1 (short) quando fast < slow,
        0 (flat) quando iguais ou NaN.
        """
        # Converte para numpy se necessario
        fast = np.asarray(fast_ma, dtype=np.float64)
        slow = np.asarray(slow_ma, dtype=np.float64)
        p = np.asarray(price, dtype=np.float64)
        
        # Sinal com estado flat (0) quando igual
        signal = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
        
        # Retorna o preco para calculo de performance
        returns = np.diff(p) / p[:-1]
        returns = np.nan_to_num(returns, nan=0.0)
        
        # Ajusta sinal ao tamanho dos retornos
        # signal[i] aplica ao retorno de i para i+1
        signal_trimmed = signal[:-1]
        
        # Retorna da estrategia
        strategy_returns = signal_trimmed * returns
        
        pnl = pd.Series(strategy_returns)
        equity = (1 + pnl).cumprod()
        
        return pnl, equity