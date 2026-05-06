# ma_lab/core/backtest_engine.py
import numpy as np
import pandas as pd
from numba import njit, prange, jit, set_num_threads
import os
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configura número de threads do Numba
set_num_threads(os.cpu_count())

@njit(cache=True, fastmath=True)
def run_backtest_vectorized(price: np.ndarray, signal: np.ndarray,
                            slippage: float = 1.0, cost: float = 0.5) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Backtest vetorizado ultra-rápido com custos e slippage.

    Args:
        price: Array de preços OHLC
        signal: Array de sinais (1=long, -1=short, 0=flat)
        slippage: Slippage em pontos por trade (execução)
        cost: Custo/COMISSÃO em pontos por trade (corretagem + emolumentos)

    Returns:
        (strategy_returns, equity_curve, n_trades)
    """
    n = len(price)

    # Pré-aloca arrays com tipos otimizados
    returns = np.zeros(n - 1)
    position = np.zeros(n - 1, dtype=np.int8)

    # Calcula retornos de uma vez (mais rápido que loop)
    for i in range(n - 1):
        returns[i] = (price[i + 1] / price[i]) - 1.0

    # Posições (excluíndo último)
    for i in range(n - 1):
        position[i] = signal[i]

    # Conta trades e aplica custos + slippage
    trades = 0
    prev_signal = signal[0]

    for i in range(1, n):
        current_signal = signal[i]
        if current_signal != prev_signal:
            trades += 1
            # Custo total por trade: slippage + comissão
            if price[i] > 1e-8:
                total_cost_pts = (cost + slippage) * 0.5  # metade na saída, metade na entrada
                cost_impact = total_cost_pts / price[i]
                returns[i-1] -= cost_impact * abs(current_signal - prev_signal)
            prev_signal = current_signal

    # Retornos da estratégia
    strategy_returns = position * returns

    # Equity curve - cálculo único
    equity = np.zeros(n)
    equity[0] = 1.0
    cumprod = 1.0
    for i in range(n - 1):
        cumprod *= (1.0 + strategy_returns[i])
        equity[i + 1] = cumprod

    return strategy_returns, equity, trades


@njit(parallel=True, cache=True)
def run_batch_backtest(prices: np.ndarray, signals_matrix: np.ndarray,
                       slippage: float = 1.0, cost: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa múltiplos backtests em paralelo com Numba
    """
    n_systems = signals_matrix.shape[0]
    n = len(prices)
    
    all_pnls = np.zeros((n_systems, n-1))
    all_equities = np.ones((n_systems, n))
    all_trades = np.zeros(n_systems, dtype=np.int32)
    
    # Paralelização automática com prange
    for sys in prange(n_systems):
        pnl, equity, trades = run_backtest_vectorized(
            prices, signals_matrix[sys], slippage, cost
        )
        all_pnls[sys] = pnl
        all_equities[sys] = equity
        all_trades[sys] = trades
    
    return all_pnls, all_equities, all_trades


@njit(cache=True)
def generate_signal_from_cross(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """Gera sinais de cruzamento entre duas médias"""
    n = len(fast)
    signal = np.zeros(n, dtype=np.int8)
    
    # Loop único otimizado
    for i in range(n):
        if fast[i] > slow[i]:
            signal[i] = 1
        elif fast[i] < slow[i]:
            signal[i] = -1
        # else mantém 0
    
    return signal


@njit(cache=True)
def calculate_metrics(pnl: np.ndarray, equity: np.ndarray, trades: int,
                      risk_free_rate: float = 0.02,
                      periods_per_year: int = 252) -> Dict[str, float]:
    """Calcula métricas de performance - ultra rápido

    Args:
        pnl: Array de retornos por período
        equity: Array da curva de equity
        trades: Número de trades
        risk_free_rate: Taxa livre de risco anual (ex: 0.02 = 2%)
        periods_per_year: Períodos por ano para anualização correta
                          (252 para daily, 19656 para 5-min bars, etc.)
    """

    n = len(pnl)
    if n == 0:
        return {
            'sharpe': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'max_dd': 0.0,
            'total_return': 0.0,
            'trades': 0
        }

    # Média e desvio (cálculo único)
    mean_pnl = 0.0
    std_pnl = 0.0
    gains = 0.0
    losses = 0.0
    wins = 0

    for i in range(n):
        val = pnl[i]
        mean_pnl += val
        if val > 0:
            gains += val
            wins += 1
        elif val < 0:
            losses += -val

    mean_pnl /= n

    # Desvio padrão amostral (ddof=1 para consistência com metrics.py)
    for i in range(n):
        std_pnl += (pnl[i] - mean_pnl) ** 2
    # Em Numba não temos ddof, calculamos manualmente: std_amostral = sqrt(sum_sq / (n-1))
    std_pnl = np.sqrt(std_pnl / (n - 1)) if n > 1 else 0.0

    # Sharpe com anualização correta para a frequência dos dados
    sharpe = 0.0
    if std_pnl > 1e-10:
        ppy = max(periods_per_year, 1)
        excess = mean_pnl - risk_free_rate / ppy
        sharpe = np.sqrt(ppy) * excess / std_pnl
    
    # Profit Factor - corrigido: retorna inf quando nao ha perdas ( nao inflacionar)
    if losses > 1e-10:
        profit_factor = gains / losses
    elif gains > 0:
        profit_factor = float('inf')  # Sistema sem perdas = PF infinito (correto)
    else:
        profit_factor = 0.0  # Sistema sem ganhos e sem perdas

    # Win Rate
    win_rate = (wins / n) * 100 if n > 0 else 0.0
    
    # Max Drawdown
    m = len(equity)
    peak = equity[0]
    max_dd = 0.0
    
    for i in range(1, m):
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Retorno total
    total_return = (equity[-1] - 1.0) * 100
    
    return {
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'max_dd': max_dd * 100,
        'total_return': total_return,
        'trades': trades
    }


# ============================================================
# UTILITÁRIO DE ANUALIZAÇÃO
# ============================================================
def periods_per_year_for_timeframe(tf_minutes: int, trading_hours: int = 9) -> int:
    """
    Calcula períodos por ano para anualização correta do Sharpe.

    Mini Índice (WIN) opera ~9h/dia (9:00-18:00).
    Para um timeframe de N minutos:
      periods_per_year = 252 * (trading_hours * 60) / N

    Args:
        tf_minutes: Timeframe em minutos (ex: 5 para 5-min)
        trading_hours: Horas de negociação por dia (default 9 para WIN)

    Returns:
        Número aproximado de períodos por ano
    """
    if tf_minutes <= 0:
        return 252
    return max(252, int(252 * trading_hours * 60 / tf_minutes))


# Versão para teste sem Numba (fallback)
def run_backtest_vectorized_python(price: np.ndarray, signal: np.ndarray,
                                   slippage: float = 1.0, cost: float = 0.5):
    """Versão Python pura para comparação"""
    n = len(price)
    returns = np.zeros(n-1)
    for i in range(n-1):
        returns[i] = (price[i+1] / price[i]) - 1.0
    
    position = signal[:-1]
    trades = np.sum(signal[1:] != signal[:-1])
    
    # Custos
    changes = signal[1:] != signal[:-1]
    cost_impact = np.zeros(n-1)
    for i in range(n-1):
        if changes[i]:
            cost_impact[i] = cost / price[i+1] if price[i+1] > 1e-8 else 0.0
    
    strategy_returns = position * (returns - cost_impact)
    
    equity = np.ones(n)
    for i in range(n-1):
        equity[i+1] = equity[i] * (1 + strategy_returns[i])
    
    return strategy_returns, equity, int(trades)