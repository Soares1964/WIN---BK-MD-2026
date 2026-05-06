# ma_lab/core/portfolio_management.py
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize


@dataclass
class PortfolioConfig:
    """Configuracao da gestao de portfolio (pos-otimizacao)"""
    enabled: bool = False
    allocation_method: str = "equal"      # equal, markowitz, risk_parity
    rebalance_enabled: bool = False
    rebalance_freq_bars: int = 1000
    dd_management_enabled: bool = False
    dd_threshold: float = 10.0
    dd_reduction: float = 0.5
    top_n: int = 10
    max_correlation: float = 0.95
    risk_free_rate: float = 0.02


@dataclass
class PortfolioResult:
    """Resultado da otimizacao de portfolio"""
    weights: np.ndarray
    correlation_matrix: np.ndarray
    system_indices: List[int]
    system_labels: List[str]
    combined_pnl: np.ndarray
    combined_equity: np.ndarray
    managed_equity: Optional[np.ndarray] = None
    exposure: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    allocation_method: str = "equal"

    def to_dict(self) -> dict:
        return {
            'pf_n_systems': len(self.system_indices),
            'pf_allocation': self.allocation_method,
            'pf_sharpe': round(self.metrics.get('pf_sharpe', 0), 3),
            'pf_profit_factor': round(self.metrics.get('pf_profit_factor', 0), 3),
            'pf_win_rate': round(self.metrics.get('pf_win_rate', 0), 1),
            'pf_max_dd': round(self.metrics.get('pf_max_dd', 0), 1),
            'pf_total_return': round(self.metrics.get('pf_total_return', 0), 1),
            'pf_trades': self.metrics.get('pf_trades', 0),
        }


def compute_correlation_matrix(pnl_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Matriz de correlacao entre sistemas.
    Alinha todos os arrays pelo menor comprimento.
    """
    n_systems = len(pnl_arrays)
    if n_systems == 0:
        return np.array([[]])
    if n_systems == 1:
        return np.ones((1, 1))

    min_len = min(len(arr) for arr in pnl_arrays)
    if min_len < 2:
        return np.eye(n_systems)

    aligned = np.zeros((n_systems, min_len))
    for i, arr in enumerate(pnl_arrays):
        aligned[i] = arr[:min_len]

    corr = np.corrcoef(aligned)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
    return corr


def _max_sharpe_objective(weights: np.ndarray,
                          mean_ret: np.ndarray,
                          cov: np.ndarray,
                          rf: float) -> float:
    """Objetivo: -Sharpe Ratio (minimizar)"""
    port_ret = np.dot(weights, mean_ret)
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_std = np.sqrt(max(port_var, 1e-16))
    excess = port_ret - rf / 252.0
    if port_std < 1e-16:
        return 0.0
    return -excess / port_std


def markowitz_optimization(mean_returns: np.ndarray,
                           cov_matrix: np.ndarray,
                           risk_free_rate: float = 0.02) -> np.ndarray:
    """
    Markowitz Mean-Variance: maximiza Sharpe Ratio.
    Restricoes: weights >= 0 (sem short), sum(weights) = 1.
    """
    n = len(mean_returns)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n

    try:
        result = minimize(
            _max_sharpe_objective, w0,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if result.success:
            weights = result.x
        else:
            weights = w0
    except Exception:
        weights = w0

    weights = np.maximum(weights, 0.0)
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        weights = np.ones(n) / n
    return weights


def _risk_parity_error(weights: np.ndarray, cov: np.ndarray) -> float:
    """Erro de risk parity: soma dos desvios da contribuicao igualitaria"""
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_std = np.sqrt(max(port_var, 1e-16))
    mrc = np.dot(cov, weights) / port_std
    rc = weights * mrc
    target = np.mean(rc)
    return float(np.sum((rc - target) ** 2))


def risk_parity_optimization(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Risk Parity: cada ativo contribui igualmente para o risco total.
    """
    n = len(cov_matrix)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n

    try:
        result = minimize(
            _risk_parity_error, w0, args=(cov_matrix,),
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        if result.success:
            weights = result.x
        else:
            weights = w0
    except Exception:
        weights = w0

    weights = np.maximum(weights, 0.0)
    total = np.sum(weights)
    if total > 0:
        weights /= total
    else:
        weights = np.ones(n) / n
    return weights


def filter_by_correlation(pnl_arrays: List[np.ndarray],
                          indices: List[int],
                          labels: List[str],
                          max_correlation: float = 0.95
                          ) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Remove sistemas altamente correlacionados (max_correlation)"""
    if len(pnl_arrays) <= 1:
        return pnl_arrays, indices, labels

    corr = compute_correlation_matrix(pnl_arrays)
    n = len(pnl_arrays)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if abs(corr[i, j]) > max_correlation:
                var_i = float(np.var(pnl_arrays[i]))
                var_j = float(np.var(pnl_arrays[j]))
                if var_i < var_j:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    filtered = ([a for a, k in zip(pnl_arrays, keep) if k],
                [i for i, k in zip(indices, keep) if k],
                [l for l, k in zip(labels, keep) if k])
    return filtered


def combine_portfolio_returns(pnl_arrays: List[np.ndarray],
                              weights: np.ndarray) -> np.ndarray:
    """Combina retornos de N sistemas com pesos w_i (mesmo comprimento)"""
    if not pnl_arrays or len(pnl_arrays) == 0:
        return np.array([])

    min_len = min(len(arr) for arr in pnl_arrays)
    if min_len == 0:
        return np.array([])

    combined = np.zeros(min_len)
    for i, arr in enumerate(pnl_arrays):
        combined += weights[i] * arr[:min_len]
    return combined


@njit(cache=True)
def compute_portfolio_equity_numba(pnl: np.ndarray) -> np.ndarray:
    """Equity curve from combined PnL"""
    n = len(pnl)
    equity = np.zeros(n + 1)
    equity[0] = 1.0
    for i in range(n):
        equity[i + 1] = equity[i] * (1.0 + pnl[i])
    return equity


@njit(cache=True)
def apply_drawdown_management_numba(equity: np.ndarray,
                                     threshold: float,
                                     reduction: float
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduz exposicao quando o drawdown ultrapassa threshold.
    Retorna (managed_equity, exposure_array).
    """
    n = len(equity)
    exposure = np.ones(n)
    peak = equity[0]

    for i in range(1, n):
        if equity[i] > peak:
            peak = equity[i]
        dd_pct = (peak - equity[i]) / peak * 100.0
        if dd_pct > threshold:
            exposure[i] = max(1.0 - reduction, 0.0)

    managed = np.zeros(n)
    managed[0] = equity[0]
    for i in range(1, n):
        period_ret = (equity[i] / equity[i - 1]) - 1.0
        managed_ret = period_ret * exposure[i - 1]
        managed[i] = managed[i - 1] * (1.0 + managed_ret)

    return managed, exposure


def _match_pnl_lengths(pnl_list: List[np.ndarray],
                       length: int) -> List[np.ndarray]:
    """Trunca ou estende PnLs para o mesmo comprimento"""
    result = []
    for arr in pnl_list:
        if len(arr) >= length:
            result.append(arr[:length].copy())
        else:
            padded = np.zeros(length)
            padded[:len(arr)] = arr
            result.append(padded)
    return result


def build_rebalance_schedule(total_bars: int,
                             freq_bars: int) -> List[int]:
    """Gera indices de rebalanceamento"""
    if freq_bars <= 0:
        return [0]
    return list(range(0, total_bars, freq_bars))


def compute_portfolio_metrics(pnl: np.ndarray,
                              equity: np.ndarray,
                              periods_per_year: int = 27216) -> Dict[str, float]:
    """Metricas do portfolio combinado"""
    from core.backtest_engine import calculate_metrics

    if len(pnl) == 0 or len(equity) == 0:
        return {'pf_sharpe': 0.0, 'pf_profit_factor': 0.0,
                'pf_win_rate': 0.0, 'pf_max_dd': 0.0,
                'pf_total_return': 0.0, 'pf_trades': 0}

    n_trades = int(np.sum(pnl != 0))
    metrics = calculate_metrics(pnl, equity, n_trades,
                                periods_per_year=periods_per_year)
    return {
        'pf_sharpe': metrics['sharpe'],
        'pf_profit_factor': metrics['profit_factor'],
        'pf_win_rate': metrics['win_rate'],
        'pf_max_dd': metrics['max_dd'],
        'pf_total_return': metrics['total_return'],
        'pf_trades': metrics['trades'],
    }
