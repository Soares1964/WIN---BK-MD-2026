# ma_lab/core/statistical_validation.py
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List
from numba import njit


@dataclass
class StatisticalValidationConfig:
    """Configuracao da validacao estatistica (pos-otimizacao)"""
    enabled: bool = False
    top_n: int = 10
    n_monte_carlo: int = 1000
    n_bootstrap: int = 1000
    n_null_hypothesis: int = 200
    mc_confidence: float = 0.95
    bootstrap_confidence: float = 0.95


def _extract_trade_returns(pnl: np.ndarray) -> np.ndarray:
    """Extrai returns nao-zero (trades) do array de PnL periodos"""
    if pnl is None or len(pnl) == 0:
        return np.array([])
    trades = pnl[pnl != 0]
    if len(trades) == 0:
        return np.array([])
    return trades


@njit(cache=True)
def _compute_sharpe_numba(returns: np.ndarray, periods_per_year: int) -> float:
    """Sharpe ratio via Numba para bootstrap"""
    n = len(returns)
    if n < 2:
        return 0.0
    mean_r = 0.0
    for i in range(n):
        mean_r += returns[i]
    mean_r /= n
    var = 0.0
    for i in range(n):
        diff = returns[i] - mean_r
        var += diff * diff
    var /= (n - 1)
    if var <= 0:
        return 0.0
    std = np.sqrt(var)
    rf_adj = 0.02 / periods_per_year
    excess = mean_r - rf_adj
    return np.sqrt(periods_per_year) * excess / std


@njit(cache=True)
def _compute_metrics_vectorized(
    returns_2d: np.ndarray,
    periods_per_year: int
) -> np.ndarray:
    """Computa Sharpe para cada linha de uma matrix 2D (resamples)"""
    n_rows, n_cols = returns_2d.shape
    sharpes = np.zeros(n_rows)
    for i in range(n_rows):
        sharpes[i] = _compute_sharpe_numba(returns_2d[i], periods_per_year)
    return sharpes


def monte_carlo_simulation(
    pnl: np.ndarray,
    n_scenarios: int = 1000,
    confidence: float = 0.95,
    periods_per_year: int = 27216
) -> Dict[str, float]:
    """
    Monte Carlo simulation via trade return shuffling.

    Shuffles individual trade returns to generate synthetic equity curves,
    then computes distribution of Sharpe, total return, and max drawdown.

    Returns:
        dict with mc_prob_loss, mc_sharpe_mean/ci_lower/ci_upper,
        mc_return_ci_lower/ci_upper, mc_dd_ci_lower/ci_upper
    """
    result = {
        'mc_prob_loss': 0.0,
        'mc_sharpe_mean': 0.0,
        'mc_sharpe_std': 0.0,
        'mc_sharpe_ci_lower': 0.0,
        'mc_sharpe_ci_upper': 0.0,
        'mc_return_ci_lower': 0.0,
        'mc_return_ci_upper': 0.0,
        'mc_dd_ci_lower': 0.0,
        'mc_dd_ci_upper': 0.0,
    }

    trade_rets = _extract_trade_returns(pnl)
    n_trades = len(trade_rets)
    if n_trades < 5 or n_scenarios < 2:
        return result

    # Sample with replacement: (n_scenarios, n_trades)
    rng = np.random.default_rng()
    sampled = rng.choice(trade_rets, size=(n_scenarios, n_trades))

    # Equity curves: cumprod(1 + returns)
    eq_curves = np.cumprod(1.0 + sampled, axis=1)

    # Final equity values
    final_eq = eq_curves[:, -1]
    total_returns = (final_eq - 1.0) * 100.0  # percent

    # Probability of loss
    result['mc_prob_loss'] = float(np.mean(total_returns < 0))

    # Sharpe per scenario via numba
    sharpes = _compute_metrics_vectorized(sampled, periods_per_year)

    result['mc_sharpe_mean'] = float(np.mean(sharpes))
    result['mc_sharpe_std'] = float(np.std(sharpes, ddof=1))

    # Percentiles
    alpha = 1.0 - confidence
    result['mc_sharpe_ci_lower'] = float(np.percentile(sharpes, alpha * 50))
    result['mc_sharpe_ci_upper'] = float(np.percentile(sharpes, (1.0 - alpha * 0.5) * 100))
    result['mc_return_ci_lower'] = float(np.percentile(total_returns, alpha * 50))
    result['mc_return_ci_upper'] = float(np.percentile(total_returns, (1.0 - alpha * 0.5) * 100))

    # Max DD per scenario (vectorized)
    peaks = np.maximum.accumulate(eq_curves, axis=1)
    dd_pct = (peaks - eq_curves) / peaks * 100.0
    max_dds = np.max(dd_pct, axis=1)
    result['mc_dd_ci_lower'] = float(np.percentile(max_dds, alpha * 50))
    result['mc_dd_ci_upper'] = float(np.percentile(max_dds, (1.0 - alpha * 0.5) * 100))

    return result


def bootstrap_sharpe_ratio(
    pnl: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    periods_per_year: int = 27216
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Resamples period returns with replacement and computes Sharpe
    for each resample. Returns CI bounds and p-value for Sharpe > 0.

    Returns:
        dict with boot_sharpe_mean, boot_sharpe_std,
        boot_sharpe_ci_lower/upper, boot_sharpe_p_value
    """
    result = {
        'boot_sharpe_mean': 0.0,
        'boot_sharpe_std': 0.0,
        'boot_sharpe_ci_lower': 0.0,
        'boot_sharpe_ci_upper': 0.0,
        'boot_sharpe_p_value': 1.0,
    }

    n = len(pnl) if pnl is not None else 0
    if n < 5 or n_bootstrap < 2:
        return result

    rng = np.random.default_rng()
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled = pnl[indices]

    sharpes = _compute_metrics_vectorized(resampled, periods_per_year)

    result['boot_sharpe_mean'] = float(np.mean(sharpes))
    result['boot_sharpe_std'] = float(np.std(sharpes, ddof=1))

    alpha = 1.0 - confidence
    result['boot_sharpe_ci_lower'] = float(np.percentile(sharpes, alpha * 50))
    result['boot_sharpe_ci_upper'] = float(np.percentile(sharpes, (1.0 - alpha * 0.5) * 100))
    result['boot_sharpe_p_value'] = float(np.mean(sharpes <= 0))

    return result


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    periods_per_year: int = 27216
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Corrects the observed Sharpe ratio for multiple testing (number of
    strategies tried). Returns the probability that the observed Sharpe
    is not due to chance (DSR), between 0 and 1.

    The formula works with the non-annualized Sharpe internally:
      SR_raw = SR_annual / sqrt(ppy)
      DSR = Phi((SR_raw * sqrt(T) - E[max Z]) * sqrt(2 ln N))

    Args:
        observed_sharpe: The annualized Sharpe ratio of the best strategy
        n_observations: Number of period returns (T)
        n_trials: Number of strategies tested (N)
        periods_per_year: Annualization factor (e.g., 252 for daily, 27216 for 5-min)
    """
    if n_trials <= 0:
        return 0.0
    if n_trials == 1:
        return 1.0 if observed_sharpe > 0 else 0.0
    if n_observations < 3 or periods_per_year <= 0:
        return 0.0

    # De-annualize Sharpe: DSR formula works with raw Sharpe
    sr_raw = observed_sharpe / np.sqrt(periods_per_year)

    # Expected maximum of N i.i.d. standard normals
    # Approximation from Bailey & Lopez de Prado
    gamma = 0.5772156649
    log_n = np.log(n_trials)

    e_max_z = np.sqrt(2.0 * log_n) - (np.log(log_n) + np.log(4.0 * np.pi) - 2.0 * gamma) / (
        2.0 * np.sqrt(2.0 * log_n)
    )

    # Standard deviation of max Z
    std_max_z = 1.0 / np.sqrt(2.0 * log_n)

    # DSR = Phi((SR_raw * sqrt(T) - E[max Z]) / std_max_z)
    z_score = (sr_raw * np.sqrt(n_observations) - e_max_z) / std_max_z

    dsr = stats.norm.cdf(z_score)
    return float(np.clip(dsr, 0.0, 1.0))


def null_hypothesis_test(
    price: np.ndarray,
    signal: np.ndarray,
    original_sharpe: float,
    n_shuffles: int = 200,
    slippage: float = 1.0,
    cost: float = 0.5,
    periods_per_year: int = 27216,
    batch_size: int = 100
) -> Dict[str, float]:
    """
    Null hypothesis test via signal permutation.

    Shuffles the signal to destroy any temporal relationship with price,
    runs backtests, and compares the real Sharpe against the null distribution.

    Returns:
        dict with null_sharpe_mean, null_sharpe_std, null_p_value
    """
    result = {
        'null_sharpe_mean': 0.0,
        'null_sharpe_std': 0.0,
        'null_p_value': 1.0,
    }

    if price is None or signal is None or n_shuffles < 10:
        return result

    from core.backtest_engine import run_batch_backtest, calculate_metrics

    n = len(price)
    all_null_sharpes = []

    # Process in batches to bound memory
    remaining = n_shuffles
    rng = np.random.default_rng()

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        shuffled_signals = np.zeros((current_batch, n), dtype=np.int8)

        for i in range(current_batch):
            shuffled_signals[i] = rng.permutation(signal)

        # Batch backtest (Numba parallel)
        batch_pnls, batch_equities, batch_trades = run_batch_backtest(
            price, shuffled_signals, slippage, cost
        )

        for i in range(current_batch):
            metrics = calculate_metrics(
                batch_pnls[i], batch_equities[i], batch_trades[i],
                periods_per_year=periods_per_year
            )
            all_null_sharpes.append(metrics['sharpe'])

        remaining -= current_batch

    null_sharpes = np.array(all_null_sharpes)

    if len(null_sharpes) < 5:
        return result

    result['null_sharpe_mean'] = float(np.mean(null_sharpes))
    result['null_sharpe_std'] = float(np.std(null_sharpes, ddof=1))
    result['null_p_value'] = float(np.mean(null_sharpes >= original_sharpe))

    return result


def parameter_sensitivity(
    result_obj,
    price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    config,
    cache,
    fp_range: int = 5,
    sp_range: int = 5
) -> Dict[str, float]:
    """
    Parameter sensitivity analysis around optimal point.

    Varies fast/slow periods around the optimal combination and measures
    impact on Sharpe. Returns robustness score (1.0 = perfectly robust).

    Args:
        result_obj: SystemResult (needs fast_indicator, slow_indicator,
                    fast_period, slow_period, timeframe)
        price, high, low: Price arrays
        config: OptimizationConfig (for slippage, cost, min_trades, etc.)
        cache: MovingAverageCache instance
        fp_range: Range of fast periods to test around optimum
        sp_range: Range of slow periods to test around optimum

    Returns:
        dict with sens_robustness, sens_sharpe_std, sens_avg_sharpe
    """
    result = {
        'sens_robustness': 0.0,
        'sens_sharpe_std': 0.0,
        'sens_avg_sharpe': 0.0,
    }

    from core.backtest_engine import (
        generate_signal_from_cross,
        run_backtest_vectorized,
        calculate_metrics,
        periods_per_year_for_timeframe,
    )

    fast_ind = result_obj.fast_indicator
    slow_ind = result_obj.slow_indicator
    fp_opt = result_obj.fast_period
    sp_opt = result_obj.slow_period
    tf = result_obj.timeframe

    if price is None or len(price) < 100:
        return result

    ppy = periods_per_year_for_timeframe(tf)
    sharpes = []

    f_min = max(2, fp_opt - fp_range)
    f_max = fp_opt + fp_range
    s_min = max(fp_opt + 1, sp_opt - sp_range)
    s_max = sp_opt + sp_range

    for fp in range(f_min, f_max + 1):
        for sp in range(s_min, s_max + 1):
            if fp >= sp:
                continue

            try:
                fast_ma = cache.get(fast_ind, fp).values
                slow_ma = cache.get(slow_ind, sp).values
                if len(fast_ma) == 0 or len(slow_ma) == 0:
                    continue

                signal = generate_signal_from_cross(fast_ma, slow_ma)
                pnl, equity, trades = run_backtest_vectorized(
                    price, signal, config.slippage, config.cost
                )

                if trades < max(config.min_trades * 0.5, 10):
                    continue

                metrics = calculate_metrics(pnl, equity, trades, periods_per_year=ppy)
                sh = metrics['sharpe']
                if not np.isinf(sh) and not np.isnan(sh):
                    sharpes.append(sh)
            except Exception:
                continue

    if len(sharpes) < 3:
        return result

    sh_arr = np.array(sharpes)
    mean_sh = float(np.mean(sh_arr))
    std_sh = float(np.std(sh_arr, ddof=1))

    result['sens_sharpe_std'] = std_sh
    result['sens_avg_sharpe'] = mean_sh
    result['sens_robustness'] = float(
        1.0 - min(std_sh / max(abs(mean_sh), 0.001), 1.0)
    )

    return result
