# ma_lab/core/transactional_costs.py
import numpy as np
from numba import njit
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TransactionalCostConfig:
    """Configuracao de custos transacionais realistas"""
    # Master switch
    tc_enabled: bool = False

    # Variable spread (ATR-based)
    tc_spread_variable: bool = False
    tc_spread_base_pts: float = 0.5
    tc_spread_vol_mult: float = 0.05
    tc_spread_max_pts: float = 5.0

    # Variable slippage (ATR-based)
    tc_slippage_variable: bool = False
    tc_slippage_base_pts: float = 0.5
    tc_slippage_vol_mult: float = 0.10
    tc_slippage_max_pts: float = 10.0

    # Brokerage and IR
    tc_brokerage_per_trade: float = 0.0
    tc_exchange_fee: float = 0.0
    tc_ir_tax_enabled: bool = False
    tc_ir_tax_rate: float = 0.0

    # Liquidity restriction
    tc_liquidity_enabled: bool = False
    tc_liquidity_min_atr_ratio: float = 0.3

    # Session gap handling
    tc_gap_enabled: bool = False
    tc_gap_max_minutes: int = 30


@njit(cache=True)
def compute_variable_spread(
    price: np.ndarray,
    atr: np.ndarray,
    base_pts: float,
    vol_mult: float,
    max_pts: float,
) -> np.ndarray:
    """Computa spread variavel por barra baseado em ATR"""
    n = len(price)
    spread = np.zeros(n)
    for i in range(n):
        raw = base_pts + atr[i] * vol_mult
        if raw > max_pts:
            raw = max_pts
        if raw < 0.0:
            raw = 0.0
        spread[i] = raw
    return spread


@njit(cache=True)
def compute_variable_slippage(
    price: np.ndarray,
    atr: np.ndarray,
    base_pts: float,
    vol_mult: float,
    max_pts: float,
) -> np.ndarray:
    """Computa slippage variavel por barra baseado em ATR"""
    n = len(price)
    slippage = np.zeros(n)
    for i in range(n):
        raw = base_pts + atr[i] * vol_mult
        if raw > max_pts:
            raw = max_pts
        if raw < 0.0:
            raw = 0.0
        slippage[i] = raw
    return slippage


@njit(cache=True)
def detect_session_gaps(
    timestamps: np.ndarray,
    max_gap_ns: float,
) -> np.ndarray:
    """Detecta gaps entre sessoes (bool mask: True = gap)"""
    n = len(timestamps)
    gap = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return gap
    for i in range(1, n):
        diff = timestamps[i] - timestamps[i - 1]
        if diff > max_gap_ns:
            gap[i] = True
    return gap


@njit(cache=True)
def compute_liquidity_mask(
    atr: np.ndarray,
    atr_mean: float,
    min_atr_ratio: float,
) -> np.ndarray:
    """Mascara de liquidez (True = liquidez suficiente)"""
    n = len(atr)
    mask = np.ones(n, dtype=np.bool_)
    if atr_mean <= 0.0 or min_atr_ratio <= 0.0:
        return mask
    threshold = atr_mean * min_atr_ratio
    for i in range(n):
        if atr[i] < threshold:
            mask[i] = False
    return mask


@njit(cache=True)
def compute_atr_stats(atr: np.ndarray) -> Tuple[float, float]:
    """Media e desvio padrao do ATR"""
    n = len(atr)
    if n == 0:
        return 0.0, 0.0
    mean = 0.0
    for i in range(n):
        mean += atr[i]
    mean /= n
    var = 0.0
    for i in range(n):
        diff = atr[i] - mean
        var += diff * diff
    std = np.sqrt(var / n) if n > 1 else 0.0
    return mean, std


@njit(cache=True, fastmath=True)
def run_backtest_realistic(
    price: np.ndarray,
    signal: np.ndarray,
    spread_arr: np.ndarray,
    slippage_arr: np.ndarray,
    gap_mask: np.ndarray,
    liq_mask: np.ndarray,
    brokerage_per_trade: float = 0.0,
    exchange_fee: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Backtest vetorizado com custos realistas (arrays por barra).

    Aplica spread, slippage, corretagem e taxa de exchange como arrays
    variaveis por barra. Respeita mascaras de gap e liquidez.

    Returns:
        (strategy_returns, equity_curve, n_trades)
    """
    n = len(price)
    m = n - 1

    returns = np.zeros(m)
    position = np.zeros(m, dtype=np.int8)

    # Raw returns
    for i in range(m):
        returns[i] = (price[i + 1] / price[i]) - 1.0

    # Effective signal considering gaps and liquidity
    eff_signal = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if gap_mask[i] or not liq_mask[i]:
            eff_signal[i] = 0
        else:
            eff_signal[i] = signal[i]

    # Position from effective signal
    for i in range(m):
        position[i] = eff_signal[i]

    # Trades and costs
    trades = 0
    prev_signal = eff_signal[0]
    for i in range(1, n):
        current_signal = eff_signal[i]
        if current_signal != prev_signal:
            trades += 1
            if price[i] > 1e-8:
                total_cost_pts = (
                    (spread_arr[i] + slippage_arr[i]) * 0.5
                    + brokerage_per_trade
                    + exchange_fee
                )
                cost_impact = total_cost_pts / price[i]
                returns[i - 1] -= cost_impact * abs(int(current_signal) - int(prev_signal))
            prev_signal = current_signal

    # Strategy returns = position * raw returns (with costs deducted)
    strat_rets = np.zeros(m)
    for i in range(m):
        strat_rets[i] = position[i] * returns[i]

    # Equity curve
    equity = np.zeros(n)
    equity[0] = 1.0
    cumprod = 1.0
    for i in range(m):
        cumprod *= 1.0 + strat_rets[i]
        equity[i + 1] = cumprod

    return strat_rets, equity, trades


@njit(cache=True)
def apply_ir_tax(
    pnl: np.ndarray,
    equity: np.ndarray,
    ir_tax_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica imposto de renda sobre o lucro liquido no final"""
    if ir_tax_rate <= 0.0:
        return pnl, equity

    net_profit = equity[-1] - 1.0
    if net_profit <= 0.0:
        return pnl, equity

    tax = net_profit * ir_tax_rate
    equity[-1] -= tax

    # Recalcula ultimo retorno para manter consistencia
    n = len(equity)
    if n > 1 and equity[-2] > 1e-10:
        pnl[-1] = equity[-1] / equity[-2] - 1.0

    return pnl, equity
