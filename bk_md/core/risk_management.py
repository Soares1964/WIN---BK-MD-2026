# ma_lab/core/risk_management.py
import numpy as np
from numba import njit


@njit(cache=True)
def compute_atr_numba(high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (Wilder smoothing) com Numba"""
    n = len(close)
    atr = np.zeros(n)

    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        if i < period:
            atr[i] = tr  # raw TR ate o periodo aquecer
        elif i == period:
            # Primeiro ATR = media simples
            s = 0.0
            for j in range(1, period + 1):
                tr_j = max(high[j] - low[j],
                           abs(high[j] - close[j - 1]),
                           abs(low[j] - close[j - 1]))
                s += tr_j
            atr[i] = s / period
        else:
            # Wilder smoothing: ATR = (ATR_prev*(p-1) + TR) / p
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period

    # Preenche atr[0] com atr[1] (nao ha TR para barra 0)
    if n > 1:
        atr[0] = atr[1]
    return atr


@njit(cache=True, fastmath=True)
def run_backtest_with_risk(
    price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal: np.ndarray,
    sl_type: int,
    sl_value: float,
    tp_type: int,
    tp_value: float,
    atr_period: int,
    slippage: float = 1.0,
    cost: float = 0.5,
) -> tuple:
    """
    Backtest vetorizado com stop-loss, take-profit e trailing stop.

    Codigos de controle:
        sl_type: 0=none, 1=fixed_pts, 2=atr_mult, 3=trailing_atr
        tp_type: 0=none, 1=fixed_pts, 2=atr_mult

    Quando sl_type=0 e tp_type=0, comportamento identico ao
    run_backtest_vectorizado (mesmo tracking de posicao).

    Retorno:
        (strategy_returns, equity_curve, trades, sl_hits, tp_hits)
    """
    n = len(price)
    returns = np.zeros(n - 1)
    equity = np.zeros(n)
    equity[0] = 1.0

    # Pre-computa ATR se necessario
    atr = np.zeros(n)
    use_risk = (sl_type > 0 or tp_type > 0)
    if sl_type >= 2 or tp_type >= 2:
        atr = compute_atr_numba(high, low, price, atr_period)

    half_tc = (cost + slippage) * 0.5

    # Variaveis de tracking de posicao
    risk_position = 0
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    trades = 0
    sl_hits = 0
    tp_hits = 0
    prev_signal = 0  # para custo no modo sem-risk

    for i in range(n - 1):
        # A posicao para esta barra depende do modo
        if use_risk:
            pos = risk_position
        else:
            # Sem risk mgmt: posicao segue o sinal exatamente
            pos = signal[i]

        period_return = 0.0
        exit_triggered = False
        exit_price = price[i + 1]
        exit_reason = 0  # 0=signal, 1=sl, 2=tp

        # --- Verifica saida (apenas para risk-tracked positions) ---
        if pos != 0 and use_risk:
            if pos == 1:  # Long
                if sl_type > 0 and low[i + 1] <= sl_level:
                    exit_price = sl_level
                    exit_triggered = True
                    exit_reason = 1
                elif tp_type > 0 and high[i + 1] >= tp_level:
                    exit_price = tp_level
                    exit_triggered = True
                    exit_reason = 2
                elif signal[i] != pos:
                    exit_triggered = True

                if sl_type == 3 and not exit_triggered:
                    new_sl = price[i + 1] - atr[i + 1] * sl_value
                    if new_sl > sl_level:
                        sl_level = new_sl
                    if low[i + 1] <= sl_level:
                        exit_price = sl_level
                        exit_triggered = True
                        exit_reason = 1

            else:  # Short
                if sl_type > 0 and high[i + 1] >= sl_level:
                    exit_price = sl_level
                    exit_triggered = True
                    exit_reason = 1
                elif tp_type > 0 and low[i + 1] <= tp_level:
                    exit_price = tp_level
                    exit_triggered = True
                    exit_reason = 2
                elif signal[i] != pos:
                    exit_triggered = True

                if sl_type == 3 and not exit_triggered:
                    new_sl = price[i + 1] + atr[i + 1] * sl_value
                    if new_sl < sl_level:
                        sl_level = new_sl
                    if high[i + 1] >= sl_level:
                        exit_price = sl_level
                        exit_triggered = True
                        exit_reason = 1

        # --- Computa retorno da barra ---
        if pos != 0:
            if exit_triggered:
                period_return = (exit_price / price[i] - 1.0) * pos
                if use_risk:
                    period_return -= half_tc / entry_price
                trades += 1
                if exit_reason == 1:
                    sl_hits += 1
                elif exit_reason == 2:
                    tp_hits += 1
                if use_risk:
                    risk_position = 0
            else:
                period_return = (price[i + 1] / price[i] - 1.0) * pos

        # --- Custo para modo sem-risk (replica run_backtest_vectorized) ---
        if not use_risk and signal[i] != prev_signal and i > 0:
            total_cost = (cost + slippage) * 0.5
            cost_impact = total_cost / price[i]
            returns[i - 1] -= cost_impact * abs(int(signal[i]) - int(prev_signal))
        if not use_risk:
            prev_signal = signal[i]

        # --- Abre nova posicao (apenas modo risk) ---
        if use_risk and signal[i] != 0 and risk_position == 0:
            risk_position = signal[i]
            entry_price = price[i]

            if risk_position == 1:  # Long
                if sl_type == 1:
                    sl_level = entry_price - sl_value
                elif sl_type >= 2:
                    sl_level = entry_price - atr[i] * sl_value
                if tp_type == 1:
                    tp_level = entry_price + tp_value
                elif tp_type >= 2:
                    tp_level = entry_price + atr[i] * tp_value
            else:  # Short
                if sl_type == 1:
                    sl_level = entry_price + sl_value
                elif sl_type >= 2:
                    sl_level = entry_price + atr[i] * sl_value
                if tp_type == 1:
                    tp_level = entry_price - tp_value
                elif tp_type >= 2:
                    tp_level = entry_price - atr[i] * tp_value

            period_return -= half_tc / entry_price

        returns[i] = period_return
        equity[i + 1] = equity[i] * (1.0 + period_return)

    return returns, equity, trades, sl_hits, tp_hits


@njit(cache=True, fastmath=True)
def apply_fixed_fractional_sizing(
    returns: np.ndarray,
    equity: np.ndarray,
    signal: np.ndarray,
    atr: np.ndarray,
    risk_per_trade: float,
    atr_mult: float,
    max_alloc_pct: float,
) -> tuple:
    """
    Aplica position sizing fracionario fixo sobre os retornos.

    Tamanho da posicao = (capital * risk_per_trade) / (atr * atr_mult)
    Limitado a capital * max_alloc_pct.

    Retorno:
        (sized_returns, sized_equity)
    """
    n = len(returns)
    sized_returns = np.zeros(n)
    sized_eq = np.zeros(n + 1)
    sized_eq[0] = 1.0
    capital = 1.0  # equity normalizada comeca em 1.0

    for i in range(n):
        if signal[i] != 0 and atr[i] > 0:
            risco_monetario = capital * risk_per_trade
            dist_stop = atr[i] * atr_mult
            size = risco_monetario / dist_stop

            # Limita alocacao maxima
            max_size = capital * max_alloc_pct
            if size > max_size:
                size = max_size

            sized_returns[i] = returns[i] * size
        else:
            sized_returns[i] = returns[i]

        capital = sized_eq[i] * (1.0 + sized_returns[i])
        sized_eq[i + 1] = capital

    return sized_returns, sized_eq


@njit(cache=True)
def compute_kelly_fraction(trade_returns: np.ndarray) -> float:
    """
    Computa a fracao otima de Kelly a partir dos retornos dos trades.

    f* = (p * b - q) / b
    onde p = win_rate, q = 1-p, b = avg_win / avg_loss

    Retorno: fracao entre 0.0 e 1.0 (limitada a 0.25 por seguranca)
    """
    n = len(trade_returns)
    if n < 10:
        return 0.0

    wins = 0
    sum_win = 0.0
    sum_loss = 0.0
    loss_count = 0

    for i in range(n):
        if trade_returns[i] > 0:
            wins += 1
            sum_win += trade_returns[i]
        elif trade_returns[i] < 0:
            sum_loss += abs(trade_returns[i])
            loss_count += 1

    if n == 0 or loss_count == 0:
        return 0.0

    win_rate = wins / n
    avg_win = sum_win / wins if wins > 0 else 0.0
    avg_loss = sum_loss / loss_count if loss_count > 0 else 0.0

    if avg_loss == 0.0:
        return 0.0

    b = avg_win / avg_loss
    if b <= 0:
        return 0.0

    kelly = (win_rate * b - (1.0 - win_rate)) / b

    # Limita a 25% por seguranca (Kelly fracionario)
    if kelly > 0.25:
        kelly = 0.25
    if kelly < 0.0:
        kelly = 0.0

    return kelly
