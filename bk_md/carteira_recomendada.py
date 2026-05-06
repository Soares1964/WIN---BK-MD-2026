# carteira_recomendada.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.data_loader import DataLoader
from core.backtest_engine import run_backtest_vectorized, generate_signal_from_cross, calculate_metrics
from core.indicators import ema_numba, sma_numba, hull_numba

# Carregar dados
loader = DataLoader()
loader.load_csv("exportacoes_mt5/WIN$N_1 minuto_2026-01-01_a_2026-03-16.csv")
price = loader.timeframe_dfs[5]['close'].values  # TF 5min

# =============================================
# SISTEMA 1 (PRINCIPAL) - 50% do capital
# EMA(26) x SMA(29)
# =============================================
fast1 = ema_numba(price, 26)
slow1 = sma_numba(price, 29)
signal1 = generate_signal_from_cross(fast1, slow1)
pnl1, equity1, trades1 = run_backtest_vectorized(price, signal1, slippage=0.5, cost=0.2)

# =============================================
# SISTEMA 2 (DIVERSIFICADOR) - 30% do capital
# HMA(10) x SMA(29) - Baixo drawdown
# =============================================
fast2 = hull_numba(price, 10)
slow2 = sma_numba(price, 29)
signal2 = generate_signal_from_cross(fast2, slow2)
pnl2, equity2, trades2 = run_backtest_vectorized(price, signal2, slippage=0.5, cost=0.2)

# =============================================
# SISTEMA 3 (BACKUP) - 20% do capital
# EMA(18) x SMA(25)
# =============================================
fast3 = ema_numba(price, 18)
slow3 = sma_numba(price, 25)
signal3 = generate_signal_from_cross(fast3, slow3)
pnl3, equity3, trades3 = run_backtest_vectorized(price, signal3, slippage=0.5, cost=0.2)

# =============================================
# CARTEIRA COMBINADA (media ponderada de PnL)
# NOTA: Combinamos os retornos (PnL), nao os trades.
# O numero de trades da carteira deve ser recalculado
# a partir do PnL combinado, nao somado ingenuamente.
# =============================================
pnl_combinado = 0.5 * pnl1 + 0.3 * pnl2 + 0.2 * pnl3
equity_combinado = np.cumprod(1 + pnl_combinado)

# Recalcula trades do PnL combinado (count de barras != 0)
trades_combinado = int(np.sum(pnl_combinado != 0))

metrics1 = calculate_metrics(pnl1, equity1, trades1)
metrics2 = calculate_metrics(pnl2, equity2, trades2)
metrics3 = calculate_metrics(pnl3, equity3, trades3)
metrics_combo = calculate_metrics(pnl_combinado, equity_combinado, trades_combinado)

print("="*70)
print("📊 COMPARACAO DE PERFORMANCE")
print("="*70)
print(f"\n{'Sistema':<20} {'Retorno':<10} {'PF':<8} {'DD':<8} {'Trades':<8}")
print("-"*60)

print(f"{'EMA26xSMA29':<20} {metrics1['total_return']:>6.1f}%   {metrics1['profit_factor']:>5.2f}   {metrics1['max_dd']:>5.1f}%   {metrics1['trades']:>5.0f}")
print(f"{'HMA10xSMA29':<20} {metrics2['total_return']:>6.1f}%   {metrics2['profit_factor']:>5.2f}   {metrics2['max_dd']:>5.1f}%   {metrics2['trades']:>5.0f}")
print(f"{'EMA18xSMA25':<20} {metrics3['total_return']:>6.1f}%   {metrics3['profit_factor']:>5.2f}   {metrics3['max_dd']:>5.1f}%   {metrics3['trades']:>5.0f}")
print("-"*60)
print(f"{'CARTEIRA (50/30/20)':<20} {metrics_combo['total_return']:>6.1f}%   {metrics_combo['profit_factor']:>5.2f}   {metrics_combo['max_dd']:>5.1f}%   {metrics_combo['trades']:>5.0f}")

# Plot
plt.figure(figsize=(15, 10))

plt.subplot(2,1,1)
plt.plot(equity1, label='EMA26xSMA29', alpha=0.7)
plt.plot(equity2, label='HMA10xSMA29', alpha=0.7)
plt.plot(equity3, label='EMA18xSMA25', alpha=0.7)
plt.plot(equity_combinado, label='CARTEIRA', linewidth=3, color='black')
plt.legend()
plt.title('Equity Curves - Sistemas Individuais vs Carteira')
plt.grid(True, alpha=0.3)

plt.subplot(2,1,2)
# Drawdown da carteira
peak = np.maximum.accumulate(equity_combinado)
drawdown = (equity_combinado - peak) / peak * 100
plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
plt.plot(drawdown, color='red', linewidth=1)
plt.title(f'Drawdown da Carteira - Maximo: {metrics_combo["max_dd"]:.1f}%')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
