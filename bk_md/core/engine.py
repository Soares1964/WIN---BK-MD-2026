"""
engine.py - Motor de otimizacao alternativo (legacy)

NOTA: Este arquivo existe para compatibilidade com codigo legacy.
O motor principal esta em core/optimizer_engine.py.
Para novo codigo, use OptimizationEngine de core.optimizer_engine.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd

from core.metrics import Metrics
from core.ma_cache import MovingAverageCache
from backtest.crossover import CrossoverBacktest


class OptimizationEngineV2:
    """Motor de otimizacao legado com multiprocessing.
    
    NOTA: Prefira usar OptimizationEngine de core.optimizer_engine
    que usa ThreadPoolExecutor e e mais eficiente para I/O bound.
    """

    def __init__(self, df):
        self.df = df
        self.price = df["close"]
        self.bt = CrossoverBacktest()
        self.cache = MovingAverageCache(self.price)

    def evaluate_system(self, args):
        fast, slow, pf, ps = args

        fast_ma = self.cache.get(fast, pf)
        slow_ma = self.cache.get(slow, ps)

        pnl, equity = self.bt.run(self.price, fast_ma, slow_ma)

        sharpe = Metrics.sharpe_ratio(pnl.values)
        pfactor = Metrics.profit_factor(pnl.values)
        win = Metrics.win_rate(pnl.values)
        dd = Metrics.max_drawdown(equity.values)

        return (
            fast,
            slow,
            sharpe,
            pfactor,
            win,
            dd
        )

    def run(self, fast_list, slow_list, fast_periods, slow_periods):
        tasks = []

        for f in fast_list:
            for s in slow_list:
                if f == s:
                    continue

                for pf in fast_periods:
                    for ps in slow_periods:
                        # pf = fast period, ps = slow period (nomes confusos mas intencionais)
                        if pf >= ps:
                            continue

                        tasks.append((f, s, pf, ps))

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self.evaluate_system, tasks)

        return results
