# ma_lab/core/walkforward.py
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime
import time
import gc

from core.optimizer_engine import (
    OptimizationEngine, OptimizationConfig, SystemResult
)
from core.ma_cache import MovingAverageCache


@dataclass
class WalkForwardConfig:
    """Configuracao do Walk-Forward Optimization"""
    enabled: bool = False
    n_windows: int = 3
    train_pct: float = 0.6
    test_pct: float = 0.2
    cv_mode: str = "rolling"  # rolling ou expanding
    top_n: int = 5
    min_train_bars: int = 1000


@dataclass
class WindowSplit:
    """Definicao de uma janela de walk-forward"""
    window_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


@dataclass
class WindowResult:
    """Resultado de uma janela de walk-forward"""
    window_idx: int
    split: WindowSplit
    train_results: List[SystemResult] = field(default_factory=list)
    oos_results: List[SystemResult] = field(default_factory=list)
    best_oos: Optional[SystemResult] = None

    @property
    def n_train_pass(self) -> int:
        return len(self.train_results)

    @property
    def n_oos(self) -> int:
        return len(self.oos_results)


@dataclass
class WalkForwardResult:
    """Resultado completo do walk-forward"""
    window_results: List[WindowResult] = field(default_factory=list)
    config: Optional[WalkForwardConfig] = None
    optimization_config: Optional[OptimizationConfig] = None

    @property
    def n_windows(self) -> int:
        return len(self.window_results)

    def get_all_oos(self) -> List[SystemResult]:
        """Retorna todos os resultados OOS de todas as janelas"""
        oos = []
        for wr in self.window_results:
            oos.extend(wr.oos_results)
        return oos

    def get_aggregated_metrics(self) -> dict:
        """Metricas agregadas OOS"""
        oos = self.get_all_oos()
        if not oos:
            return {}

        sharpes = [r.sharpe for r in oos if not np.isinf(r.sharpe) and not np.isnan(r.sharpe)]
        pfs = [r.profit_factor for r in oos if not np.isinf(r.profit_factor) and not np.isnan(r.profit_factor)]
        returns = [r.total_return for r in oos]
        dds = [r.max_dd for r in oos]
        trades = [r.trades for r in oos]

        def mean_std(arr):
            return float(np.mean(arr)), float(np.std(arr))

        sh_mean, sh_std = mean_std(sharpes) if sharpes else (0, 0)
        pf_mean, pf_std = mean_std(pfs) if pfs else (0, 0)
        ret_mean, ret_std = mean_std(returns)
        dd_mean, dd_std = mean_std(dds)

        stability = sh_std / max(abs(sh_mean), 0.001) if sh_mean != 0 else 999

        return {
            'avg_sharpe': round(sh_mean, 3),
            'std_sharpe': round(sh_std, 3),
            'avg_pf': round(pf_mean, 3),
            'std_pf': round(pf_std, 3),
            'avg_return': round(ret_mean, 1),
            'std_return': round(ret_std, 1),
            'avg_dd': round(dd_mean, 1),
            'std_dd': round(dd_std, 1),
            'avg_trades': int(np.mean(trades)),
            'n_windows': self.n_windows,
            'n_total_oos': len(oos),
            'stability_score': round(stability, 3),
        }


class WalkForwardEngine:
    """Motor de Walk-Forward Optimization"""

    def __init__(self, df_dict: Dict[int, pd.DataFrame]):
        self.df_dict = df_dict
        self.result: Optional[WalkForwardResult] = None

    def estimate_total(self, config) -> int:
        """Estima total de combinacoes (delega para OptimizationEngine)"""
        from core.optimizer_engine import OptimizationEngine
        fake_engine = OptimizationEngine(self.df_dict)
        return fake_engine.estimate_total(config)

    def generate_splits(self, n_bars: int,
                        config: WalkForwardConfig) -> List[WindowSplit]:
        """Gera janelas de walk-forward para o dado"""
        splits = []

        if n_bars < config.min_train_bars * 2:
            return splits

        train_bars = int(n_bars * config.train_pct)
        test_bars = int(n_bars * config.test_pct)
        step = train_bars // max(config.n_windows, 1)

        if config.cv_mode == "expanding":
            # Expanding: treino cresce, teste desliza
            for w in range(config.n_windows):
                t_start = 0
                t_end = min(train_bars + w * (train_bars // config.n_windows), n_bars - test_bars)
                v_start = t_end
                v_end = min(t_end + test_bars, n_bars)

                if v_end - v_start < 50:
                    continue

                splits.append(WindowSplit(w, t_start, t_end, v_start, v_end))
        else:
            # Rolling: janela deslizante
            overlap = max(train_bars - step, 0)
            start_pos = 0

            for w in range(config.n_windows):
                t_start = start_pos
                t_end = min(t_start + train_bars, n_bars - test_bars)
                v_start = t_end
                v_end = min(t_end + test_bars, n_bars)

                if t_end - t_start < config.min_train_bars or v_end - v_start < 50:
                    # Ajusta ultima janela para usar o maximo de dados
                    if w > 0:
                        t_start = n_bars - train_bars - test_bars
                        t_end = n_bars - test_bars
                        v_start = t_end
                        v_end = n_bars
                        if t_end - t_start >= config.min_train_bars:
                            splits[-1] = WindowSplit(w - 1, t_start, t_end, v_start, v_end)
                    break

                splits.append(WindowSplit(w, t_start, t_end, v_start, v_end))
                start_pos += step

        return splits

    def run(self, opt_config: OptimizationConfig,
            wf_config: Optional[WalkForwardConfig] = None,
            progress_callback: Optional[Callable] = None,
            result_callback: Optional[Callable] = None) -> WalkForwardResult:
        """
        Executa walk-forward optimization completa.

        1. Gera splits para cada timeframe (usa o primeiro como referencia)
        2. Para cada janela: otimiza em treino, testa top-N em OOS
        3. Agrega resultados
        """
        if wf_config is None:
            wf_config = WalkForwardConfig(
                enabled=True,
                n_windows=opt_config.wf_n_windows,
                train_pct=opt_config.wf_train_pct,
                test_pct=opt_config.wf_test_pct,
                cv_mode=opt_config.wf_cv_mode,
                top_n=opt_config.wf_top_n
            )

        # Usa o primeiro timeframe como referencia para splits
        ref_tf = opt_config.timeframes[0] if opt_config.timeframes else 5
        ref_df = self.df_dict.get(ref_tf)
        if ref_df is None or ref_df.empty:
            raise ValueError(f"Timeframe {ref_tf} sem dados")

        n_bars = len(ref_df)
        splits = self.generate_splits(n_bars, wf_config)

        if not splits:
            raise ValueError(
                f"Dados insuficientes para walk-forward: {n_bars} barras, "
                f"minimo ~{wf_config.min_train_bars * 2}"
            )

        result = WalkForwardResult(
            config=wf_config,
            optimization_config=opt_config
        )

        start_time = time.time()
        total_windows = len(splits)

        for w_idx, split in enumerate(splits):
            window_start = time.time()

            if progress_callback:
                progress_callback(
                    (w_idx / total_windows) * 100,
                    w_idx, total_windows, 0, 0, None, "Preparando janela..."
                )

            # Cria dados de treino (subsets)
            train_dict = {}
            for tf in opt_config.timeframes:
                df = self.df_dict.get(tf)
                if df is None:
                    continue
                # Ajusta indices do split para o tamanho deste timeframe
                tf_ratio = len(df) / n_bars
                t_start = int(split.train_start * tf_ratio)
                t_end = int(split.train_end * tf_ratio)
                t_start = max(0, t_start)
                t_end = min(len(df), t_end)
                if t_end - t_start > 100:
                    train_dict[tf] = df.iloc[t_start:t_end].copy()

            if not train_dict:
                continue

            # Cria engine de treino e executa otimizacao
            train_engine = OptimizationEngine(train_dict)
            train_results = train_engine.run(
                opt_config,
                progress_callback=None,  # silencia callbacks internos
                result_callback=None
            )

            if not train_results:
                continue

            # Pega top-N resultados do treino
            top_n_results = train_results[:min(wf_config.top_n, len(train_results))]

            # Testa cada top-N no OOS
            window_result = WindowResult(window_idx=split.window_idx, split=split)
            window_result.train_results = top_n_results

            for tf in opt_config.timeframes:
                df = self.df_dict.get(tf)
                if df is None:
                    continue

                tf_ratio = len(df) / n_bars
                t_start = int(split.test_start * tf_ratio)
                t_end = int(split.test_end * tf_ratio)
                t_start = max(0, t_start)
                t_end = min(len(df), t_end)

                if t_end - t_start < 50:
                    continue

                test_df = df.iloc[t_start:t_end].copy()

                # Cria cache para dados de teste
                test_cache = MovingAverageCache(
                    test_df['close'] if 'close' in test_df.columns else test_df['Close']
                )

                price_arr = test_df['close'].values.astype(np.float64)
                high_arr = test_df['high'].values.astype(np.float64) if 'high' in test_df.columns else price_arr
                low_arr = test_df['low'].values.astype(np.float64) if 'low' in test_df.columns else price_arr

                # Cria engine temporario para usar test_single_system
                test_engine = OptimizationEngine({tf: test_df})

                for train_result in top_n_results:
                    oos_result = test_engine.test_single_system(
                        train_result.fast_indicator,
                        train_result.slow_indicator,
                        train_result.fast_period,
                        train_result.slow_period,
                        tf,
                        opt_config,
                        price_arr, high_arr, low_arr,
                        test_cache
                    )

                    if oos_result is not None:
                        oos_result.window_id = w_idx
                        oos_result.is_oos = True
                        oos_result.train_score = train_result.score()
                        window_result.oos_results.append(oos_result)

                        if result_callback:
                            result_callback(oos_result)

                del test_cache
                gc.collect()

            # Melhor OOS desta janela
            if window_result.oos_results:
                window_result.best_oos = max(
                    window_result.oos_results,
                    key=lambda r: r.score()
                )

            result.window_results.append(window_result)

            # Progresso
            if progress_callback:
                elapsed = time.time() - start_time
                pct = ((w_idx + 1) / total_windows) * 100
                rate = (w_idx + 1) / max(elapsed, 0.001)
                remaining = (total_windows - w_idx - 1) / max(rate, 0.001)

                best_str = ""
                if window_result.best_oos:
                    best_str = f"{window_result.best_oos.fast_indicator}({window_result.best_oos.fast_period})x{window_result.best_oos.slow_indicator}({window_result.best_oos.slow_period})"

                progress_callback(
                    pct, w_idx + 1, total_windows, rate, remaining,
                    window_result.best_oos, best_str
                )

            # Limpeza
            del train_engine
            gc.collect()

        return result
