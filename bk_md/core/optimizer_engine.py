# ma_lab/core/optimizer_engine.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading  # Adicionado: lock correto para threads
from dataclasses import dataclass, field
from datetime import datetime
import time
import gc
import traceback
import os
from functools import partial

from core.ma_cache import MovingAverageCache
from core.indicators import INDICATORS, get_all_indicators
from core.backtest_engine import (
    run_backtest_vectorized,
    generate_signal_from_cross,
    calculate_metrics,
    periods_per_year_for_timeframe
)
from core.risk_management import (
    run_backtest_with_risk,
    compute_atr_numba,
    apply_fixed_fractional_sizing,
    compute_kelly_fraction
)
from core.transactional_costs import (
    compute_variable_spread,
    compute_variable_slippage,
    detect_session_gaps,
    compute_liquidity_mask,
    compute_atr_stats,
    run_backtest_realistic,
    apply_ir_tax,
)
from core.portfolio_management import (
    PortfolioConfig,
    compute_correlation_matrix,
    markowitz_optimization,
    risk_parity_optimization,
    filter_by_correlation,
    combine_portfolio_returns,
    compute_portfolio_equity_numba,
    apply_drawdown_management_numba,
    compute_portfolio_metrics,
    PortfolioResult,
)


@dataclass
class OptimizationConfig:
    """Configuração da otimização"""
    # Indicadores
    fast_indicators: List[str]
    slow_indicators: List[str]

    # Períodos
    fast_min: int = 5
    fast_max: int = 50
    fast_step: int = 1

    slow_min: int = 20
    slow_max: int = 200
    slow_step: int = 5

    # Timeframes
    timeframes: List[int] = field(default_factory=lambda: list(range(1, 11)))

    # Filtros de qualidade
    min_trades: int = 100
    min_sharpe: float = 1.2
    min_profit_factor: float = 1.5
    max_dd: float = 8.0

    # Custos
    slippage: float = 1.0
    cost: float = 0.5

    # Performance
    batch_size: int = 1000
    n_jobs: int = mp.cpu_count()
    use_cache: bool = True

    # --- Risk Management ---
    sl_type: int = 0           # 0=none, 1=fixed_pts, 2=atr_mult, 3=trailing
    sl_value: float = 0.0      # pontos ou multiplicador ATR
    tp_type: int = 0           # 0=none, 1=fixed_pts, 2=atr_mult
    tp_value: float = 0.0      # pontos ou multiplicador ATR
    atr_period: int = 14

    # --- Position Sizing ---
    sizing_method: int = 0     # 0=none(1 contrato), 1=fixed_fractional, 2=kelly
    risk_per_trade: float = 0.01     # % do capital por trade (metodo 1)
    kelly_fraction: float = 0.25     # fracao do Kelly otimo (metodo 2)
    max_alloc_pct: float = 1.0       # alocacao maxima por posicao

    # --- Walk-Forward ---
    use_walkforward: bool = False
    wf_n_windows: int = 3
    wf_train_pct: float = 0.6
    wf_test_pct: float = 0.2
    wf_cv_mode: str = "rolling"   # rolling ou expanding
    wf_top_n: int = 5

    # --- Statistical Validation ---
    run_statistical_validation: bool = False
    stat_top_n: int = 10
    stat_n_monte_carlo: int = 1000
    stat_n_bootstrap: int = 1000
    stat_n_null_hypothesis: int = 200
    stat_mc_confidence: float = 0.95
    stat_bootstrap_confidence: float = 0.95

    # --- Transactional Costs (Realismo Transacional) ---
    tc_enabled: bool = False
    tc_spread_variable: bool = False
    tc_spread_base_pts: float = 0.5
    tc_spread_vol_mult: float = 0.05
    tc_spread_max_pts: float = 5.0
    tc_slippage_variable: bool = False
    tc_slippage_base_pts: float = 0.5
    tc_slippage_vol_mult: float = 0.10
    tc_slippage_max_pts: float = 10.0
    tc_brokerage_per_trade: float = 0.0
    tc_exchange_fee: float = 0.0
    tc_ir_tax_enabled: bool = False
    tc_ir_tax_rate: float = 0.0
    tc_liquidity_enabled: bool = False
    tc_liquidity_min_atr_ratio: float = 0.3
    tc_gap_enabled: bool = False
    tc_gap_max_minutes: int = 30

    # --- Portfolio Management ---
    pf_enabled: bool = False
    pf_allocation_method: str = "equal"       # equal, markowitz, risk_parity
    pf_rebalance_enabled: bool = False
    pf_rebalance_freq_bars: int = 1000
    pf_dd_management_enabled: bool = False
    pf_dd_threshold: float = 10.0
    pf_dd_reduction: float = 0.5
    pf_top_n: int = 10
    pf_max_correlation: float = 0.95

    # Debug
    debug: bool = False


@dataclass
class SystemResult:
    """Resultado de um sistema testado"""
    fast_indicator: str
    slow_indicator: str
    fast_period: int
    slow_period: int
    timeframe: int

    sharpe: float
    profit_factor: float
    win_rate: float
    max_dd: float
    total_return: float
    trades: int

    # Risk management diagnostics
    sl_hits: int = 0
    tp_hits: int = 0

    # Walk-forward metadata
    window_id: int = -1       # -1 = nao WF
    is_oos: bool = False      # True = resultado out-of-sample
    train_score: float = 0.0  # score no treino (para comparacao WF)

    # Statistical validation fields (populated by _run_statistical_validation)
    mc_prob_loss: float = 0.0
    mc_sharpe_mean: float = 0.0
    mc_sharpe_std: float = 0.0
    mc_sharpe_ci_lower: float = 0.0
    mc_sharpe_ci_upper: float = 0.0
    mc_return_ci_lower: float = 0.0
    mc_return_ci_upper: float = 0.0
    mc_dd_ci_lower: float = 0.0
    mc_dd_ci_upper: float = 0.0
    boot_sharpe_mean: float = 0.0
    boot_sharpe_std: float = 0.0
    boot_sharpe_ci_lower: float = 0.0
    boot_sharpe_ci_upper: float = 0.0
    boot_sharpe_p_value: float = 0.0
    dsr: float = 0.0
    null_sharpe_mean: float = 0.0
    null_sharpe_std: float = 0.0
    null_p_value: float = 0.0
    sens_robustness: float = 0.0
    sens_sharpe_std: float = 0.0
    sens_avg_sharpe: float = 0.0

    # Internal: raw data for statistical post-processing (NOT serialized)
    _pnl_array: Optional[np.ndarray] = None
    _signal_array: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = {
            'fast': self.fast_indicator,
            'slow': self.slow_indicator,
            'fp': self.fast_period,
            'sp': self.slow_period,
            'tf': self.timeframe,
            'sharpe': round(self.sharpe, 3),
            'pf': round(self.profit_factor, 3),
            'win': round(self.win_rate, 1),
            'dd': round(self.max_dd, 1),
            'ret': round(self.total_return, 1),
            'trades': self.trades,
            'sl_hits': self.sl_hits,
            'tp_hits': self.tp_hits,
        }
        if self.window_id >= 0:
            d['window_id'] = self.window_id
            d['is_oos'] = self.is_oos
            d['train_score'] = round(self.train_score, 3)

        # Statistical validation fields (only if populated)
        if self.dsr != 0.0 or self.mc_prob_loss != 0.0:
            d['mc_prob_loss'] = round(self.mc_prob_loss, 4)
            d['mc_sharpe_mean'] = round(self.mc_sharpe_mean, 3)
            d['mc_sharpe_std'] = round(self.mc_sharpe_std, 3)
            d['mc_sharpe_ci_lower'] = round(self.mc_sharpe_ci_lower, 3)
            d['mc_sharpe_ci_upper'] = round(self.mc_sharpe_ci_upper, 3)
            d['mc_return_ci_lower'] = round(self.mc_return_ci_lower, 1)
            d['mc_return_ci_upper'] = round(self.mc_return_ci_upper, 1)
            d['mc_dd_ci_lower'] = round(self.mc_dd_ci_lower, 1)
            d['mc_dd_ci_upper'] = round(self.mc_dd_ci_upper, 1)
            d['boot_sharpe_mean'] = round(self.boot_sharpe_mean, 3)
            d['boot_sharpe_std'] = round(self.boot_sharpe_std, 3)
            d['boot_sharpe_ci_lower'] = round(self.boot_sharpe_ci_lower, 3)
            d['boot_sharpe_ci_upper'] = round(self.boot_sharpe_ci_upper, 3)
            d['boot_sharpe_p_value'] = round(self.boot_sharpe_p_value, 4)
            d['dsr'] = round(self.dsr, 4)
            d['null_sharpe_mean'] = round(self.null_sharpe_mean, 3)
            d['null_sharpe_std'] = round(self.null_sharpe_std, 3)
            d['null_p_value'] = round(self.null_p_value, 4)
            d['sens_robustness'] = round(self.sens_robustness, 4)
            d['sens_sharpe_std'] = round(self.sens_sharpe_std, 3)
            d['sens_avg_sharpe'] = round(self.sens_avg_sharpe, 3)

        return d

    def score(self) -> float:
        """Calcula score global do sistema"""
        # Trata inf/nan para calculo de score
        pf = self.profit_factor
        if np.isinf(pf):
            pf = 10.0  # Valor alto mas finito para sistemas sem perdas
        if np.isnan(pf):
            pf = 0.0
        
        sh = self.sharpe
        if np.isinf(sh) or np.isnan(sh):
            sh = 0.0
        
        return (
            sh * 3.0 +
            pf * 2.5 +
            self.win_rate * 0.25 -
            self.max_dd * 0.3
        )


class OptimizationEngine:
    """Motor de otimização com Threads (alta performance)"""

    def __init__(self, df_dict: Dict[int, pd.DataFrame]):
        """
        Args:
            df_dict: Dicionário com DataFrames para cada timeframe
                     {timeframe: df com coluna 'close'}
        """
        self.df_dict = df_dict
        self.prices = {}
        self.highs = {}
        self.lows = {}
        self.caches = {}
        self.timestamps = {}

        for tf, df in df_dict.items():
            if df is None or df.empty:
                raise ValueError(f"DataFrame vazio para timeframe {tf}")

            if 'close' in df.columns:
                self.prices[tf] = df['close'].values.astype(np.float64)
            elif 'Close' in df.columns:
                self.prices[tf] = df['Close'].values.astype(np.float64)
            else:
                raise ValueError(f"DataFrame precisa ter coluna 'close' para timeframe {tf}")

            # Armazena high/low para risk management
            if 'high' in df.columns:
                self.highs[tf] = df['high'].values.astype(np.float64)
            if 'low' in df.columns:
                self.lows[tf] = df['low'].values.astype(np.float64)

            # Cria cache por timeframe
            self.caches[tf] = MovingAverageCache(
                df['close'] if 'close' in df.columns else df['Close']
            )

            # Timestamps for gap detection (int64 nanoseconds)
            if isinstance(df.index, pd.DatetimeIndex):
                self.timestamps[tf] = df.index.values.astype(np.int64)
            else:
                self.timestamps[tf] = np.arange(len(df), dtype=np.int64)

        self.results: List[SystemResult] = []
        self.best_result: Optional[SystemResult] = None
        self.portfolio_result: Optional[PortfolioResult] = None
        self.is_running = False
        self.is_paused = False
        self.progress = 0.0
        self.total_systems = 0
        self.processed_systems = 0
        self.start_time = None
        self._lock = threading.Lock()  # Corrigido: lock de thread ( nao multiprocessing)

    def estimate_total(self, config: OptimizationConfig) -> int:
        """Estima número total de combinações"""
        fast_periods = len(range(config.fast_min, config.fast_max + 1, config.fast_step))
        slow_periods = len(range(config.slow_min, config.slow_max + 1, config.slow_step))

        return (
            len(config.fast_indicators) *
            len(config.slow_indicators) *
            fast_periods *
            slow_periods *
            len(config.timeframes)
        )

    def _process_single(self, fast_ind: str, slow_ind: str,
                        fp: int, sp: int, tf: int,
                        config: OptimizationConfig,
                        price_arr: Optional[np.ndarray] = None,
                        high_arr: Optional[np.ndarray] = None,
                        low_arr: Optional[np.ndarray] = None,
                        cache_override: Optional = None) -> Optional[SystemResult]:
        """Processa um único sistema

        Args:
            price_arr: opcional, para usar dados especificos (ex: janela WF)
            high_arr: opcional, high para SL/TP
            low_arr: opcional, low para SL/TP
            cache_override: opcional, cache para dados especificos
        """
        try:
            # Usa dados fornecidos ou os defaults do engine
            prices = price_arr if price_arr is not None else self.prices[tf]
            cache = cache_override if cache_override is not None else self.caches[tf]
            highs = high_arr if high_arr is not None else self.highs.get(tf)
            lows = low_arr if low_arr is not None else self.lows.get(tf)

            # Obtém médias do cache
            fast_ma = cache.get(fast_ind, fp).values
            slow_ma = cache.get(slow_ind, sp).values

            # Verifica se as médias são válidas
            if len(fast_ma) == 0 or len(slow_ma) == 0:
                return None

            # Gera sinais
            signal = generate_signal_from_cross(fast_ma, slow_ma)

            # Executa backtest (com ou sem risk management / custos realistas)
            use_risk = config.sl_type > 0 or config.tp_type > 0

            if config.tc_enabled and not use_risk:
                # --- Transactional costs path (realistic) ---
                has_any_feature = (
                    config.tc_spread_variable
                    or config.tc_slippage_variable
                    or config.tc_gap_enabled
                    or config.tc_liquidity_enabled
                    or config.tc_brokerage_per_trade > 0
                    or config.tc_exchange_fee > 0
                    or config.tc_ir_tax_enabled
                )

                if not has_any_feature:
                    # tc_enabled=True but no features active: use standard vectorized
                    pnl, equity, trades = run_backtest_vectorized(
                        prices, signal, config.slippage, config.cost
                    )
                    sl_hits = 0
                    tp_hits = 0
                else:
                    # Compute ATR once for variable spread/slippage/liquidity
                    h_for_atr = self.highs.get(tf) if highs is None else highs
                    l_for_atr = self.lows.get(tf) if lows is None else lows
                    if h_for_atr is None:
                        h_for_atr = prices
                    if l_for_atr is None:
                        l_for_atr = prices
                    atr = compute_atr_numba(h_for_atr, l_for_atr, prices, config.atr_period)

                    # 1) Spread array (variable or constant)
                    if config.tc_spread_variable:
                        spread_arr = compute_variable_spread(
                            prices, atr,
                            config.tc_spread_base_pts,
                            config.tc_spread_vol_mult,
                            config.tc_spread_max_pts,
                        )
                    else:
                        spread_arr = np.full(len(prices), config.tc_spread_base_pts)

                    # 2) Slippage array (variable or constant)
                    if config.tc_slippage_variable:
                        slippage_arr = compute_variable_slippage(
                            prices, atr,
                            config.tc_slippage_base_pts,
                            config.tc_slippage_vol_mult,
                            config.tc_slippage_max_pts,
                        )
                    else:
                        slippage_arr = np.full(len(prices), config.tc_slippage_base_pts)

                    # 3) Gap mask
                    if config.tc_gap_enabled:
                        ts_arr = self.timestamps.get(tf, np.arange(len(prices), dtype=np.int64))
                        max_gap_ns = config.tc_gap_max_minutes * 60 * 1_000_000_000
                        gap_mask = detect_session_gaps(ts_arr, float(max_gap_ns))
                    else:
                        gap_mask = np.zeros(len(prices), dtype=np.bool_)

                    # 4) Liquidity mask
                    if config.tc_liquidity_enabled:
                        atr_mean, _ = compute_atr_stats(atr)
                        liq_mask = compute_liquidity_mask(
                            atr, atr_mean, config.tc_liquidity_min_atr_ratio
                        )
                    else:
                        liq_mask = np.ones(len(prices), dtype=np.bool_)

                    # 5) Run realistic backtest
                    pnl, equity, trades = run_backtest_realistic(
                        prices, signal,
                        spread_arr, slippage_arr,
                        gap_mask, liq_mask,
                        config.tc_brokerage_per_trade,
                        config.tc_exchange_fee,
                    )
                    sl_hits = 0
                    tp_hits = 0

                    # 6) IR tax post-processing
                    if config.tc_ir_tax_enabled and config.tc_ir_tax_rate > 0:
                        pnl, equity = apply_ir_tax(pnl, equity, config.tc_ir_tax_rate)

            elif use_risk and highs is not None and lows is not None:
                pnl, equity, trades, sl_hits, tp_hits = run_backtest_with_risk(
                    prices, highs, lows, signal,
                    config.sl_type, config.sl_value,
                    config.tp_type, config.tp_value,
                    config.atr_period,
                    config.slippage, config.cost
                )
            else:
                pnl, equity, trades = run_backtest_vectorized(
                    prices, signal, config.slippage, config.cost
                )
                sl_hits = 0
                tp_hits = 0

            # Apply position sizing if enabled
            if config.sizing_method == 1 and use_risk:
                atr = compute_atr_numba(highs, lows, prices, config.atr_period)
                pnl, equity = apply_fixed_fractional_sizing(
                    pnl, equity, signal, atr,
                    config.risk_per_trade, config.sl_value if config.sl_value > 0 else 1.0,
                    config.max_alloc_pct
                )
            elif config.sizing_method == 2:
                # Kelly sizing: apply fraction to entire equity curve
                kelly_frac = compute_kelly_fraction(pnl[pnl != 0])
                if kelly_frac > 0:
                    kelly_frac = min(kelly_frac, config.kelly_fraction)
                    pnl = pnl * kelly_frac
                    equity = np.zeros(len(prices))
                    equity[0] = 1.0
                    for i in range(len(pnl)):
                        equity[i + 1] = equity[i] * (1.0 + pnl[i])

            # Filtro de trades mínimos
            if trades < config.min_trades:
                return None

            # Calcula métricas (com anualização correta para o timeframe)
            ppy = periods_per_year_for_timeframe(tf)
            metrics = calculate_metrics(pnl, equity, trades, periods_per_year=ppy)

            # Trata inf/nan nos filtros
            pf = metrics['profit_factor']
            if np.isinf(pf):
                pf = 10.0  # Valor alto mas finito
            
            sh = metrics['sharpe']
            if np.isinf(sh) or np.isnan(sh):
                sh = -999.0

            # Aplica filtros de qualidade
            if sh < config.min_sharpe:
                return None
            if pf < config.min_profit_factor:
                return None
            if metrics['max_dd'] > config.max_dd:
                return None

            result = SystemResult(
                fast_indicator=fast_ind,
                slow_indicator=slow_ind,
                fast_period=fp,
                slow_period=sp,
                timeframe=tf,
                sharpe=metrics['sharpe'],
                profit_factor=metrics['profit_factor'],
                win_rate=metrics['win_rate'],
                max_dd=metrics['max_dd'],
                total_return=metrics['total_return'],
                trades=metrics['trades'],
                sl_hits=sl_hits,
                tp_hits=tp_hits
            )

            # Store raw arrays for statistical validation / portfolio (post-processing)
            if config.run_statistical_validation or config.pf_enabled:
                result._pnl_array = pnl.copy()
                result._signal_array = signal.copy()

            return result

        except Exception as e:
            if config.debug:
                print(f"Erro: {fast_ind}({fp}) x {slow_ind}({sp}) TF{tf}: {e}")
            return None

    def _process_batch(self, batch_tasks: List[tuple],
                       config: OptimizationConfig) -> List[SystemResult]:
        """Processa um batch de sistemas"""
        results = []
        for task in batch_tasks:
            if not self.is_running:
                break
            fast_ind, slow_ind, fp, sp, tf = task
            result = self._process_single(fast_ind, slow_ind, fp, sp, tf, config)
            if result:
                results.append(result)
                # Atualiza melhor resultado (thread-safe)
                with self._lock:
                    if self.best_result is None or result.score() > self.best_result.score():
                        self.best_result = result
        return results

    def _generate_all_tasks(self, config: OptimizationConfig) -> List[tuple]:
        """Gera todas as tarefas possíveis"""
        tasks = []

        fast_periods = list(range(config.fast_min, config.fast_max + 1, config.fast_step))
        slow_periods = list(range(config.slow_min, config.slow_max + 1, config.slow_step))

        for fast_ind in config.fast_indicators:
            for slow_ind in config.slow_indicators:
                if fast_ind == slow_ind:
                    continue

                for fp in fast_periods:
                    for sp in slow_periods:
                        if fp >= sp:
                            continue

                        for tf in config.timeframes:
                            tasks.append((fast_ind, slow_ind, fp, sp, tf))

        return tasks

    def run(self, config: OptimizationConfig,
            progress_callback: Optional[Callable] = None,
            result_callback: Optional[Callable] = None) -> List[SystemResult]:
        """
        Executa otimização completa usando ThreadPoolExecutor

        Args:
            config: Configuração da otimização
            progress_callback: Função para reportar progresso
            result_callback: Função para reportar resultados parciais

        Returns:
            Lista de resultados
        """
        self.is_running = True
        self.is_paused = False
        self.results = []
        self.best_result = None
        self.processed_systems = 0
        self.total_systems = self.estimate_total(config)
        self.start_time = time.time()

        if self.total_systems == 0:
            print("Aviso: Nenhum sistema para testar")
            return []

        # Gera todas as combinações
        all_tasks = self._generate_all_tasks(config)

        if len(all_tasks) == 0:
            print("Aviso: Nenhuma tarefa gerada")
            return []

        if config.debug:
            print(f"📦 Total de tarefas: {len(all_tasks)}")
            print(f"⚡ Threads: {config.n_jobs}")

        # Divide em batches
        batches = []
        for i in range(0, len(all_tasks), config.batch_size):
            batches.append(all_tasks[i:i + config.batch_size])

        # Processa batches com ThreadPoolExecutor
        for batch_idx, batch_tasks in enumerate(batches):
            if not self.is_running:
                break

            while self.is_paused:
                time.sleep(0.1)
                if not self.is_running:
                    break

            batch_start = time.time()

            # Divide batch em sub-batches para paralelização
            n_workers = min(config.n_jobs, len(batch_tasks))
            sub_batch_size = max(1, len(batch_tasks) // n_workers)
            sub_batches = []

            for i in range(0, len(batch_tasks), sub_batch_size):
                sub_batches.append(batch_tasks[i:i + sub_batch_size])

            # Processa sub-batches em paralelo com threads
            batch_results = []

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for sub_batch in sub_batches:
                    future = executor.submit(self._process_batch, sub_batch, config)
                    futures.append(future)

                for future in as_completed(futures):
                    if not self.is_running:
                        break
                    try:
                        sub_results = future.result(timeout=30)
                        batch_results.extend(sub_results)
                        for result in sub_results:
                            if result_callback:
                                result_callback(result)
                    except Exception as e:
                        if config.debug:
                            print(f"Erro no batch: {e}")
                        continue

            self.results.extend(batch_results)
            self.processed_systems += len(batch_tasks)

            # Calcula progresso
            self.progress = (self.processed_systems / self.total_systems) * 100

            # Estatísticas de performance
            elapsed = time.time() - self.start_time
            batch_time = time.time() - batch_start

            if self.processed_systems > 0:
                rate = self.processed_systems / elapsed
                remaining = (self.total_systems - self.processed_systems) / rate if rate > 0 else 0
            else:
                rate = 0
                remaining = 0

            # Log de performance (debug)
            if config.debug and batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(batches)}: {len(batch_tasks)} tasks em {batch_time:.2f}s "
                      f"({len(batch_tasks)/batch_time:.0f} tasks/s)")

            # Reporta progresso
            if progress_callback:
                try:
                    progress_callback(
                        self.progress,
                        self.processed_systems,
                        self.total_systems,
                        rate,
                        remaining,
                        self.best_result
                    )
                except Exception as e:
                    if config.debug:
                        print(f"Erro no progress_callback: {e}")

            # Limpa memória periodicamente
            if batch_idx % 10 == 0:
                gc.collect()

        # Ordena resultados por score
        self.results.sort(key=lambda x: x.score(), reverse=True)

        # Statistical validation (pos-processamento sobre top-N)
        if config.run_statistical_validation and self.results:
            self._run_statistical_validation(config, progress_callback)

        # Portfolio optimization (pos-processamento sobre top-N)
        if config.pf_enabled and self.results:
            try:
                self._run_portfolio_optimization(config)
            except Exception as e:
                if config.debug:
                    print(f"Erro na otimizacao de portfolio: {e}")
                self.portfolio_result = None

        self.is_running = False

        return self.results

    def pause(self):
        """Pausa a otimização"""
        self.is_paused = True

    def resume(self):
        """Continua a otimização"""
        self.is_paused = False

    def stop(self):
        """Para a otimização"""
        self.is_running = False
        self.is_paused = False

    def _run_statistical_validation(
        self,
        config: OptimizationConfig,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """
        Pos-processamento: executa validacao estatistica nos top-N sistemas.

        Roda Monte Carlo, Bootstrap, DSR, Null Hypothesis e Sensitivity
        em cada um dos top-N resultados (por score). Modifica os objetos
        SystemResult in-place.
        """
        from core.statistical_validation import (
            monte_carlo_simulation,
            bootstrap_sharpe_ratio,
            deflated_sharpe_ratio,
            null_hypothesis_test,
            parameter_sensitivity,
        )
        from core.backtest_engine import periods_per_year_for_timeframe

        top_n = min(config.stat_top_n, len(self.results))
        if top_n == 0:
            return

        top_systems = self.results[:top_n]
        total_steps = top_n * 4  # MC + Bootstrap + DSR + Null; Sensitivity is extra
        step = 0

        for idx, result in enumerate(top_systems):
            if not self.is_running:
                break

            pnl = result._pnl_array
            signal = result._signal_array
            tf = result.timeframe
            ppy = periods_per_year_for_timeframe(tf)
            price = self.prices.get(tf)

            if pnl is None or signal is None or len(pnl) < 5:
                continue

            # 1) Monte Carlo
            try:
                mc_res = monte_carlo_simulation(
                    pnl, config.stat_n_monte_carlo,
                    config.stat_mc_confidence, ppy
                )
                for k, v in mc_res.items():
                    setattr(result, k, v)
            except Exception:
                pass
            step += 1

            # 2) Bootstrap
            try:
                boot_res = bootstrap_sharpe_ratio(
                    pnl, config.stat_n_bootstrap,
                    config.stat_bootstrap_confidence, ppy
                )
                for k, v in boot_res.items():
                    setattr(result, k, v)
            except Exception:
                pass
            step += 1

            # 3) Deflated Sharpe Ratio
            try:
                n_obs = len(pnl)
                n_trials = self.estimate_total(config)
                result.dsr = deflated_sharpe_ratio(
                    result.sharpe, n_obs, n_trials,
                    periods_per_year=ppy
                )
            except Exception:
                pass
            step += 1

            # 4) Null Hypothesis Test
            if price is not None and len(price) > 100:
                try:
                    null_res = null_hypothesis_test(
                        price, signal, result.sharpe,
                        config.stat_n_null_hypothesis,
                        config.slippage, config.cost, ppy
                    )
                    for k, v in null_res.items():
                        setattr(result, k, v)
                except Exception:
                    pass
            step += 1

            # 5) Parameter Sensitivity (only if cache exists)
            if price is not None and tf in self.caches:
                try:
                    cache = self.caches.get(tf)
                    high_arr = self.highs.get(tf) or price
                    low_arr = self.lows.get(tf) or price
                    sens_res = parameter_sensitivity(
                        result, price, high_arr, low_arr,
                        config, cache, fp_range=5, sp_range=5
                    )
                    for k, v in sens_res.items():
                        setattr(result, k, v)
                except Exception:
                    pass

            # Clean up raw arrays to free memory
            result._pnl_array = None
            result._signal_array = None

            # Report progress
            if progress_callback:
                try:
                    pct = 100.0  # validation runs at end
                    progress_callback(
                        pct, self.processed_systems, self.total_systems,
                        0, 0, self.best_result
                    )
                except Exception:
                    pass

    def _run_portfolio_optimization(
        self,
        config: OptimizationConfig
    ) -> None:
        """
        Pos-processamento: constroi portfolio a partir dos top-N sistemas.

        1. Seleciona top-N por score
        2. Filtra por correlacao (diversificacao)
        3. Computa alocacao otima (Markowitz / Risk Parity / Equal)
        4. Combina retornos
        5. Aplica DD management (se habilitado)
        6. Computa metricas do portfolio

        Armazena resultado em self.portfolio_result.
        """
        from core.backtest_engine import periods_per_year_for_timeframe

        self.portfolio_result = None

        top_n = min(config.pf_top_n, len(self.results))
        if top_n < 2:
            return

        top_systems = self.results[:top_n]

        pnl_list = []
        sys_indices = []
        labels = []
        timeframes = []

        for idx, result in enumerate(top_systems):
            pnl = result._pnl_array
            if pnl is None or len(pnl) < 5:
                continue
            pnl_list.append(pnl)
            sys_indices.append(idx)
            labels.append(
                f"{result.fast_indicator}({result.fast_period})"
                f"x{result.slow_indicator}({result.slow_period})"
            )
            timeframes.append(result.timeframe)

        if len(pnl_list) < 2:
            return

        # Filtra por correlacao
        if config.pf_max_correlation < 1.0:
            pnl_list, sys_indices, labels = filter_by_correlation(
                pnl_list, sys_indices, labels, config.pf_max_correlation
            )

        if len(pnl_list) < 2:
            return

        min_len = min(len(arr) for arr in pnl_list)
        if min_len < 5:
            return

        aligned = [arr[:min_len].copy() for arr in pnl_list]
        corr_matrix = compute_correlation_matrix(aligned)

        method = config.pf_allocation_method

        if method in ("markowitz", "risk_parity"):
            stacked = np.column_stack(aligned)
            cov = np.cov(stacked, rowvar=False)
            if method == "markowitz":
                mean_ret = np.mean(stacked, axis=0)
                weights = markowitz_optimization(
                    mean_ret, cov, 0.02
                )
            else:
                weights = risk_parity_optimization(cov)
        else:
            weights = np.ones(len(aligned)) / len(aligned)

        # Rebalanceamento periodico
        if config.pf_rebalance_enabled and config.pf_rebalance_freq_bars > 0:
            freq = config.pf_rebalance_freq_bars
            n_bars = min_len
            n_sys = len(aligned)
            combined_pnl = np.zeros(n_bars)

            for start in range(0, n_bars, freq):
                end = min(start + freq, n_bars)
                window = np.column_stack([a[start:end] for a in aligned])
                if window.shape[0] >= 5:
                    w_mean = np.mean(window, axis=0)
                    w_cov = np.cov(window, rowvar=False)
                    try:
                        if method == "markowitz":
                            w = markowitz_optimization(
                                w_mean, w_cov, 0.02
                            )
                        elif method == "risk_parity":
                            w = risk_parity_optimization(w_cov)
                        else:
                            w = np.ones(n_sys) / n_sys
                    except Exception:
                        w = weights
                else:
                    w = weights

                for i in range(start, end):
                    val = 0.0
                    for j in range(n_sys):
                        val += w[j] * aligned[j][i]
                    combined_pnl[i] = val
        else:
            combined_pnl = combine_portfolio_returns(aligned, weights)

        if len(combined_pnl) == 0:
            return

        combined_equity = compute_portfolio_equity_numba(combined_pnl)

        managed_equity = None
        exposure = None
        if config.pf_dd_management_enabled:
            managed_equity, exposure = apply_drawdown_management_numba(
                combined_equity,
                config.pf_dd_threshold,
                config.pf_dd_reduction,
            )

        from collections import Counter
        tf_counter = Counter(timeframes)
        most_common_tf = tf_counter.most_common(1)[0][0]
        ppy = periods_per_year_for_timeframe(most_common_tf)

        metrics = compute_portfolio_metrics(combined_pnl, combined_equity, ppy)

        if managed_equity is not None:
            managed_pnl = np.zeros(len(combined_pnl))
            for i in range(len(combined_pnl)):
                managed_pnl[i] = (
                    managed_equity[i + 1] / managed_equity[i]
                ) - 1.0
            managed_metrics = compute_portfolio_metrics(
                managed_pnl, managed_equity, ppy
            )
        else:
            managed_metrics = None

        self.portfolio_result = PortfolioResult(
            weights=weights,
            correlation_matrix=corr_matrix,
            system_indices=sys_indices,
            system_labels=labels,
            combined_pnl=combined_pnl,
            combined_equity=combined_equity,
            managed_equity=managed_equity,
            exposure=exposure,
            metrics=metrics,
            allocation_method=method,
        )

        if managed_metrics:
            for k, v in managed_metrics.items():
                mk = k.replace("pf_", "pf_managed_")
                self.portfolio_result.metrics[mk] = v

    def get_timeframe_data(self, tf: int) -> tuple:
        """Retorna arrays de precos para um timeframe (para walk-forward)"""
        price = self.prices.get(tf)
        high = self.highs.get(tf)
        low = self.lows.get(tf)
        return price, high, low

    def test_single_system(self, fast_ind: str, slow_ind: str,
                           fp: int, sp: int, tf: int,
                           config: OptimizationConfig,
                           price_arr: np.ndarray,
                           high_arr: np.ndarray,
                           low_arr: np.ndarray,
                           cache_override) -> Optional[SystemResult]:
        """Testa um sistema especifico em dados fornecidos (para WF OOS)"""
        return self._process_single(
            fast_ind, slow_ind, fp, sp, tf, config,
            price_arr=price_arr, high_arr=high_arr,
            low_arr=low_arr, cache_override=cache_override
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Retorna resultados como DataFrame"""
        if not self.results:
            return pd.DataFrame()

        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)

    def get_ranking(self, metric: str = 'sharpe') -> pd.DataFrame:
        """
        Retorna ranking dos sistemas

        Args:
            metric: Métrica para ordenação ('sharpe', 'pf', 'score', etc)
        """
        df = self.get_results_dataframe()
        if df.empty:
            return df

        # Adiciona coluna de score
        df['score'] = df.apply(
            lambda row: row['sharpe'] * 3 + row['pf'] * 2.5 +
                       row['win'] * 0.25 - row['dd'] * 0.3,
            axis=1
        )

        # Ordena
        if metric == 'score':
            df = df.sort_values('score', ascending=False)
        else:
            df = df.sort_values(metric, ascending=False)

        return df

    def get_aggregated_ranking(self) -> pd.DataFrame:
        """
        Retorna ranking agregado por indicador
        """
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Agrega por indicador FAST
        fast_agg = df.groupby('fast').agg({
            'sharpe': 'mean',
            'pf': 'mean',
            'win': 'mean',
            'dd': 'mean',
            'fast': 'count'
        }).rename(columns={'fast': 'count'})

        fast_agg['score'] = (
            fast_agg['sharpe'] * 3 +
            fast_agg['pf'] * 2.5 +
            fast_agg['win'] * 0.25 -
            fast_agg['dd'] * 0.3
        )

        return fast_agg.sort_values('score', ascending=False)

    def get_heatmap_data(self, metric: str = 'sharpe') -> pd.DataFrame:
        """
        Retorna dados para heatmap de cruzamentos
        """
        df = self.get_results_dataframe()
        if df.empty:
            return pd.DataFrame()

        # Pivot table
        pivot = df.pivot_table(
            values=metric,
            index='fast',
            columns='slow',
            aggfunc='mean'
        )

        return pivot

    def get_stats(self) -> dict:
        """Retorna estatísticas da otimização"""
        return {
            'total_systems': self.total_systems,
            'processed_systems': self.processed_systems,
            'results_found': len(self.results),
            'hit_rate': len(self.results) / self.processed_systems * 100 if self.processed_systems > 0 else 0,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'best_score': self.best_result.score() if self.best_result else 0
        }


# Aliases para compatibilidade
OptimizationEngineV2 = OptimizationEngine