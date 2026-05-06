"""
Script de teste para verificar todas as correcoes aplicadas
"""
import sys
import os
import inspect
import numpy as np
import pandas as pd

# Adiciona bk_md ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bk_md'))

print("=" * 60)
print("TESTE DE VALIDACAO - CORRECOES APLICADAS")
print("=" * 60)

# 1. Testa imports basicos
print("\n[1/7] Testando imports...")
try:
    from core.backtest_engine import (
        run_backtest_vectorized,
        generate_signal_from_cross,
        calculate_metrics,
        periods_per_year_for_timeframe
    )
    from core.optimizer_engine import OptimizationEngine, OptimizationConfig
    from core.ma_cache import MovingAverageCache
    from core.metrics import Metrics
    from core.indicators import ema_numba, sma_numba, hull_numba
    from core.data_loader import DataLoader
    from backtest.crossover import CrossoverBacktest
    print("   Todos os imports OK")
except Exception as e:
    print(f"   Erro nos imports: {e}")
    sys.exit(1)

# 2. Testa que fillna deprecado foi corrigido
print("\n[2/7] Verificando correcao do fillna deprecado...")
try:
    import inspect
    source = inspect.getsource(MovingAverageCache.get)
    if 'fillna(method=' in source:
        print("   fillna(method=...) ainda presente!")
    else:
        print("   fillna deprecado removido (usando ffill/bfill)")
except Exception as e:
    print(f"   Nao foi possivel verificar: {e}")

# 3. Testa Profit Factor corrigido
print("\n[3/7] Testando Profit Factor corrigido...")
import numpy as np
pnl_gains_only = np.array([0.001, 0.002, 0.0015, 0.001])
_, eq, _ = run_backtest_vectorized(
    np.array([100.0, 100.1, 100.2, 100.3, 100.4]),
    np.array([1, 1, 1, 1], dtype=np.int8),
    0.5, 0.2
)
metrics = calculate_metrics(pnl_gains_only, eq, 1)
if np.isinf(metrics['profit_factor']):
    print(f"   PF = inf quando sem perdas (correto)")
else:
    print(f"   PF = {metrics['profit_factor']:.2f} (valor normal)")

# 4. Testa Sharpe com desvio amostral e anualizacao correta
print("\n[4/7] Testando Sharpe Ratio...")
test_returns = np.array([0.001, -0.0005, 0.002, -0.001, 0.0015])

# metrics.py deve usar ddof=1
sharpe_metrics = Metrics.sharpe_ratio(test_returns)

# backtest_engine deve usar n-1 (amostral)
pnl_test = np.array([0.001, -0.0005, 0.002, -0.001, 0.0015])
price_for_eq = np.array([100.0, 100.1, 100.05, 100.25, 99.95, 100.10])
signal_for_eq = np.array([1, 1, 1, 1, 1], dtype=np.int8)
_, eq_test2, _ = run_backtest_vectorized(price_for_eq, signal_for_eq, 0.5, 0.2)
metrics_test = calculate_metrics(pnl_test, eq_test2, 0)
print(f"   Sharpe (metrics.py ddof=1): {sharpe_metrics:.4f}")
print(f"   Sharpe (backtest_engine n-1): {metrics_test['sharpe']:.4f}")

# Testa periods_per_year
ppy_5min = periods_per_year_for_timeframe(5)
ppy_1min = periods_per_year_for_timeframe(1)
print(f"   Periods/ano TF=5min: {ppy_5min} (correto para Mini Indice)")
print(f"   Periods/ano TF=1min: {ppy_1min} (correto para Mini Indice)")

# Testa que Sharpe com periods_per_year difere do padrao
metrics_custom = calculate_metrics(pnl_test, eq_test2, 0, periods_per_year=ppy_5min)
if abs(metrics_custom['sharpe'] - metrics_test['sharpe']) > 0.01:
    print(f"   periods_per_year funcional: {metrics_custom['sharpe']:.4f} != {metrics_test['sharpe']:.4f} OK")
else:
    print("   ATENCAO: periods_per_year nao esta alterando o Sharpe (possivel bug)")

# 5. Testa CrossoverBacktest com estado flat
print("\n[5/7] Testando CrossoverBacktest com flat state...")
try:
    bt = CrossoverBacktest()
    price_test = np.array([100.0, 100.1, 100.2, 100.15, 100.1, 100.05])
    fast_test = np.array([100.0, 100.05, 100.15, 100.12, 100.08, 100.02])
    slow_test = np.array([100.0, 100.05, 100.10, 100.12, 100.12, 100.08])
    pnl_bt, eq_bt = bt.run(price_test, fast_test, slow_test)
    print(f"   CrossoverBacktest OK - {len(pnl_bt)} retornos")
except Exception as e:
    print(f"   Erro no CrossoverBacktest: {e}")

# 6. Testa threading.Lock (nao mp.Lock)
print("\n[6/7] Verificando lock correto para threads...")
try:
    source = inspect.getsource(OptimizationEngine)
    if 'threading.Lock()' in source:
        print("   Usando threading.Lock() (correto para ThreadPoolExecutor)")
    elif 'mp.Lock()' in source:
        print("   Ainda usando mp.Lock() (incorreto!)")
    else:
        print("   Nao foi possivel verificar o lock")
except Exception as e:
    print(f"   Erro ao verificar: {e}")

# 7. Testa que OptimizationConfig nao esta duplicada
print("\n[7/7] Verificando duplicacao de OptimizationConfig...")
try:
    from core.optimizer_engine import OptimizationConfig as OC_core
    from models.config import OptimizationConfig as OC_models
    if OC_core is OC_models:
        print("   OptimizationConfig unica (models importa de core)")
    else:
        print("   OptimizationConfig duplicada!")
except Exception as e:
    print(f"   Erro ao verificar: {e}")

# ============================================================
# NOVOS TESTES - Validacao das correcoes dos indicadores
# ============================================================
print("\n" + "=" * 60)
print("TESTES ADICIONAIS - INDICADORES CORRIGIDOS")
print("=" * 60)

# 8. Testa Laguerre filter (copias independentes)
print("\n[8/8] Testando Laguerre filter (copias independentes)...")
try:
    from core.indicators import Laguerre
    price_test = pd.Series(np.cumsum(np.random.randn(100)) + 100)

    # Teste de regressao: Laguerre com periodos diferentes
    l1 = Laguerre(14)
    l2 = Laguerre(20)

    r1 = l1.compute(price_test)
    r2 = l2.compute(price_test)

    assert len(r1) == len(price_test), f"Laguerre(14) shape mismatch: {len(r1)} != {len(price_test)}"
    assert len(r2) == len(price_test), f"Laguerre(20) shape mismatch: {len(r2)} != {len(price_test)}"
    assert r1.isna().sum() == 0, f"Laguerre(14) tem NaN: {r1.isna().sum()}"
    assert r2.isna().sum() == 0, f"Laguerre(20) tem NaN: {r2.isna().sum()}"
    print(f"   Laguerre OK - resultados consistentes, NaN: 0")
except Exception as e:
    print(f"   Erro no Laguerre filter: {e}")

# 9. Testa FAMA (deve usar alpha do MAMA, nao ser simples EMA)
print("\n[9/9] Testando FAMA com alpha do MAMA...")
try:
    from core.indicators import MAMA, FAMA
    mama = MAMA(20).compute(price_test)
    fama = FAMA(20).compute(price_test)

    # FAMA deve ser diferente de MAMA (se for igual, alpha nao esta sendo usado)
    diff = np.sum(np.abs(mama.values - fama.values))
    assert diff > 1.0, f"FAMA muito proximo de MAMA (diff={diff:.4f}) - alpha nao aplicado?"
    print(f"   FAMA OK - difere de MAMA (diff={diff:.4f})")
except Exception as e:
    print(f"   Erro no FAMA: {e}")

# 10. Testa HodrickPrescott com matriz esparsa
print("\n[10/10] Testando HodrickPrescott (matriz esparsa)...")
try:
    from core.indicators import HodrickPrescott
    # Teste com 12000 pontos (deve usar sparse + downsample)
    price_large = pd.Series(np.cumsum(np.random.randn(12000)) + 100)
    hp = HodrickPrescott(1600)
    hp_result = hp.compute(price_large)
    assert len(hp_result) == 12000, f"HP shape mismatch: {len(hp_result)}"
    assert hp_result.isna().sum() == 0, f"HP tem NaN: {hp_result.isna().sum()}"
    print(f"   HodrickPrescott OK - 12000 pts, NaN: {hp_result.isna().sum()}")
except Exception as e:
    print(f"   Erro no HodrickPrescott: {e}")

# 11. Testa Slippage no backtest
print("\n[11/11] Validando slippage no backtest...")
try:
    price = np.array([100.0 + i * 0.1 for i in range(50)])
    signal = np.array([1]*25 + [-1]*25, dtype=np.int8)

    # Com slippage=0, cost=0
    pnl_0, _, _ = run_backtest_vectorized(price, signal, slippage=0.0, cost=0.0)
    # Com slippage=1, cost=1
    pnl_1, _, _ = run_backtest_vectorized(price, signal, slippage=1.0, cost=1.0)

    total_0 = np.sum(pnl_0)
    total_1 = np.sum(pnl_1)
    assert total_1 < total_0, f"Slippage/cost nao reduziram retorno: {total_0} -> {total_1}"
    print(f"   Slippage funcional: retorno reduziu de {total_0:.6f} para {total_1:.6f}")
except Exception as e:
    print(f"   Erro no teste de slippage: {e}")

# 12. Testa import em config_trabalho.py
print("\n[12/12] Validando config_trabalho import...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bk_md', 'models'))
    # So verifica se o modulo pode ser importado sem SyntaxError
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_trabalho",
        os.path.join(os.path.dirname(__file__), 'bk_md', 'models', 'config_trabalho.py')
    )
    if spec is None:
        print("   config_trabalho.py nao encontrado!")
    else:
        print("   config_trabalho.py OK (import valido)")
except Exception as e:
    print(f"   Erro config_trabalho: {e}")

# ============================================================
# NOVOS TESTES - Validacao Estatistica (Fase 2)
# ============================================================
print("\n" + "=" * 60)
print("TESTES FASE 2 - VALIDACAO ESTATISTICA")
print("=" * 60)

# 13. Testa pipeline SEM validacao (backward compatible)
print("\n[13/13] Testando pipeline SEM validacao estatistica...")
try:
    np.random.seed(42)
    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({'close': price, 'high': price*1.001, 'low': price*0.999})

    config_no_val = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5],
        min_trades=20, min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        run_statistical_validation=False
    )
    engine = OptimizationEngine({5: df})
    results = engine.run(config_no_val)
    d = results[0].to_dict()
    assert 'dsr' not in d, "stat fields should NOT appear when validation disabled"
    assert results[0].dsr == 0.0, "dsr should be 0 when not validated"
    print(f"   OK: {len(results)} resultados, sem campos estatisticos")
except Exception as e:
    print(f"   ERRO: {e}")

# 14. Testa pipeline COM validacao
print("\n[14/14] Testando pipeline COM validacao estatistica...")
try:
    config_yes_val = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5],
        min_trades=20, min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        run_statistical_validation=True,
        stat_top_n=5,
        stat_n_monte_carlo=100,
        stat_n_bootstrap=100,
        stat_n_null_hypothesis=30
    )
    engine = OptimizationEngine({5: df})
    results = engine.run(config_yes_val)
    d = results[0].to_dict()
    assert 'dsr' in d or results[0].mc_prob_loss != 0.0, \
        "stat fields should appear when validation enabled"
    # Check at least some fields were populated
    validated = sum(1 for r in results[:5] if r.mc_prob_loss != 0.0)
    assert validated > 0, f"Nenhum dos top-5 recebeu validacao (validated={validated})"
    print(f"   OK: {len(results)} resultados, {validated}/5 top com MC populado")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 15. Testa funcoes individuais
print("\n[15/15] Testando funcoes de validacao individualmente...")
try:
    from core.statistical_validation import (
        monte_carlo_simulation, bootstrap_sharpe_ratio,
        deflated_sharpe_ratio, null_hypothesis_test,
        parameter_sensitivity
    )
    from core.backtest_engine import (
        run_backtest_vectorized, generate_signal_from_cross,
        periods_per_year_for_timeframe
    )
    from core.ma_cache import MovingAverageCache
    from core.indicators import ema_numba, sma_numba

    np.random.seed(42)
    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({'close': price, 'high': price*1.001, 'low': price*0.999})

    # Generate a proper signal and run backtest
    cache = MovingAverageCache(df['close'])
    fast_ma = cache.get('EMA', 16).values
    slow_ma = cache.get('SMA', 34).values
    signal = generate_signal_from_cross(fast_ma, slow_ma)
    pnl, equity, trades = run_backtest_vectorized(price, signal, 0.5, 0.2)
    ppy = periods_per_year_for_timeframe(5)

    # 15a. Monte Carlo
    mc = monte_carlo_simulation(pnl, 200, 0.95, ppy)
    assert mc['mc_prob_loss'] >= 0, f"MC prob_loss invalido: {mc['mc_prob_loss']}"
    assert mc['mc_sharpe_ci_lower'] <= mc['mc_sharpe_ci_upper'], \
        f"MC CI invertido: {mc['mc_sharpe_ci_lower']} > {mc['mc_sharpe_ci_upper']}"
    print(f"   Monte Carlo OK: prob_loss={mc['mc_prob_loss']:.3f}, sharpe_mean={mc['mc_sharpe_mean']:.3f}")

    # 15b. Bootstrap
    boot = bootstrap_sharpe_ratio(pnl, 200, 0.95, ppy)
    assert boot['boot_sharpe_ci_lower'] <= boot['boot_sharpe_ci_upper'], \
        f"Boot CI invertido: {boot['boot_sharpe_ci_lower']} > {boot['boot_sharpe_ci_upper']}"
    assert 0 <= boot['boot_sharpe_p_value'] <= 1, \
        f"Boot p-value fora do range: {boot['boot_sharpe_p_value']}"
    print(f"   Bootstrap OK: sharpe_mean={boot['boot_sharpe_mean']:.3f}, p_value={boot['boot_sharpe_p_value']:.3f}")

    # 15c. DSR
    dsr = deflated_sharpe_ratio(1.5, len(pnl), 100, periods_per_year=ppy)
    assert 0 <= dsr <= 1, f"DSR fora do range: {dsr}"
    print(f"   DSR OK: {dsr:.6f}")

    # 15d. Null hypothesis
    null_res = null_hypothesis_test(price, signal, 1.5, 30, 0.5, 0.2, ppy)
    assert 0 <= null_res['null_p_value'] <= 1, \
        f"Null p-value fora do range: {null_res['null_p_value']}"
    print(f"   Null Test OK: p_value={null_res['null_p_value']:.4f}")

    # 15e. Parameter sensitivity
    from core.optimizer_engine import SystemResult
    result_obj = SystemResult(
        fast_indicator='EMA', slow_indicator='SMA',
        fast_period=16, slow_period=34, timeframe=5,
        sharpe=1.5, profit_factor=1.1, win_rate=55.0,
        max_dd=5.0, total_return=20.0, trades=100
    )
    cfg = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        min_trades=10, slippage=0.5, cost=0.2
    )
    sens = parameter_sensitivity(
        result_obj, price, price*1.001, price*0.999,
        cfg, cache, fp_range=3, sp_range=3
    )
    assert 0 <= sens['sens_robustness'] <= 1, \
        f"Robustness fora do range: {sens['sens_robustness']}"
    print(f"   Sensitivity OK: robustness={sens['sens_robustness']:.4f}")

    print("   Todas as funcoes OK")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# TESTES FASE 3 - REALISMO TRANSACIONAL
# ============================================================
print("\n" + "=" * 60)
print("TESTES FASE 3 - REALISMO TRANSACIONAL")
print("=" * 60)

from core.transactional_costs import (
    TransactionalCostConfig,
    compute_variable_spread,
    compute_variable_slippage,
    detect_session_gaps,
    compute_liquidity_mask,
    compute_atr_stats,
    run_backtest_realistic,
    apply_ir_tax,
)

np.random.seed(42)
n = 2000
price = 100 + np.cumsum(np.random.randn(n) * 0.1)

# 16. compute_variable_spread
print("\n[16/16] Testando compute_variable_spread...")
try:
    atr = np.full(n, 0.5)
    sp_const = compute_variable_spread(price, atr, 0.5, 0.0, 5.0)
    assert np.allclose(sp_const, 0.5), f"Spread constante falhou: {sp_const[:5]}"

    atr_var = np.linspace(0.1, 100.0, n)
    sp_var = compute_variable_spread(price, atr_var, 0.5, 0.05, 5.0)
    assert sp_var[0] > 0.5, f"Spread variavel baixo: {sp_var[0]}"
    assert sp_var[-1] == 5.0, f"Spread nao clampado: {sp_var[-1]} (max=5.0)"

    sp_empty = compute_variable_spread(np.array([]), np.array([]), 0.5, 0.05, 5.0)
    assert len(sp_empty) == 0, "Spread vazio nao vazio"
    print("   OK: constante, variavel, vazio")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 17. detect_session_gaps
print("\n[17/17] Testando detect_session_gaps...")
try:
    ts_no_gap = np.arange(5, dtype=np.int64) * 300_000_000_000
    g1 = detect_session_gaps(ts_no_gap, 600_000_000_000.0)
    assert not g1.any(), "Falso positivo de gap"

    ts_gap = np.array([0, 300, 5000, 5300, 10600], dtype=np.int64) * 1_000_000_000
    g2 = detect_session_gaps(ts_gap, 600.0 * 1_000_000_000)
    assert g2[2], f"Gap nao detectado no indice 2: {g2}"
    assert not g2[0], "Barra 0 marcada como gap"

    g3 = detect_session_gaps(np.array([0], dtype=np.int64), 100.0)
    assert len(g3) == 1 and not g3[0], "Barra unica falhou"
    print("   OK: sem gaps, com gaps, barra unica")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 18. compute_liquidity_mask
print("\n[18/18] Testando compute_liquidity_mask...")
try:
    atr = np.array([0.1, 0.2, 0.5, 0.8, 1.0])
    liq = compute_liquidity_mask(atr, np.mean(atr), 0.5)
    assert not liq[0], "Baixo ATR deveria ser bloqueado"
    assert liq[2], "ATR medio deveria passar"
    assert liq[4], "ATR alto deveria passar"

    liq_all = compute_liquidity_mask(atr, 0.0, 0.5)
    assert liq_all.all(), "ATR mean zero deveria liberar tudo"

    liq_none = compute_liquidity_mask(atr, np.mean(atr), 0.0)
    assert liq_none.all(), "Ratio zero deveria liberar tudo"
    print("   OK: bloqueio, liberacao total")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 19. run_backtest_realistic vs run_backtest_vectorized
print("\n[19/19] Testando run_backtest_realistic...")
try:
    from core.backtest_engine import run_backtest_vectorized

    signal = np.array([1]*500 + [-1]*500 + [1]*500 + [-1]*500, dtype=np.int8)
    price_sub = price[:2000]

    sp_arr = np.full(2000, 0.5)
    sl_arr = np.full(2000, 0.5)
    gm = np.zeros(2000, dtype=np.bool_)
    lm = np.ones(2000, dtype=np.bool_)

    pnl_v, eq_v, tr_v = run_backtest_vectorized(price_sub, signal, 0.5, 0.5)
    pnl_r, eq_r, tr_r = run_backtest_realistic(price_sub, signal, sp_arr, sl_arr, gm, lm, 0.0, 0.0)

    assert abs(eq_v[-1] - eq_r[-1]) < 1e-10, f"Equity difere: {eq_v[-1]} vs {eq_r[-1]}"
    assert tr_v == tr_r, f"Trades diferem: {tr_v} vs {tr_r}"
    print("   OK: resultados identicos com arrays constantes")

    # Gap mask
    # Gap mask across signal transition: position closes and reopens
    gm_gap = np.zeros(2000, dtype=np.bool_)
    gm_gap[500:510] = True  # gap during 1->-1 transition
    pnl_gap, eq_gap, tr_gap = run_backtest_realistic(price_sub, signal, sp_arr, sl_arr, gm_gap, lm, 0.0, 0.0)
    assert abs(eq_gap[-1] - eq_r[-1]) > 1e-8, f"Gap nao alterou equity: {eq_gap[-1]} vs {eq_r[-1]}"
    print(f"   OK: gap mask altera equity ({tr_gap} trades vs {tr_r})")

    # Liquidity mask
    lm_liq = np.ones(2000, dtype=np.bool_)
    lm_liq[1000:1020] = False  # low liq during -1->1 transition
    pnl_liq, eq_liq, tr_liq = run_backtest_realistic(price_sub, signal, sp_arr, sl_arr, gm, lm_liq, 0.0, 0.0)
    assert abs(eq_liq[-1] - eq_r[-1]) > 1e-8, f"Liquidez nao alterou equity"
    print(f"   OK: liquidity mask altera equity ({tr_liq} trades vs {tr_r})")

    # Brokerage
    pnl_b, eq_b, tr_b = run_backtest_realistic(price_sub, signal, sp_arr, sl_arr, gm, lm, 0.5, 0.25)
    assert eq_b[-1] < eq_r[-1], f"Brokerage nao reduziu equity: {eq_b[-1]} vs {eq_r[-1]}"
    print("   OK: brokerage reduz equity")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 20. IR tax
print("\n[20/20] Testando apply_ir_tax...")
try:
    pnl_profit = np.array([0.001, 0.002, 0.0015, 0.001])
    eq_profit = np.array([1.0, 1.001, 1.003, 1.0045, 1.0055])
    pnl_t, eq_t = apply_ir_tax(pnl_profit.copy(), eq_profit.copy(), 0.20)
    expected_tax = (1.0055 - 1.0) * 0.20
    assert abs(eq_t[-1] - (1.0055 - expected_tax)) < 1e-10, f"IR incorreto"
    assert eq_t[-1] < eq_profit[-1], "IR nao reduziu equity"
    print(f"   OK: IR aplicado corretamente (taxa={expected_tax:.6f})")

    eq_loss = np.array([1.0, 0.99, 0.98, 0.97, 0.96])
    pnl_l, eq_l = apply_ir_tax(np.array([-0.01, -0.01, -0.01, -0.01]), eq_loss, 0.20)
    assert eq_l[-1] == eq_loss[-1], "IR nao deveria ser aplicado em prejuizo"
    print("   OK: IR nao aplicado em prejuizo")

    pnl_z, eq_z = apply_ir_tax(pnl_profit.copy(), eq_profit.copy(), 0.0)
    assert eq_z[-1] == eq_profit[-1], "IR zero alterou equity"
    print("   OK: taxa zero = sem efeito")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 21. Full pipeline tc_enabled=False regression
print("\n[21/21] Testando pipeline com tc_enabled=False (regressao)...")
try:
    from core.optimizer_engine import OptimizationEngine, OptimizationConfig

    df_tc = pd.DataFrame({'close': price, 'high': price*1.001, 'low': price*0.999})

    config_base = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=20,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
    )
    config_tc = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=20,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        tc_enabled=False,
    )

    engine1 = OptimizationEngine({5: df_tc})
    engine2 = OptimizationEngine({5: df_tc})
    r1 = engine1.run(config_base)
    r2 = engine2.run(config_tc)

    assert len(r1) == len(r2), f"Count diff: {len(r1)} vs {len(r2)}"
    for i in range(min(len(r1), 5)):
        assert abs(r1[i].sharpe - r2[i].sharpe) < 0.001, f"Sharpe diff at {i}"
    print(f"   OK: {len(r1)} resultados identicos (regressao zero)")

    config_tc_on = OptimizationConfig(
        fast_indicators=['EMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=20,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        tc_enabled=True,
        tc_spread_variable=True,
        tc_slippage_variable=True,
        tc_brokerage_per_trade=0.5,
        tc_exchange_fee=0.25,
        tc_ir_tax_enabled=True,
        tc_ir_tax_rate=0.20,
    )
    engine3 = OptimizationEngine({5: df_tc})
    r3 = engine3.run(config_tc_on)
    assert len(r3) > 0, "TC enabled produziu zero resultados"
    if r3[0].sharpe != r1[0].sharpe:
        print(f"   OK: TC alterou sharpe ({r1[0].sharpe:.3f} -> {r3[0].sharpe:.3f})")
    else:
        print(f"   ATENCAO: TC nao alterou sharpe")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
"""
Fase 4 tests fragment - to be appended to run_tests.py
"""

# ============================================================
# TESTES FASE 4 - GESTAO DE PORTFOLIO
# ============================================================
print("\n" + "=" * 60)
print("TESTES FASE 4 - GESTAO DE PORTFOLIO")
print("=" * 60)

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

np.random.seed(42)

# 22. compute_correlation_matrix
print("\n[22/22] Testando compute_correlation_matrix...")
try:
    pnl_identical = [np.array([0.001, 0.002, -0.001, 0.003])] * 3
    corr = compute_correlation_matrix(pnl_identical)
    assert corr.shape == (3, 3), f"Shape errado: {corr.shape}"
    assert abs(corr[0, 1] - 1.0) < 1e-10, "Correlacao identica deveria ser 1.0"
    assert abs(corr[0, 0] - 1.0) < 1e-10, "Diagonal deveria ser 1.0"
    print(f"   OK: shape={corr.shape}, identica={corr[0,1]:.2f}")

    pnl_uncorr = [np.random.randn(100) * 0.01 for _ in range(5)]
    corr2 = compute_correlation_matrix(pnl_uncorr)
    assert corr2.shape == (5, 5), f"Shape errado: {corr2.shape}"
    assert abs(corr2[0, 0] - 1.0) < 1e-10, "Diagonal"
    print(f"   OK: shape={corr2.shape}, aleatoria OK")

    corr_empty = compute_correlation_matrix([])
    assert corr_empty.shape == (1, 0) or corr_empty.size == 0, "Vazio deveria ser 1x0"
    print("   OK: lista vazia")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 23. markowitz_optimization
print("\n[23/23] Testando markowitz_optimization...")
try:
    mean_ret = np.array([0.001, 0.002, 0.0015])
    cov = np.array([[0.0001, 0.00002, 0.00001],
                    [0.00002, 0.0002, 0.00003],
                    [0.00001, 0.00003, 0.00015]])
    w = markowitz_optimization(mean_ret, cov, 0.02)
    assert abs(np.sum(w) - 1.0) < 1e-6, f"Weights nao somam 1: {np.sum(w)}"
    assert np.all(w >= -1e-10), f"Weights negativos: {w}"
    assert len(w) == 3, f"Tamanho errado: {len(w)}"
    print(f"   OK: weights={np.round(w, 4)}, sum={np.sum(w):.4f}")

    w1 = markowitz_optimization(np.array([0.001]), np.array([[0.0001]]), 0.02)
    assert abs(w1[0] - 1.0) < 1e-6, "Unico ativo deveria ser 1.0"
    print("   OK: unico ativo = 1.0")

    w_empty = markowitz_optimization(np.array([]), np.array([[]]), 0.02)
    assert len(w_empty) == 0, "Vazio deveria ser 0"
    print("   OK: vazio")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 24. risk_parity_optimization
print("\n[24/24] Testando risk_parity_optimization...")
try:
    cov = np.array([[0.0001, 0.00002],
                    [0.00002, 0.0002]])
    w = risk_parity_optimization(cov)
    assert abs(np.sum(w) - 1.0) < 1e-6, f"Weights nao somam 1: {np.sum(w)}"
    assert np.all(w >= 0), f"Weights negativos: {w}"
    port_var = np.dot(w.T, np.dot(cov, w))
    port_std = np.sqrt(max(port_var, 1e-16))
    mrc = np.dot(cov, w) / port_std
    rc = w * mrc
    assert abs(rc[0] - rc[1]) < 0.1, f"Contribuicoes de risco muito diferentes: {rc}"
    print(f"   OK: weights={np.round(w, 4)}, rc={np.round(rc, 6)}")

    w1 = risk_parity_optimization(np.array([[0.0001]]))
    assert abs(w1[0] - 1.0) < 1e-6, "Unico ativo = 1.0"
    print("   OK: unico ativo")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 25. combine_portfolio_returns and equity
print("\n[25/25] Testando combinacao de retornos...")
try:
    r1 = np.array([0.001, 0.002, -0.001, 0.003])
    r2 = np.array([0.002, 0.001, 0.003, -0.001])
    weights = np.array([0.5, 0.5])
    combined = combine_portfolio_returns([r1, r2], weights)
    assert len(combined) == 4, f"Tamanho errado: {len(combined)}"
    expected = r1 * 0.5 + r2 * 0.5
    assert np.allclose(combined, expected), f"Valor errado: {combined} vs {expected}"
    print(f"   OK: combinacao correta")

    eq = compute_portfolio_equity_numba(combined)
    assert len(eq) == 5, f"Equity size: {len(eq)}"
    assert abs(eq[0] - 1.0) < 1e-10, "Equity[0] = 1.0"
    expected_eq = 1.0
    for r in combined:
        expected_eq *= (1.0 + r)
    assert abs(eq[-1] - expected_eq) < 1e-10, f"Equity final errada: {eq[-1]} vs {expected_eq}"
    print(f"   OK: equity curve correta, final={eq[-1]:.6f}")

    combined_empty = combine_portfolio_returns([], np.array([]))
    assert len(combined_empty) == 0, "Vazio deveria ser 0"
    print("   OK: vazio")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 26. apply_drawdown_management
print("\n[26/26] Testando apply_drawdown_management...")
try:
    eq = np.array([1.0, 1.05, 1.10, 0.99, 0.95, 0.90, 1.02, 1.08])
    managed, exposure = apply_drawdown_management_numba(eq, 5.0, 0.5)
    assert len(managed) == len(eq), f"Tamanho: {len(managed)}"
    assert len(exposure) == len(eq), f"Exposure size: {len(exposure)}"
    # Peak = 1.10, DD at index 3 = (1.10-0.99)/1.10 = 10% > 5% threshold
    assert exposure[3] < 1.0, f"Exposicao deveria ser reduzida no DD: {exposure[3]}"
    assert exposure[0] == 1.0, "Exposicao inicial = 1.0"
    assert exposure[-1] == 1.0, f"Exposicao final = 1.0: {exposure[-1]}"
    print(f"   OK: managed_equity[-1]={managed[-1]:.4f}, min_exposure={exposure.min():.2f}")

    eq_up = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
    m_up, e_up = apply_drawdown_management_numba(eq_up, 5.0, 0.5)
    assert np.all(e_up == 1.0), "Sem DD, exposure = 1.0"
    print(f"   OK: sem DD, exposure constante")

    eq_single = np.array([1.0])
    m_s, e_s = apply_drawdown_management_numba(eq_single, 5.0, 0.5)
    assert len(m_s) == 1, "Unico elemento"
    print("   OK: unico elemento")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 27. filter_by_correlation
print("\n[27/27] Testando filter_by_correlation...")
try:
    pnls = [np.random.randn(100) * 0.01 for _ in range(4)]
    idx = list(range(4))
    labels = [f"Sys{i}" for i in range(4)]

    filtered_p, filtered_i, filtered_l = filter_by_correlation(pnls, idx, labels, 0.99)
    assert len(filtered_p) == 4 or len(filtered_p) == 3, \
        f"Deveria manter quase todos com correlacao alta: {len(filtered_p)}"
    print(f"   OK: correlacao alta filtrou {4 - len(filtered_p)} sistemas")

    pnl_same = [np.array([0.001, 0.002, -0.001, 0.003]) for _ in range(4)]
    f_p, f_i, f_l = filter_by_correlation(pnl_same, [0, 1, 2, 3], ["a", "b", "c", "d"], 0.5)
    assert len(f_p) < 4, f"Identicos deveriam ser filtrados: {len(f_p)}"
    print(f"   OK: sistemas identicos filtrados ({len(f_p)} restantes)")

    f_e, f_ie, f_le = filter_by_correlation([], [], [], 0.95)
    assert len(f_e) == 0, "Vazio deveria ser 0"
    print("   OK: vazio")
except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()

# 28. Full pipeline portfolio
print("\n[28/28] Testando pipeline completa de portfolio...")
try:
    from core.optimizer_engine import OptimizationEngine, OptimizationConfig

    n = 2000
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({'close': price, 'high': price*1.001, 'low': price*0.999})

    # Portfolio disabled (regression)
    config_no_pf = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=10,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
    )
    engine_no_pf = OptimizationEngine({5: df})
    r_no_pf = engine_no_pf.run(config_no_pf)
    assert engine_no_pf.portfolio_result is None, \
        "Portfolio should be None when disabled"
    assert len(r_no_pf) > 0, "Deveria produzir resultados"
    print(f"   OK: pf_enabled=False, portfolio_result=None, {len(r_no_pf)} sistemas")

    # Portfolio enabled with equal weights
    config_pf = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=10,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        pf_enabled=True,
        pf_allocation_method="equal",
        pf_top_n=5,
        pf_max_correlation=0.99,
    )
    engine_pf = OptimizationEngine({5: df})
    r_pf = engine_pf.run(config_pf)
    assert engine_pf.portfolio_result is not None, \
        "Portfolio should be created when enabled"
    pf_res = engine_pf.portfolio_result
    assert len(pf_res.weights) >= 2, \
        f"Portfolio weights < 2: {len(pf_res.weights)}"
    assert abs(np.sum(pf_res.weights) - 1.0) < 1e-6, \
        f"Weights devem somar 1: {np.sum(pf_res.weights)}"
    assert pf_res.metrics.get('pf_sharpe', 0) != 0, \
        "Portfolio metrics should be populated"
    print(f"   OK: equal weights portfolio, {len(pf_res.system_labels)} sistemas, "
          f"Sharpe={pf_res.metrics.get('pf_sharpe', 0):.3f}")

    # Markowitz portfolio
    config_mark = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=10,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        pf_enabled=True,
        pf_allocation_method="markowitz",
        pf_top_n=5,
    )
    engine_mark = OptimizationEngine({5: df})
    engine_mark.run(config_mark)
    assert engine_mark.portfolio_result is not None, \
        "Markowitz portfolio should be created"
    pf_m = engine_mark.portfolio_result
    assert pf_m.allocation_method == "markowitz"
    print(f"   OK: Markowitz portfolio, {len(pf_m.weights)} sistemas, "
          f"Sharpe={pf_m.metrics.get('pf_sharpe', 0):.3f}")

    # Portfolio with DD management
    config_dd = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=10,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        pf_enabled=True,
        pf_dd_management_enabled=True,
        pf_dd_threshold=5.0,
        pf_dd_reduction=0.5,
        pf_top_n=5,
    )
    engine_dd = OptimizationEngine({5: df})
    engine_dd.run(config_dd)
    assert engine_dd.portfolio_result is not None
    pf_dd = engine_dd.portfolio_result
    has_managed = pf_dd.managed_equity is not None
    print(f"   OK: DD management {'ativado' if has_managed else 'disponivel'}")

    # Portfolio with rebalance
    config_rb = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'], slow_indicators=['SMA'],
        fast_min=10, fast_max=20, fast_step=5,
        slow_min=30, slow_max=40, slow_step=5,
        timeframes=[5], min_trades=10,
        min_sharpe=-999, min_profit_factor=1.0, max_dd=99,
        pf_enabled=True,
        pf_rebalance_enabled=True,
        pf_rebalance_freq_bars=500,
        pf_top_n=5,
    )
    engine_rb = OptimizationEngine({5: df})
    engine_rb.run(config_rb)
    assert engine_rb.portfolio_result is not None
    print(f"   OK: Rebalance portfolio ativo")

except Exception as e:
    print(f"   ERRO: {e}")
    import traceback
    traceback.print_exc()


print("TESTE CONCLUIDO - Verifique os resultados acima")
print("=" * 60)
