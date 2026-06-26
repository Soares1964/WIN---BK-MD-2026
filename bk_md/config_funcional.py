#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONFIGURAÇÃO PROFISSIONAL PARA PRODUÇÃO
========================================
Esta configuração implementa as melhores práticas para backtest
e otimização de sistemas de trading com validação estatística.

Para 32GB RAM: Usa até 75% dos cores, cache inteligente e batches otimizados.
"""

from core.optimizer_engine import OptimizationConfig


def get_config_producao_conservadora():
    """
    Configuração CONSERVADORA para produção real
    - Validação estatística rigorosa
    - Custos realistas elevados
    - Walk-forward obrigatório
    - Filtros de qualidade altos
    """
    
    config = OptimizationConfig(
        # =========================================================
        # INDICADORES - Apenas os mais robustos
        # =========================================================
        fast_indicators=['EMA', 'HMA', 'KAMA'],  # Redução de lag + adaptativas
        slow_indicators=['SMA', 'EMA'],           # Clássicas como referência
        
        # =========================================================
        # PERÍODOS - Range equilibrado
        # =========================================================
        fast_min=8,
        fast_max=40,
        fast_step=2,
        
        slow_min=30,
        slow_max=100,
        slow_step=5,
        
        # =========================================================
        # TIMEFRAMES - Múltiplos para robustez
        # =========================================================
        timeframes=[5, 10, 15],  # 5min (principal), 10min, 15min
        
        # =========================================================
        # FILTROS RIGOROSOS (padrão profissional)
        # =========================================================
        min_trades=150,              # Significância estatística
        min_sharpe=1.5,              # Sharpe anualizado mínimo
        min_profit_factor=1.8,       # PF mínimo conservador
        max_dd=6.0,                  # Drawdown máximo aceitável
        
        # =========================================================
        # CUSTOS REALISTAS (conservadores)
        # =========================================================
        slippage=1.5,                # Slippage elevado para segurança
        cost=0.6,                    # Custo total realista
        
        # =========================================================
        # CUSTOS TRANSACIONAIS AVANÇADOS
        # =========================================================
        tc_enabled=True,
        tc_spread_variable=True,     # Spread varia com volatilidade
        tc_spread_base_pts=0.5,
        tc_spread_vol_mult=0.05,
        tc_spread_max_pts=5.0,
        
        tc_slippage_variable=True,   # Slippage varia com volatilidade
        tc_slippage_base_pts=0.8,
        tc_slippage_vol_mult=0.10,
        tc_slippage_max_pts=10.0,
        
        tc_brokerage_per_trade=0.0,  # Corretagem zero (day trade)
        tc_exchange_fee=0.0,         # Taxas B3 embutidas no cost
        
        tc_ir_tax_enabled=False,     # IR calculado separadamente
        tc_ir_tax_rate=0.20,         # 20% para day trade
        
        tc_liquidity_enabled=True,   # Filtra períodos de baixa liquidez
        tc_liquidity_min_atr_ratio=0.3,
        
        tc_gap_enabled=True,         # Evita trades em gaps
        tc_gap_max_minutes=30,
        
        # =========================================================
        # RISK MANAGEMENT
        # =========================================================
        sl_type=2,                   # Stop por ATR múltiplo
        sl_value=2.0,                # 2x ATR de stop
        tp_type=0,                   # Sem TP fixo (deixa correr)
        tp_value=0.0,
        atr_period=14,
        
        # Position sizing conservador
        sizing_method=1,             # Fixed fractional
        risk_per_trade=0.01,         # 1% do capital por trade
        max_alloc_pct=0.20,          # Máximo 20% por sistema
        
        # =========================================================
        # VALIDAÇÃO ESTATÍSTICA COMPLETA
        # =========================================================
        run_statistical_validation=True,
        stat_top_n=10,
        stat_n_monte_carlo=2000,     # 2000 cenários para precisão
        stat_n_bootstrap=2000,
        stat_n_null_hypothesis=300,
        stat_mc_confidence=0.95,
        stat_bootstrap_confidence=0.95,
        
        # =========================================================
        # WALK-FORWARD OPTIMIZATION
        # =========================================================
        use_walkforward=True,
        wf_n_windows=4,              # 4 janelas para validação cruzada
        wf_train_pct=0.50,           # 50% treino
        wf_test_pct=0.25,            # 25% teste
        wf_cv_mode="rolling",        # Janelas deslizantes
        wf_top_n=5,                  # Top 5 de cada janela
        
        # =========================================================
        # PORTFOLIO MANAGEMENT
        # =========================================================
        pf_enabled=True,
        pf_allocation_method="risk_parity",  # Equaliza risco
        pf_rebalance_enabled=False,          # Rebalanceamento desliga
        pf_rebalance_freq_bars=1000,
        pf_dd_management_enabled=True,       # Proteção por drawdown
        pf_dd_threshold=10.0,                # Ativa em 10% DD
        pf_dd_reduction=0.5,                 # Reduz 50% exposição
        pf_top_n=10,                         # Top 10 sistemas
        pf_max_correlation=0.70,             # Diversificação real
        
        # =========================================================
        # PERFORMANCE OTIMIZADA (32GB RAM)
        # =========================================================
        batch_size=3000,           # Balanceia memória/throughput
        n_jobs=-1,                 # Todos os cores disponíveis
        use_cache=True,
        debug=False
    )
    
    return config


def get_config_pesquisa_rapida():
    """
    Configuração RÁPIDA para screening inicial
    - Sem validação estatística (fase exploratória)
    - Custos reduzidos para não eliminar candidatos
    - Filtros relaxados
    """
    
    config = OptimizationConfig(
        # Indicadores diversificados para pesquisa
        fast_indicators=['SMA', 'EMA', 'HMA', 'WMA', 'KAMA', 'FRAMA', 'DEMA', 'TEMA'],
        slow_indicators=['SMA', 'EMA', 'HMA', 'WMA', 'KAMA'],
        
        # Range amplo para descoberta
        fast_min=5,
        fast_max=50,
        fast_step=3,
        
        slow_min=20,
        slow_max=150,
        slow_step=10,
        
        # Múltiplos timeframes
        timeframes=[1, 3, 5, 10, 15],
        
        # Filtros mínimos
        min_trades=50,
        min_sharpe=0.5,
        min_profit_factor=1.2,
        max_dd=12.0,
        
        # Custos padrão
        slippage=1.0,
        cost=0.5,
        
        # Sem custos avançados (rápido)
        tc_enabled=False,
        
        # Sem validação estatística (rápido)
        run_statistical_validation=False,
        
        # Sem walk-forward (rápido)
        use_walkforward=False,
        
        # Portfolio simples
        pf_enabled=False,
        
        # Performance máxima
        batch_size=5000,
        n_jobs=-1,
        use_cache=True,
        debug=False
    )
    
    return config


def get_config_validacao_final():
    """
    Configuração para VALIDAÇÃO FINAL dos melhores sistemas
    - Validação estatística completa
    - Walk-forward rigoroso
    - Custos realistas máximos
    """
    
    config = OptimizationConfig(
        # Foco nos melhores indicadores já identificados
        fast_indicators=['EMA', 'HMA'],
        slow_indicators=['SMA'],
        
        # Range estreito ao redor dos ótimos conhecidos
        fast_min=15,
        fast_max=35,
        fast_step=1,               # Step 1 para precisão máxima
        
        slow_min=25,
        slow_max=45,
        slow_step=1,               # Step 1 para precisão máxima
        
        # Apenas timeframe principal
        timeframes=[5],
        
        # Filtros intermediários (já sabe que funciona)
        min_trades=100,
        min_sharpe=1.0,
        min_profit_factor=1.5,
        max_dd=8.0,
        
        # Custos realistas
        slippage=1.5,
        cost=0.6,
        
        # Custos avançados ativados
        tc_enabled=True,
        tc_spread_variable=True,
        tc_slippage_variable=True,
        tc_gap_enabled=True,
        tc_liquidity_enabled=True,
        
        # Validação estatística máxima
        run_statistical_validation=True,
        stat_top_n=20,             # Valida mais sistemas
        stat_n_monte_carlo=3000,   # Mais cenários
        stat_n_bootstrap=3000,
        stat_n_null_hypothesis=500,
        
        # Walk-forward completo
        use_walkforward=True,
        wf_n_windows=5,            # 5 janelas para robustez
        wf_train_pct=0.50,
        wf_test_pct=0.25,
        wf_top_n=10,
        
        # Portfolio analysis
        pf_enabled=True,
        pf_allocation_method="markowitz",
        pf_top_n=15,
        pf_max_correlation=0.65,
        
        # Performance
        batch_size=2000,
        n_jobs=-1,
        use_cache=True,
        debug=True                 # Debug para monitoramento
    )
    
    return config


def get_config_exemplo_teste():
    """
    Configuração de EXEMPLO para testes rápidos
    - Reproduz resultados dos testes iniciais
    - Ideal para validar instalação
    """
    
    config = OptimizationConfig(
        fast_indicators=['SMA', 'EMA', 'HMA'],
        slow_indicators=['SMA', 'EMA'],
        
        fast_min=10,
        fast_max=30,
        fast_step=2,
        
        slow_min=25,
        slow_max=50,
        slow_step=2,
        
        timeframes=[5],
        
        # Filtros relaxados como nos testes
        min_trades=30,
        min_sharpe=-999,           # Ignora Sharpe inicialmente
        min_profit_factor=1.05,
        max_dd=20.0,
        
        slippage=0.5,
        cost=0.2,
        
        # Sem features avançadas (rápido)
        tc_enabled=False,
        run_statistical_validation=False,
        use_walkforward=False,
        pf_enabled=False,
        
        batch_size=5000,
        n_jobs=8,
        use_cache=True,
        debug=False
    )
    
    return config


# =============================================================
# GUIA DE SELEÇÃO DE CONFIGURAÇÃO
# =============================================================
"""
QUAL CONFIGURAÇÃO USAR?

1. PRIMEIRO USO / TESTE DE INSTALAÇÃO:
   → get_config_exemplo_teste()
   - Rápido, valida que o sistema funciona
   
2. PESQUISA INICIAL / SCREENING:
   → get_config_pesquisa_rapida()
   - Explora amplamente o espaço de parâmetros
   - Identifica regiões promissoras
   
3. OTIMIZAÇÃO DETALHADA:
   → Use resultados da pesquisa para definir ranges estreitos
   - Execute com step=1 nos ranges identificados
   
4. VALIDAÇÃO FINAL:
   → get_config_validacao_final()
   - Validação estatística completa
   - Walk-forward rigoroso
   
5. PRODUÇÃO REAL:
   → get_config_producao_conservadora()
   - Máxima segurança e realismo
   - Pronto para trading com capital real

CHECKLIST PRÉ-PRODUÇÃO:
□ mc_prob_loss < 0.30
□ boot_sharpe_p_value < 0.05
□ dsr > 0.7
□ null_p_value < 0.05
□ sens_robustness > 0.6
□ avg_sharpe_OOS > 1.0
□ stability_score < 0.5
□ Correlação média < 0.7
□ Drawdown máximo < 8%
"""


# =============================================================
# EXEMPLO DE USO
# =============================================================
if __name__ == "__main__":
    print("="*70)
    print("🎯 CONFIGURAÇÕES PROFISSIONAIS DISPONÍVEIS")
    print("="*70)
    
    configs = {
        "Produção Conservadora": get_config_producao_conservadora,
        "Pesquisa Rápida": get_config_pesquisa_rapida,
        "Validação Final": get_config_validacao_final,
        "Exemplo Teste": get_config_exemplo_teste,
    }
    
    for nome, func in configs.items():
        cfg = func()
        print(f"\n{nome}:")
        print(f"  - Indicadores FAST: {cfg.fast_indicators}")
        print(f"  - Indicadores SLOW: {cfg.slow_indicators}")
        print(f"  - Timeframes: {cfg.timeframes}")
        print(f"  - Min Trades: {cfg.min_trades}")
        print(f"  - Min Sharpe: {cfg.min_sharpe}")
        print(f"  - Min PF: {cfg.min_profit_factor}")
        print(f"  - Max DD: {cfg.max_dd}%")
        print(f"  - Validação Estatística: {'✅' if cfg.run_statistical_validation else '❌'}")
        print(f"  - Walk-Forward: {'✅' if cfg.use_walkforward else '❌'}")
        print(f"  - Portfolio: {'✅' if cfg.pf_enabled else '❌'}")
        print(f"  - Custos Realistas: {'✅' if cfg.tc_enabled else '❌'}")
    
    print("\n" + "="*70)
    print("📖 Consulte MELHORIAS_PROFISSIONAIS.md para detalhes completos")
    print("="*70)