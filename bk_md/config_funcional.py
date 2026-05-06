#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONFIGURAÇÃO QUE FUNCIONOU NOS TESTES
======================================
Esta configuração reproduz exatamente os parâmetros que geraram
os resultados positivos nos testes de validação.
"""

from core.optimizer_engine import OptimizationConfig

def get_config_funcional():
    """
    Retorna configuração que funcionou nos testes
    Sem filtros rigorosos, apenas parâmetros que geraram resultados
    """
    
    config = OptimizationConfig(
        # =========================================================
        # INDICADORES - OS QUE APARECERAM NOS TESTES
        # =========================================================
        fast_indicators=['SMA', 'EMA', 'HMA'],  # HMA apareceu como melhor
        slow_indicators=['SMA', 'EMA'],         # SMA dominou como lenta
        
        # =========================================================
        # PERÍODOS - RANGE QUE PRODUZIU OS MELHORES RESULTADOS
        # =========================================================
        # FAST: entre 10 e 30 (onde estavam EMA18, EMA24, EMA26, EMA28, HMA10)
        fast_min=10,
        fast_max=30,
        fast_step=2,      # Step 2 para teste rápido (depois pode refinar)
        
        # SLOW: entre 25 e 50 (onde estavam SMA25, SMA27, SMA29, SMA31)
        slow_min=25,
        slow_max=50,
        slow_step=2,
        
        # =========================================================
        # TIMEFRAMES - 5 MIN FOI O MELHOR
        # =========================================================
        timeframes=[5],   # Foco no timeframe que deu certo
        
        # =========================================================
        # FILTROS RELAXADOS (como nos testes)
        # =========================================================
        min_trades=30,              # Baixo para não eliminar sistemas
        min_sharpe=-999,            # Ignora Sharpe (sabemos que está negativo)
        min_profit_factor=1.05,      # Apenas > 1.0 (lucro mínimo)
        max_dd=20.0,                # Aceita drawdown alto
        
        # =========================================================
        # CUSTOS REALISTAS (como nos testes)
        # =========================================================
        slippage=0.5,                # 0.5 ponto de slippage
        cost=0.2,                    # 0.2 ponto de custo
        
        # =========================================================
        # PERFORMANCE
        # =========================================================
        batch_size=5000,              # Processa 5000 combinações por vez
        n_jobs=8,                     # Usa 8 threads
        use_cache=True,                # Usa cache de médias
        debug=False                    # Sem debug para produção
    )
    
    return config


def get_config_rapido():
    """
    Versão ainda mais rápida para testes exploratórios
    """
    config = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'],     # Só os melhores
        slow_indicators=['SMA'],            # Só SMA
        fast_min=15,
        fast_max=30,
        fast_step=2,
        slow_min=25,
        slow_max=35,
        slow_step=1,                        # Step 1 para precisão
        timeframes=[5],
        min_trades=20,
        min_sharpe=-999,
        min_profit_factor=1.03,
        max_dd=25.0,
        slippage=0.5,
        cost=0.2,
        batch_size=2000,
        n_jobs=8,
        use_cache=True,
        debug=False
    )
    return config


def get_config_completo():
    """
    Versão completa com todos os indicadores e timeframes
    (mais lenta, mas mais abrangente)
    """
    config = OptimizationConfig(
        fast_indicators=['SMA', 'EMA', 'HMA', 'WMA', 'KAMA', 'FRAMA'],
        slow_indicators=['SMA', 'EMA', 'HMA', 'WMA'],
        fast_min=5,
        fast_max=50,
        fast_step=2,
        slow_min=20,
        slow_max=100,
        slow_step=3,
        timeframes=[1, 3, 5, 10],
        min_trades=30,
        min_sharpe=-999,
        min_profit_factor=1.05,
        max_dd=20.0,
        slippage=0.5,
        cost=0.2,
        batch_size=10000,
        n_jobs=8,
        use_cache=True,
        debug=False
    )
    return config


def get_config_validacao():
    """
    Configuração para validar os melhores sistemas encontrados
    (usa step=1 para encontrar a combinação exata)
    """
    config = OptimizationConfig(
        fast_indicators=['EMA', 'HMA'],
        slow_indicators=['SMA'],
        fast_min=15,
        fast_max=30,
        fast_step=1,           # Step 1 para encontrar o ponto exato
        slow_min=25,
        slow_max=35,
        slow_step=1,           # Step 1 para encontrar o ponto exato
        timeframes=[5],
        min_trades=30,
        min_sharpe=-999,
        min_profit_factor=1.1,
        max_dd=10.0,
        slippage=0.5,
        cost=0.2,
        batch_size=2000,
        n_jobs=8,
        use_cache=True,
        debug=True              # Debug para ver progresso
    )
    return config


# =============================================================
# EXEMPLO DE USO
# =============================================================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from core.data_loader import DataLoader
    from core.optimizer_engine import OptimizationEngine
    
    print("="*70)
    print("🚀 TESTE COM CONFIGURAÇÃO FUNCIONAL")
    print("="*70)
    
    # Carrega dados
    loader = DataLoader()
    import glob
    csv_files = glob.glob("exportacoes_mt5/*.csv")
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        sys.exit(1)
    
    print(f"📁 Arquivo: {os.path.basename(csv_files[0])}")
    df = loader.load_csv(csv_files[0])
    
    # Pega configuração
    config = get_config_funcional()
    
    # Mostra configuração
    print("\n📊 CONFIGURAÇÃO:")
    print(f"   FAST: {config.fast_indicators} ({config.fast_min}-{config.fast_max} step={config.fast_step})")
    print(f"   SLOW: {config.slow_indicators} ({config.slow_min}-{config.slow_max} step={config.slow_step})")
    print(f"   TIMEFRAMES: {config.timeframes}")
    print(f"   MIN_TRADES: {config.min_trades}")
    print(f"   MIN_PF: {config.min_profit_factor}")
    print(f"   MAX_DD: {config.max_dd}%")
    
    # Estima total de combinações
    engine = OptimizationEngine(loader.timeframe_dfs)
    total = engine.estimate_total(config)
    print(f"\n📈 Total estimado: {total} combinações")
    
    # Pergunta se quer executar
    resposta = input("\n🔧 Executar otimização? (s/N): ")
    if resposta.lower() == 's':
        print("\n⏳ Executando...")
        
        def progress_cb(progress, processed, total, rate, remaining, best):
            if processed % 500 == 0:
                print(f"   Progresso: {progress:.1f}% ({processed}/{total}) | {rate:.0f} sys/s")
        
        results = engine.run(config, progress_callback=progress_cb)
        
        print(f"\n✅ Encontrados {len(results)} sistemas com PF > {config.min_profit_factor}")
        
        if results:
            print("\n🏆 TOP 10 por Profit Factor:")
            print("-" * 70)
            print(f"{'#':<3} {'FAST':<12} {'SLOW':<12} {'PF':<8} {'Retorno':<10} {'DD':<8} {'Trades':<8}")
            print("-" * 70)
            
            for i, r in enumerate(sorted(results, key=lambda x: x.profit_factor, reverse=True)[:10]):
                print(f"{i+1:<3} {r.fast_indicator}({r.fast_period})  {r.slow_indicator}({r.slow_period})  "
                      f"{r.profit_factor:>6.2f}  {r.total_return:>7.1f}%  {r.max_dd:>6.1f}%  {r.trades:>6.0f}")
    
    print("\n" + "="*70)
    print("✅ CONFIGURAÇÃO FUNCIONAL PRONTA PARA USO")
    print("="*70)