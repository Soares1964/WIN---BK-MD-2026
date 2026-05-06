# ma_lab/models/config.py
"""
Configurações do sistema de backtest - Versão Oficial
Baseada nos testes reais com o Mini Índice (WIN)

NOTE: OptimizationConfig é definida em core.optimizer_engine.
Este arquivo usa import para evitar duplicação.
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Importa a versão canônica do core
from core.optimizer_engine import OptimizationConfig


class ConfigManager:
    """Gerenciador de configurações - Versão Aprimorada"""
    
    @staticmethod
    def save_config(config: Dict[str, Any], path: str):
        """Salva configuração em arquivo JSON"""
        config['saved_at'] = datetime.now().isoformat()
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuração salva em: {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Carrega configuração de arquivo JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Configuração carregada de: {path}")
        return config
    
    @staticmethod
    def get_config_rapido() -> Dict[str, Any]:
        """Configuração rápida para testes exploratórios"""
        return {
            'fast_indicators': ['EMA', 'HMA'],
            'slow_indicators': ['SMA'],
            'fast_min': 10,
            'fast_max': 30,
            'fast_step': 2,
            'slow_min': 25,
            'slow_max': 35,
            'slow_step': 1,
            'timeframes': [5],
            'min_trades': 50,
            'min_sharpe': -999,
            'min_profit_factor': 1.05,
            'max_dd': 15.0,
            'slippage': 0.5,
            'cost': 0.2,
            'batch_size': 2000,
            'n_jobs': 8,
            'use_cache': True,
            'description': "Configuração rápida para testes"
        }
    
    @staticmethod
    def get_config_completo() -> Dict[str, Any]:
        """Configuração completa para otimização abrangente"""
        return {
            'fast_indicators': ['SMA', 'EMA', 'HMA', 'WMA', 'KAMA', 'FRAMA'],
            'slow_indicators': ['SMA', 'EMA', 'HMA', 'WMA'],
            'fast_min': 5,
            'fast_max': 50,
            'fast_step': 2,
            'slow_min': 20,
            'slow_max': 100,
            'slow_step': 3,
            'timeframes': [1, 3, 5, 10],
            'min_trades': 50,
            'min_sharpe': -999,
            'min_profit_factor': 1.05,
            'max_dd': 20.0,
            'slippage': 0.5,
            'cost': 0.2,
            'batch_size': 10000,
            'n_jobs': 8,
            'use_cache': True,
            'description': "Configuração completa para varredura ampla"
        }
    
    @staticmethod
    def get_config_validacao() -> Dict[str, Any]:
        """Configuração para validar sistemas específicos (step=1)"""
        return {
            'fast_indicators': ['EMA', 'HMA'],
            'slow_indicators': ['SMA'],
            'fast_min': 15,
            'fast_max': 30,
            'fast_step': 1,
            'slow_min': 25,
            'slow_max': 35,
            'slow_step': 1,
            'timeframes': [5],
            'min_trades': 50,
            'min_sharpe': -999,
            'min_profit_factor': 1.10,
            'max_dd': 10.0,
            'slippage': 0.5,
            'cost': 0.2,
            'batch_size': 2000,
            'n_jobs': 8,
            'use_cache': True,
            'debug': True,
            'description': "Configuração de validação com step=1"
        }
    
    @staticmethod
    def get_sistemas_recomendados() -> List[Dict[str, Any]]:
        """Retorna lista dos melhores sistemas encontrados nos testes"""
        return [
            {
                'nome': 'CAMPEÃO',
                'fast': 'EMA',
                'fast_period': 26,
                'slow': 'SMA',
                'slow_period': 29,
                'timeframe': 5,
                'retorno': 30.5,
                'pf': 1.13,
                'dd': 6.0,
                'win_rate': 50.4,
                'trades': 229
            },
            {
                'nome': 'SEGURO',
                'fast': 'HMA',
                'fast_period': 10,
                'slow': 'SMA',
                'slow_period': 29,
                'timeframe': 5,
                'retorno': 27.5,
                'pf': 1.12,
                'dd': 3.8,
                'win_rate': 50.1,
                'trades': 296
            },
            {
                'nome': 'CONSISTENTE',
                'fast': 'EMA',
                'fast_period': 18,
                'slow': 'SMA',
                'slow_period': 25,
                'timeframe': 5,
                'retorno': 28.9,
                'pf': 1.13,
                'dd': 6.6,
                'win_rate': 50.3,
                'trades': 244
            },
            {
                'nome': 'CRESCIMENTO',
                'fast': 'EMA',
                'fast_period': 28,
                'slow': 'SMA',
                'slow_period': 31,
                'timeframe': 5,
                'retorno': 29.3,
                'pf': 1.13,
                'dd': 7.5,
                'win_rate': 50.1,
                'trades': 219
            }
        ]


# Instância global para fácil acesso
default_config = OptimizationConfig(
    fast_indicators=['EMA', 'HMA'],
    slow_indicators=['SMA']
)


# =============================================================
# EXEMPLO DE USO
# =============================================================
if __name__ == "__main__":
    print("="*70)
    print("📋 CONFIGURAÇÕES DISPONÍVEIS")
    print("="*70)
    
    # Mostra configuração padrão
    print("\n📊 Configuração Padrão (Otimizada para WIN):")
    print(f"   FAST: {default_config.fast_indicators} ({default_config.fast_min}-{default_config.fast_max})")
    print(f"   SLOW: {default_config.slow_indicators} ({default_config.slow_min}-{default_config.slow_max})")
    print(f"   TIMEFRAMES: {default_config.timeframes}")
    print(f"   MIN_PF: {default_config.min_profit_factor}")
    print(f"   MAX_DD: {default_config.max_dd}%")
    
    # Mostra sistemas recomendados
    print("\n🏆 Sistemas Recomendados:")
    for s in ConfigManager.get_sistemas_recomendados():
        print(f"   • {s['nome']}: {s['fast']}({s['fast_period']}) x {s['slow']}({s['slow_period']}) "
              f"- Ret: {s['retorno']}% PF: {s['pf']} DD: {s['dd']}%")
    
    print("\n" + "="*70)
    print("✅ CONFIGURAÇÕES CARREGADAS COM SUCESSO")
    print("="*70)