# ma_lab/core/ma_cache.py
import pandas as pd
from typing import Dict, Optional, Tuple
import hashlib
import numpy as np
from core.indicators import INDICATORS, Indicator


class MovingAverageCache:
    """Cache inteligente para médias móveis com sistema de memória"""
    
    def __init__(self, price: pd.Series, max_cache: int = 1000):
        self.price = price
        self.cache: Dict[str, pd.Series] = {}
        self.hits = 0
        self.misses = 0
        self.max_cache = max_cache
        self.access_count: Dict[str, int] = {}
    
    def _make_key(self, name: str, period: int) -> str:
        """Gera chave única para o cache"""
        return f"{name}_{period}"
    
    def _clean_cache(self):
        """Remove itens menos acessados se cache estiver cheio"""
        if len(self.cache) >= self.max_cache:
            # Ordena por número de acessos
            sorted_items = sorted(
                self.access_count.items(),
                key=lambda x: x[1]
            )
            # Remove 20% dos menos acessados
            to_remove = int(self.max_cache * 0.2)
            for key, _ in sorted_items[:to_remove]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
    
    def get(self, name: str, period: int) -> pd.Series:
        """Obtém média do cache ou calcula"""
        key = self._make_key(name, period)
        
        if key in self.cache:
            self.hits += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        self.misses += 1
        
        if name not in INDICATORS:
            raise ValueError(f"Indicador {name} não encontrado")
        
        indicator_class = INDICATORS[name]
        indicator = indicator_class(period)
        ma = indicator.compute(self.price)
        
        # Pré-processamento - preenche NaNs iniciais com forward fill depois back fill
        ma = ma.ffill().bfill()
        
        self.cache[key] = ma
        self.access_count[key] = 1
        
        self._clean_cache()
        
        return ma
    
    def get_many(self, items: list) -> dict:
        """Obtém múltiplas médias de uma vez"""
        results = {}
        for name, period in items:
            results[f"{name}_{period}"] = self.get(name, period)
        return results
    
    def clear(self):
        """Limpa o cache"""
        self.cache.clear()
        self.access_count.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do cache"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache': self.max_cache
        }