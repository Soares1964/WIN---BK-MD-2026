# ma_lab/core/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class Metrics:
    """Cálculo de métricas de performance"""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """Calcula Sharpe Ratio anualizado

        Args:
            returns: Array de retornos por período
            risk_free: Taxa livre de risco anual (ex: 0.02 = 2%)
            periods_per_year: Períodos por ano para anualização
                              (252 daily, 19656 para 5-min, etc.)
        """
        if len(returns) == 0:
            return 0.0

        std = returns.std(ddof=1)  # Desvio amostral (ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0

        ppy = max(periods_per_year, 1)
        excess_returns = returns - risk_free / ppy
        return np.sqrt(ppy) * excess_returns.mean() / std
    
    @staticmethod
    def profit_factor(pnl: np.ndarray) -> float:
        """Calcula Profit Factor"""
        gains = np.sum(pnl[pnl > 0])
        losses = np.abs(np.sum(pnl[pnl < 0]))
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
        
        return gains / losses
    
    @staticmethod
    def win_rate(pnl: np.ndarray) -> float:
        """Calcula Win Rate percentual"""
        if len(pnl) == 0:
            return 0.0
        
        wins = np.sum(pnl > 0)
        return wins / len(pnl) * 100
    
    @staticmethod
    def max_drawdown(equity: np.ndarray) -> float:
        """Calcula Maximum Drawdown percentual"""
        if len(equity) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return np.abs(np.min(dd)) * 100
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, equity: np.ndarray) -> float:
        """Calcula Calmar Ratio"""
        total_return = (equity[-1] - 1) * 100 if len(equity) > 0 else 0
        max_dd = Metrics.max_drawdown(equity)
        
        if max_dd == 0:
            return 0.0
        
        return total_return / max_dd
    
    @staticmethod
    def avg_trade(pnl: np.ndarray) -> float:
        """Calcula retorno médio por trade"""
        if len(pnl) == 0:
            return 0.0
        
        return np.mean(pnl) * 100
    
    @staticmethod
    def trade_frequency(pnl: np.ndarray, periods_per_year: int = 252) -> float:
        """Calcula frequência de trades"""
        trades = np.sum(pnl != 0)
        if trades == 0:
            return 0.0
        
        return trades / len(pnl) * periods_per_year
    
    @staticmethod
    def ulcer_index(equity: np.ndarray) -> float:
        """Calcula Ulcer Index"""
        if len(equity) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity)
        dd_pct = (equity - peak) / peak * 100
        squared_dd = dd_pct ** 2
        
        return np.sqrt(np.mean(squared_dd))
    
    @staticmethod
    def pain_index(equity: np.ndarray) -> float:
        """Calcula Pain Index"""
        if len(equity) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        
        return np.mean(np.abs(dd))
    
    @staticmethod
    def recovery_factor(total_return: float, max_dd: float) -> float:
        """Calcula Recovery Factor"""
        if max_dd == 0:
            return 0.0
        
        return total_return / max_dd
    
    @staticmethod
    def risk_of_ruin(pnl: np.ndarray, capital: float = 10000) -> float:
        """Calcula Risco de Ruína simplificado"""
        if len(pnl) == 0:
            return 1.0
        
        avg_loss = np.mean(pnl[pnl < 0]) if np.sum(pnl < 0) > 0 else 0
        if avg_loss >= 0:
            return 0.0
        
        trades = len(pnl)
        win_rate = np.sum(pnl > 0) / trades
        
        # Fórmula simplificada
        k = abs(avg_loss) / capital
        ror = ((1 - win_rate) / win_rate) ** (1/k) if win_rate > 0 and win_rate < 1 else 1.0
        
        return min(ror, 1.0)
    
    @staticmethod
    def all_metrics(pnl: np.ndarray, equity: np.ndarray) -> Dict[str, float]:
        """Retorna todas as métricas em um dicionário"""
        return {
            'sharpe': Metrics.sharpe_ratio(pnl),
            'profit_factor': Metrics.profit_factor(pnl),
            'win_rate': Metrics.win_rate(pnl),
            'max_dd': Metrics.max_drawdown(equity),
            'calmar': Metrics.calmar_ratio(pnl, equity),
            'avg_trade': Metrics.avg_trade(pnl),
            'ulcer': Metrics.ulcer_index(equity),
            'pain': Metrics.pain_index(equity),
            'trades': np.sum(pnl != 0)
        }
    
    @staticmethod
    def filter_systems(df: pd.DataFrame,
                       min_sharpe: float = 1.2,
                       min_pf: float = 1.5,
                       max_dd: float = 8.0,
                       min_trades: int = 100) -> pd.DataFrame:
        """Aplica filtros de qualidade a um DataFrame de resultados"""
        filtered = df.copy()
        
        if 'sharpe' in filtered.columns:
            filtered = filtered[filtered['sharpe'] >= min_sharpe]
        
        if 'pf' in filtered.columns or 'profit_factor' in filtered.columns:
            pf_col = 'pf' if 'pf' in filtered.columns else 'profit_factor'
            filtered = filtered[filtered[pf_col] >= min_pf]
        
        if 'dd' in filtered.columns or 'max_dd' in filtered.columns:
            dd_col = 'dd' if 'dd' in filtered.columns else 'max_dd'
            filtered = filtered[filtered[dd_col] <= max_dd]
        
        if 'trades' in filtered.columns:
            filtered = filtered[filtered['trades'] >= min_trades]
        
        return filtered