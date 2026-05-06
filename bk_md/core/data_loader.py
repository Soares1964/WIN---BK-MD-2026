# ma_lab/core/data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import os


class DataLoader:
    """Carregador e validador de dados para múltiplos timeframes"""
    
    def __init__(self):
        self.original_df: Optional[pd.DataFrame] = None
        self.timeframe_dfs: Dict[int, pd.DataFrame] = {}
        self.metadata: Dict = {}
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Carrega arquivo CSV e prepara dados
        
        Args:
            path: Caminho do arquivo CSV
        
        Returns:
            DataFrame carregado
        """
        # Tenta diferentes codificações
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Se nenhuma codificação funcionar, tenta com engine python
            df = pd.read_csv(path, encoding='latin1')
        
        # Normaliza nomes das colunas
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Verifica colunas necessárias
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                # Tenta encontrar coluna similar
                found = False
                for c in df.columns:
                    if 'close' in c or 'fech' in c:
                        df.rename(columns={c: 'close'}, inplace=True)
                        found = True
                        break
                
                if not found:
                    raise ValueError(f"Coluna '{col}' não encontrada no CSV")
        
        # Converte preços para numérico
        for col in required:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove linhas com NaN
        df = df.dropna(subset=['close'])
        
        # Converte data se existir
        date_cols = [c for c in df.columns if 'date' in c or 'time' in c or 'data' in c]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.dropna(subset=[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
        
        # Ordena por data
        df = df.sort_index()
        
        self.original_df = df
        self.metadata = {
            'path': path,
            'rows': len(df),
            'start': df.index[0] if isinstance(df.index, pd.DatetimeIndex) else None,
            'end': df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None,
            'columns': list(df.columns)
        }
        
        # Gera timeframes
        self._generate_timeframes()
        
        return df
    
    def _generate_timeframes(self):
        """Gera dataframes para múltiplos timeframes"""
        if self.original_df is None:
            return
        
        if not isinstance(self.original_df.index, pd.DatetimeIndex):
            # Se não tem índice temporal, cria um artificial
            self.original_df.index = pd.date_range(
                start='2025-01-01',
                periods=len(self.original_df),
                freq='1min'
            )
        
        # Gera para timeframes de 1 a 10 minutos
        for tf in range(1, 11):
            rule = f'{tf}min'
            resampled = self.original_df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            self.timeframe_dfs[tf] = resampled
    
    def validate(self) -> Tuple[bool, str]:
        """
        Valida a qualidade dos dados
        
        Returns:
            (is_valid, message)
        """
        if self.original_df is None:
            return False, "Nenhum dado carregado"
        
        df = self.original_df
        
        # Verifica quantidade mínima de dados
        if len(df) < 1000:
            return False, f"Poucos dados: {len(df)} linhas (mínimo 1000)"
        
        # Verifica valores negativos
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and (df[col] <= 0).any():
                return False, f"Valores <= 0 encontrados em {col}"
        
        # Verifica gaps temporais
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            expected = pd.Timedelta(minutes=1)
            gaps = time_diffs[time_diffs > expected * 1.5]
            
            if len(gaps) > len(df) * 0.01:  # Mais de 1% de gaps
                return False, f"Muitos gaps temporais: {len(gaps)} encontrados"
        
        # Verifica outliers
        for col in ['close']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[abs(df[col] - mean) > 5 * std]
                
                if len(outliers) > len(df) * 0.001:  # Mais de 0.1% outliers
                    return False, f"Muitos outliers em {col}"
        
        return True, "Dados validados com sucesso"
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas dos dados"""
        if self.original_df is None:
            return {}
        
        stats = {
            'arquivo': self.metadata.get('path', ''),
            'linhas': len(self.original_df),
            'inicio': str(self.metadata.get('start', '')),
            'fim': str(self.metadata.get('end', '')),
            'timeframes_disponiveis': list(self.timeframe_dfs.keys()),
            'colunas': self.metadata.get('columns', [])
        }
        
        # Adiciona tamanho de cada timeframe
        for tf, df in self.timeframe_dfs.items():
            stats[f'tf_{tf}_linhas'] = len(df)
        
        return stats
    
    def get_timeframe(self, tf: int) -> Optional[pd.DataFrame]:
        """Retorna dados para um timeframe específico"""
        return self.timeframe_dfs.get(tf)