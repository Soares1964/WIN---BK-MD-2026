# ma_lab/utils/dataset_validator.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """
    Valida a qualidade do dataset
    
    Args:
        df: DataFrame com dados OHLCV
    
    Returns:
        (is_valid, message, stats)
    """
    stats = {
        'rows': len(df),
        'columns': list(df.columns),
        'missing': {},
        'outliers': {},
        'gaps': 0
    }
    
    # Verifica colunas necessárias
    required = ['close']
    if 'close' not in df.columns and 'Close' in df.columns:
        df = df.rename(columns={'Close': 'close'})
    
    if 'close' not in df.columns:
        return False, "Coluna 'close' não encontrada", stats
    
    # Verifica quantidade mínima
    if len(df) < 100:
        return False, f"Poucos dados: {len(df)} linhas", stats
    
    # Verifica valores nulos
    null_count = df['close'].isnull().sum()
    stats['missing']['close'] = null_count
    
    if null_count > len(df) * 0.01:  # Mais de 1% nulos
        return False, f"Muitos valores nulos em close: {null_count}", stats
    
    # Verifica valores negativos
    neg_count = (df['close'] <= 0).sum()
    if neg_count > 0:
        return False, f"Valores <= 0 em close: {neg_count}", stats
    
    # Verifica outliers (5 sigmas)
    mean = df['close'].mean()
    std = df['close'].std()
    outliers = df[abs(df['close'] - mean) > 5 * std]
    stats['outliers']['close'] = len(outliers)
    
    if len(outliers) > len(df) * 0.001:
        return False, f"Muitos outliers em close: {len(outliers)}", stats
    
    # Verifica gaps temporais se índice for datetime
    if isinstance(df.index, pd.DatetimeIndex):
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            if len(time_diffs) > 1:
                median_diff = time_diffs.median()
                gaps = time_diffs[time_diffs > median_diff * 1.5]
                stats['gaps'] = len(gaps)
                
                if len(gaps) > len(df) * 0.01:  # Mais de 1% gaps
                    return False, f"Muitos gaps temporais: {len(gaps)}", stats
    
    # Verifica variação de preço
    returns = df['close'].pct_change().dropna()
    if returns.std() == 0:
        return False, "Série sem variação (std=0)", stats
    
    # Estatísticas adicionais
    stats['mean'] = df['close'].mean()
    stats['std'] = df['close'].std()
    stats['min'] = df['close'].min()
    stats['max'] = df['close'].max()
    stats['returns_mean'] = returns.mean()
    stats['returns_std'] = returns.std()
    
    return True, "Dataset validado com sucesso", stats


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Retorna métricas de qualidade dos dados
    
    Args:
        df: DataFrame com dados OHLCV
    
    Returns:
        Dicionário com métricas de qualidade
    """
    quality = {}
    
    if 'close' in df.columns:
        close = df['close']
        
        # Estatísticas básicas
        quality['rows'] = len(df)
        quality['missing'] = close.isnull().sum()
        quality['missing_pct'] = quality['missing'] / len(df) * 100
        
        # Distribuição
        quality['mean'] = close.mean()
        quality['std'] = close.std()
        quality['min'] = close.min()
        quality['max'] = close.max()
        
        # Retornos
        returns = close.pct_change().dropna()
        quality['returns_mean'] = returns.mean()
        quality['returns_std'] = returns.std()
        quality['returns_skew'] = returns.skew()
        quality['returns_kurt'] = returns.kurtosis()
        
        # Autocorrelação
        if len(returns) > 1:
            quality['autocorr_1'] = returns.autocorr(1)
            quality['autocorr_5'] = returns.autocorr(5)
        else:
            quality['autocorr_1'] = 0
            quality['autocorr_5'] = 0
    
    # Verifica gaps
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        time_diffs = df.index.to_series().diff()
        quality['time_diff_mean'] = time_diffs.mean().total_seconds() / 60
        quality['time_diff_std'] = time_diffs.std().total_seconds() / 60
        
        expected = pd.Timedelta(minutes=1)
        gaps = time_diffs[time_diffs > expected * 1.5]
        quality['gaps'] = len(gaps)
        quality['gaps_pct'] = len(gaps) / len(df) * 100
    else:
        quality['gaps'] = 0
        quality['gaps_pct'] = 0
    
    return quality


def prepare_timeframe(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Prepara dados para um timeframe específico
    
    Args:
        df: DataFrame com dados originais
        tf_minutes: Timeframe em minutos
    
    Returns:
        DataFrame resampleado
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        # Se não tem índice temporal, retorna cópia
        return df.copy()
    
    rule = f'{tf_minutes}min'
    
    # Define agregações
    agg_dict = {}
    
    if 'open' in df.columns:
        agg_dict['open'] = 'first'
    if 'high' in df.columns:
        agg_dict['high'] = 'max'
    if 'low' in df.columns:
        agg_dict['low'] = 'min'
    if 'close' in df.columns:
        agg_dict['close'] = 'last'
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'
    
    if not agg_dict:
        # Se não tem colunas padrão, usa close
        agg_dict = {'close': 'last'}
    
    # Resample
    resampled = df.resample(rule).agg(agg_dict)
    
    # Remove NaN
    resampled = resampled.dropna()
    
    return resampled