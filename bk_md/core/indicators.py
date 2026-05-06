# ma_lab/core/indicators.py
import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Type, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# NUMBA CORE FUNCTIONS
# ============================================================

@njit
def ema_numba(price: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average com Numba"""
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros(len(price))
    ema[0] = price[0]
    for i in range(1, len(price)):
        ema[i] = alpha * price[i] + (1 - alpha) * ema[i - 1]
    return ema

@njit
def wma_numba(price: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average com Numba"""
    weights = np.arange(1, period + 1)
    weight_sum = weights.sum()
    result = np.zeros(len(price))
    
    for i in range(period - 1, len(price)):
        window = price[i - period + 1:i + 1]
        result[i] = np.sum(window * weights) / weight_sum
    
    return result

@njit
def sma_numba(price: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average com Numba"""
    result = np.zeros(len(price))
    cumsum = np.cumsum(price)
    
    for i in range(period - 1, len(price)):
        if i == period - 1:
            result[i] = cumsum[i] / period
        else:
            result[i] = (cumsum[i] - cumsum[i - period]) / period
    
    return result

@njit
def hull_numba(price: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average com Numba"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA de período/2
    wma_half = np.zeros(len(price))
    weights_half = np.arange(1, half_period + 1)
    weight_sum_half = weights_half.sum()
    
    for i in range(half_period - 1, len(price)):
        window = price[i - half_period + 1:i + 1]
        wma_half[i] = np.sum(window * weights_half) / weight_sum_half
    
    # WMA de período completo
    wma_full = np.zeros(len(price))
    weights_full = np.arange(1, period + 1)
    weight_sum_full = weights_full.sum()
    
    for i in range(period - 1, len(price)):
        window = price[i - period + 1:i + 1]
        wma_full[i] = np.sum(window * weights_full) / weight_sum_full
    
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    
    # WMA do raw HMA
    result = np.zeros(len(price))
    for i in range(sqrt_period - 1, len(price)):
        window = raw_hma[i - sqrt_period + 1:i + 1]
        weights = np.arange(1, sqrt_period + 1)
        result[i] = np.sum(window * weights) / weights.sum()
    
    return result

# ============================================================
# BASE CLASS
# ============================================================

class Indicator:
    """Classe base para todos os indicadores"""
    name = "base"
    category = "Outros"
    description = ""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def compute(self, price: pd.Series) -> pd.Series:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.name}({self.period})"


# ============================================================
# CLÁSSICAS
# ============================================================

class SMA(Indicator):
    name = "SMA"
    category = "Clássicas"
    description = "Simple Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return price.rolling(self.period).mean()


class EMA(Indicator):
    name = "EMA"
    category = "Clássicas"
    description = "Exponential Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return pd.Series(
            ema_numba(price.values, self.period),
            index=price.index
        )


class WMA(Indicator):
    name = "WMA"
    category = "Clássicas"
    description = "Weighted Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return pd.Series(
            wma_numba(price.values, self.period),
            index=price.index
        )


class VWMA(Indicator):
    name = "VWMA"
    category = "Clássicas"
    description = "Volume Weighted Moving Average"

    def compute(self, price: pd.Series) -> pd.Series:
        # Aproxima volume via range price (high-low) quando não disponível
        # Isso dá mais peso a barras com maior volatilidade/atividade
        try:
            if hasattr(price, 'name') and price.name and price.name in price.index:
                pass
        except Exception:
            pass

        # Constrói volume proxy a partir da variação absoluta do preço
        volume_proxy = price.diff().abs().fillna(1.0) + 1e-10
        pv = price * volume_proxy
        return pv.rolling(self.period).sum() / volume_proxy.rolling(self.period).sum()


class SMMA(Indicator):
    name = "SMMA"
    category = "Clássicas"
    description = "Smoothed Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return price.ewm(alpha=1/self.period, adjust=False).mean()


class RMA(Indicator):
    name = "RMA"
    category = "Clássicas"
    description = "Running Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 1/self.period
        return price.ewm(alpha=alpha, adjust=False).mean()


class TMA(Indicator):
    name = "TMA"
    category = "Clássicas"
    description = "Triangular Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return price.rolling(self.period).mean().rolling(self.period).mean()


class VWAP(Indicator):
    name = "VWAP"
    category = "Clássicas"
    description = "Volume Weighted Average Price (intraday)"

    def compute(self, price: pd.Series) -> pd.Series:
        # VWAP = Σ(Price × Volume) / Σ(Volume)
        # Proxy de volume via variacao absoluta do preco
        proxy = price.diff().abs().fillna(1e-10) + 1e-10
        return (price * proxy).rolling(self.period).sum() / proxy.rolling(self.period).sum()


# ============================================================
# REDUÇÃO DE LAG
# ============================================================

class DEMA(Indicator):
    name = "DEMA"
    category = "Redução de Lag"
    description = "Double Exponential Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        ema1 = price.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        return 2 * ema1 - ema2


class TEMA(Indicator):
    name = "TEMA"
    category = "Redução de Lag"
    description = "Triple Exponential Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        ema1 = price.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3


class ZLEMA(Indicator):
    name = "ZLEMA"
    category = "Redução de Lag"
    description = "Zero Lag Exponential Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        lag = (self.period - 1) // 2
        price_adj = price + (price - price.shift(lag))
        return price_adj.ewm(span=self.period, adjust=False).mean()


class HMA(Indicator):
    name = "HMA"
    category = "Redução de Lag"
    description = "Hull Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return pd.Series(
            hull_numba(price.values, self.period),
            index=price.index
        )


class T3(Indicator):
    name = "T3"
    category = "Redução de Lag"
    description = "T3 Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        v = 0.7
        ema1 = price.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        ema4 = ema3.ewm(span=self.period, adjust=False).mean()
        ema5 = ema4.ewm(span=self.period, adjust=False).mean()
        ema6 = ema5.ewm(span=self.period, adjust=False).mean()
        
        c1 = -v**3
        c2 = 3*v**2 + 3*v**3
        c3 = -6*v**2 - 3*v - 3*v**3
        c4 = 1 + 3*v + v**3 + 3*v**2
        
        return c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3


class ALMA(Indicator):
    name = "ALMA"
    category = "Redução de Lag"
    description = "Arnaud Legoux Moving Average"

    def __init__(self, period: int = 20, offset: float = 0.85, sigma: float = 6.0):
        super().__init__(period)
        self.offset = offset
        self.sigma = sigma

    def compute(self, price: pd.Series) -> pd.Series:
        window = self.period
        m = self.offset * (window - 1)
        s = window / self.sigma
        
        weights = np.exp(-((np.arange(window) - m) ** 2) / (2 * s * s))
        weights /= weights.sum()
        
        return price.rolling(window).apply(
            lambda x: np.dot(x, weights), raw=True
        )


class Laguerre(Indicator):
    name = "Laguerre"
    category = "Redução de Lag"
    description = "Laguerre Filter"

    def compute(self, price: pd.Series) -> pd.Series:
        gamma = 0.8
        # CRÍTICO: cada variável precisa ser uma cópia INDEPENDENTE
        l0 = price.copy()
        l1 = price.copy()
        l2 = price.copy()
        l3 = price.copy()

        for i in range(1, len(price)):
            l0.iloc[i] = (1 - gamma) * price.iloc[i] + gamma * l0.iloc[i-1]
            l1.iloc[i] = -gamma * l0.iloc[i] + l0.iloc[i-1] + gamma * l1.iloc[i-1]
            l2.iloc[i] = -gamma * l1.iloc[i] + l1.iloc[i-1] + gamma * l2.iloc[i-1]
            l3.iloc[i] = -gamma * l2.iloc[i] + l2.iloc[i-1] + gamma * l3.iloc[i-1]

        return (l0 + 2*l1 + 2*l2 + l3) / 6


class ZeroLagHMA(Indicator):
    name = "ZeroLagHMA"
    category = "Redução de Lag"
    description = "Zero Lag Hull Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        hma = HMA(self.period).compute(price)
        lag = (self.period - 1) // 2
        return hma + (hma - hma.shift(lag))


# ============================================================
# ADAPTATIVAS
# ============================================================

class KAMA(Indicator):
    name = "KAMA"
    category = "Adaptativas"
    description = "Kaufman Adaptive Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        change = abs(price - price.shift(self.period))
        volatility = price.diff().abs().rolling(self.period).sum()
        
        er = change / volatility
        er = er.fillna(0)
        
        fast = 2.0 / (2.0 + 1.0)
        slow = 2.0 / (30.0 + 1.0)
        
        sc = (er * (fast - slow) + slow) ** 2
        
        kama = price.copy()
        for i in range(1, len(price)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i-1])
        
        return kama


class VIDYA(Indicator):
    name = "VIDYA"
    category = "Adaptativas"
    description = "Variable Index Dynamic Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        vol = price.diff().abs().rolling(self.period).mean()
        vol_ma = vol.rolling(self.period).mean()
        
        alpha = 2.0 / (self.period + 1.0)
        k = vol / vol_ma
        k = k.fillna(1.0)
        
        vidya = price.copy()
        for i in range(1, len(price)):
            vidya.iloc[i] = vidya.iloc[i-1] + alpha * k.iloc[i] * (price.iloc[i] - vidya.iloc[i-1])
        
        return vidya


class FRAMA(Indicator):
    name = "FRAMA"
    category = "Adaptativas"
    description = "Fractal Adaptive Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        n = self.period
        frama = price.copy()
        
        for i in range(n, len(price)):
            hl1 = price.iloc[i-n//2:i].max() - price.iloc[i-n//2:i].min()
            hl2 = price.iloc[i-n:i-n//2].max() - price.iloc[i-n:i-n//2].min()
            hl = price.iloc[i-n:i].max() - price.iloc[i-n:i].min()
            
            if hl > 0 and hl1 > 0 and hl2 > 0:
                dim = (np.log(hl1 + hl2) - np.log(hl)) / np.log(2)
            else:
                dim = 1.0
            
            alpha = np.exp(-4.6 * (dim - 1))
            alpha = np.clip(alpha, 0.01, 1.0)
            
            frama.iloc[i] = alpha * price.iloc[i] + (1 - alpha) * frama.iloc[i-1]
        
        for i in range(1, n):
            frama.iloc[i] = price.iloc[i]
        
        return frama


class MAMA(Indicator):
    name = "MAMA"
    category = "Adaptativas"
    description = "MESA Adaptive Moving Average"

    def compute(self, price: pd.Series) -> pd.Series:
        mama, _, _ = self._compute_internal(price)
        return mama

    def _compute_internal(self, price: pd.Series) -> tuple:
        """Retorna (mama, fama, alpha_series) para reuso pelo FAMA"""
        fast_limit = 0.5
        slow_limit = 0.05

        mama = price.copy()
        fama = price.copy()
        alpha_series = pd.Series(0.0, index=price.index)

        # Series para armazenar historico dos filtros (necessario para o Hilbert)
        smooth_series = price.copy()
        detrender_series = pd.Series(0.0, index=price.index)

        for i in range(5, len(price)):
            smooth = (price.iloc[i] + 2*price.iloc[i-1] + 2*price.iloc[i-2] + price.iloc[i-3]) / 6
            smooth_series.iloc[i] = smooth
            detrender = (0.0962*smooth + 0.5769*smooth_series.iloc[i-2] -
                         0.5769*smooth_series.iloc[i-4] - 0.0962*smooth_series.iloc[i-6])
            detrender_series.iloc[i] = detrender

            Q1 = (0.1181*detrender + 0.6923*detrender_series.iloc[i-3] -
                  0.6923*detrender_series.iloc[i-5] - 0.1181*detrender_series.iloc[i-7])
            I1 = detrender_series.iloc[i-3]

            if Q1 != 0:
                delta = np.arctan(I1 / Q1)
            else:
                delta = 0

            if delta > 0:
                alpha = fast_limit / delta
            else:
                alpha = fast_limit

            alpha = np.clip(alpha, slow_limit, fast_limit)
            alpha_series.iloc[i] = alpha

            mama.iloc[i] = alpha * price.iloc[i] + (1 - alpha) * mama.iloc[i-1]
            fama.iloc[i] = 0.5 * alpha * mama.iloc[i] + (1 - 0.5 * alpha) * fama.iloc[i-1]

        return mama, fama, alpha_series


class FAMA(Indicator):
    name = "FAMA"
    category = "Adaptativas"
    description = "Following Adaptive Moving Average (Filtro Adaptativo)"

    def compute(self, price: pd.Series) -> pd.Series:
        # Reusa o alpha do MAMA para computar FAMA corretamente
        _, fama, _ = MAMA(self.period)._compute_internal(price)
        return fama


class JMA(Indicator):
    name = "JMA"
    category = "Adaptativas"
    description = "Jurik Moving Average (aproximacao 3-estagios)"

    def __init__(self, period: int = 20, phase: float = 0.0, power: float = 2.0):
        super().__init__(period)
        self.phase = np.clip(phase, -100, 100)
        self.power = np.clip(power, 1.0, 10.0)

    def compute(self, price: pd.Series) -> pd.Series:
        phase_ratio = self.phase / 100.0
        pow_val = self.power

        # 3-stage adaptive filter (aproximacao Jurik)
        jma = price.copy()
        e0 = price.copy()
        e1 = price.copy()
        e2 = price.copy()

        beta = 0.45 * (self.period - 1) / (0.45 * (self.period - 1) + 2.0)
        alpha = beta ** pow_val

        for i in range(1, len(price)):
            e0.iloc[i] = (1 - alpha) * price.iloc[i] + alpha * e0.iloc[i-1]
            e1.iloc[i] = (1 - alpha) * e0.iloc[i] + alpha * e1.iloc[i-1]
            e2.iloc[i] = (1 - alpha) * e1.iloc[i] + alpha * e2.iloc[i-1]

            jma.iloc[i] = e0.iloc[i] + phase_ratio * (e0.iloc[i] - e1.iloc[i]) + \
                          (e0.iloc[i] - 2*e1.iloc[i] + e2.iloc[i]) * (1 - phase_ratio)

        return jma


class AdaptiveEMA(Indicator):
    name = "AdaptiveEMA"
    category = "Adaptativas"
    description = "Adaptive Exponential Moving Average"

    def compute(self, price: pd.Series) -> pd.Series:
        # Usa desvio padrão dos retornos como medida de volatilidade
        returns = price.pct_change()
        volatility = returns.rolling(20).std()
        norm_vol = volatility / (volatility.rolling(100).max() + 1e-10)

        alpha_base = 2.0 / (self.period + 1.0)
        alpha = alpha_base * (1 + norm_vol.clip(upper=2.0))
        alpha = alpha.fillna(alpha_base)

        aema = price.copy()
        for i in range(1, len(price)):
            aema.iloc[i] = aema.iloc[i-1] + alpha.iloc[i] * (price.iloc[i] - aema.iloc[i-1])

        return aema


class EfficiencyRatioMA(Indicator):
    name = "EfficiencyRatioMA"
    category = "Adaptativas"
    description = "Efficiency Ratio Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        change = abs(price - price.shift(self.period))
        volatility = price.diff().abs().rolling(self.period).sum()
        
        er = change / volatility
        er = er.fillna(0)
        
        er_ma = price.copy()
        for i in range(1, len(price)):
            alpha = er.iloc[i] * 0.5
            er_ma.iloc[i] = er_ma.iloc[i-1] + alpha * (price.iloc[i] - er_ma.iloc[i-1])
        
        return er_ma


# ============================================================
# FILTROS DSP (EHLERS)
# ============================================================

class SuperSmoother(Indicator):
    name = "SuperSmoother"
    category = "Filtros DSP"
    description = "Ehlers Super Smoother"
    
    def compute(self, price: pd.Series) -> pd.Series:
        a1 = np.exp(-1.414 * np.pi / self.period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / self.period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = price.copy()
        
        for i in range(2, len(price)):
            filt.iloc[i] = c1 * (price.iloc[i] + price.iloc[i-1]) / 2 + \
                          c2 * filt.iloc[i-1] + c3 * filt.iloc[i-2]
        
        return filt


class RoofingFilter(Indicator):
    name = "RoofingFilter"
    category = "Filtros DSP"
    description = "Ehlers Roofing Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        hp_period = int(self.period * 0.5)
        lp_period = self.period
        
        alpha1 = (np.cos(2*np.pi/hp_period) + np.sin(2*np.pi/hp_period) - 1) / np.cos(2*np.pi/hp_period)
        hp = price.copy()
        
        for i in range(2, len(price)):
            hp.iloc[i] = (1 - alpha1/2)**2 * (price.iloc[i] - 2*price.iloc[i-1] + price.iloc[i-2]) + \
                        2*(1 - alpha1) * hp.iloc[i-1] - (1 - alpha1)**2 * hp.iloc[i-2]
        
        a2 = np.exp(-2*np.pi/lp_period)
        b2 = 2 * a2 * np.cos(2*np.pi/lp_period)
        c2 = b2
        c3 = -a2 * a2
        c1 = 1 - c2 - c3
        
        filt = hp.copy()
        for i in range(2, len(hp)):
            filt.iloc[i] = c1 * (hp.iloc[i] + hp.iloc[i-1]) / 2 + \
                          c2 * filt.iloc[i-1] + c3 * filt.iloc[i-2]
        
        return filt


class Decycler(Indicator):
    name = "Decycler"
    category = "Filtros DSP"
    description = "Ehlers Decycler"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = (np.cos(2*np.pi/self.period) + np.sin(2*np.pi/self.period) - 1) / np.cos(2*np.pi/self.period)
        
        decycler = price.copy()
        
        for i in range(2, len(price)):
            decycler.iloc[i] = (1 - alpha/2)**2 * (price.iloc[i] - 2*price.iloc[i-1] + price.iloc[i-2]) + \
                               2*(1 - alpha) * decycler.iloc[i-1] - (1 - alpha)**2 * decycler.iloc[i-2]
        
        return decycler


class CyberCycle(Indicator):
    name = "CyberCycle"
    category = "Filtros DSP"
    description = "Ehlers Cyber Cycle"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 0.07
        cycle = pd.Series(0, index=price.index)
        
        for i in range(2, len(price)):
            cycle.iloc[i] = (1 - 0.5*alpha) * (price.iloc[i] - 2*price.iloc[i-1] + price.iloc[i-2]) + \
                            (1 - alpha) * cycle.iloc[i-1]
        
        return cycle


class MesaSineWave(Indicator):
    name = "MesaSineWave"
    category = "Filtros DSP"
    description = "MESA Sine Wave Indicator"
    
    def compute(self, price: pd.Series) -> pd.Series:
        smooth = (price + 2*price.shift(1) + 2*price.shift(2) + price.shift(3)) / 6
        
        cycle = (1 - 0.5*0.07) * (smooth - 2*smooth.shift(1) + smooth.shift(2)) + \
                (1 - 0.07) * smooth.shift(1)
        
        return cycle


class HilbertTransform(Indicator):
    name = "HilbertTransform"
    category = "Filtros DSP"
    description = "Hilbert Transform (8-tap)"

    def compute(self, price: pd.Series) -> pd.Series:
        # Suavizacao inicial (Butterworth-style 4-pole)
        smooth = price.ewm(span=4, adjust=False).mean()

        # Hilbert 8-tap: coeficientes anti-simetricos em lags pares
        # Lags: 0, 2, 4, 6, 8, 10, 12, 14
        coefs = np.array([0.0315, 0.1108, 0.2122, 0.3536,
                          -0.3536, -0.2122, -0.1108, -0.0315])
        lags = [0, 2, 4, 6, 8, 10, 12, 14]

        hilbert = pd.Series(0.0, index=price.index)
        for i in range(lags[-1], len(price)):
            val = 0.0
            for c, lag in zip(coefs, lags):
                val += c * smooth.iloc[i - lag]
            hilbert.iloc[i] = val

        return hilbert


class DominantCycle(Indicator):
    name = "DominantCycle"
    category = "Filtros DSP"
    description = "Dominant Cycle"
    
    def compute(self, price: pd.Series) -> pd.Series:
        smooth = (4*price + 3*price.shift(1) + 2*price.shift(2) + price.shift(3)) / 10
        
        period = pd.Series(self.period, index=price.index)
        
        for i in range(10, len(price)):
            real = smooth.iloc[i] - smooth.iloc[i-7]
            if real > 0:
                period.iloc[i] = period.iloc[i-1] + 0.25 * (real - period.iloc[i-1])
        
        return period


class InstantTrend(Indicator):
    name = "InstantTrend"
    category = "Filtros DSP"
    description = "Instantaneous Trend"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        
        itrend = price.copy()
        trigger = price.copy()
        
        for i in range(2, len(price)):
            itrend.iloc[i] = (alpha - alpha**2/4) * price.iloc[i] + \
                            (alpha**2/2) * price.iloc[i-1] - \
                            (alpha - 3*alpha**2/4) * price.iloc[i-2] + \
                            2*(1 - alpha) * itrend.iloc[i-1] - \
                            (1 - alpha)**2 * itrend.iloc[i-2]
            
            trigger.iloc[i] = 2 * itrend.iloc[i] - itrend.iloc[i-1]
        
        return trigger


# ============================================================
# FILTROS GAUSSIANOS
# ============================================================

class GaussianMA(Indicator):
    name = "GaussianMA"
    category = "Filtros Gaussianos"
    description = "Gaussian Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        window = self.period
        sigma = window / 3
        x = np.arange(window) - window/2
        weights = np.exp(-(x**2) / (2 * sigma**2))
        weights /= weights.sum()
        
        return price.rolling(window).apply(
            lambda x: np.dot(x, weights), raw=True
        )


class Gaussian2Pole(Indicator):
    name = "Gaussian2Pole"
    category = "Filtros Gaussianos"
    description = "2-Pole Gaussian Filter (Ehlers)"

    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)

        filt = price.copy()

        for i in range(2, len(price)):
            filt.iloc[i] = alpha * price.iloc[i] + \
                          2 * (1 - alpha) * filt.iloc[i-1] - \
                          (1 - alpha)**2 * filt.iloc[i-2]

        return filt


class Gaussian3Pole(Indicator):
    name = "Gaussian3Pole"
    category = "Filtros Gaussianos"
    description = "3-Pole Gaussian Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        
        filt = price.copy()
        
        for i in range(3, len(price)):
            filt.iloc[i] = alpha * price.iloc[i] + \
                          3 * (1 - alpha) * filt.iloc[i-1] - \
                          3 * (1 - alpha)**2 * filt.iloc[i-2] + \
                          (1 - alpha)**3 * filt.iloc[i-3]
        
        return filt


class Gaussian4Pole(Indicator):
    name = "Gaussian4Pole"
    category = "Filtros Gaussianos"
    description = "4-Pole Gaussian Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        
        filt = price.copy()
        
        for i in range(4, len(price)):
            filt.iloc[i] = alpha * price.iloc[i] + \
                          4 * (1 - alpha) * filt.iloc[i-1] - \
                          6 * (1 - alpha)**2 * filt.iloc[i-2] + \
                          4 * (1 - alpha)**3 * filt.iloc[i-3] - \
                          (1 - alpha)**4 * filt.iloc[i-4]
        
        return filt


class GaussianBandpass(Indicator):
    name = "GaussianBandpass"
    category = "Filtros Gaussianos"
    description = "Gaussian Bandpass Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        low = GaussianMA(int(self.period * 0.5)).compute(price)
        high = GaussianMA(self.period).compute(price)
        return low - high


# ============================================================
# FILTROS ESTATÍSTICOS
# ============================================================

class LSMA(Indicator):
    name = "LSMA"
    category = "Filtros Estatísticos"
    description = "Least Squares Moving Average"
    
    def compute(self, price: pd.Series) -> pd.Series:
        def linreg(x):
            if len(x) < 2:
                return x[-1] if len(x) > 0 else 0
            t = np.arange(len(x))
            slope, intercept = np.polyfit(t, x, 1)
            return slope * (len(x) - 1) + intercept
        
        return price.rolling(self.period).apply(linreg, raw=True)


class PolynomialReg(Indicator):
    name = "PolynomialReg"
    category = "Filtros Estatísticos"
    description = "Polynomial Regression"
    
    def compute(self, price: pd.Series) -> pd.Series:
        def polyreg(x):
            if len(x) < 3:
                return x[-1] if len(x) > 0 else 0
            t = np.arange(len(x))
            coeffs = np.polyfit(t, x, 2)
            return np.polyval(coeffs, len(x) - 1)
        
        return price.rolling(self.period).apply(polyreg, raw=True)


class Kalman(Indicator):
    name = "Kalman"
    category = "Filtros Estatísticos"
    description = "Kalman Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        Q = 1e-5  # Process noise
        R = 1e-2  # Measurement noise
        
        x_est = price.copy()
        p_est = pd.Series(1.0, index=price.index)
        
        for i in range(1, len(price)):
            # Prediction
            x_pred = x_est.iloc[i-1]
            p_pred = p_est.iloc[i-1] + Q
            
            # Update
            K = p_pred / (p_pred + R)
            x_est.iloc[i] = x_pred + K * (price.iloc[i] - x_pred)
            p_est.iloc[i] = (1 - K) * p_pred
        
        return x_est


class RMS(Indicator):
    name = "RMS"
    category = "Filtros Estatísticos"
    description = "Root Mean Square"
    
    def compute(self, price: pd.Series) -> pd.Series:
        return np.sqrt((price ** 2).rolling(self.period).mean())


class Median(Indicator):
    name = "Median"
    category = "Filtros Estatísticos"
    description = "Median Filter"

    def compute(self, price: pd.Series) -> pd.Series:
        return price.rolling(self.period).median()


class FisherTransform(Indicator):
    name = "FisherTransform"
    category = "Filtros Estatísticos"
    description = "Ehlers Fisher Transform"

    def compute(self, price: pd.Series) -> pd.Series:
        # Normaliza preco para [-1, 1] usando min/max do periodo
        min_p = price.rolling(self.period).min()
        max_p = price.rolling(self.period).max()
        rng = (max_p - min_p).clip(lower=1e-10)

        # Mapeia para [-0.99, 0.99] com suavizacao EMA(0.33)
        x = ((price - min_p) / rng - 0.5) * 2
        x = x.ewm(alpha=0.33, adjust=False).mean()
        x = x.clip(-0.999, 0.999)

        # Fisher Transform: 0.5 * ln((1+x)/(1-x))
        fisher = 0.5 * np.log((1 + x) / (1 - x))
        return fisher


# ============================================================
# SUAVIZAÇÃO AVANÇADA
# ============================================================

class Butterworth(Indicator):
    name = "Butterworth"
    category = "Suavização Avançada"
    description = "Butterworth 2-pole Filter (estável)"

    def compute(self, price: pd.Series) -> pd.Series:
        # Implementação padrão estável usando transformada bilinear com prewarping
        wc = 2.0 * np.pi / max(self.period, 4)  # cutoff frequency
        C = 1.0 / np.tan(wc / 2.0)               # prewarping
        D = C * C + np.sqrt(2.0) * C + 1.0

        b0 = 1.0 / D
        b1 = 2.0 / D
        b2 = 1.0 / D
        a0 = 2.0 * (C * C - 1.0) / D
        a1 = -(C * C - np.sqrt(2.0) * C + 1.0) / D

        filt = price.copy()

        for i in range(2, len(price)):
            filt.iloc[i] = (b0 * price.iloc[i] +
                            b1 * price.iloc[i-1] +
                            b2 * price.iloc[i-2] +
                            a0 * filt.iloc[i-1] +
                            a1 * filt.iloc[i-2])

        return filt


class SavitzkyGolay(Indicator):
    name = "SavitzkyGolay"
    category = "Suavização Avançada"
    description = "Savitzky-Golay Filter"
    
    def compute(self, price: pd.Series) -> pd.Series:
        window = self.period
        order = 2
        
        if window % 2 == 0:
            window += 1
        
        half = window // 2
        
        def sg_smooth(x):
            if len(x) < window:
                return x[len(x)//2] if len(x) > 0 else 0
            t = np.arange(window) - half
            A = np.vstack([t**i for i in range(order + 1)]).T
            coeffs = np.linalg.lstsq(A, x, rcond=None)[0]
            return np.polyval(coeffs, 0)
        
        return price.rolling(window, center=True).apply(sg_smooth, raw=True)


class HodrickPrescott(Indicator):
    name = "HodrickPrescott"
    category = "Suavização Avançada"
    description = "Hodrick-Prescott Filter (matriz esparsa)"

    def compute(self, price: pd.Series) -> pd.Series:
        lam = 1600  # Smoothing parameter
        n = len(price)

        # Limite prático: para >10k pontos usamos downsample + interpolação
        if n > 10000:
            step = max(1, n // 5000)
            idx_original = np.arange(n)
            idx_down = idx_original[::step]
            price_down = price.values[::step]
            trend_down = self._solve_hp(price_down, lam, len(price_down))
            trend = np.interp(idx_original, idx_down, trend_down)
            return pd.Series(trend, index=price.index)

        return pd.Series(self._solve_hp(price.values, lam, n), index=price.index)

    @staticmethod
    def _solve_hp(values: np.ndarray, lam: float, n: int) -> np.ndarray:
        """Resolve HP filter usando matriz esparsa pentadiagonal (O(n) memória)"""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        # Matriz de segunda diferença (esparsa)
        e = np.ones(n)
        D2 = sparse.spdiags([e, -2*e, e], [0, 1, 2], n-2, n, format='csr')

        # I + λ * D'D  (pentadiagonal, super eficiente)
        A = sparse.eye(n, format='csr') + lam * D2.T @ D2

        return spsolve(A, values).astype(np.float64)


class ExpSmoothing(Indicator):
    name = "ExpSmoothing"
    category = "Suavização Avançada"
    description = "Exponential Smoothing"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        return price.ewm(alpha=alpha, adjust=False).mean()


class DoubleExp(Indicator):
    name = "DoubleExp"
    category = "Suavização Avançada"
    description = "Double Exponential Smoothing"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        beta = 0.1
        
        s = price.copy()
        b = pd.Series(0, index=price.index)
        
        for i in range(1, len(price)):
            s.iloc[i] = alpha * price.iloc[i] + (1 - alpha) * (s.iloc[i-1] + b.iloc[i-1])
            b.iloc[i] = beta * (s.iloc[i] - s.iloc[i-1]) + (1 - beta) * b.iloc[i-1]
        
        return s + b


class TripleExp(Indicator):
    name = "TripleExp"
    category = "Suavização Avançada"
    description = "Triple Exponential Smoothing"
    
    def compute(self, price: pd.Series) -> pd.Series:
        alpha = 2.0 / (self.period + 1.0)
        beta = 0.1
        gamma = 0.1
        
        s = price.copy()
        b = pd.Series(0, index=price.index)
        c = pd.Series(0, index=price.index)
        
        for i in range(1, len(price)):
            s_old = s.iloc[i-1]
            b_old = b.iloc[i-1]
            
            s.iloc[i] = alpha * (price.iloc[i] - c.iloc[i-1]) + (1 - alpha) * (s_old + b_old)
            b.iloc[i] = beta * (s.iloc[i] - s_old) + (1 - beta) * b_old
            c.iloc[i] = gamma * (price.iloc[i] - s.iloc[i]) + (1 - gamma) * c.iloc[i-1]
        
        return s + b + c


# ============================================================
# REGISTRO DE INDICADORES
# ============================================================

def get_all_indicators() -> Dict[str, Type[Indicator]]:
    """Retorna todos os indicadores disponíveis"""
    indicators = {}
    for cls in globals().values():
        if isinstance(cls, type) and issubclass(cls, Indicator) and cls != Indicator:
            indicators[cls.name] = cls
    return indicators


def get_indicators_by_category(category: str) -> List[Type[Indicator]]:
    """Retorna indicadores de uma categoria específica"""
    return [cls for cls in get_all_indicators().values() if cls.category == category]


def get_categories() -> List[str]:
    """Retorna todas as categorias disponíveis"""
    categories = set()
    for cls in get_all_indicators().values():
        categories.add(cls.category)
    return sorted(list(categories))


# Instância global para fácil acesso
INDICATORS = get_all_indicators()