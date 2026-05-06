# CONTEXTO DO PROJETO: WIN — BK MD 2026 (Laboratório de Médias Móveis)

## VISÃO GERAL

Sistema profissional para backtest, otimização e análise de estratégias de crossover de médias móveis para o Mini Índice Brasileiro (WIN). O projeto permite testar centenas de milhares de combinações de indicadores, períodos e timeframes com alta performance usando Numba JIT e paralelização com ThreadPoolExecutor.

**Principais funcionalidades:**
- Otimização massiva de estratégias de crossover (FAST x SLOW)
- Suporte a +50 indicadores (médias móveis clássicas, adaptativas, filtros DSP, Gaussianos)
- Backtest vetorizado com Numba (alta performance)
- Cache inteligente de médias móveis
- Interface gráfica completa com PySide6
- Ranking de sistemas por Sharpe, Profit Factor, Win Rate
- Heatmap de cruzamentos para visualizacao de hotspots
- Validacao de qualidade de dados

## ESTRUTURA DE DIRETORIOS
WIN - BK MD 2026/
- requirements.txt # Dependencias do projeto
- exportacoes_mt5/ # Dados exportados do MT5
  - grupo1.json # Configuracoes salvas
- bk_md/ # Codigo fonte principal
  - main.py # Ponto de entrada (interface grafica)
  - check_installation.py # Diagnostico de instalacao
  - carteira_recomendada.py # Script de carteira com sistemas validados
  - config_funcional.py # Configuracoes que funcionaram nos testes
  - run_tests.py # Script de validacao das correcoes
  - core/ # Nucleo do sistema
    - indicators.py # +50 indicadores (Numba otimizado)
    - backtest_engine.py # Backtest vetorizado com Numba
    - optimizer_engine.py # Motor de otimizacao com threads
    - ma_cache.py # Cache inteligente de medias
    - data_loader.py # Carregador e validador de dados
    - metrics.py # Metricas de performance
    - engine.py # Motor legado (compatibilidade)
  - ui/ # Interface PySide6
    - main_window.py # Janela principal
    - tab_config.py # Aba de configuracao/otimizacao
    - tab_ranking.py # Aba de ranking de sistemas
    - tab_heatmap.py # Aba de heatmap de cruzamentos
    - status_bar.py # Barra de status personalizada
  - models/ # Modelos de configuracao
    - config.py # ConfigManager e configuracoes
    - config_producao.py # Configuracao para producao
    - config_trabalho.py # Configuracao para trabalho
  - analytics/ # Modulos de analise
    - heatmap_engine.py # Geracao de heatmaps
    - ranking_engine.py # Ranking de sistemas
  - backtest/ # Backtest alternativo
    - crossover.py # Crossover com estado flat
  - utils/ # Utilitarios
    - dataset_validator.py # Validacao de datasets

## PRINCIPAIS COMPONENTES

### 1. Core/indicators.py
- **+50 indicadores** organizados por categorias:
  - Classicas: SMA, EMA, WMA, VWMA, SMMA, RMA, TMA
  - Reducao de Lag: DEMA, TEMA, ZLEMA, HMA, T3, ALMA, Laguerre, ZeroLagHMA
  - Adaptativas: KAMA, VIDYA, FRAMA, MAMA, FAMA, JMA, AdaptiveEMA, EfficiencyRatioMA
  - Filtros DSP: SuperSmoother, RoofingFilter, Decycler, CyberCycle, MesaSineWave, HilbertTransform, DominantCycle, InstantTrend
  - Filtros Gaussianos: GaussianMA, Gaussian2Pole, Gaussian3Pole, Gaussian4Pole, GaussianBandpass
  - Filtros Estatisticos: LSMA, PolynomialReg, Kalman, RMS, Median
  - Suavizacao Avancada: Butterworth, SavitzkyGolay, HodrickPrescott, ExpSmoothing, DoubleExp, TripleExp
- **Todas as funcoes criticas usam @njit da Numba** para performance maxima
- Cache de medias moveis via `MovingAverageCache`

### 2. Core/backtest_engine.py
- `run_backtest_vectorized`: Backtest ultra-rapido com Numba JIT
- `run_batch_backtest`: Execucao paralela com `prange`
- `generate_signal_from_cross`: Gera sinais 1/-1/0 baseado em cruzamento
- `calculate_metrics`: Calcula Sharpe, PF, Win Rate, Max DD, Retorno Total
- `periods_per_year_for_timeframe()`: Corrige anualizacao do Sharpe para dados intraday
- **Correcoes aplicadas**:
  - Profit Factor retorna `inf` quando nao ha perdas (nao inflaciona)
  - Sharpe Ratio usa desvio amostral (ddof=1) e anualizacao parametrizada
  - Slippage e Custos aplicados no momento da mudanca de posicao

### 3. Core/optimizer_engine.py
- `OptimizationEngine`: Motor principal com ThreadPoolExecutor
- `OptimizationConfig`: Dataclass com todos os parametros de otimizacao
- `SystemResult`: Resultado de um sistema testado
- **Caracteristicas**:
  - Processa batches em paralelo com threads
  - Cache inteligente (evita recalculo de medias)
  - Filtros de qualidade configuraveis
  - Reporta progresso em tempo real
  - **Correcoes**: Usa `threading.Lock()` (nao mp.Lock()) para ThreadPoolExecutor

### 4. Core/data_loader.py
- Carrega arquivos CSV com deteccao automatica de encoding
- Normaliza nomes de colunas (open, high, low, close)
- Gera automaticamente timeframes de 1 a 10 minutos via resample
- Validacao completa: gaps temporais, outliers, valores negativos

### 5. UI (PySide6)
- **ConfigTab**: Configuracao de indicadores, periodos, timeframes e filtros
- **RankingTab**: Tabela com ranking, filtros e insights automaticos
- **HeatmapTab**: Matriz de cruzamentos com codigo de cores
- **StatusBar**: Progresso em tempo real, velocidade, melhor sistema atual

## CONFIGURACAO E EXECUCAO

### Instalacao

```bash
# Criar ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Execucao

```bash
# Interface grafica
python bk_md/main.py

# Diagnostico de instalacao
python bk_md/check_installation.py

# Executar carteira recomendada
python bk_md/carteira_recomendada.py

# Validar correcoes
python bk_md/run_tests.py
```

### Dependencias Principais

| Pacote    | Versao     | Uso                          |
|-----------|-----------|------------------------------|
| numpy     | >=1.24.0  | Calculos numericos           |
| pandas    | >=2.0.0   | Manipulacao de dados         |
| numba     | >=0.58.0  | Compilacao JIT (performance) |
| PySide6   | >=6.5.0   | Interface grafica            |
| scipy     | >=1.11.0  | Calculos estatisticos        |
| matplotlib| >=3.7.0   | Visualizacao                 |
| psutil    | >=5.9.0   | Monitoramento de recursos    |

## CORRECOES IMPORTANTES APLICADAS

### v1 (Original)
- fillna deprecado: Substituido .fillna(method='ffill') por .ffill().bfill()
- Profit Factor: Retorna inf quando nao ha perdas (correto, nao inflaciona)
- Sharpe Ratio: Desvio amostral (ddof=1) e anualizacao via periods_per_year
- Lock threading: threading.Lock() (nao mp.Lock()) para ThreadPoolExecutor
- Crossover com estado flat: Sinal 0 quando medias iguais
- Memory management: GC coletado periodicamente durante otimizacao

### v2 (2026-05-04) - Correcoes Criticas
- **Laguerre Filter (BUG GRAVE)**: 4 variaveis apontavam para o MESMO objeto - filtro corrompido
- **Sharpe Intraday**: sqrt(252) hardcoded assumia dados diarios. Agora parametrizado via periods_per_year_for_timeframe()
- **Slippage ignorado**: Parametro era passado mas NUNCA usado no backtest - corrigido
- **HP Filter**: Matriz densa nxn (explodia memoria para >10k pts). Substituido por scipy.sparse
- **FAMA sem alpha**: Usava 0.5 fixo em vez de 0.5*alpha do MAMA - refatorado com exposicao correta
- **Butterworth instavel**: Coeficientes instaveis para certos periodos. Substituido por bilinear standard
- **VWMA**: Volume = np.ones (SMA disfarcada). Trocado para proxy via variacao de preco
- **AdaptiveEMA**: abs().rolling(20).std() incorreto. Trocado para pct_change().rolling(20).std()
- **Gaussian2Pole**: beta incorreto removido do coeficiente de feedback
- **Analytics**: Colunas fast_ma/slow_ma/profit nao existiam - corrigido para fast/slow/pf
- **config_trabalho.py**: Import de OptimizationConfig faltando - adicionado
- **data_loader.py**: 'T' deprecado no pandas - trocado para 'min'
- **Testes**: Expandidos de 7 para 12 testes cobrindo todas as correcoes
- **requirements.txt**: Recriado com versoes minimas corretas (removidos pacotes locais)

## SISTEMAS RECOMENDADOS (VALIDADOS)

| Sistema     | FAST       | SLOW       | Retorno | PF   | DD   | Trades |
|------------|-----------|-----------|---------|------|------|--------|
| CAMPEAO    | EMA(26)   | SMA(29)   | 30.5%   | 1.13 | 6.0% | 229    |
| SEGURO     | HMA(10)   | SMA(29)   | 27.5%   | 1.12 | 3.8% | 296    |
| CONSISTENTE| EMA(18)   | SMA(25)   | 28.9%   | 1.13 | 6.6% | 244    |
| CRESCIMENTO| EMA(28)   | SMA(31)   | 29.3%   | 1.13 | 7.5% | 219    |

## CONFIGURACAO OTIMIZADA PARA WIN

```python
config = OptimizationConfig(
    fast_indicators=['EMA', 'HMA'],
    slow_indicators=['SMA'],
    fast_min=10, fast_max=30, fast_step=2,
    slow_min=25, slow_max=35, slow_step=1,
    timeframes=[5],
    min_trades=50,
    min_sharpe=-999,      # Ignora Sharpe
    min_profit_factor=1.05,
    max_dd=15.0,
    slippage=0.5,
    cost=0.2
)
```

## PADROES DE CODIGO

### Backtest

```python
from core.backtest_engine import run_backtest_vectorized, generate_signal_from_cross, calculate_metrics, periods_per_year_for_timeframe

# Gerar sinal
signal = generate_signal_from_cross(fast_ma, slow_ma)

# Executar backtest
pnl, equity, trades = run_backtest_vectorized(price, signal, slippage=0.5, cost=0.2)

# Calcular metricas (anualizacao correta para o timeframe)
ppy = periods_per_year_for_timeframe(5)  # 27216 para TF 5-min
metrics = calculate_metrics(pnl, equity, trades, periods_per_year=ppy)
```

### Indicadores Numba

```python
from core.indicators import ema_numba, sma_numba, hull_numba

fast_ma = ema_numba(price, 26)
slow_ma = sma_numba(price, 29)
hma = hull_numba(price, 10)
```

### Cache de Medias

```python
from core.ma_cache import MovingAverageCache

cache = MovingAverageCache(price_series)
ema_20 = cache.get('EMA', 20)
sma_50 = cache.get('SMA', 50)
```

### Otimizacao

```python
from core.optimizer_engine import OptimizationEngine, OptimizationConfig

config = OptimizationConfig(...)
engine = OptimizationEngine(df_dict)
results = engine.run(config, progress_callback=my_callback)
```

## ARQUIVOS A IGNORAR / NAO MODIFICAR
- __pycache__/ - Cache do Python
- venv/ - Ambiente virtual
- *.log - Logs rotacionados
- exportacoes_mt5/*.csv - Dados exportados (runtime)

## TESTES

```bash
# Executar suite de testes
python bk_md/run_tests.py

# Verificar instalacao
python bk_md/check_installation.py
```

### Cobertura de testes:
- fillna deprecado
- Profit Factor com inf
- Sharpe Ratio amostral com periods_per_year
- threading.Lock correto
- OptimizationConfig unica
- Laguerre filter (copias independentes)
- FAMA com alpha do MAMA
- HodrickPrescott (matriz esparsa)
- Slippage no backtest
- Import config_trabalho

## NOTAS PARA DESENVOLVIMENTO
- **Performance**: Use Numba para loops criticos (decorador @njit)
- **Paralelismo**: ThreadPoolExecutor para I/O bound, Numba prange para CPU bound
- **Cache**: MovingAverageCache evita recalculo de medias
- **Timeframes**: Gerados automaticamente via resample (1-10 minutos)
- **Dados**: Espera colunas 'open', 'high', 'low', 'close'
- **Sinais**: 1 = LONG, -1 = SHORT, 0 = FLAT
- **Custos**: Aplicados no momento da mudanca de posicao
- **Sharpe Intraday**: Sempre usar periods_per_year_for_timeframe(tf) para anualizacao correta

## LICENCA
Projeto proprietario - uso interno.

Este arquivo deve ser salvo como `CLAUDE.md` na raiz do projeto (`WIN - BK MD 2026/`).
