# ma_lab/analytics/heatmap_engine.py
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(results):
    """
    Gera heatmap de Sharpe por combinação de indicadores FAST x SLOW.

    Args:
        results: Lista de SystemResult ou DataFrame com colunas
                 'fast', 'slow', 'sharpe' (e opcionalmente 'pf', 'win', 'ret')
    """
    if isinstance(results, list):
        df = pd.DataFrame([r.to_dict() for r in results])
    else:
        df = results.copy()

    if df.empty:
        print("Nenhum dado para gerar heatmap.")
        return

    pivot = df.pivot_table(
        values="sharpe",
        index="fast",
        columns="slow",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="Sharpe")
    plt.title("Heatmap de Cruzamentos (Sharpe médio)")
    plt.tight_layout()
    plt.show()
