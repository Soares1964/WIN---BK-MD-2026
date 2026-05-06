# ma_lab/analytics/ranking_engine.py
import pandas as pd


def build_ranking(results):
    """
    Constrói ranking agregado por indicador FAST.

    Args:
        results: Lista de SystemResult ou DataFrame com colunas
                 'fast', 'slow', 'sharpe', 'pf' (profit factor)

    Returns:
        DataFrame com ranking por indicador FAST
    """
    if isinstance(results, list):
        df = pd.DataFrame([r.to_dict() for r in results])
    else:
        df = results.copy()

    if df.empty:
        return pd.DataFrame()

    ranking = df.groupby("fast").agg({
        "sharpe": "mean",
        "pf": "mean",
        "win": "mean",
        "dd": "mean",
        "fast": "count"
    }).rename(columns={"fast": "count"})

    ranking = ranking.sort_values("sharpe", ascending=False)

    return ranking
