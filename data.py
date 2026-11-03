"""
Shared data preparation for the Danish stock panel.

build_panel() returns cleaned monthly firm-level data that every factor script uses:
  ret      monthly total returns (from RI), NaN when the firm is not trading
  mv       firm market value at month end (sum of all Copenhagen share classes), DKK millions
  price    unadjusted price of the return line (DKK)
  spread   relative bid-ask spread (ask - bid) / mid
  names    firm names by column
  novo     column id of Novo Nordisk
Columns are firm ids (row numbers in DK_active + DK_dead); the index is the month start date.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from feasibility import NOVO, drop_flat_tail, firm_key, load_series, load_static

TEMP_LINES = r"NIL PAID|FULLY PAID|RIGHTS|NEW SHS|SPUN-OFF"


@lru_cache(maxsize=1)
def build_panel() -> dict:
    s = load_static()
    ri, ri_codes = load_series("DK_RI.csv")
    mv, mv_codes = load_series("DK_MV.csv")
    up, _ = load_series("DK_UP.csv")
    pa, _ = load_series("DKK_PA.csv")
    pb, _ = load_series("DKK_PB.csv")

    errors = ri_codes.isna() | mv_codes.isna()
    cse_equity = (
        (s["bourse_mnemonic"] == "CSE")
        & ~s["industry_group"].isin(["Unquoted equities", "Unclassified"])
        & ~errors
        & ~s["name"].str.contains(TEMP_LINES, regex=True)
    )
    s["firm"] = s["name"].map(firm_key)

    # one return line per firm: the major line with the most observations
    major = s[cse_equity & (s["major_flag"] == "Y")].assign(nobs=ri.notna().sum())
    ret_line = major.sort_values("nobs", ascending=False).groupby("firm").head(1)
    u = ret_line.index.sort_values()

    # firm market value = sum over all its Copenhagen share classes
    class_lines = s[cse_equity & s["firm"].isin(ret_line["firm"])]
    to_firm = class_lines["firm"].map(ret_line.reset_index().set_index("firm")["index"])
    R_all = ri[class_lines.index].apply(drop_flat_tail)
    mv_lines = mv[class_lines.index].where(mv[class_lines.index] > 0).where(R_all.notna())
    firm_mv = mv_lines.T.groupby(to_firm).sum(min_count=1).T.reindex(columns=u)

    R = ri[u].apply(drop_flat_tail)
    alive = R.notna()
    ret = R.pct_change(fill_method=None)
    sp = (pa[u] - pb[u]) / ((pa[u] + pb[u]) / 2)

    return {
        "ret": ret,
        "mv": firm_mv.where(alive),
        "price": up[u].where(alive),
        "spread": sp.where(alive & (sp > 0) & (sp < 1)),
        "names": s.loc[u, "name"],
        "status": s.loc[u, "status"],
        "industry": s.loc[u, "industry_group"],
        "novo": int(s.index[(s["type"] == NOVO) & s.index.isin(u)][0]),
    }


# ---------------------------------------------------------------- performance helpers
def newey_west_t(x: pd.Series, lags: int = 6) -> float:
    """t-statistic of the mean with Newey-West standard errors."""
    x = x.dropna().to_numpy()
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(var / n)


def alpha_beta(y: pd.Series, x: pd.Series, lags: int = 6) -> tuple[float, float, float]:
    """CAPM-style regression y = a + b x + e. Returns annualised alpha, its Newey-West t-stat, beta."""
    d = pd.concat([y, x], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.iloc[:, 1]])
    b = np.linalg.lstsq(X, d.iloc[:, 0], rcond=None)[0]
    e = d.iloc[:, 0].to_numpy() - X @ b
    n = len(d)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lags + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / n)
    se = np.sqrt(np.diag(Q @ S @ Q / n))
    t = b[0] / se[0] if abs(b[0]) > 1e-10 else np.nan   # regressing the market on itself
    return b[0] * 12, t, b[1]


def perf_table(rets: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    rows = {}
    for c, r in rets.items():
        r = r.dropna()
        cum = (1 + r).cumprod()
        a, ta, b = alpha_beta(r, bench)
        rows[c] = {
            "ann_return": cum.iloc[-1] ** (12 / len(r)) - 1,
            "ann_mean": r.mean() * 12,
            "t_mean": newey_west_t(r),
            "ann_vol": r.std() * np.sqrt(12),
            "mean_per_vol": r.mean() * 12 / (r.std() * np.sqrt(12)),
            "max_drawdown": (cum / cum.cummax() - 1).min(),
            "alpha_vs_mkt": a, "t_alpha": ta, "beta": b,
        }
    return pd.DataFrame(rows).T
