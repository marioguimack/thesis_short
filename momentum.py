"""
Momentum factor for the Danish market.

Signal (Jegadeesh & Titman 1993; Carhart 1997): cumulative return from month t-12 to t-2,
skipping the most recent month to avoid short-term reversal. At least 8 of the 11 months
must be available.

Each formation date f (end of month t-1), stocks are eligible if they trade, have a positive
market value and an unadjusted price of at least 1 DKK. They are sorted into terciles on the
signal (all-stock breakpoints). Portfolios earn the return of month t.

Portfolios
  W / M / L        winners, middle, losers (value- and equal-weighted)
  WML              winners minus losers (the academic long-short factor, for reference)
  W_capped         long-only winners, value-weighted with a 10% cap per stock (the tilt
                   an investor could hold)
  MKT, MKT_capped  benchmarks on the same eligible universe

Rebalancing: monthly, and quarterly (membership fixed for three months, weights drift
with market values). Transaction costs: half the bid-ask spread on each unit of turnover.

Run:  python momentum.py      Outputs: ./output/momentum_*
"""
import numpy as np
import pandas as pd

from data import build_panel, perf_table
from feasibility import OUT, capped_weights

LOOKBACK, SKIP, MIN_OBS = 11, 1, 8      # months t-12..t-2
MIN_PRICE = 1.0                         # DKK
CAP = 0.10
N_GROUPS = 3
PERIODS = {"2005-2016": ("2005", "2016"), "2017-Jun 2024 (boom)": ("2017", "2024-06"),
           "Jul 2024-2026 (fall)": ("2024-07", "2026")}


def momentum_signal(ret: pd.DataFrame) -> pd.DataFrame:
    """Signal known at the end of month f: cumulative return over f-11..f-1."""
    logr = np.log1p(ret)
    cum = logr.rolling(LOOKBACK, min_periods=MIN_OBS).sum()
    return np.expm1(cum.shift(SKIP))


def assign_groups(signal: pd.Series) -> pd.Series:
    """Tercile 1 = losers, 3 = winners."""
    return pd.qcut(signal.rank(method="first"), N_GROUPS, labels=False) + 1


def run(panel: dict, freq: str) -> dict:
    ret, mv, price, spread = panel["ret"], panel["mv"], panel["price"], panel["spread"]
    sig = momentum_signal(ret)
    dates = ret.index
    out = {k: {} for k in ["ret", "turnover", "cost", "n", "novo_in_W", "novo_w_W"]}
    groups = None
    prev_w = {}

    for i in range(1, len(dates)):
        f, t = dates[i - 1], dates[i]
        rebalance = freq == "M" or f.month in (12, 3, 6, 9) or groups is None
        elig = mv.loc[f].gt(0) & price.loc[f].ge(MIN_PRICE) & ret.loc[f].notna()
        if rebalance:
            s = sig.loc[f][elig].dropna()
            if len(s) < 3 * N_GROUPS:
                continue
            groups = assign_groups(s)
        g = groups[groups.index.isin(elig[elig].index)]       # drop firms that delisted
        r_t = ret.loc[t]
        half_sp = spread.loc[f].fillna(spread.loc[f].median()) / 2

        ports = {
            "W": g.index[g == N_GROUPS], "M": g.index[g == 2], "L": g.index[g == 1],
            "MKT": elig[elig].index,
        }
        weights = {}
        for k, idx in ports.items():
            idx = [c for c in idx if pd.notna(r_t.get(c))]   # renormalise over stocks with a return
            if not idx:
                continue
            m = mv.loc[f, idx]
            weights[f"{k}_vw"] = m / m.sum()
            weights[f"{k}_ew"] = pd.Series(1 / len(idx), index=idx)
            if k in ("W", "MKT"):
                weights[f"{k}_capped"] = capped_weights(m.copy(), CAP)

        for k, w in weights.items():
            out["ret"].setdefault(k, {})[t] = (w * r_t[w.index]).sum()
            # turnover vs drifted previous weights
            pw = prev_w.get(k)
            if pw is None:
                to = 1.0
                trade = w
            else:
                drift = pw * (1 + ret.loc[f, pw.index].fillna(0))
                drift = drift / drift.sum()
                trade = w.sub(drift, fill_value=0).abs()
                to = trade.sum() / 2
            out["turnover"].setdefault(k, {})[t] = to
            out["cost"].setdefault(k, {})[t] = (trade * half_sp.reindex(trade.index).fillna(half_sp.median())).sum()
            prev_w[k] = w
        out["n"].setdefault("W", {})[t] = len(ports["W"])
        out["n"].setdefault("L", {})[t] = len(ports["L"])
        out["n"].setdefault("MKT", {})[t] = len(ports["MKT"])
        out["novo_in_W"][t] = panel["novo"] in ports["W"]
        out["novo_w_W"][t] = weights.get("W_vw", pd.Series(dtype=float)).get(panel["novo"], 0.0)

    res = {k: pd.DataFrame(v) if k not in ("novo_in_W", "novo_w_W") else pd.Series(v)
           for k, v in out.items()}
    R = res["ret"]
    R["WML_vw"] = R["W_vw"] - R["L_vw"]
    R["WML_ew"] = R["W_ew"] - R["L_ew"]
    net = R.drop(columns=["WML_vw", "WML_ew"]) - res["cost"]
    net["WML_vw"] = net["W_vw"] - net["L_vw"]
    net["WML_ew"] = net["W_ew"] - net["L_ew"]
    res["net"] = net
    return res


def main() -> None:
    panel = build_panel()
    names = panel["names"]
    lines = ["DANISH MOMENTUM FACTOR (12-1, terciles)", ""]
    cols = ["W_vw", "W_capped", "W_ew", "L_vw", "L_ew", "WML_vw", "WML_ew", "MKT_vw", "MKT_capped", "MKT_ew"]
    store = {}

    for freq, label in [("M", "monthly"), ("Q", "quarterly")]:
        res = run(panel, freq)
        R, N = res["ret"][cols], res["net"][cols]
        store[freq] = res
        bench = R["MKT_vw"]
        if freq == "M":
            lines += [f"Sample: {R.index[0]:%b %Y} - {R.index[-1]:%b %Y} ({len(R)} months)",
                      f"Stocks per tercile: mean {res['n']['W'].mean():.0f}, min {res['n']['W'].min()}", ""]
        lines += [f"=== {label.upper()} REBALANCING ===", "", "Gross returns (alpha and beta vs value-weighted market)",
                  perf_table(R, bench).round(3).to_string(), "",
                  "Net of half-spread costs", perf_table(N, N["MKT_vw"]).round(3)[
                      ["ann_return", "ann_mean", "t_mean", "mean_per_vol", "alpha_vs_mkt", "t_alpha"]].to_string(), "",
                  "Average one-way turnover per month",
                  res["turnover"][[c for c in cols if c in res["turnover"]]].mean().round(3).to_string(), "",
                  "Annualised gross return by period",
                  pd.concat({k: R.loc[a:b].apply(lambda r: (1 + r).prod() ** (12 / len(r)) - 1)
                             for k, (a, b) in PERIODS.items()}, axis=1).round(3).to_string(), ""]
        R.to_csv(OUT / f"momentum_returns_{label}.csv")
        N.to_csv(OUT / f"momentum_returns_{label}_net.csv")

    m = store["M"]
    novo_in = m["novo_in_W"]
    yearly = pd.DataFrame({"novo_in_winners_share_of_months": novo_in.groupby(novo_in.index.year).mean(),
                           "novo_weight_in_W_vw": m["novo_w_W"].groupby(m["novo_w_W"].index.year).mean()})
    lines += ["=== NOVO NORDISK IN THE WINNER PORTFOLIO (monthly rebalancing) ===",
              f"In winners in {novo_in.mean():.0%} of months", yearly.round(2).to_string()]
    yearly.to_csv(OUT / "momentum_novo_in_winners.csv")

    report = "\n".join(lines)
    (OUT / "momentum_report.txt").write_text(report)
    print(report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = store["M"]["ret"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c, lab in [("W_vw", "Winners VW"), ("W_capped", "Winners VW capped 10%"),
                   ("MKT_vw", "Market VW"), ("MKT_capped", "Market capped 10%")]:
        axes[0].plot((1 + R[c]).cumprod(), label=lab)
    axes[0].set(title="Long-only: growth of 1 DKK (monthly rebalancing)", yscale="log")
    axes[0].legend(frameon=False, fontsize=8)
    for c, lab in [("WML_vw", "WML value-weighted"), ("WML_ew", "WML equal-weighted")]:
        axes[1].plot((1 + R[c]).cumprod(), label=lab)
    axes[1].axhline(1, c="grey", lw=0.8)
    axes[1].set(title="Long-short momentum factor (winners minus losers)", yscale="log")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "momentum.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
