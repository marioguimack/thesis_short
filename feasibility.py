"""
Feasibility and concentration analysis for the Danish equity market (LSEG Datastream data).

Run:  python feasibility.py
Outputs go to ./output/ (CSV tables and PNG charts).

Steps
1. Load the static lists (DK_active, DK_dead) and the monthly series (RI, MV, UP, VO, PA, PB, NOSHFF).
2. Build the universe: Copenhagen listings (CSE), major security flag, no investment funds,
   no lines that returned #ERROR.
3. Clean: remove the flat tail Datastream adds after a stock dies (repeated last value).
4. Feasibility: firms per month, zero-return months, coverage of each variable, bid-ask spreads.
5. Concentration: Novo Nordisk's weight, top-5 share, Herfindahl index (HHI), effective number of stocks.
6. Market returns: value-weighted, equal-weighted, value-weighted ex-Novo, and a 10%-capped index.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent
OUT = DATA / "output"
OUT.mkdir(exist_ok=True)

NOVO = "DK:NON"          # Datastream mnemonic for Novo Nordisk B
CAP = 0.10               # weight cap for the capped index
MIN_FIRMS = 80           # rule of thumb for tercile portfolios
STALE_LIMIT = 0.30       # flag stocks with more than 30% zero-return months


# ---------------------------------------------------------------- loading
def load_static() -> pd.DataFrame:
    act = pd.read_csv(DATA / "DK_active.csv").assign(status="active")
    dead = pd.read_csv(DATA / "DK_dead.csv").assign(status="dead")
    s = pd.concat([act, dead], ignore_index=True)
    s.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in s.columns]
    return s


def load_series(fname: str) -> tuple[pd.DataFrame, pd.Series]:
    """Datastream export: row 0 = names, row 1 = codes, rows 2+ = monthly values.
    Columns are in the same order as DK_active followed by DK_dead."""
    raw = pd.read_csv(DATA / fname, header=None, low_memory=False)
    codes = raw.iloc[1, 1:].reset_index(drop=True)
    df = raw.iloc[2:, 1:].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(raw.iloc[2:, 0]).rename("date")
    df.columns = range(df.shape[1])
    return df, codes


def drop_flat_tail(x: pd.Series, min_run: int = 3) -> pd.Series:
    """Datastream keeps repeating the last value after delisting. If the series ends with
    `min_run` or more identical values, keep the first and set the rest to NaN."""
    v = x.dropna()
    if len(v) < min_run:
        return x
    last = v.iloc[-1]
    diff = (v != last).to_numpy()[::-1]
    run = diff.argmax() if diff.any() else len(v)
    if run >= min_run:
        x = x.copy()
        x.loc[v.index[-run + 1:]] = np.nan
    return x


def firm_key(name: str) -> str:
    """Group share classes of the same firm: 'CARLSBERG A' and 'CARLSBERG B' -> 'CARLSBERG'."""
    n = re.split(r" DEAD| \(SUSP| \(", name)[0].replace("'", "").strip()
    return re.sub(r"( SER\.? ?\d| [AB])$", "", n).strip()


def capped_weights(w: pd.Series, cap: float) -> pd.Series:
    """Iteratively cap weights at `cap` and redistribute the excess pro rata."""
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        w[free] += excess * w[free] / w[free].sum()
    return w


# ---------------------------------------------------------------- main
def main() -> None:
    s = load_static()
    ri, ri_codes = load_series("DK_RI.csv")
    mv, mv_codes = load_series("DK_MV.csv")

    # ---- universe
    errors = ri_codes.isna() | mv_codes.isna()
    temp_line = s["name"].str.contains(r"NIL PAID|FULLY PAID|RIGHTS|NEW SHS|SPUN-OFF", regex=True)
    cse_equity = (
        (s["bourse_mnemonic"] == "CSE")
        & ~s["industry_group"].isin(["Unquoted equities", "Unclassified"])
        & ~errors & ~temp_line
    )
    s["firm"] = s["name"].map(firm_key)
    # return line per firm: the major line with the most return observations
    nobs = ri.notna().sum()
    major = s[cse_equity & (s["major_flag"] == "Y")].assign(nobs=nobs)
    ret_line = major.sort_values("nobs", ascending=False).groupby("firm").head(1)
    u = ret_line.index.sort_values()
    keep = s.index.isin(u)
    names = s.loc[u, "name"]
    novo_col = s.index[(s["type"] == NOVO) & keep][0]
    # all Copenhagen share classes of each firm (for firm-level market value)
    class_lines = s[cse_equity & s["firm"].isin(ret_line["firm"])]
    line_to_firm_col = class_lines["firm"].map(ret_line.reset_index().set_index("firm")["index"])
    multi = class_lines.groupby("firm").size()
    multi_class = class_lines[class_lines["firm"].isin(multi[multi > 1].index)][["firm", "name", "major_flag"]]
    multi_class.to_csv(OUT / "share_classes_merged.csv", index=False)

    universe_summary = pd.Series({
        "lines in static lists": len(s),
        "Copenhagen (CSE) lines": int((s["bourse_mnemonic"] == "CSE").sum()),
        "#ERROR in RI or MV": int(errors.sum()),
        "share-class lines merged into firms": len(multi_class),
        "final universe (firms)": len(u),
        "  of which active": int((s.loc[u, "status"] == "active").sum()),
        "  of which dead": int((s.loc[u, "status"] == "dead").sum()),
    })
    missing_in_sample = s.loc[errors & (s["bourse_mnemonic"] == "CSE")
                              & (s["major_flag"] == "Y")
                              & ~s["industry_group"].isin(["Unquoted equities"]),
                              ["name", "datastream_code", "isin_code", "status"]]
    missing_in_sample.to_csv(OUT / "missing_series_to_redownload.csv", index=False)

    # ---- clean returns; firm market value = sum over its Copenhagen share classes
    R = ri[u].apply(drop_flat_tail)
    lines = class_lines.index
    R_all = ri[lines].apply(drop_flat_tail)
    MV_lines = mv[lines].where(mv[lines] > 0).where(R_all.notna())   # nothing after a line dies
    M = MV_lines.T.groupby(line_to_firm_col).sum(min_count=1).T
    M = M.reindex(columns=u).where(R.notna())
    ret = R.pct_change(fill_method=None)
    lag_mv = M.shift(1)
    valid = ret.notna() & lag_mv.notna()
    ret = ret.where(valid).iloc[1:]
    lag_mv = lag_mv.where(valid).iloc[1:]
    valid = valid.iloc[1:]

    # ---- feasibility
    n_firms = valid.sum(axis=1)
    zero = (ret == 0)
    zero_share_stock = zero.sum() / valid.sum()
    stale_stocks = zero_share_stock[zero_share_stock > STALE_LIMIT].index
    n_liquid = valid.drop(columns=stale_stocks).sum(axis=1)

    coverage = {}
    for f in ["DK_UP.csv", "DK_VO.csv", "DK_NOSH.csv", "DK_NOSHFF.csv", "DKK_PA.csv", "DKK_PB.csv"]:
        x, _ = load_series(f)
        x = x[u].iloc[1:]
        coverage[f.replace(".csv", "")] = (x.notna() & valid).sum(axis=1) / n_firms

    pa, _ = load_series("DKK_PA.csv")
    pb, _ = load_series("DKK_PB.csv")
    spread = (pa[u] - pb[u]) / ((pa[u] + pb[u]) / 2)
    spread = spread.iloc[1:].where(valid & (spread.iloc[1:] > 0) & (spread.iloc[1:] < 1))

    feas = pd.DataFrame({
        "n_firms": n_firms,
        "n_firms_excl_stale": n_liquid,
        "zero_return_share": zero.sum(axis=1) / n_firms,
        "median_spread": spread.median(axis=1),
        "extreme_returns": ((ret > 3) | (ret < -0.9)).sum(axis=1),
    }).join(pd.DataFrame(coverage).add_prefix("cov_"))
    feas.to_csv(OUT / "feasibility_monthly.csv")

    # ---- concentration (weights at start of each month = lagged MV)
    w = lag_mv.div(lag_mv.sum(axis=1), axis=0)
    conc = pd.DataFrame({
        "novo_weight": w[novo_col],
        "top1_weight": w.max(axis=1),
        "top5_share": w.apply(lambda r: r.nlargest(5).sum(), axis=1),
        "top10_share": w.apply(lambda r: r.nlargest(10).sum(), axis=1),
        "hhi": (w ** 2).sum(axis=1),
    })
    conc["effective_n"] = 1 / conc["hhi"]
    conc["largest_stock"] = w.idxmax(axis=1).map(names)
    conc.to_csv(OUT / "concentration_monthly.csv")

    # ---- market return series
    vw = (w * ret).sum(axis=1)
    ew = ret.mean(axis=1)
    w_ex = lag_mv.drop(columns=novo_col)
    vw_ex = (w_ex.div(w_ex.sum(axis=1), axis=0) * ret.drop(columns=novo_col)).sum(axis=1)
    capped = pd.Series({d: (capped_weights(lag_mv.loc[d].dropna(), CAP)
                            * ret.loc[d].dropna()).sum() for d in ret.index})
    mkt = pd.DataFrame({"VW": vw, "EW": ew, "VW_ex_Novo": vw_ex, f"VW_capped_{int(CAP*100)}pct": capped,
                        "Novo": ret[novo_col]})
    mkt.to_csv(OUT / "market_returns_monthly.csv")

    def stats(r: pd.Series) -> pd.Series:
        r = r.dropna()
        cum = (1 + r).cumprod()
        ann_ret = cum.iloc[-1] ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        dd = (cum / cum.cummax() - 1).min()
        return pd.Series({"ann_return": ann_ret, "ann_vol": vol,
                          "return_per_vol": ann_ret / vol, "max_drawdown": dd,
                          "corr_with_Novo": r.corr(mkt["Novo"])})

    perf = mkt.apply(stats).T
    # sub-periods: before the Novo boom, the boom, the fall
    periods = {"2004-2016": ("2004", "2016"), "2017-Jun 2024 (boom)": ("2017", "2024-06"),
               "Jul 2024-2026 (fall)": ("2024-07", "2026")}
    sub = pd.concat({k: mkt.loc[a:b].apply(lambda r: (1 + r).prod() ** (12 / len(r)) - 1)
                     for k, (a, b) in periods.items()}, axis=1)
    novo_contrib = (w[novo_col] * ret[novo_col]).groupby(ret.index.year).sum()
    total_vw = vw.groupby(vw.index.year).sum()
    contrib = pd.DataFrame({"VW_return_sum": total_vw, "Novo_contribution": novo_contrib,
                            "Novo_share_of_VW": novo_contrib / total_vw})
    contrib.to_csv(OUT / "novo_contribution_by_year.csv")

    # ---- report
    pct = lambda x: f"{x:.1%}"
    lines = [
        "DANISH MARKET – FEASIBILITY AND CONCENTRATION",
        f"Sample: {ret.index[0]:%b %Y} – {ret.index[-1]:%b %Y} ({len(ret)} months)\n",
        "Universe", universe_summary.to_string(), "",
        "Feasibility",
        f"  firms per month: mean {n_firms.mean():.0f}, min {n_firms.min()} ({n_firms.idxmin():%b %Y}), "
        f"max {n_firms.max()}",
        f"  months below {MIN_FIRMS} firms: {(n_firms < MIN_FIRMS).sum()}",
        f"  excluding {len(stale_stocks)} stale stocks (> {STALE_LIMIT:.0%} zero months): "
        f"mean {n_liquid.mean():.0f}, min {n_liquid.min()}",
        f"  zero-return share of observations: {pct(zero.sum().sum() / valid.sum().sum())}",
        f"  extreme returns (>300% or <-90%): {int(feas['extreme_returns'].sum())}",
        "  coverage of valid obs: " + ", ".join(f"{k} {pct(v.mean())}" for k, v in coverage.items()),
        f"  median bid-ask spread: {pct(spread.median(axis=1).mean())} "
        f"(max year {spread.median(axis=1).groupby(spread.index.year).mean().idxmax()})",
        f"  series to re-download: {len(missing_in_sample)} (see missing_series_to_redownload.csv)", "",
        "Concentration",
        f"  Novo weight: {pct(conc.novo_weight.iloc[0])} (start) -> peak {pct(conc.novo_weight.max())} "
        f"({conc.novo_weight.idxmax():%b %Y}) -> {pct(conc.novo_weight.iloc[-1])} (end)",
        f"  top-5 share: {pct(conc.top5_share.iloc[0])} -> peak {pct(conc.top5_share.max())} -> "
        f"{pct(conc.top5_share.iloc[-1])}",
        f"  effective number of stocks: {conc.effective_n.iloc[0]:.1f} -> low {conc.effective_n.min():.1f} "
        f"({conc.effective_n.idxmin():%b %Y}) -> {conc.effective_n.iloc[-1]:.1f}",
        f"  months where Novo is the largest stock: {(conc.largest_stock == names[novo_col]).sum()} of {len(conc)}",
        "", "Market returns (full sample)", perf.round(3).to_string(),
        "", "Annualised return by period", sub.round(3).to_string(),
        "", "Novo's contribution to the VW return (sum of monthly contributions)",
        contrib.round(3).to_string(),
    ]
    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report)
    print(report)

    # ---- charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping charts")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(n_firms, label="All firms")
    ax.plot(n_liquid, label="Excl. stale stocks")
    ax.axhline(MIN_FIRMS, ls="--", c="grey", lw=1, label=f"{MIN_FIRMS}-firm threshold")
    ax.set(title="Danish stocks with a return and lagged market value", ylabel="Firms")
    ax.legend(frameon=False); ax.set_ylim(0)
    fig.tight_layout(); fig.savefig(OUT / "firms_per_month.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(conc.novo_weight, label="Novo Nordisk weight")
    ax.plot(conc.top5_share, label="Top-5 share")
    ax.axhline(CAP, ls="--", c="grey", lw=1, label=f"{CAP:.0%} cap")
    ax.set(title="Concentration of the Danish market", ylabel="Share of total market value")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1)); ax.set_ylim(0)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "concentration.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for c in ["VW", "EW", "VW_ex_Novo", f"VW_capped_{int(CAP*100)}pct"]:
        ax.plot((1 + mkt[c]).cumprod(), label=c.replace("_", " "))
    ax.set(title="Growth of 1 DKK", yscale="log")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "cumulative_returns.png", dpi=150); plt.close(fig)
    print(f"\nCharts and tables saved to {OUT}")


if __name__ == "__main__":
    main()
