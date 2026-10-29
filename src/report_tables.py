"""Numbers and tables for REAL_BENCHMARK_REPORT.md, computed from the run artefacts.

Writes ``figures/data/report_tables.md`` (markdown tables) and
``figures/data/report_numbers.json``.  Every comparison is paired over common
seeds, Wilcoxon signed-rank, Holm-corrected within one metric family.

    python -m src.report_tables
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .run_real_benchmark import merge_runs

OUT = Path("figures/data")
SYSTEM_DIRS = [Path("real_benchmark_outputs/llmrouterbench_v3"),
               Path("real_benchmark_outputs/llmrouterbench_v4_blocks")]
DEFAULT_W = -0.08
LEARNED = ("aspsa", "spsa", "zo_gt", "local_rls")


def paired(frame: pd.DataFrame, key: str, metric: str, reference: str, others, lower_is_better: bool):
    """Paired Wilcoxon of reference vs each other, Holm within this metric."""
    ref = frame[frame[key] == reference].set_index("seed")[metric]
    rows = []
    for other in others:
        cmp = frame[frame[key] == other].set_index("seed")[metric]
        both = pd.concat([ref.rename("r"), cmp.rename("o")], axis=1).dropna()
        diff = both["r"] - both["o"]
        p = float(stats.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
        better = int(((diff < 0) if lower_is_better else (diff > 0)).sum())
        rows.append({"other": other, "ref_mean": float(both["r"].mean()), "other_mean": float(both["o"].mean()),
                     "diff": float(diff.mean()),
                     "rel": float(diff.mean() / abs(both["o"].mean())) if both["o"].mean() else np.nan,
                     "p": p, "better_seeds": better, "n": int(len(diff))})
    order = np.argsort([r["p"] for r in rows])
    running, total = 0.0, len(rows)
    for rank, idx in enumerate(order, start=1):
        running = max(running, min(1.0, (total - rank + 1) * rows[idx]["p"]))
        rows[idx]["p_holm"] = running
    return rows


def _fmt_rows(rows, pct=False):
    out = []
    for r in rows:
        diff = f"{100 * r['rel']:+.1f} %" if pct else f"{r['diff']:+.4f}"
        out.append(f"| {r['other']} | {r['ref_mean']:.4f} | {r['other_mean']:.4f} | {diff} | "
                   f"{r['better_seeds']}/{r['n']} | {r['p_holm']:.4f} |")
    return out


def system_section(md: list[str], numbers: dict) -> None:
    dirs = [d for d in SYSTEM_DIRS if d.exists()]
    runs, provenance = merge_runs(dirs)
    numbers["system_sources"] = provenance
    runs = runs[np.isclose(runs["price_weight"], DEFAULT_W)]
    for regime in ("api", "capacity", "stress"):
        part = runs[runs["regime"] == regime]
        if part.empty:
            continue
        md.append(f"\n### System, regime `{regime}` (π = 0.08, 10 seeds)\n")
        cols = ["mean_quality", "mean_cost", "deadline_hit_rate", "goodput_per_s", "p95_latency_s"]
        table = part.groupby("system")[cols].mean().sort_values("mean_quality", ascending=False)
        md.append("| system | quality | $/req | DHR | goodput/s | p95 s |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for name, row in table.iterrows():
            md.append(f"| {name} | {row.mean_quality:.4f} | {row.mean_cost:.4f} | {row.deadline_hit_rate:.4f} | "
                      f"{row.goodput_per_s:.4f} | {row.p95_latency_s:.0f} |")
        numbers[f"system_{regime}"] = table.round(6).to_dict(orient="index")
        others = [s for s in part["system"].unique() if s != "aspsa"]
        for metric, lower in (("mean_quality", False), ("deadline_hit_rate", False),
                              ("goodput_per_s", False), ("p95_latency_s", True)):
            rows = paired(part, "system", metric, "aspsa", others, lower)
            numbers[f"system_{regime}_{metric}_vs"] = rows
            md.append(f"\nA-SPSA system vs others, `{metric}`:\n")
            md.append("| other | A-SPSA | other | diff | A-SPSA better | p_holm |")
            md.append("|---|---:|---:|---:|---:|---:|")
            md.extend(_fmt_rows(rows))
        if regime == "api":
            ref = part.iloc[0]
            numbers["api_references"] = {
                "openrouter_quality": float(part["reference_router_quality"].mean()),
                "openrouter_cost": float(part["reference_router_cost_usd"].mean()),
                "best_single_quality": float(part["best_single_quality"].mean()),
                "best_single_cost": float(part["best_single_cost_usd"].mean()),
                "oracle_quality": float(part["oracle_quality"].mean()),
                "oracle_cost": float(part["oracle_cost_usd"].mean()),
            }


def prediction_section(md: list[str], numbers: dict) -> None:
    source = SYSTEM_DIRS[1] if SYSTEM_DIRS[1].exists() else SYSTEM_DIRS[0]
    manifest = json.loads((source / "experiment_manifest.json").read_text(encoding="utf-8"))
    numbers["prediction_source"] = str(source)
    numbers["prediction_selected_steps"] = manifest.get("selected_steps")
    numbers["prediction_prior"] = manifest.get("selected_rls_prior_strength")
    metrics = (("phi_p_auc", "Φ_p"), ("phi_w_auc", "Φ_w"), ("phi_q_auc", "Φ_q"),
               ("f1", "F1 out of sample"), ("f2", "F2 out of sample"), ("f3", "F3 out of sample"),
               ("true_processing_mape", "MAPE time, held-out"), ("true_quality_rmse", "RMSE quality, held-out"))
    for regime in ("api", "capacity", "stress"):
        path = source / f"runs_{regime}.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        part = part[np.isclose(part["price_weight"], DEFAULT_W) & part["system"].isin(LEARNED)]
        md.append(f"\n### Prediction layer, regime `{regime}` ({source.name})\n")
        md.append("| metric | " + " | ".join(LEARNED) + " |")
        md.append("|---|" + "---:|" * len(LEARNED))
        for metric, name in metrics:
            means = part.groupby("system")[metric].mean()
            md.append(f"| {name} | " + " | ".join(f"{means.get(s, np.nan):.4f}" for s in LEARNED) + " |")
        for metric, name in metrics:
            rows = paired(part, "system", metric, "aspsa", [s for s in LEARNED if s != "aspsa"], True)
            numbers[f"prediction_{regime}_{metric}_vs"] = rows
            md.append(f"\nA-SPSA vs others, {name} (lower is better):\n")
            md.append("| other | A-SPSA | other | rel. diff | A-SPSA better | p_holm |")
            md.append("|---|---:|---:|---:|---:|---:|")
            md.extend(_fmt_rows(rows, pct=True))


EQUAL_STEP_DIR = Path("real_benchmark_outputs/llmrouterbench_v5_equal_step")
EQUAL_STEP_METRICS = ("f1", "true_processing_mape", "phi_p_auc", "f2", "f3", "phi_w_auc", "phi_q_auc")


def _holm(rows: list[dict]) -> None:
    order = np.argsort([r["p"] for r in rows])
    running, total = 0.0, len(rows)
    for rank, idx in enumerate(order, start=1):
        running = max(running, min(1.0, (total - rank + 1) * rows[idx]["p"]))
        rows[idx]["p_holm"] = running


def _lower(ref: pd.Series, other: pd.Series) -> dict:
    """Paired Wilcoxon of A-SPSA against one other system; lower is better."""
    both = pd.concat([ref.rename("r"), other.rename("o")], axis=1).dropna()
    diff = both["r"] - both["o"]
    p = float(stats.wilcoxon(diff).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
    return {"aspsa": float(both["r"].mean()), "other_mean": float(both["o"].mean()),
            "rel": float(100 * diff.mean() / abs(both["o"].mean())), "aspsa_lower": int((diff < 0).sum()),
            "n": int(len(diff)), "p": p}


def equal_step_section(md: list[str], numbers: dict) -> None:
    """Prediction layer with every method at one common step h.

    A-SPSA against SPSA and ZO-GT: Holm within (regime, metric) over every
    step of the run and both competitors.  A-SPSA at h = 0.01 against RLS
    alone (the v4 rows, same seeds and prior): Holm over regimes and metrics.
    """
    if not (EQUAL_STEP_DIR / "runs_api.csv").exists():
        return
    rows, versus_rls = [], []
    for regime in ("api", "capacity", "stress"):
        runs = pd.read_csv(EQUAL_STEP_DIR / f"runs_{regime}.csv")
        v4 = pd.read_csv(SYSTEM_DIRS[1] / f"runs_{regime}.csv")
        rls = v4[(v4["system"] == "local_rls") & np.isclose(v4["price_weight"], DEFAULT_W)].set_index("seed")
        for metric in EQUAL_STEP_METRICS:
            family = []
            for step in sorted(runs["common_step"].unique()):
                part = runs[np.isclose(runs["common_step"], step)]
                ref = part[part["system"] == "aspsa"].set_index("seed")[metric]
                for other in ("spsa", "zo_gt"):
                    cmp = part[part["system"] == other].set_index("seed")[metric]
                    family.append({"regime": regime, "h": float(step), "metric": metric, "other": other,
                                   **_lower(ref, cmp)})
                if np.isclose(step, 0.01) and metric in ("f1", "true_processing_mape", "phi_p_auc"):
                    versus_rls.append({"regime": regime, "h": float(step), "metric": metric,
                                       "other": "local_rls", **_lower(ref, rls[metric])})
            _holm(family)
            rows.extend(family)
    _holm(versus_rls)
    pd.DataFrame(rows).to_csv(OUT / "v5_equal_step_tests.csv", index=False)
    pd.DataFrame(versus_rls).to_csv(OUT / "v5_vs_rls_tests.csv", index=False)
    numbers["equal_step_tests"] = rows
    numbers["equal_step_vs_rls"] = versus_rls
    md.append(f"\n### Prediction layer, equal step ({EQUAL_STEP_DIR.name})\n")
    md.append("| regime | h | metric | vs | A-SPSA | other | rel. diff | A-SPSA lower | p_holm |")
    md.append("|---|---:|---|---|---:|---:|---:|---:|---:|")
    for r in rows + versus_rls:
        md.append(f"| {r['regime']} | {r['h']:g} | {r['metric']} | {r['other']} | {r['aspsa']:.4f} | "
                  f"{r['other_mean']:.4f} | {r['rel']:+.1f} % | {r['aspsa_lower']}/{r['n']} | "
                  f"{r['p_holm']:.4f} |")


RUNS = Path("real_benchmark_outputs")
V6_MLP = RUNS / "llmrouterbench_v6_mlp"
V6_ORDERS = RUNS / "llmrouterbench_v6_orders"
SPROUT_DIRS = [RUNS / "sprout_v1", RUNS / "sprout_v1_optimizers"]
LATENCY_ROOT = RUNS / "latency_sensitivity"
LATENCY_VARIANTS = ("default", "slow", "fast", "shuffle1", "shuffle2", "shuffle3")
# (metric, lower is better) at the default operating point
SYSTEM_METRICS = (("mean_quality", False), ("mean_cost", True), ("deadline_hit_rate", False),
                  ("goodput_per_s", False), ("p95_latency_s", True))


def _versus(md: list[str], numbers: dict, key: str, part: pd.DataFrame, reference: str, others,
            metrics=SYSTEM_METRICS) -> None:
    """Paired tests of one system against several, Holm within each metric."""
    others = [o for o in others if o in set(part["system"])]
    if reference not in set(part["system"]) or not others:
        return
    md.append(f"\n{reference} against {', '.join(others)}:\n")
    md.append("| metric | other | reference | other | diff | reference better | p_holm |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for metric, lower in metrics:
        rows = paired(part, "system", metric, reference, others, lower)
        numbers[f"{key}_{reference}_{metric}"] = rows
        for r in rows:
            md.append(f"| {metric} | {r['other']} | {r['ref_mean']:.4f} | {r['other_mean']:.4f} | "
                      f"{r['diff']:+.4f} | {r['better_seeds']}/{r['n']} | {r['p_holm']:.4f} |")


def _means(md: list[str], part: pd.DataFrame, systems) -> None:
    cols = [m for m, _ in SYSTEM_METRICS]
    table = part[part["system"].isin(systems)].groupby("system")[cols].mean()
    md.append("| system | quality | $/req | DHR | goodput/s | p95 s |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in [s for s in systems if s in table.index]:
        row = table.loc[name]
        md.append(f"| {name} | {row.mean_quality:.4f} | {row.mean_cost:.4f} | {row.deadline_hit_rate:.4f} | "
                  f"{row.goodput_per_s:.4f} | {row.p95_latency_s:.0f} |")


def trained_router_section(md: list[str], numbers: dict) -> None:
    """The MLP router and the architecture with the MLP estimate as its prior."""
    if not (V6_MLP / "runs_api.csv").exists():
        return
    runs, _ = merge_runs([d for d in SYSTEM_DIRS + [V6_MLP] if d.exists()])
    runs = runs[np.isclose(runs["price_weight"], DEFAULT_W)]
    systems = ["aspsa", "aspsa_mlp", "mlp_router", "knn_router", "static_domain"]
    for regime in ("api", "capacity", "stress"):
        part = runs[runs["regime"] == regime]
        md.append(f"\n### Trained router, regime `{regime}` (π = 0.08, 10 seeds)\n")
        _means(md, part, systems)
        _versus(md, numbers, f"router_{regime}", part, "aspsa_mlp", ["mlp_router", "aspsa", "knn_router"])
        _versus(md, numbers, f"router_{regime}", part, "aspsa", ["mlp_router"])


def orders_section(md: list[str], numbers: dict) -> None:
    """The weighted buffer against the classic rules, same predictors (A-SPSA, h = 0.01)."""
    if not (V6_ORDERS / "runs_capacity.csv").exists():
        return
    for regime in ("capacity", "stress"):
        base = pd.read_csv(EQUAL_STEP_DIR / f"runs_{regime}.csv")
        base = base[(base["system"] == "aspsa") & np.isclose(base["common_step"], 0.01)]
        part = pd.concat([base, pd.read_csv(V6_ORDERS / f"runs_{regime}.csv")], ignore_index=True)
        md.append(f"\n### Buffer order, regime `{regime}` (A-SPSA, h = 0.01, 10 seeds)\n")
        systems = ["aspsa", "aspsa_edf", "aspsa_llf", "aspsa_sjf"]
        _means(md, part, systems)
        _versus(md, numbers, f"orders_{regime}", part, "aspsa", systems[1:])


def sprout_section(md: list[str], numbers: dict) -> None:
    """Second benchmark, nothing tuned on it but the shared RLS prior."""
    dirs = [d for d in SPROUT_DIRS if (d / "runs_api.csv").exists()]
    if not dirs:
        return
    runs, _ = merge_runs(dirs)
    frontier = pd.read_csv(SPROUT_DIRS[0] / "frontier_by_seed.csv") if (SPROUT_DIRS[0] / "frontier_by_seed.csv").exists() else None
    default = runs[np.isclose(runs["price_weight"], DEFAULT_W)]
    systems = ["aspsa", "aspsa_mlp", "local_rls", "knn_router", "mlp_router", "static_domain", "quality_greedy",
               "cost_greedy", "shortest_queue", "random", "oracle"]
    for regime in ("api", "capacity", "stress"):
        part = default[default["regime"] == regime]
        md.append(f"\n### SPROUT, regime `{regime}` (π = 0.08, 10 seeds)\n")
        _means(md, part, systems)
        numbers[f"sprout_{regime}_references"] = {k: float(part[k].mean()) for k in (
            "best_single_quality", "best_single_cost_usd", "oracle_quality", "oracle_cost_usd") if k in part}
        _versus(md, numbers, f"sprout_{regime}", part, "aspsa",
                ["knn_router", "mlp_router", "static_domain", "quality_greedy", "local_rls"])
        _versus(md, numbers, f"sprout_{regime}", part, "aspsa_mlp", ["mlp_router"])
        if frontier is not None:
            aiq = frontier[frontier["regime"] == regime].rename(columns={"aiq": "aiq_value"})
            _versus(md, numbers, f"sprout_{regime}_aiq", aiq, "aspsa", ["knn_router", "mlp_router"],
                    metrics=(("aiq_value", False),))
        prediction = part[part["system"].isin(["aspsa", "spsa", "zo_gt", "local_rls"])]
        _versus(md, numbers, f"sprout_{regime}_prediction", prediction, "aspsa", ["spsa", "zo_gt", "local_rls"],
                metrics=(("f1", True), ("true_processing_mape", True), ("phi_p_auc", True)))


def latency_section(md: list[str], numbers: dict) -> None:
    """The system against its main baselines under other latency profiles."""
    variants = [v for v in LATENCY_VARIANTS if (LATENCY_ROOT / v / "runs_capacity.csv").exists()]
    if not variants:
        return
    md.append("\n### Latency-profile sensitivity (A-SPSA h = 0.01, π = 0.08, 10 seeds)\n")
    md.append("| variant | regime | DHR ours | DHR kNN | DHR static | p95 ours | p95 kNN | p95 static | "
              "DHR vs kNN better | p_holm | DHR vs static better | p_holm |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        for regime in ("capacity", "stress"):
            path = LATENCY_ROOT / variant / f"runs_{regime}.csv"
            if not path.exists():
                continue
            part = pd.read_csv(path)
            rows = paired(part, "system", "deadline_hit_rate", "aspsa", ["knn_router", "static_domain"], False)
            tail = paired(part, "system", "p95_latency_s", "aspsa", ["knn_router", "static_domain"], True)
            numbers[f"latency_{variant}_{regime}_dhr"] = rows
            numbers[f"latency_{variant}_{regime}_p95"] = tail
            by = {r["other"]: r for r in rows}
            p95 = {r["other"]: r for r in tail}
            md.append(f"| {variant} | {regime} | {by['knn_router']['ref_mean']:.4f} | "
                      f"{by['knn_router']['other_mean']:.4f} | {by['static_domain']['other_mean']:.4f} | "
                      f"{p95['knn_router']['ref_mean']:.0f} | {p95['knn_router']['other_mean']:.0f} | "
                      f"{p95['static_domain']['other_mean']:.0f} | {by['knn_router']['better_seeds']}/10 | "
                      f"{by['knn_router']['p_holm']:.4f} | {by['static_domain']['better_seeds']}/10 | "
                      f"{by['static_domain']['p_holm']:.4f} |")


POLICY_SETS = {
    "five weights": [("fixed step", RUNS / "policy_real_trace"), ("tuned step", RUNS / "policy_real_tuned_wide"),
                     ("tuned step", RUNS / "policy_real_tuned_noisy"), ("ablation", RUNS / "policy_real_ablation"),
                     ("rules", RUNS / "policy_real_rules")],
    "six weights": [("fixed step", RUNS / "policy6_trace"), ("tuned step", RUNS / "policy6_tuned"),
                    ("tuned step", RUNS / "policy6_tuned_noisy"), ("ablation", RUNS / "policy6_ablation")],
}
RULES = ("edf", "llf", "sjf")


def _against_rules(md: list[str], numbers: dict, key: str, raw: pd.DataFrame) -> None:
    """Optimizer seeds against each deterministic rule: seeds above it and a one-sample Wilcoxon."""
    rules = raw[raw["method"].isin(RULES + ("hand-set defaults",))].groupby("method")["deadline_hit_rate"].mean()
    if rules.empty:
        return
    md.append("\n| reference | DHR | " + " | ".join(f"{m} ({n}) seeds above, p" for n in ("low_noise", "noisy")
                                                   for m in ("aspsa",)) + " |")
    md.append("|---|---:|---:|---:|")
    for name, value in rules.items():
        cells = []
        for noise in ("low_noise", "noisy"):
            seeds = raw[(raw["protocol"] == "fixed step") & (raw["noise"] == noise) & (raw["method"] == "aspsa")]
            diff = seeds["deadline_hit_rate"].to_numpy() - value
            if len(diff) == 0:
                cells.append("—")
                continue
            p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            cells.append(f"{int((diff > 0).sum())}/{len(diff)}, p = {p:.4f}")
            numbers[f"{key}_{noise}_aspsa_vs_{name}"] = {"rule": float(value), "aspsa": float(diff.mean() + value),
                                                         "above": int((diff > 0).sum()), "n": len(diff), "p": p}
        md.append(f"| {name} | {value:.4f} | " + " | ".join(cells) + " |")


def policy_section(md: list[str], numbers: dict) -> None:
    for name, dirs in POLICY_SETS.items():
        _policy_set(md, numbers, name, dirs)


def _policy_set(md: list[str], numbers: dict, name: str, dirs) -> None:
    frames = []
    for label, directory in dirs:
        path = directory / "policy_real_raw.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(protocol=label))
    if not frames:
        return
    tag = "policy" if name == "five weights" else "policy6"
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"optimizer_seed": "seed"})
    md.append(f"\n## Policy layer, {name}\n")
    _against_rules(md, numbers, f"{tag}_rules", raw)
    for protocol in ("fixed step", "tuned step"):
        part = raw[raw["protocol"] == protocol]
        if part.empty:
            continue
        for noise in sorted(n for n in part["noise"].unique() if n != "reference"):
            sub = part[part["noise"] == noise]
            if protocol == "fixed step":
                abl = raw[(raw["protocol"] == "ablation") & (raw["noise"] == noise)]
                sub = pd.concat([sub, abl], ignore_index=True)
            summary = sub.groupby("method")[["deadline_hit_rate", "goodput", "p95_wait", "mean_quality",
                                              "mean_cost"]].agg(["mean", "std", "min", "max"])
            md.append(f"\n### Policy layer ({name}), {protocol}, {noise} oracle\n")
            md.append("| method | DHR mean | std | min | max | goodput | p95 wait | quality |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            order = summary[("deadline_hit_rate", "mean")].sort_values(ascending=False).index
            for m in order:
                r = summary.loc[m]
                md.append(f"| {m} | {r[('deadline_hit_rate','mean')]:.4f} | {r[('deadline_hit_rate','std')]:.4f} | "
                          f"{r[('deadline_hit_rate','min')]:.4f} | {r[('deadline_hit_rate','max')]:.4f} | "
                          f"{r[('goodput','mean')]:.3f} | {r[('p95_wait','mean')]:.3f} | {r[('mean_quality','mean')]:.4f} |")
            numbers[f"{tag}_{protocol}_{noise}"] = {
                m: {k: float(summary.loc[m][(k, "mean")]) for k in ("deadline_hit_rate", "goodput", "p95_wait")}
                for m in order}
            if "aspsa" in set(sub["method"]):
                for metric, lower in (("deadline_hit_rate", False), ("goodput", False), ("p95_wait", True)):
                    rows = paired(sub, "method", metric, "aspsa",
                                  [m for m in sub["method"].unique() if m != "aspsa"], lower)
                    numbers[f"{tag}_{protocol}_{noise}_{metric}_vs"] = rows
                    md.append(f"\nA-SPSA vs others, `{metric}`:\n")
                    md.append("| other | A-SPSA | other | diff | A-SPSA better | p_holm |")
                    md.append("|---|---:|---:|---:|---:|---:|")
                    md.extend(_fmt_rows(rows))
    for label, directory in dirs:
        if label == "tuned step" and (directory / "experiment_manifest.json").exists():
            manifest = json.loads((directory / "experiment_manifest.json").read_text(encoding="utf-8"))
            numbers[f"{tag}_selected_steps_{directory.name}"] = manifest.get("selected_steps")
            md.append(f"\nSelected steps ({directory.name}): `{manifest.get('selected_steps')}`\n")
    fixed = raw[raw["protocol"] == "fixed step"]
    if "trace" in fixed.columns:
        md.append(f"\n### Convergence on the training objective ({name}, fixed step)\n")
        md.append("| noise | method | mean objective over rounds | objective, last 5 rounds | "
                  "first round within 0.01 of own final |")
        md.append("|---|---|---:|---:|---:|")
        conv = {}
        learned = fixed[~fixed["method"].isin(RULES + ("hand-set defaults",))]
        for (noise, method), group in learned.groupby(["noise", "method"]):
            traces = np.array([json.loads(t) for t in group["trace"]], dtype=float)
            mean_curve = traces.mean(axis=0)
            final = mean_curve[-5:].mean()
            within = int(np.argmax(np.convolve(mean_curve, np.ones(3) / 3, "same") <= final + 0.01)) + 1
            conv[f"{noise}/{method}"] = {"auc": float(mean_curve.mean()), "final": float(final), "round": within}
            md.append(f"| {noise} | {method} | {mean_curve.mean():.4f} | {final:.4f} | {within} |")
        numbers[f"{tag}_convergence"] = conv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Report tables (generated by src.report_tables)"]
    numbers: dict = {}
    system_section(md, numbers)
    prediction_section(md, numbers)
    equal_step_section(md, numbers)
    trained_router_section(md, numbers)
    orders_section(md, numbers)
    sprout_section(md, numbers)
    latency_section(md, numbers)
    policy_section(md, numbers)
    (OUT / "report_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "report_numbers.json").write_text(json.dumps(numbers, indent=2, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
