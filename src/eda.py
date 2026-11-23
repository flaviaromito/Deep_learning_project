"""Analisi Esplorativa dei Dati (EDA) per il progetto di Deep Learning.

Analizza tutti e 4 gli split (train, id, ood_s, ood_t) in DATA/:
- volumi, aree geografiche, overlap tra split e stagionalità;
- lunghezze delle sequenze variabili, orizzonti temporali effettivi e coerenza maschere;
- percentuali di NaN per feature in history e future;
- classi climatiche di Köppen-Geiger e shift OOD spaziale/temporale;
- distribuzioni dei target (scala originale e standardization_arcsinh), correlazioni
  e benchmark immediato di persistenza;
- entità del rumore nel meteo futuro (future_noise).

Uso:
    python3 src/eda.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config


def apply_arcsinh_scaler(x: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float) -> np.ndarray:
    z = (x - mean) / np.sqrt(var + eps)
    return np.arcsinh(z)


def inspect_split(split_name: str, split_dir: Path, feature_names: list[str], target_names: list[str]) -> dict:
    npz_files = sorted(split_dir.glob("*.npz"))
    areas = [p.stem for p in npz_files]

    area_counts = {}
    climate_counter = Counter()
    month_counter = Counter()
    year_counter = Counter()

    hist_lens = []
    fut_lens = []
    target_positions_list = []
    hist_clear_counts = []
    last_hist_valid = 0
    first_hist_valid = 0

    n_samples = 0
    hist_total_rows = 0
    fut_total_rows = 0
    hist_nan_counts = np.zeros(len(feature_names), dtype=np.int64)
    fut_nan_counts = np.zeros(len(feature_names), dtype=np.int64)

    mask_mismatch_hist = 0
    mask_mismatch_fut = 0
    target_nan_count = 0
    target_mask_true_count = 0

    all_targets = []
    all_last_hist_targets = []

    # Statistiche incrementali sul train/split per feature grezze (somma, somma quadrati, conteggio)
    feat_sum = np.zeros(len(feature_names), dtype=np.float64)
    feat_sq_sum = np.zeros(len(feature_names), dtype=np.float64)
    feat_count = np.zeros(len(feature_names), dtype=np.int64)
    feat_min = np.full(len(feature_names), np.inf, dtype=np.float64)
    feat_max = np.full(len(feature_names), -np.inf, dtype=np.float64)

    # Analisi rumore futuro: std per colonna e per posizione temporale (giorno 1..30)
    noise_sq_sum = np.zeros(len(feature_names), dtype=np.float64)
    noise_count = np.zeros(len(feature_names), dtype=np.int64)
    noise_by_day_sq = np.zeros((35, 11), dtype=np.float64)
    noise_by_day_cnt = np.zeros((35, 11), dtype=np.int64)

    # Campioni per grafici (subsampling deterministico)
    sample_examples = []
    hist_vals_sample = []

    min_ts = None
    max_ts = None

    for npz_path in npz_files:
        area = npz_path.stem
        with np.load(npz_path, allow_pickle=True) as d:
            h_arr = d["history"]
            hm_arr = d["history_mask"]
            hts_arr = d["history_timestamps"]
            f_arr = d["future"]
            fm_arr = d["future_mask"]
            fn_arr = d["future_noise"]
            fts_arr = d["future_timestamps"]
            t_arr = d["target"]          # (N, 3, 3)
            tm_arr = d["target_mask"]    # (N, 3, 3)
            pos_arr = d["future_target_positions"]  # (N, 3)
            clim_arr = d["climate"]

            n = len(t_arr)
            area_counts[area] = n
            n_samples += n

            climate_counter.update(clim_arr.tolist())
            target_positions_list.append(pos_arr)
            all_targets.append(t_arr.astype(np.float32))
            target_nan_count += int(np.isnan(t_arr).sum())
            target_mask_true_count += int(tm_arr.sum())

            last_h_vals = np.full((n, 3), np.nan, dtype=np.float32)

            for i in range(n):
                h = h_arr[i]
                hm = hm_arr[i]
                f = f_arr[i]
                fm = fm_arr[i]
                fn = fn_arr[i]

                hl = h.shape[0]
                fl = f.shape[0]
                hist_lens.append(hl)
                fut_lens.append(fl)
                hist_total_rows += hl
                fut_total_rows += fl

                h_isnan = np.isnan(h)
                f_isnan = np.isnan(f)
                hist_nan_counts += h_isnan.sum(axis=0)
                fut_nan_counts += f_isnan.sum(axis=0)

                if not np.array_equal(h_isnan, hm):
                    mask_mismatch_hist += 1
                if not np.array_equal(f_isnan, fm):
                    mask_mismatch_fut += 1

                # Quanti clear-sky validi in history (controlliamo colonna 18 = NDVI)
                valid_idx = np.where(~h_isnan[:, 18])[0]
                hist_clear_counts.append(len(valid_idx))
                if len(valid_idx) > 0:
                    if valid_idx[0] == 0:
                        first_hist_valid += 1
                    if valid_idx[-1] == hl - 1:
                        last_hist_valid += 1
                    last_h_vals[i] = h[valid_idx[-1], 18:21]

                # Statistiche feature su history
                valid_mask = ~h_isnan
                h_zero = np.where(valid_mask, h, 0.0)
                feat_sum += h_zero.sum(axis=0)
                feat_sq_sum += (h_zero ** 2).sum(axis=0)
                feat_count += valid_mask.sum(axis=0)
                with np.errstate(invalid="ignore"):
                    h_for_min = np.where(valid_mask, h, np.inf)
                    h_for_max = np.where(valid_mask, h, -np.inf)
                    feat_min = np.minimum(feat_min, h_for_min.min(axis=0))
                    feat_max = np.maximum(feat_max, h_for_max.max(axis=0))

                # Rumore futuro
                fn_valid = ~np.isnan(fn)
                fn_zero = np.where(fn_valid, fn, 0.0)
                noise_sq_sum += (fn_zero ** 2).sum(axis=0)
                noise_count += fn_valid.sum(axis=0)
                d_lim = min(fl, 35)
                noise_by_day_sq[:d_lim] += (fn_zero[:d_lim, :11] ** 2)
                noise_by_day_cnt[:d_lim] += fn_valid[:d_lim, :11]

                # Date (prendiamo primo e ultimo timestamp della finestra)
                t0 = str(hts_arr[i][0])[:10]
                t_end = str(fts_arr[i][-1])[:10]
                t_now = str(hts_arr[i][-1])[:10]
                if min_ts is None or t0 < min_ts:
                    min_ts = t0
                if max_ts is None or t_end > max_ts:
                    max_ts = t_end
                year_counter[t_now[:4]] += 1
                month_counter[int(t_now[5:7])] += 1

                # Subsample per plot distribuzioni feature (solo train, ~1 campione ogni 100)
                if split_name == "train" and (i % 100 == 0):
                    hist_vals_sample.append(h)

                # Salva qualche esempio rappresentativo dal train per visualizzazione traiettorie
                if split_name == "train" and len(sample_examples) < 6 and i == 0 and hl >= 15 and fl >= 15:
                    sample_examples.append({
                        "area": area,
                        "climate": str(clim_arr[i]),
                        "history": h.copy(),
                        "future": f.copy(),
                        "target": t_arr[i].copy(),
                        "pos": pos_arr[i].copy(),
                        "history_timestamps": [str(x)[:10] for x in hts_arr[i]],
                        "future_timestamps": [str(x)[:10] for x in fts_arr[i]],
                    })

            all_last_hist_targets.append(last_h_vals)

    targets_all = np.concatenate(all_targets, axis=0)            # (N, 3, 3)
    last_hist_all = np.concatenate(all_last_hist_targets, axis=0)  # (N, 3)
    positions_all = np.concatenate(target_positions_list, axis=0)  # (N, 3)

    # Persistenza: previsione = ultimo valore clear-sky osservato in history
    # Confronto su step 1, 2, 3 e media globale
    diff = targets_all - last_hist_all[:, None, :]  # (N, 3, 3)
    persistence_mae = {}
    persistence_rmse = {}
    for t_idx, t_name in enumerate(target_names):
        short_name = t_name.replace("avg_", "").replace("_clear_sky", "")
        step_maes = [float(np.nanmean(np.abs(diff[:, s, t_idx]))) for s in range(3)]
        step_rmses = [float(np.sqrt(np.nanmean(diff[:, s, t_idx] ** 2))) for s in range(3)]
        persistence_mae[short_name] = {
            "step_1": step_maes[0],
            "step_2": step_maes[1],
            "step_3": step_maes[2],
            "overall": float(np.nanmean(np.abs(diff[:, :, t_idx]))),
        }
        persistence_rmse[short_name] = {
            "step_1": step_rmses[0],
            "step_2": step_rmses[1],
            "step_3": step_rmses[2],
            "overall": float(np.sqrt(np.nanmean(diff[:, :, t_idx] ** 2))),
        }

    # Statistiche target sulla scala originale
    target_stats = {}
    for t_idx, t_name in enumerate(target_names):
        short_name = t_name.replace("avg_", "").replace("_clear_sky", "")
        vals = targets_all[:, :, t_idx].ravel()
        vals = vals[~np.isnan(vals)]
        target_stats[short_name] = {
            "min": float(np.min(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
            "out_of_bounds": int(np.sum((vals < -1.0) | (vals > 1.0))),
        }

    # Correlazione tra i 3 target
    flat_targets = targets_all.reshape(-1, 3)
    valid_t_rows = ~np.isnan(flat_targets).any(axis=1)
    target_corr = np.corrcoef(flat_targets[valid_t_rows], rowvar=False).tolist()

    hist_lens_np = np.array(hist_lens)
    fut_lens_np = np.array(fut_lens)
    horizons_days = positions_all + 1  # giorni dopo history_end (1-indexed)

    def pct_stats(arr: np.ndarray) -> dict:
        return {
            "min": int(np.min(arr)),
            "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
        }

    feat_mean = feat_sum / np.maximum(feat_count, 1)
    feat_var = np.maximum((feat_sq_sum / np.maximum(feat_count, 1)) - feat_mean ** 2, 0.0)
    noise_rms = np.sqrt(noise_sq_sum / np.maximum(noise_count, 1))
    noise_by_day_rms = np.sqrt(noise_by_day_sq / np.maximum(noise_by_day_cnt, 1))

    return {
        "split": split_name,
        "n_areas": len(areas),
        "areas": areas,
        "area_counts": area_counts,
        "n_samples": n_samples,
        "date_range": {"min": min_ts, "max": max_ts},
        "years": dict(sorted(year_counter.items())),
        "months": dict(sorted(month_counter.items())),
        "climate_counts": dict(climate_counter.most_common()),
        "hist_len_stats": pct_stats(hist_lens_np),
        "fut_len_stats": pct_stats(fut_lens_np),
        "horizon_days_stats": {
            f"step_{s+1}": pct_stats(horizons_days[:, s]) for s in range(3)
        },
        "hist_clear_counts": dict(Counter(hist_clear_counts).most_common()),
        "first_hist_valid_pct": float(100.0 * first_hist_valid / max(n_samples, 1)),
        "last_hist_valid_pct": float(100.0 * last_hist_valid / max(n_samples, 1)),
        "mask_mismatch_hist": mask_mismatch_hist,
        "mask_mismatch_fut": mask_mismatch_fut,
        "target_nan_count": target_nan_count,
        "target_mask_true_count": target_mask_true_count,
        "hist_nan_pct": {
            name: float(100.0 * hist_nan_counts[idx] / max(hist_total_rows, 1))
            for idx, name in enumerate(feature_names)
        },
        "fut_nan_pct": {
            name: float(100.0 * fut_nan_counts[idx] / max(fut_total_rows, 1))
            for idx, name in enumerate(feature_names)
        },
        "target_stats": target_stats,
        "target_corr": target_corr,
        "persistence_mae": persistence_mae,
        "persistence_rmse": persistence_rmse,
        "feature_stats": {
            name: {
                "mean": float(feat_mean[idx]),
                "std": float(np.sqrt(feat_var[idx])),
                "min": float(feat_min[idx]),
                "max": float(feat_max[idx]),
            }
            for idx, name in enumerate(feature_names)
        },
        "noise_rms": {name: float(noise_rms[idx]) for idx, name in enumerate(feature_names)},
        "noise_by_day_rms": noise_by_day_rms,
        # Arrays per plotting (non salvati nel JSON)
        "_hist_lens": hist_lens_np,
        "_fut_lens": fut_lens_np,
        "_horizons_days": horizons_days,
        "_targets": targets_all,
        "_sample_examples": sample_examples,
        "_hist_vals_sample": np.concatenate(hist_vals_sample, axis=0) if hist_vals_sample else None,
    }


def make_plots(results: dict[str, dict], feature_names: list[str], scaler: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    splits = ["train", "id", "ood_s", "ood_t"]
    split_labels = {"train": "Train", "id": "ID (2020)", "ood_s": "OOD-S (Spaziale)", "ood_t": "OOD-T (Temporale)"}
    colors = {"train": "#1b5e20", "id": "#0277bd", "ood_s": "#ef6c00", "ood_t": "#8e24aa"}

    # ------------------------------------------------------------------
    # Plot 1: Lunghezze sequenze (history, future) e orizzonti in giorni
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for s in splits:
        sns.kdeplot(results[s]["_hist_lens"], ax=axes[0], label=split_labels[s], color=colors[s], lw=2, clip=(0, 80))
        sns.kdeplot(results[s]["_fut_lens"], ax=axes[1], label=split_labels[s], color=colors[s], lw=2, clip=(0, 80))
    axes[0].set_title("Lunghezza sequenza History (giorni)")
    axes[0].set_xlabel("Numero di giorni")
    axes[0].set_xlim(0, 60)
    axes[0].legend()

    axes[1].set_title("Lunghezza sequenza Future (giorni)")
    axes[1].set_xlabel("Numero di giorni")
    axes[1].set_xlim(0, 60)
    axes[1].legend()

    # Orizzonte in giorni sul train per step 1, 2, 3
    h_days = results["train"]["_horizons_days"]
    step_colors = ["#2e7d32", "#f9a825", "#c62828"]
    for idx, c in enumerate(step_colors):
        vals = h_days[:, idx]
        med = np.median(vals)
        sns.kdeplot(vals, ax=axes[2], label=f"Step {idx+1} (mediana {med:.0f} gg)", color=c, lw=2, clip=(0, 60))
    axes[2].axvline(14, color="black", ls="--", lw=1.2, alpha=0.7, label="14 giorni nominali")
    axes[2].set_title("Orizzonte effettivo dei 3 target (Train)")
    axes[2].set_xlabel("Giorni dopo l'ultima osservazione storica")
    axes[2].set_xlim(0, 50)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "01_sequence_lengths_and_horizons.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Distribuzioni Target OOD + Correlazione
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    t_short = ["NDVI", "GNDVI", "EVI"]
    for t_idx, name in enumerate(t_short):
        for s in splits:
            vals = results[s]["_targets"][:, :, t_idx].ravel()
            vals = vals[~np.isnan(vals)]
            # Subsample per velocità KDE se > 40k
            if len(vals) > 40000:
                rng = np.random.default_rng(42)
                vals = rng.choice(vals, 40000, replace=False)
            sns.kdeplot(vals, ax=axes[t_idx], label=split_labels[s], color=colors[s], lw=2)
        axes[t_idx].set_title(f"Distribuzione target: {name}")
        axes[t_idx].set_xlabel(f"{name} (scala originale)")
        axes[t_idx].set_xlim(-0.1, 1.0)
        axes[t_idx].legend(fontsize=8)

    corr_mat = np.array(results["train"]["target_corr"])
    sns.heatmap(
        corr_mat, annot=True, fmt=".3f", cmap="YlGn", vmin=0.5, vmax=1.0,
        xticklabels=t_short, yticklabels=t_short, ax=axes[3], square=True
    )
    axes[3].set_title("Correlazione Target (Train)")

    plt.tight_layout()
    plt.savefig(out_dir / "02_target_distributions_ood.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Classi climatiche di Köppen e Stagionalità (mesi)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    all_climates = sorted(
        set().union(*(results[s]["climate_counts"].keys() for s in splits)),
        key=lambda c: -results["train"]["climate_counts"].get(c, 0),
    )
    clim_df = pd.DataFrame({
        split_labels[s]: [
            100.0 * results[s]["climate_counts"].get(c, 0) / results[s]["n_samples"]
            for c in all_climates
        ]
        for s in splits
    }, index=all_climates)
    clim_df.plot(kind="bar", ax=axes[0], color=[colors[s] for s in splits], width=0.8)
    axes[0].set_title("Distribuzione classi climatiche di Köppen per split (%)")
    axes[0].set_ylabel("% campioni nello split")
    axes[0].set_xlabel("Classe climatica")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=9)

    months = list(range(1, 13))
    month_df = pd.DataFrame({
        split_labels[s]: [
            100.0 * results[s]["months"].get(m, 0) / results[s]["n_samples"]
            for m in months
        ]
        for s in splits
    }, index=[f"M{m:02d}" for m in months])
    month_df.plot(kind="bar", ax=axes[1], color=[colors[s] for s in splits], width=0.8)
    axes[1].set_title("Distribuzione mensile (stagionalità) per split (%)")
    axes[1].set_ylabel("% campioni nello split")
    axes[1].set_xlabel("Mese (data fine history)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "03_climate_and_seasonality.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: Effetto standardization_arcsinh + Rumore meteo per orizzonte
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    sample_h = results["train"]["_hist_vals_sample"]
    mean_arr = np.array(scaler["mean"], dtype=np.float64)
    var_arr = np.array(scaler["var"], dtype=np.float64)
    eps = float(scaler["eps"])

    showcase_feats = [
        (3, "rainfall (mm)"),
        (8, "thick_clouds"),
        (18, "NDVI (clear-sky)"),
    ]
    for col_plot, (f_idx, label) in enumerate(showcase_feats):
        raw = sample_h[:, f_idx]
        raw = raw[~np.isnan(raw)]
        trans = apply_arcsinh_scaler(raw, mean_arr[f_idx], var_arr[f_idx], eps)
        # Clip per leggibilità visiva del grezzo
        p995 = np.percentile(raw, 99.5)
        sns.histplot(raw[raw <= p995], bins=45, ax=axes[0, col_plot], color="#546e7a", stat="density")
        axes[0, col_plot].set_title(f"Grezzo: {label}")
        sns.histplot(trans, bins=45, ax=axes[1, col_plot], color="#2e7d32", stat="density")
        axes[1, col_plot].set_title(f"Dopo standardization_arcsinh: {label}")

    plt.tight_layout()
    plt.savefig(out_dir / "04_scaler_effect.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 5: Esempi di traiettorie + Benchmark Persistenza + Rumore vs Orizzonte
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    # Riga superiore: 3 traiettorie concrete dal train
    examples = results["train"]["_sample_examples"][:3]
    for idx, ex in enumerate(examples):
        ax = fig.add_subplot(gs[0, idx])
        hl = ex["history"].shape[0]
        fl = ex["future"].shape[0]
        t_hist = np.arange(-hl + 1, 1)
        t_fut = np.arange(1, fl + 1)

        # NDVI storico (punti validi) e target futuri
        ndvi_h = ex["history"][:, 18]
        valid_h = ~np.isnan(ndvi_h)
        ax.scatter(t_hist[valid_h], ndvi_h[valid_h], color="#1b5e20", s=65, zorder=5, label="NDVI history (clear-sky)")
        ax.plot(t_hist[valid_h], ndvi_h[valid_h], color="#1b5e20", ls=":", alpha=0.6)

        t_targets = ex["pos"] + 1
        ax.scatter(t_targets, ex["target"][:, 0], color="#e65100", s=75, marker="D", zorder=6, label="NDVI target (futuro)")
        ax.plot(t_targets, ex["target"][:, 0], color="#e65100", ls="--", alpha=0.7)

        # Temperatura media su asse secondario
        ax2 = ax.twinx()
        temp_all = np.concatenate([ex["history"][:, 5], ex["future"][:, 5]])
        t_all = np.concatenate([t_hist, t_fut])
        ax2.plot(t_all, temp_all, color="#0288d1", alpha=0.35, lw=1.5, label="Temp media (°C)")
        ax2.set_ylabel("Temp (°C)", color="#0288d1", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#0288d1", labelsize=8)

        ax.axvline(0.5, color="black", ls="--", lw=1)
        ax.set_title(f"Area {ex['area']} ({ex['climate']}) | H={hl}g, F={fl}g", fontsize=10)
        ax.set_xlabel("Giorni relativi a t_now (0)")
        ax.set_ylabel("NDVI")
        if idx == 0:
            ax.legend(loc="lower left", fontsize=8)

    # Riga inferiore sx+centro: MAE Persistenza per split e step (NDVI)
    ax_p = fig.add_subplot(gs[1, :2])
    rows = []
    for s in splits:
        for t_name in ["NDVI", "GNDVI", "EVI"]:
            for step_i in [1, 2, 3]:
                rows.append({
                    "Split": split_labels[s],
                    "Indice": t_name,
                    "Step": f"Step {step_i}",
                    "MAE": results[s]["persistence_mae"][t_name][f"step_{step_i}"],
                })
    pers_df = pd.DataFrame(rows)
    ndvi_pers = pers_df[pers_df["Indice"] == "NDVI"]
    sns.barplot(data=ndvi_pers, x="Split", y="MAE", hue="Step", ax=ax_p, palette=["#66bb6a", "#ffa726", "#ef5350"])
    ax_p.set_title("Baseline ingenua (Persistenza ultimo clear-sky) — MAE su NDVI per orizzonte e split")
    ax_p.set_ylabel("MAE (scala originale)")

    # Riga inferiore dx: RMS del rumore futuro vs giorno dell'orizzonte (normalizzato su std feature)
    ax_n = fig.add_subplot(gs[1, 2])
    noise_day = results["train"]["noise_by_day_rms"]  # (35, 11)
    days = np.arange(1, 26)
    for f_idx, f_lbl in [(5, "avg_temp"), (3, "rainfall"), (2, "solar_irr"), (8, "thick_clouds")]:
        f_std = results["train"]["feature_stats"][feature_names[f_idx]]["std"]
        ax_n.plot(days, noise_day[:25, f_idx] / max(f_std, 1e-6), lw=2, label=f_lbl)
    ax_n.set_title("Entità rumore futuro vs giorno (RMS / std feature)")
    ax_n.set_xlabel("Giorno futuro (t + k)")
    ax_n.set_ylabel("Noise RMS / Feature Std")
    ax_n.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "05_trajectories_persistence_noise.png", dpi=150)
    plt.close()


def main() -> None:
    cfg = load_config()
    data_root = PROJECT_ROOT / cfg["data"]["root"]
    splits_map = cfg["data"]["splits"]
    scaler_path = data_root / cfg["data"]["scaler_file"]

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    # Leggi feature_names e target_names dal primo json di area del train (escludendo scaler.json)
    train_dir = data_root / splits_map["train"]
    first_json = [p for p in sorted(train_dir.glob("*.json")) if p.name != "scaler.json"][0]
    with open(first_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]
    target_names = meta["target_names"]

    results = {}
    for split_key, folder_name in splits_map.items():
        print(f"Analisi split '{split_key}' ({folder_name})...", flush=True)
        results[split_key] = inspect_split(split_key, data_root / folder_name, feature_names, target_names)

    # Overlap aree tra split
    area_sets = {k: set(v["areas"]) for k, v in results.items()}
    overlaps = {}
    keys = list(area_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            inter = sorted(area_sets[k1] & area_sets[k2])
            overlaps[f"{k1}__vs__{k2}"] = {"count": len(inter), "areas": inter[:10]}

    # Classi climatiche unseen rispetto al train
    train_clim = set(results["train"]["climate_counts"].keys())
    unseen_climates = {
        k: sorted(set(v["climate_counts"].keys()) - train_clim)
        for k, v in results.items() if k != "train"
    }

    out_dir = PROJECT_ROOT / "runs" / "eda"
    make_plots(results, feature_names, scaler, out_dir)

    # Salva riepilogo JSON (senza array grezzi interni)
    summary = {
        "feature_names": feature_names,
        "target_names": target_names,
        "scaler_mode": scaler["mode"],
        "area_overlaps": overlaps,
        "unseen_climates_vs_train": unseen_climates,
        "splits": {
            k: {sk: sv for sk, sv in v.items() if not sk.startswith("_") and sk != "noise_by_day_rms"}
            for k, v in results.items()
        },
    }
    summary_path = out_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Stampa riassunto leggibile a console
    print("\n================ RIEPILOGO EDA ================")
    for k in ["train", "id", "ood_s", "ood_t"]:
        r = results[k]
        print(
            f"[{k:5s}] aree={r['n_areas']:3d} | campioni={r['n_samples']:6d} | "
            f"date={r['date_range']['min']}..{r['date_range']['max']} | "
            f"hist_len med={r['hist_len_stats']['median']:.0f} (max {r['hist_len_stats']['max']}) | "
            f"fut_len med={r['fut_len_stats']['median']:.0f} (max {r['fut_len_stats']['max']}) | "
            f"target_NaN={r['target_nan_count']} | mask_mismatch=({r['mask_mismatch_hist']},{r['mask_mismatch_fut']})"
        )
    print("\nOverlap aree:", {k: v["count"] for k, v in overlaps.items()})
    print("Classi climatiche unseen rispetto al train:", unseen_climates)
    print("\nPersistenza MAE (overall):")
    for k in ["train", "id", "ood_s", "ood_t"]:
        p = results[k]["persistence_mae"]
        print(f"  {k:5s}: NDVI={p['NDVI']['overall']:.4f}, GNDVI={p['GNDVI']['overall']:.4f}, EVI={p['EVI']['overall']:.4f}")
    print(f"\nGrafici e riepilogo salvati in: {out_dir}")


if __name__ == "__main__":
    main()
