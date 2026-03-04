import pyabf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import ttest_ind

# ---------- PARAMETERS ----------
folder = r"x:\Kristian\Projects\Deep_Sup\Scripts\FI_curves\Action_potential_data\FI_curve_traces"
control_folder = os.path.join(folder, "control")
experimental_folder = os.path.join(folder, "experimental")

ap_pre_ms = 5
ap_post_ms = 20
spike_height_above_baseline = 61.64
pre_step_window = 0.1

colors = {"Control": "blue", "Experimental": "red"}
results = []

# ---------- HELPER FUNCTIONS ----------
def detect_spikes(t, y, threshold, data_rate):
    crossings = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    spike_indices = []
    search_window = int(2e-3 * data_rate)
    for c in crossings:
        window = slice(c, min(c + search_window, len(y)))
        if window.stop > window.start:
            local_peak = window.start + np.argmax(y[window])
            if y[local_peak] >= threshold:
                spike_indices.append(local_peak)
    return np.array(spike_indices)

def extract_ap_features(t, y, spike_indices, data_rate):
    ap_waveforms, amplitudes, fAHPs, mAHPs, max_slopes = [], [], [], [], []

    pre_pts = int(ap_pre_ms / 1000 * data_rate)
    post_pts = int(ap_post_ms / 1000 * data_rate)

    for idx in spike_indices:
        start = max(idx - pre_pts, 0)
        end = min(idx + post_pts, len(y))

        ap_wave = y[start:end] - y[start]
        ap_waveforms.append(ap_wave)

        amplitudes.append(np.max(y[start:end]) - y[start])

        # fAHP: 0-5 ms after peak
        fAHP_end = min(idx + int(0.005 * data_rate), len(y))
        fAHP_val = np.min(y[idx:fAHP_end] - y[start])
        fAHPs.append(fAHP_val)

        # mAHP: 5-20 ms after peak
        mAHP_start = fAHP_end
        mAHP_end = min(idx + int(0.02 * data_rate), len(y))
        mAHP_val = np.min(y[mAHP_start:mAHP_end] - y[start]) if mAHP_end > mAHP_start else np.nan
        mAHPs.append(mAHP_val)

        dv = np.diff(y[start:end])
        dt = np.diff(t[start:end])
        slope = dv / dt / 1000
        max_slopes.append(np.max(slope))

    return {"ap_waveforms": ap_waveforms, "amplitudes": amplitudes,
            "fAHPs": fAHPs, "mAHPs": mAHPs, "max_slopes": max_slopes}

def process_folder(folder_path, group):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".abf"):
            continue
        abf_path = os.path.join(folder_path, filename)
        abf = pyabf.ABF(abf_path)
        cell_id = filename.split(".")[0]

        if "-" in cell_id:
            marker = cell_id.split("-")[1][0].lower()
            cell_type = "Sup" if marker == "s" else "Deep" if marker == "d" else "Unknown"
        else:
            cell_type = "Unknown"

        first_spike_sweep = None
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweep)
            t, y = abf.sweepX, abf.sweepY
            baseline = np.mean(y[t < pre_step_window])
            threshold = baseline + spike_height_above_baseline
            spike_indices = detect_spikes(t, y, threshold, abf.dataRate)
            if len(spike_indices) > 0:
                first_spike_sweep = sweep
                break
        if first_spike_sweep is None:
            continue

        target_sweep = first_spike_sweep + 4
        if target_sweep >= abf.sweepCount:
            continue

        abf.setSweep(target_sweep)
        t, y = abf.sweepX, abf.sweepY
        baseline = np.mean(y[t < pre_step_window])
        threshold = baseline + spike_height_above_baseline
        spike_indices = detect_spikes(t, y, threshold, abf.dataRate)
        if len(spike_indices) == 0:
            continue

        features = extract_ap_features(t, y, spike_indices, abf.dataRate)

        results.append({"CellID": cell_id, "Group": group, "CellType": cell_type,
                        "Sweep": target_sweep, "dataRate": abf.dataRate,
                        "ap_waveforms": features["ap_waveforms"],
                        "amplitudes": features["amplitudes"],
                        "fAHPs": features["fAHPs"],
                        "mAHPs": features["mAHPs"],
                        "max_slopes": features["max_slopes"]})

# ---------- RUN ----------
process_folder(control_folder, "Control")
process_folder(experimental_folder, "Experimental")

# ---------- FUNCTION FOR COHEN'S D ----------
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1))/dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

def mean_sd(x):
    return np.mean(x), np.std(x, ddof=1)

def welch_df(x, y):
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    return (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))

def pval_to_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

# ---------- FEATURE EXTRACTION ----------
props = []
for r in results:
    if r["fAHPs"]:
        props.append({"CellID": r["CellID"], "Group": r["Group"], "CellType": r["CellType"], "Feature":"fAHP", "Value":np.mean(r["fAHPs"])})
    if r["mAHPs"]:
        props.append({"CellID": r["CellID"], "Group": r["Group"], "CellType": r["CellType"], "Feature":"mAHP", "Value":np.mean(r["mAHPs"])})
    if r["max_slopes"]:
        props.append({"CellID": r["CellID"], "Group": r["Group"], "CellType": r["CellType"], "Feature":"MaxSlope", "Value":np.mean(r["max_slopes"])})
    if r["ap_waveforms"]:
        amp = [np.max(ap) for ap in r["ap_waveforms"]]
        props.append({"CellID": r["CellID"], "Group": r["Group"], "CellType": r["CellType"], "Feature":"Amplitude", "Value":np.mean(amp)})

df_props = pd.DataFrame(props)

# ---------- FEATURE PLOTS WITH FULL STATISTICS ----------
features = ["fAHP","mAHP","MaxSlope","Amplitude"]
cell_types = ["Sup","Deep"]
fig, axes = plt.subplots(len(features), len(cell_types), figsize=(4,10))

for i, feature in enumerate(features):
    for j, ct in enumerate(cell_types):
        ax = axes[i,j]
        subset = df_props[(df_props["Feature"]==feature) & (df_props["CellType"]==ct)]
        if subset.empty:
            ax.set_visible(False)
            continue

        groups_order = ["Control","Experimental"]
        data = [subset[subset["Group"]==g].groupby("CellID")["Value"].mean().values for g in groups_order]

        # --- t-test + Cohen's d + mean±SD ---
        if all(len(d) > 1 for d in data):
            t_stat, p_val = ttest_ind(data[0], data[1], equal_var=False)
            df_w = welch_df(data[0], data[1])
            d = cohen_d(data[0], data[1])
            mean0, sd0 = mean_sd(data[0])
            mean1, sd1 = mean_sd(data[1])
            signif = pval_to_stars(p_val)

            print(f"{ct} {feature}:")
            print(f"  Control      : {mean0:.2f} ± {sd0:.2f} (SD), n={len(data[0])}")
            print(f"  Experimental : {mean1:.2f} ± {sd1:.2f} (SD), n={len(data[1])}")
            print(f"  Welch t-test : t({df_w:.1f}) = {t_stat:.3f}, p = {p_val:.4f} ({signif})")
            print(f"  Effect size  : Cohen's d = {d:.3f}")

        # --- Boxplot ---
        bplot = ax.boxplot(
            data, patch_artist=True, labels=groups_order,
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2, alpha=0.7),
            whiskerprops=dict(color='black', linewidth=2),
            capprops=dict(color='black', linewidth=2),
            medianprops=dict(color='black', linewidth=2),
            showfliers=False
        )
        for patch, g in zip(bplot['boxes'], groups_order):
            patch.set_facecolor(colors[g])

        # Scatter points
        for k, g in enumerate(groups_order):
            vals = subset[subset["Group"]==g].groupby("CellID")["Value"].mean().values
            x_pos = k + 1 - 0.15 if g=="Control" else k+1+0.15
            ax.scatter(x_pos + np.random.uniform(-0.05,0.05,len(vals)), vals,
                       color=colors[g], edgecolor='black', alpha=0.7, s=30, zorder=10)

        # Significance line
        if all(len(d) > 1 for d in data):
            y_max = max([max(d) for d in data])
            y_min = min([min(d) for d in data])
            y_range = y_max - y_min
            y_line = y_max + 0.12*y_range
            tick_height = 0.03*y_range
            y_text = y_line + 0.06*y_range

            x_control, x_exp = 1,2
            ax.plot([x_control, x_exp],[y_line,y_line], color='k', linewidth=2)
            ax.plot([x_control,x_control],[y_line-tick_height,y_line],color='k',linewidth=2)
            ax.plot([x_exp,x_exp],[y_line-tick_height,y_line],color='k',linewidth=2)
            ax.text((x_control+x_exp)/2, y_text, signif, ha='center', va='bottom', fontsize=16)

        # Titles & labels
        n1 = subset[subset["Group"]=="Control"]["CellID"].nunique()
        n2 = subset[subset["Group"]=="Experimental"]["CellID"].nunique()
        ax.set_title(f"{ct} {feature}")
        ax.set_xticks([1,2])
        ylabel = "ΔmV/ms" if feature=="MaxSlope" else "mV" if feature=="Amplitude" else "ΔmV"
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(groups_order)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

plt.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.savefig(r"Z:\Kristian\Projects\Deep_Sup\Python_figures\feature_boxplots_with_stats.svg", dpi=1200, bbox_inches='tight')
plt.show()
