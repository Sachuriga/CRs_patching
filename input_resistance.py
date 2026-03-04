import os
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

# ---------- PARAMETERS ----------
base_dir = r"x:\Kristian\Projects\Deep_Sup\Scripts\Input_resitance"
control_dir = os.path.join(base_dir, "Control")
experimental_dir = os.path.join(base_dir, "Experimental")
save_dir = r"x:\Kristian\Projects\Deep_Sup\Python_figures"

current_step = -100e-12  # -100 pA in Amperes

# ---------- HELPER FUNCTIONS ----------
def compute_input_resistance(abf_file, current_step, sweep_index=0, debug=False):
    """Compute input resistance in MOhm from a sweep using steady-state voltage."""
    try:
        abf = pyabf.ABF(abf_file)
        abf.setSweep(sweep_index)
        v = np.array(abf.sweepY)  # mV
        fs = abf.dataRate
        n = len(v)

        # Step parameters
        step_start_s = 0.1
        step_duration_s = 1.0
        start_idx = int(step_start_s * fs)
        end_idx = start_idx + int(step_duration_s * fs)
        if end_idx > n:
            end_idx = n

        # Baseline: 100 ms before step
        pre_samples = int(0.1 * fs)
        baseline_window = v[start_idx - pre_samples:start_idx]

        # Steady-state: last 200 ms of step
        ss_samples = int(0.2 * fs)
        steady_window = v[end_idx - ss_samples:end_idx]

        if len(baseline_window) == 0 or len(steady_window) == 0:
            if debug:
                print(f"{abf_file}: empty baseline or steady window")
            return np.nan

        baseline = np.mean(baseline_window)
        steady = np.mean(steady_window)
        delta_v_V = (steady - baseline) * 1e-3  # mV → V

        if current_step == 0:
            if debug:
                print(f"{abf_file}: current step is zero")
            return np.nan

        R_ohm = delta_v_V / current_step
        R_MOhm = np.abs(R_ohm) / 1e6

        if not np.isfinite(R_MOhm) or R_MOhm > 1e4:
            if debug:
                print(f"{abf_file}: R_MOhm={R_MOhm} outside expected range")
            return np.nan

        if debug:
            print(f"{abf_file}: baseline={baseline:.2f} mV, steady={steady:.2f} mV, R={R_MOhm:.1f} MΩ")

        return R_MOhm

    except Exception as e:
        if debug:
            print(f"Error computing Rin for {abf_file}: {e}")
        return np.nan


def pval_to_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."


def mean_sd(data):
    """Return mean and SD, ignoring NaNs."""
    return np.nanmean(data), np.nanstd(data, ddof=1)


def cohen_d(x, y):
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    s_pooled = np.sqrt(((nx - 1)*np.nanvar(x, ddof=1) + (ny - 1)*np.nanvar(y, ddof=1)) / (nx + ny - 2))
    return (np.nanmean(x) - np.nanmean(y)) / s_pooled


def welch_df(x, y):
    """Compute Welch–Satterthwaite degrees of freedom."""
    sx2 = np.nanvar(x, ddof=1)
    sy2 = np.nanvar(y, ddof=1)
    nx, ny = len(x), len(y)
    num = (sx2/nx + sy2/ny)**2
    denom = (sx2**2 / (nx**2 * (nx-1))) + (sy2**2 / (ny**2 * (ny-1)))
    return num / denom


def process_folder(folder, group_name, debug=False):
    """Process all ABF files in a folder and compute input resistance."""
    results = []

    for fname in os.listdir(folder):
        if not fname.endswith(".abf"):
            continue
        fpath = os.path.join(folder, fname)
        try:
            if fname.startswith(("D-s02", "I-d30", "H-s30","H-d02","G-d30","F-d30","E-s30","C-d30","B-d30",
                                 "G-d18","G-d04","F-s18","F-d26","E-s01","E-d18","D-s01","D-d18",
                                 "C-s18","C-s01","B-s01","B-d18","A-s26","A-s01","A-d18")):
                sweep_idx = 2
            else:
                sweep_idx = 0

            rin = compute_input_resistance(fpath, current_step, sweep_index=sweep_idx, debug=debug)
            cell_type = "Deep" if fname[2].lower() == "d" else "Sup"
            results.append({
                "File": fname,
                "Group": group_name,
                "CellType": cell_type,
                "InputResistance_MOhm": rin
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            cell_type = "Deep" if fname[2].lower() == "d" else "Sup"
            results.append({
                "File": fname,
                "Group": group_name,
                "CellType": cell_type,
                "InputResistance_MOhm": np.nan
            })

    return pd.DataFrame(results)


# ---------- PROCESS DATA ----------
print("Processing control folder...")
df_control = process_folder(control_dir, "Control", debug=True)

print("Processing experimental folder...")
df_experimental = process_folder(experimental_dir, "Experimental", debug=True)

df_all = pd.concat([df_control, df_experimental], ignore_index=True)

# ---------- SAVE TO EXCEL ----------
control_xlsx = os.path.join(control_dir, "InputResistance_Control.xlsx")
experimental_xlsx = os.path.join(experimental_dir, "InputResistance_Experimental.xlsx")
summary_xlsx = os.path.join(save_dir, "InputResistance_All.xlsx")

df_control.to_excel(control_xlsx, index=False)
df_experimental.to_excel(experimental_xlsx, index=False)
df_all.to_excel(summary_xlsx, index=False)
print(f"\nSaved results to:\n{summary_xlsx}")


# ---------- STATISTICS AND PLOTS ----------
colors = {"Control": "#0000FF", "Experimental": "#FF0000"}
cell_types = ["Sup", "Deep"]
groups_order = ["Control", "Experimental"]

sns.set(style="ticks", font_scale=1.0)

all_values = df_all["InputResistance_MOhm"].dropna().values
y_min, y_max = 0, np.ceil(all_values.max() * 1.1)

print("\n==================== INPUT RESISTANCE STATISTICS ====================")
for cell_type in cell_types:
    subset = df_all[df_all["CellType"] == cell_type]
    data_control = subset[subset["Group"] == "Control"]["InputResistance_MOhm"].dropna().values
    data_experimental = subset[subset["Group"] == "Experimental"]["InputResistance_MOhm"].dropna().values

    if len(data_control) > 1 and len(data_experimental) > 1:
        # Welch's t-test
        t_stat, p_val = ttest_ind(data_control, data_experimental, equal_var=False)
        df_welch = welch_df(data_control, data_experimental)
        d = cohen_d(data_control, data_experimental)

        mean_ctrl, sd_ctrl = mean_sd(data_control)
        mean_exp, sd_exp = mean_sd(data_experimental)

        print(f"\n{cell_type} cells:")
        print(f"  Control       : {mean_ctrl:.2f} ± {sd_ctrl:.2f} (SD), n = {len(data_control)}")
        print(f"  Experimental  : {mean_exp:.2f} ± {sd_exp:.2f} (SD), n = {len(data_experimental)}")
        print(f"  Welch’s t-test: t({df_welch:.1f}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Effect size   : Cohen’s d = {d:.3f}")

    # ---------- PLOTS ----------
    fig, ax = plt.subplots(figsize=(2.5, 3), dpi=1200)
    data = [data_control, data_experimental]
    bplot = ax.boxplot(
        data,
        patch_artist=True,
        labels=groups_order,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2),
        medianprops=dict(color='black', linewidth=2),
        showfliers=False
    )
    for patch, g in zip(bplot['boxes'], groups_order):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.7)

    offsets = [-0.15, 0.15]
    for k, g in enumerate(groups_order):
        vals = data[k]
        if len(vals) == 0:
            continue
        x_center = k + 1
        jitter = np.random.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(
            x_center + offsets[k] + jitter,
            vals,
            color=colors[g],
            edgecolor='black',
            alpha=0.7,
            s=45,
            zorder=10
        )

    # Significance bar
    if len(data_control) > 1 and len(data_experimental) > 1:
        signif = pval_to_stars(p_val)
        y_line = y_max * 1.05
        tick_height = y_max * 0.02
        y_text = y_line + tick_height
        ax.plot([1, 2], [y_line, y_line], color='k', linewidth=2)
        ax.plot([1, 1], [y_line - tick_height, y_line], color='k', linewidth=2)
        ax.plot([2, 2], [y_line - tick_height, y_line], color='k', linewidth=2)
        ax.text(1.5, y_text, signif, ha='center', va='bottom', fontsize=16)

    ax.set_title(f"{cell_type} Cells", fontsize=16)
    ax.set_ylabel("Input Resistance (MΩ)", fontsize=16)
    ax.set_xticklabels(groups_order, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    n_control = len(data_control)
    n_experimental = len(data_experimental)
    ax.set_ylim(bottom=-0.1*y_max, top=y_max*1.1)
    for i, n_val in enumerate([n_control, n_experimental]):
        ax.text(i + 1, -0.05*y_max, f"N = {n_val}", ha='center', va='top', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"InputResistance_Boxplot_{cell_type}.svg")
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
    print(f"Saved plot: {save_path}")
