import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
import os

# ---------- PARAMETERS ----------
control_file = r"x:\Kristian\Projects\Deep_Sup\Scripts\FI_curves\Control_FI.xlsx"
experimental_file = r"x:\Kristian\Projects\Deep_Sup\Scripts\FI_curves\Experimental_FI.xlsx"
save_folder = r"x:\Kristian\Projects\Deep_Sup\Python_figures"
os.makedirs(save_folder, exist_ok=True)

current_min = 0
current_max = 400

# ---------- FONT SIZES ----------
title_fontsize = 16
ylabel_fontsize = 16
xlabel_fontsize = 14
xtick_fontsize = 16
ytick_fontsize = 16

# ---------- LOAD DATA ----------
control_df = pd.read_excel(control_file)
experimental_df = pd.read_excel(experimental_file)

control_df.rename(columns={control_df.columns[0]: "Current"}, inplace=True)
experimental_df.rename(columns={experimental_df.columns[0]: "Current"}, inplace=True)

control_df = control_df[(control_df["Current"] >= current_min) & (control_df["Current"] <= current_max)]
experimental_df = experimental_df[(experimental_df["Current"] >= current_min) & (experimental_df["Current"] <= current_max)]

# ---------- SPLIT SUP AND DEEP ----------
def split_cells(df):
    sup_cols = [col for col in df.columns if len(col) > 2 and col[2].lower() == 's']
    deep_cols = [col for col in df.columns if len(col) > 2 and col[2].lower() == 'd']
    sup = df[sup_cols]
    deep = df[deep_cols]
    return sup, deep

control_sup, control_deep = split_cells(control_df)
experimental_sup, experimental_deep = split_cells(experimental_df)

# ---------- HELPER FUNCTIONS ----------
def first_firing_current(group_df, full_df):
    first_currents = []
    for col in group_df.columns:
        firing = group_df[col]
        if (firing > 0).any():
            idx_first_fire = (firing > 0).idxmax()
            current_first_fire = full_df.loc[idx_first_fire, "Current"]
        else:
            current_first_fire = np.nan
        first_currents.append(current_first_fire)
    return np.array(first_currents)

def first_max_current(group_df, full_df):
    first_currents = []
    for col in group_df.columns:
        firing = group_df[col]
        max_firing = firing.max()
        idx_first_max = (firing == max_firing).idxmax()
        current_first_max = full_df.loc[idx_first_max, "Current"]
        first_currents.append(current_first_max)
    return np.array(first_currents)

def prepare_boxplot_df(control_data, experimental_data, group_name):
    df = pd.DataFrame({
        "Current": np.concatenate([control_data, experimental_data]),
        "Group": ["Control"] * len(control_data) + ["Experimental"] * len(experimental_data),
        "CellType": [group_name] * (len(control_data) + len(experimental_data))
    })
    return df

def pval_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def welch_df(a, b):
    """Calculate Welch–Satterthwaite degrees of freedom."""
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    n1 = len(a)
    n2 = len(b)
    num = (s1/n1 + s2/n2)**2
    denom = ((s1/n1)**2)/(n1-1) + ((s2/n2)**2)/(n2-1)
    return num / denom

def cohen_d(a, b):
    """Compute Cohen’s d for unequal sample sizes."""
    a, b = np.array(a), np.array(b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_sd

def mean_sd(data):
    """Return mean and standard deviation, ignoring NaNs."""
    mean = np.nanmean(data)
    sd = np.nanstd(data, ddof=1)
    return mean, sd

# ---------- COMPUTE n VALUES ----------
n_control_sup = control_sup.shape[1]
n_experimental_sup = experimental_sup.shape[1]
n_control_deep = control_deep.shape[1]
n_experimental_deep = experimental_deep.shape[1]

# ---------- LOOP OVER MODES ----------
modes = {"Fmin": first_firing_current, "Fmax": first_max_current}
sns.set(style="ticks", font_scale=1.0)
colors = {"Control": "#0000FF", "Experimental": "#FF0000"}

for mode_name, calc_func in modes.items():
    # Compute data
    control_sup_first = calc_func(control_sup, control_df)
    experimental_sup_first = calc_func(experimental_sup, experimental_df)
    control_deep_first = calc_func(control_deep, control_df)
    experimental_deep_first = calc_func(experimental_deep, experimental_df)

    sup_df = prepare_boxplot_df(control_sup_first, experimental_sup_first, "Sup")
    deep_df = prepare_boxplot_df(control_deep_first, experimental_deep_first, "Deep")

    comparisons = {
        "Sup": (control_sup_first, experimental_sup_first),
        "Deep": (control_deep_first, experimental_deep_first)
    }

    print(f"\n==================== {mode_name} ====================")
    for layer_name, (ctrl, exp) in comparisons.items():
        ctrl = ctrl[~np.isnan(ctrl)]
        exp = exp[~np.isnan(exp)]

        # Welch’s t-test
        t_stat, p_val = ttest_ind(ctrl, exp, equal_var=False)
        df = welch_df(ctrl, exp)
        d = cohen_d(ctrl, exp)

        # Means ± SEM
        mean_ctrl, sd_ctrl = mean_sd(ctrl)
        mean_exp, sd_exp = mean_sd(exp)

        print(f"\n{layer_name} layer:")
        print(f"  Control       : {mean_ctrl:.2f} ± {sd_ctrl:.2f} (SD), n = {len(ctrl)}")
        print(f"  Experimental  : {mean_exp:.2f} ± {sd_exp:.2f} (SD), n = {len(exp)}")
        print(f"  Welch’s t-test: t({df:.1f}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Effect size   : Cohen’s d = {d:.3f}")

    # --- Plotting ---
    for cell_type, df_subset, p_val in zip(
        ["Sup", "Deep"],
        [sup_df, deep_df],
        [ttest_ind(control_sup_first, experimental_sup_first, equal_var=False)[1],
         ttest_ind(control_deep_first, experimental_deep_first, equal_var=False)[1]]
    ):
        fig, ax = plt.subplots(figsize=(2.5, 3), dpi=1200)

        groups_order = ["Control", "Experimental"]
        data = [df_subset[df_subset["Group"] == g]["Current"].dropna().values for g in groups_order]
        data_clean = [d if len(d) > 0 else np.array([]) for d in data]

        # --- Boxplot ---
        bplot = ax.boxplot(
            data_clean,
            patch_artist=True,
            labels=groups_order,
            boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=2),
            capprops=dict(color='black', linewidth=2),
            medianprops=dict(color='black', linewidth=2),
            showfliers=False
        )

        # Color boxes
        for patch, g in zip(bplot['boxes'], groups_order):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)

        # Scatter points
        offsets = [-0.15, 0.15]
        for k, g in enumerate(groups_order):
            vals = data_clean[k]
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
        if all(len(d) > 1 for d in data_clean):
            signif = pval_to_stars(p_val)
            y_max = max([np.max(d) for d in data_clean])
            offset_val = 0.05 * y_max if y_max > 0 else 10
            y_line = y_max + offset_val
            tick_height = 0.02 * y_max if y_max > 0 else 5
            y_text = y_line + tick_height

            ax.plot([1, 2], [y_line, y_line], color='k', linewidth=2)
            ax.plot([1, 1], [y_line - tick_height, y_line], color='k', linewidth=2)
            ax.plot([2, 2], [y_line - tick_height, y_line], color='k', linewidth=2)
            ax.text(1.5, y_text, signif, ha='center', va='bottom', fontsize=16)

        # --- Styling ---
        ax.set_title(f"{mode_name} - {cell_type}", fontsize=title_fontsize)
        ylabel_text = "Current at first Fmax (pA)" if mode_name == "Fmax" else "Current at Fmin (pA)"
        ax.set_ylabel(ylabel_text, fontsize=ylabel_fontsize)
        ax.set_xticklabels(groups_order, fontsize=xtick_fontsize)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ---------- Add N labels ----------
        if cell_type == "Sup":
            ns = [n_control_sup, n_experimental_sup]
        else:
            ns = [n_control_deep, n_experimental_deep]

        ax.set_ylim(-50, 450)

        for i, n_val in enumerate(ns):
            ax.text(
                i + 1, -25,
                f"N = {n_val}",
                ha='center',
                va='top',
                fontsize=12
            )

        plt.tight_layout()
        save_path = os.path.join(save_folder, f"{mode_name}_boxplot_{cell_type}.svg")
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")

