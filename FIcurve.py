import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- PARAMETERS ----------
control_file = r"x:\Kristian\Projects\Deep_Sup\Scripts\FI_curves\Control_FI.xlsx"
experimental_file = r"x:\Kristian\Projects\Deep_Sup\Scripts\FI_curves\Experimental_FI.xlsx"
save_folder = r"x:\Kristian\Projects\Deep_Sup\Python_figures"
os.makedirs(save_folder, exist_ok=True)

current_min = 0    # minimum current to include
current_max = 400 # maximum current to include

# ---------- LOAD DATA ----------
control_df = pd.read_excel(control_file)
experimental_df = pd.read_excel(experimental_file)

# Ensure first column is named 'Current' for consistency
control_df.rename(columns={control_df.columns[0]: "Current"}, inplace=True)
experimental_df.rename(columns={experimental_df.columns[0]: "Current"}, inplace=True)

# Filter by current range
control_df = control_df[(control_df["Current"] >= current_min) & (control_df["Current"] <= current_max)]
experimental_df = experimental_df[(experimental_df["Current"] >= current_min) & (experimental_df["Current"] <= current_max)]

# ---------- SEPARATE SUP AND DEEP BASED ON THIRD CHARACTER ----------
def split_cells(df):
    sup_cols = [col for col in df.columns if len(col) > 2 and col[2].lower() == 's']
    deep_cols = [col for col in df.columns if len(col) > 2 and col[2].lower() == 'd']
    sup = df[sup_cols]
    deep = df[deep_cols]
    return sup, deep

control_sup, control_deep = split_cells(control_df)
experimental_sup, experimental_deep = split_cells(experimental_df)

currents = control_df["Current"]


# ---------- COMPUTE n VALUES ----------
n_control_sup = control_sup.shape[1]
n_experimental_sup = experimental_sup.shape[1]
n_control_deep = control_deep.shape[1]
n_experimental_deep = experimental_deep.shape[1]

# ---------- PLOT ----------
plt.figure(figsize=(6, 6))

# Sup group (if needed)
plt.plot(currents, control_sup.mean(axis=1), '-', color="#0000FF", label=f"Experimental Sup (n={n_control_sup})", linewidth=3)
plt.fill_between(currents,
                 control_sup.mean(axis=1) - control_sup.sem(axis=1),
                 control_sup.mean(axis=1) + control_sup.sem(axis=1),
                 color="#0000FF", alpha=0.3)
plt.plot(currents, experimental_sup.mean(axis=1), '-', color="#FF0000", label=f"Experimental Sup (n={n_experimental_sup})", linewidth=3)
plt.fill_between(currents,
                 experimental_sup.mean(axis=1) - experimental_sup.sem(axis=1),
                 experimental_sup.mean(axis=1) + experimental_sup.sem(axis=1),
                 color="#FF0000", alpha=0.3)

# # Deep group
# plt.plot(currents, control_deep.mean(axis=1), '-', color="#0000FF", label=f"Experimental Sup (n={n_control_deep})", linewidth=3)
# plt.fill_between(currents,
#                  control_deep.mean(axis=1) - control_deep.sem(axis=1),
#                  control_deep.mean(axis=1) + control_deep.sem(axis=1),
#                  color="#0000FF", alpha=0.3)
# plt.plot(currents, experimental_deep.mean(axis=1), '-', color="#FF0000", label=f"Experimental Sup (n={n_experimental_deep})", linewidth=3)
# plt.fill_between(currents,
#                  experimental_deep.mean(axis=1) - experimental_deep.sem(axis=1),
#                  experimental_deep.mean(axis=1) + experimental_deep.sem(axis=1),
#                  color="#FF0000", alpha=0.3)

plt.xlabel("Current (pA)", fontsize=20)
plt.ylabel("Firing Frequency (Hz)", fontsize=20)
plt.title("F-I curve deep cells (Mean ± SEM)", fontsize=20)
plt.legend(loc='upper left', fontsize=13)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax.tick_params(width=2.0, length=3)
plt.tight_layout()

# Save figure
save_path = os.path.join(save_folder, "FI_curve_deep.svg")
plt.savefig(save_path, dpi=1200)
plt.show()

print(f"F-I curve saved to: {save_path}")
