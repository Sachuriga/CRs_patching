import pyabf
import numpy as np
import pandas as pd
import os

# ---------- PARAMETERS ----------
base_folder = r"Z:\Kristian\Projects\Deep_Sup\Scripts\FI_curves"  # contains 'Control' and 'Experimental'
folders = ["Control", "Experimental"]

voltage_channel = 0
current_channel = 1

spike_height_above_baseline = 61.64  # mV threshold above baseline
pre_step_window = 0.1                # seconds before step
step_window = (0.2, 0.6)             # time window during step

# ---------- FUNCTIONS ----------
def detect_spikes(t, y, threshold, data_rate):
    crossings = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    spike_indices = []
    search_window = int(2e-3 * data_rate)
    for c in crossings:
        window = slice(c, min(c + search_window, len(y)))
        if window.stop > window.start:
            peak_idx = window.start + np.argmax(y[window])
            if y[peak_idx] >= threshold:
                spike_indices.append(peak_idx)
    return spike_indices

def get_baseline(y, t, pre_step_window=0.1):
    return np.mean(y[t < pre_step_window])

def measure_step_amplitude(t, y, step_window):
    mask = (t >= step_window[0]) & (t <= step_window[1])
    return np.mean(y[mask])

def process_folder(folder_path, output_path):
    abf_files = [f for f in os.listdir(folder_path) if f.endswith(".abf")]
    all_data = {}

    for f in abf_files:
        abf_path = os.path.join(folder_path, f)
        abf = pyabf.ABF(abf_path)
        cell_id = f[:9]
        print(f"Processing {cell_id}...")

        currents = []
        spike_counts = []

        for sweep in range(abf.sweepCount):
            # --- Voltage channel ---
            abf.setSweep(sweep, channel=voltage_channel)
            t_v, y_v = abf.sweepX, abf.sweepY
            baseline_v = get_baseline(y_v, t_v, pre_step_window)
            threshold = baseline_v + spike_height_above_baseline
            spikes = detect_spikes(t_v, y_v, threshold, abf.dataRate)
            spike_counts.append(len(spikes))

            # --- Current channel ---
            abf.setSweep(sweep, channel=current_channel)
            t_i, y_i = abf.sweepX, abf.sweepY
            baseline_i = get_baseline(y_i, t_i, pre_step_window)
            y_i_zeroed = y_i - baseline_i
            step_amp = measure_step_amplitude(t_i, y_i_zeroed, step_window)
            currents.append(step_amp)  # keep exact value

        # Store as DataFrame per cell
        df_cell = pd.DataFrame({"Current": currents, "Spikes": spike_counts})
        all_data[cell_id] = df_cell

    # Combine into one FI DataFrame
    df_final = pd.DataFrame({"Current": range(0, 650, 25)})
    for cell_id, df_cell in all_data.items():
        # Assign each sweep to nearest 25 pA bin
        df_cell["Bin"] = (df_cell["Current"] / 25).round() * 25
        spikes_per_bin = df_cell.groupby("Bin")["Spikes"].mean()
        df_final[cell_id] = df_final["Current"].map(spikes_per_bin)

    # Save to Excel
    df_final.to_excel(output_path, index=False)
    print(f"✅ Saved: {output_path}")


# ---------- RUN ----------
for folder_name in folders:
    folder_path = os.path.join(base_folder, folder_name)
    output_path = os.path.join(base_folder, f"{folder_name}_FI.xlsx")

    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    process_folder(folder_path, output_path)
# %%

