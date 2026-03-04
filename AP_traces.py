import pyabf
import numpy as np
import matplotlib.pyplot as plt
import os

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

# ---------- FUNCTIONS ----------
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

def get_baseline(y, t, pre_step_window=0.1):
    return np.mean(y[t < pre_step_window])

# ---------- PROCESS FOLDER ----------
def process_folder(folder_path, group):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".abf"):
            continue

        abf_path = os.path.join(folder_path, filename)
        abf = pyabf.ABF(abf_path)
        cell_id = filename[:9]

        # Determine cell type
        if "-" in cell_id:
            marker = cell_id.split("-")[1][0].lower()
            cell_type = "Sup" if marker == "s" else "Deep" if marker == "d" else "Unknown"
        else:
            cell_type = "Unknown"

        # --- Find first sweep with spike ---
        first_spike_sweep = None
        first_spike_index = None

        for sweep in range(abf.sweepCount):
            abf.setSweep(sweep)
            t, y = abf.sweepX, abf.sweepY
            baseline = get_baseline(y, t, pre_step_window)
            threshold = baseline + spike_height_above_baseline
            spikes = detect_spikes(t, y, threshold, abf.dataRate)

            if len(spikes) > 0:
                first_spike_sweep = sweep
                first_spike_index = spikes[0]
                break

        if first_spike_sweep is None:
            continue

        # --- Extract first AP ---
        abf.setSweep(first_spike_sweep)
        t, y = abf.sweepX, abf.sweepY

        pre_pts = int(ap_pre_ms / 1000 * abf.dataRate)
        post_pts = int(ap_post_ms / 1000 * abf.dataRate)

        start = max(first_spike_index - pre_pts, 0)
        end = min(first_spike_index + post_pts, len(y))

        baseline = y[start]
        ap_wave = y[start:end] - baseline
        amp = np.max(ap_wave)

        # fAHP (0–5 ms)
        f_start = first_spike_index
        f_end = min(first_spike_index + int(0.005 * abf.dataRate), len(y))
        fAHP = np.min(y[f_start:f_end] - baseline)

        # mAHP (5–50 ms)
        m_start = first_spike_index + int(0.005 * abf.dataRate)
        m_end = min(first_spike_index + int(0.05 * abf.dataRate), len(y))
        mAHP = np.min(y[m_start:m_end] - baseline)

        # Max slope (mV/ms)
        dv = np.diff(ap_wave)
        dt = np.diff(t[start:end])
        max_slope = np.max((dv / dt) / 1000)

        results.append({
            "CellID": cell_id,
            "Group": group,
            "CellType": cell_type,
            "dataRate": abf.dataRate,
            "ap_waveforms": [ap_wave],
            "fAHPs": [fAHP],
            "mAHPs": [mAHP],
            "max_slopes": [max_slope],
            "Amplitude": [amp]
        })

# ---------- RUN ----------
process_folder(control_folder, "Control")
process_folder(experimental_folder, "Experimental")

# ---------- PLOT AVERAGE APs PER CELL ----------
fig, axes = plt.subplots(2, 3, figsize=(6, 6))
axes = axes.flatten()

groups = [
    ("Control", "Sup"),
    ("Control", "Deep"),
    ("Experimental", "Sup"),
    ("Experimental", "Deep")
]

for i, (g, ct) in enumerate(groups):
    ax = axes[i]

    subset = [r for r in results if r["Group"] == g and r["CellType"] == ct]
    if not subset:
        ax.set_visible(False)
        continue

    cell_ids = list(set([r["CellID"] for r in subset]))
    cell_mean_aps = []

    for cid in cell_ids:
        aps = np.vstack([
            np.mean(np.vstack(r["ap_waveforms"]), axis=0)
            for r in subset if r["CellID"] == cid
        ])

        mean_ap = np.mean(aps, axis=0)
        cell_mean_aps.append(mean_ap)

        t_axis = np.linspace(-ap_pre_ms, ap_post_ms, len(mean_ap))
        ax.plot(t_axis, mean_ap, color='gray', alpha=0.5)

    grand_mean = np.mean(np.vstack(cell_mean_aps), axis=0)
    t_axis = np.linspace(-ap_pre_ms, ap_post_ms, len(grand_mean))
    ax.plot(t_axis, grand_mean, color=colors[g], lw=2)

    ax.set_title(f"{g} {ct} (n={len(cell_mean_aps)})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ΔVm (mV)")
    ax.set_xlim(-ap_pre_ms, ap_post_ms)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ---------- COMPARISONS ----------
comparisons = [
    (("Control","Sup"),("Experimental","Sup")),
    (("Control","Deep"),("Experimental","Deep"))
]

def avg_per_cell(subset):
    cell_ids = list(set([r["CellID"] for r in subset]))
    cell_means = []
    for cid in cell_ids:
        aps = np.vstack([
            np.mean(np.vstack(r["ap_waveforms"]), axis=0)
            for r in subset if r["CellID"] == cid
        ])
        cell_means.append(np.mean(aps, axis=0))
    return np.mean(np.vstack(cell_means), axis=0)

for j, ((g1, ct1), (g2, ct2)) in enumerate(comparisons):
    ax = axes[4 + j]

    subset1 = [r for r in results if r["Group"] == g1 and r["CellType"] == ct1]
    subset2 = [r for r in results if r["Group"] == g2 and r["CellType"] == ct2]

    if not subset1 or not subset2:
        ax.set_visible(False)
        continue

    mean1 = avg_per_cell(subset1)
    mean2 = avg_per_cell(subset2)

    t_axis = np.linspace(-ap_pre_ms, ap_post_ms, len(mean1))

    ax.plot(t_axis, mean1, color=colors[g1], lw=2,
            label=f"{g1} (n={len(set([r['CellID'] for r in subset1]))})")
    ax.plot(t_axis, mean2, color=colors[g2], lw=2,
            label=f"{g2} (n={len(set([r['CellID'] for r in subset2]))})")

    ax.set_title(f"{ct1} comparison")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("ΔmV")
    ax.set_xlim(-ap_pre_ms, ap_post_ms)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(r"Z:\Kristian\Projects\Deep_Sup\Figures\actionpotentials.svg",
            dpi=1200, bbox_inches='tight')
plt.show()