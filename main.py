import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import skew, kurtosis
from erdcsp import ANALYSIS as ANALYSIS
from classifier import CLASSIFIER as CLASSIFIER

raw = None
filename = None
events = None
event_id = None
fs = 250.0
classifier = CLASSIFIER()
txt_low = None
txt_high = None
btn_load = None
lbl_status = None
btn_erd = None
btn_csp = None
btn_train = None
btn_save = None
btn_load_model = None
btn_predict = None

def load_data():
    global raw, filename, events, event_id, fs
    global btn_erd, btn_csp, btn_train, btn_predict, lbl_status

    file_path = filedialog.askopenfilename(filetypes=[("GDF files", "*.gdf")])
    if not file_path:
        return

    try:
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

        target_channels = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
        existing_chs = raw.ch_names
        to_pick = [ch for ch in target_channels if ch in existing_chs]

        if len(to_pick) != 3:
            messagebox.showwarning("Channel Warning", f"Expected 3 channels, found: {to_pick}")

        raw.pick_channels(to_pick)
        rename_dict = {ch: ch.replace('EEG:', '') for ch in to_pick}
        raw.rename_channels(rename_dict)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        events_map, event_id_map = mne.events_from_annotations(raw, verbose=False)
        events = events_map
        fs = raw.info['sfreq']

        event_id = {}
        for key, val in event_id_map.items():
            if '769' in key:
                event_id['Left Hand'] = val
            elif '770' in key:
                event_id['Right Hand'] = val
            elif '783' in key:
                event_id['Unknown'] = val

        filename = file_path.split('/')[-1]
        lbl_status.config(text=f"Loaded: {filename}\nEvents: {list(event_id.keys())}", fg="green")

        btn_erd.config(state=tk.NORMAL)
        btn_csp.config(state=tk.NORMAL)

        if 'Left Hand' in event_id and 'Right Hand' in event_id:
            btn_train.config(state=tk.NORMAL)
        else:
            btn_train.config(state=tk.DISABLED)

        if 'Unknown' in event_id:
            btn_predict.config(state=tk.NORMAL)
        else:
            btn_predict.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def get_filtered_data():
    global raw, fs, txt_low, txt_high
    if raw is None:
        raise RuntimeError("No raw data loaded")
    
    try:
        l_cut = float(txt_low.get())
        h_cut = float(txt_high.get())
    except (ValueError, AttributeError):
        l_cut, h_cut = 8.0, 30.0
        
    raw_data = raw.get_data() * 1e6
    return ANALYSIS.butter_bandpass_filter(raw_data, l_cut, h_cut, fs, order=4)
def run_erd_ers():
    global raw, events, event_id, fs
    if raw is None:
        return

    data = get_filtered_data()
    labeled_ids = {k: v for k, v in event_id.items() if k in ['Left Hand', 'Right Hand']}

    times, results = ANALYSIS.compute_erd_ers(
        data, events, labeled_ids, fs, tmin=-1.5, tmax=4.5, ref_tmin=-1.0, ref_tmax=0.0
    )

    stats_data = [] 
    mask_active = (times >= 0.0) & (times <= 4.0)
    channels = raw.ch_names

    for i, ch_name in enumerate(channels):
        if 'Left Hand' in results:
            sig = results['Left Hand'][i][mask_active]
            stats_data.append((
                ch_name, "Left", 
                f"{np.mean(sig):.1f}%", 
                f"{np.var(sig):.1f}", 
                f"{np.min(sig):.1f}%", 
                f"{np.max(sig):.1f}%",
                f"{skew(sig):.2f}",
                f"{kurtosis(sig):.2f}"
            ))

        if 'Right Hand' in results:
            sig = results['Right Hand'][i][mask_active]
            stats_data.append((
                ch_name, "Right", 
                f"{np.mean(sig):.1f}%", 
                f"{np.var(sig):.1f}", 
                f"{np.min(sig):.1f}%", 
                f"{np.max(sig):.1f}%",
                f"{skew(sig):.2f}",
                f"{kurtosis(sig):.2f}"
            ))

    stats_win = tk.Toplevel()
    stats_win.title(f"Time Domain Features - {filename}")
    stats_win.geometry("800x300")
    columns = ("Channel", "Class", "Mean", "Var", "Peak ERD", "Peak ERS", "Skew", "Kurtosis")
    tree = ttk.Treeview(stats_win, columns=columns, show='headings')

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=90, anchor='center')

    for row in stats_data:
        tree.insert('', tk.END, values=row)

    scrollbar = ttk.Scrollbar(stats_win, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.pack(fill=tk.BOTH, expand=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    for i, ax in enumerate(axes):
        ch_name = channels[i]
        
        if 'Left Hand' in results:
            ax.plot(times, results['Left Hand'][i], 'b', label='Left', linewidth=1)
        
        if 'Right Hand' in results:
            ax.plot(times, results['Right Hand'][i], 'r', label='Right', linewidth=1)

        ax.set_title(f"ERD/ERS: {ch_name}", fontsize=10)
        ax.axvline(0, color='g', linestyle='--', alpha=0.7)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_ylabel("% Pow")
        
        if i == 0: ax.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"ERD/ERS Analysis\nFile: {filename}", fontsize=12)
    plt.show()

def run_cwt_analysis():
    global raw, events, event_id, fs, txt_low, txt_high
    if raw is None: return

    available_classes = []
    if 'Left Hand' in event_id: available_classes.append('Left Hand')
    if 'Right Hand' in event_id: available_classes.append('Right Hand')
    
    if not available_classes:
        messagebox.showwarning("Data", "No Left/Right labels found.")
        return

    n_channels = 3
    n_classes = len(available_classes)
    
    fig, axes = plt.subplots(n_channels, n_classes, figsize=(4 * n_classes, 8), 
                             sharex=True, sharey=True)
    
    if n_classes == 1:
        axes = axes[:, np.newaxis]

    data_raw = raw.get_data() * 1e6
    
    for col_idx, label_name in enumerate(available_classes):
        target_id = event_id[label_name]
        
        epochs, _ = ANALYSIS.get_epochs_manual(data_raw, events, target_id, fs, -1.0, 4.0)
        
        if len(epochs) == 0: continue

        freqs, power = ANALYSIS.cwt(epochs, fs, fmin=2, fmax=40, wavelet='cmor0.5-1.5')
        
        for row_idx in range(n_channels):
            ax = axes[row_idx, col_idx]
            
            im = ax.imshow(power[row_idx], extent=[-1.0, 4.0, 2, 40], aspect='auto', 
                           cmap='jet', origin='lower')
            
            if row_idx == 0:
                ax.set_title(f"{label_name}", fontsize=12, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f"{raw.ch_names[row_idx]}\nFreq (Hz)", fontsize=10)
            
            ax.axvline(0, color='white', linestyle='--', alpha=0.7)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Power (uV^2)')

    fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=12)
    plt.suptitle(f"Time-Frequency Analysis (CWT: cmor0.5-1.5)\nFile: {filename}", fontsize=14)
    
    plt.subplots_adjust(right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.show()

def run_csp_plot():
    global raw, events, event_id, fs
    if raw is None:
        return

    data = get_filtered_data()
    labeled_ids = {k: v for k, v in event_id.items() if k in ['Left Hand', 'Right Hand']}

    try:
        W = ANALYSIS.gen_csp(data, events, labeled_ids, fs, 0.5, 3.5)
        patterns = scipy.linalg.pinv(W).T

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        info = raw.info
        mne.viz.plot_topomap(patterns[0, :], info, axes=axes[0], show=False, contours=0, sensors=True, names=raw.ch_names)
        axes[0].set_title("Component 1 (Left)")
        mne.viz.plot_topomap(patterns[-1, :], info, axes=axes[1], show=False, contours=0, sensors=True, names=raw.ch_names)
        axes[1].set_title("Component 3 (Right)")
        plt.show()

    except Exception as e:
        messagebox.showerror("CSP Error", str(e))


def train_model():
    global raw, events, event_id, fs, classifier, btn_save
    if raw is None:
        return

    try:
        data = get_filtered_data()
        labeled_ids = {k: v for k, v in event_id.items() if k in ['Left Hand', 'Right Hand']}

        acc, kappa, cm, loss_curve, total_trials = classifier.train_model(data, events, labeled_ids, fs)

        btn_save.config(state=tk.NORMAL)

        id_left = labeled_ids['Left Hand']
        id_right = labeled_ids['Right Hand']
        t_course, acc_course = classifier.perform_sliding_window_analysis(data, events, labeled_ids, fs)

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left', 'Right'])
        disp.plot(cmap='Purples', ax=ax1, colorbar=False)
        ax1.set_title(f"ANN Confusion Matrix\nAcc: {acc:.2%} | Kappa: {kappa:.2f}")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        text_info = (
            f"Neural Network Performance\n"
            f"Topology: Input [2] -> [20, 10] -> Output [2]\n"
            f"Total Trials: {total_trials}\n\n"
            f"Overall Accuracy: {acc:.2%}\n"
        )
        ax2.text(0.1, 0.5, text_info, fontsize=11, verticalalignment='center')

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(loss_curve, color='red')
        ax3.set_title("Training Loss (Convergence)")
        ax3.set_xlabel("Iterations")
        ax3.set_ylabel("Loss")
        ax3.grid(True)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(t_course, acc_course, 'o-', color='purple', linewidth=2, markersize=4)
        ax4.axhline(0.5, color='gray', linestyle='--', label='Chance')
        ax4.axvline(0, color='green', linestyle='-', label='Cue')
        ax4.set_ylim(0, 1.05)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Performance Over Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Training Error", str(e))


def save_model():
    global classifier
    if not hasattr(classifier, 'trained_clf') or classifier.trained_clf is None:
        messagebox.showwarning("No Model", "No trained model to save.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".dat", filetypes=[("DAT files", "*.dat")])
    if not file_path:
        return

    try:
        classifier.save_model(file_path)
        messagebox.showinfo("Saved", f"Model saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Save Error", str(e))


def load_model():
    global classifier
    file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
    if not file_path:
        return

    try:
        classifier.load_model(file_path)
        messagebox.showinfo("Loaded", "Model loaded successfully.")
    except Exception as e:
        messagebox.showerror("Load Error", str(e))


def predict_unknowns():
    global raw, events, event_id, fs, classifier, filename
    if not hasattr(classifier, 'trained_clf') or classifier.trained_clf is None:
        messagebox.showwarning("No Model")
        return

    if 'Unknown' not in event_id:
        messagebox.showwarning("No Data", "Current file does not have 'Unknown' (783) events.")
        return

    try:
        data = get_filtered_data()
        id_unknown = event_id['Unknown']

        preds, count_left, count_right = classifier.forward_prop(data, events, id_unknown, fs)

        plt.figure(figsize=(10, 4))
        plt.plot(preds, 'o-', label='Prediction (0=Left, 1=Right)')
        plt.yticks([0, 1], ['Left', 'Right'])
        plt.xlabel("Trial Number")
        plt.title(f"Predictions for {filename}\nLeft: {count_left}, Right: {count_right}")
        plt.grid(True, axis='y')
        plt.legend()
        plt.show()

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

def main(root):
    global btn_load, lbl_status, btn_erd, btn_csp, btn_train, btn_save, btn_load_model, btn_predict, txt_low, txt_high

    root.title("ASN - EEG Analysis")
    root.geometry("500x750")

    tk.Label(root, text="ASN - EEG Analysis", font=("Arial", 14, "bold")).pack(pady=10)

    frame_load = tk.LabelFrame(root, text="Data Loading", padx=10, pady=5)
    frame_load.pack(fill="x", padx=10, pady=5)

    btn_load = tk.Button(frame_load, text="Load GDF File", command=load_data)
    btn_load.pack(fill="x")
    lbl_status = tk.Label(frame_load, text="-", fg="gray")
    lbl_status.pack()

    frame_analysis = tk.LabelFrame(root, text="Analysis", padx=10, pady=5)
    frame_analysis.pack(fill="x", padx=10, pady=5)

    
    frame_freq = tk.Frame(frame_analysis)
    frame_freq.pack(fill="x", pady=5)
    
    tk.Label(frame_freq, text="BP Filter (Hz):").pack(side=tk.LEFT)
    
    txt_low = tk.Entry(frame_freq, width=5)
    txt_low.insert(0, "8.0")
    txt_low.pack(side=tk.LEFT, padx=5)
    
    tk.Label(frame_freq, text="-").pack(side=tk.LEFT)
    
    txt_high = tk.Entry(frame_freq, width=5)
    txt_high.insert(0, "30.0")
    txt_high.pack(side=tk.LEFT, padx=5)

    btn_cwt = tk.Button(frame_analysis, text="CWT", command=run_cwt_analysis)
    btn_cwt.pack(fill="x", pady=2)

    btn_erd = tk.Button(frame_analysis, text="ERD/ERS", command=run_erd_ers, state=tk.DISABLED)
    btn_erd.pack(fill="x", pady=2)
    btn_csp = tk.Button(frame_analysis, text="CSP Plot", command=run_csp_plot, state=tk.DISABLED)
    btn_csp.pack(fill="x", pady=2)
    frame_ml = tk.LabelFrame(root, text="Feature Extraction", padx=10, pady=5)
    frame_ml.pack(fill="x", padx=10, pady=5)

    btn_train = tk.Button(frame_ml, text="Train Model", command=train_model, state=tk.DISABLED)
    btn_train.pack(fill="x", pady=2)

    btn_save = tk.Button(frame_ml, text="Save Model", command=save_model, state=tk.DISABLED)
    btn_save.pack(fill="x", pady=2)

    btn_load_model = tk.Button(frame_ml, text="Load Model", command=load_model)
    btn_load_model.pack(fill="x", pady=2)

    tk.Label(frame_ml, text="Run Model").pack(pady=2)

    btn_predict = tk.Button(frame_ml, text="Forward Propagate", command=predict_unknowns, state=tk.DISABLED)
    btn_predict.pack(fill="x", pady=2)


root = tk.Tk()
main(root)
root.mainloop()