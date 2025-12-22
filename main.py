import tkinter as tk
from tkinter import filedialog, messagebox
import mne
from mne.decoding import CSP
import numpy as np
import matplotlib.pyplot as plt

class bci:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI EEG Analysis (Graz Dataset B)")
        self.root.geometry("400x350")

        # Variables
        self.raw = None
        self.filename = None
        self.events = None
        self.event_id = None
        
        # UI Elements
        tk.Label(root, text="BCI Competition 2008 - Graz Data Analysis", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.btn_load = tk.Button(root, text="1. Load GDF File", command=self.load_data, width=25)
        self.btn_load.pack(pady=5)

        self.lbl_status = tk.Label(root, text="No file loaded", fg="gray")
        self.lbl_status.pack(pady=5)

        tk.Label(root, text="Analysis Methods", font=("Arial", 10, "bold")).pack(pady=10)

        self.btn_erd = tk.Button(root, text="2. Run ERD/ERS Analysis", command=self.erd_ers, state=tk.DISABLED, width=25)
        self.btn_erd.pack(pady=5)

        self.btn_csp = tk.Button(root, text="3. Run CSP Analysis", command=self.csp, state=tk.DISABLED, width=25)
        self.btn_csp.pack(pady=5)

    def load_data(self):
        """Loads the GDF file, picks channels, and renames them for standard montage."""
        file_path = filedialog.askopenfilename(filetypes=[("GDF files", "*.gdf")])
        if not file_path:
            return

        try:
            self.raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            
            target_channels = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
            
            existing_chs = self.raw.ch_names
            to_pick = [ch for ch in target_channels if ch in existing_chs]
            
            if len(to_pick) != 3:
                messagebox.showwarning("Channel Warning", f"Expected 3 channels, found: {to_pick}")
            
            self.raw.pick_channels(to_pick)

            rename_dict = {ch: ch.replace('EEG:', '') for ch in to_pick}
            self.raw.rename_channels(rename_dict)

            events, event_id = mne.events_from_annotations(self.raw, verbose=False)
            
            target_event_id = {}
            for key, val in event_id.items():
                if '769' in key:
                    target_event_id['Left Hand'] = val
                elif '770' in key:
                    target_event_id['Right Hand'] = val
            
            if not target_event_id:
                messagebox.showerror("Error", "Could not find Left/Right hand event markers (769/770).")
                return

            self.events = events
            self.event_id = target_event_id
            self.filename = file_path.split("/")[-1]
            
            self.lbl_status.config(text=f"Loaded: {self.filename}\nChannels: {self.raw.ch_names}", fg="green")
            self.btn_erd.config(state=tk.NORMAL)
            self.btn_csp.config(state=tk.NORMAL)
            
            print(f"Data loaded. Channels renamed to: {self.raw.ch_names}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def erd_ers(self):
        if self.raw is None: return

        raw_band = self.raw.copy().filter(8., 30., fir_design='firwin', verbose=False)

        tmin, tmax = -1.5, 4.5
        epochs = mne.Epochs(raw_band, self.events, self.event_id, tmin, tmax, 
                            baseline=None, preload=True, verbose=False)

        data = epochs.get_data(copy=True) 
        power = data ** 2
    
        idx_left = epochs.events[:, 2] == self.event_id['Left Hand']
        idx_right = epochs.events[:, 2] == self.event_id['Right Hand']

        power_left = np.mean(power[idx_left], axis=0)
        power_right = np.mean(power[idx_right], axis=0)

        times = epochs.times
        ref_mask = (times >= -1.0) & (times <= 0.0)
        
        R_left = np.mean(power_left[:, ref_mask], axis=1, keepdims=True)
        R_right = np.mean(power_right[:, ref_mask], axis=1, keepdims=True)

        erd_left = ((power_left - R_left) / R_left) * 100
        erd_right = ((power_right - R_right) / R_right) * 100

        self._plot_erd(times, erd_left, erd_right)

    def _plot_erd(self, times, erd_left, erd_right):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channels = self.raw.ch_names
        
        for i, ax in enumerate(axes):
            ax.plot(times, erd_left[i], label='Left Hand', color='blue')
            ax.plot(times, erd_right[i], label='Right Hand', color='red')
            
            ax.set_title(f"ERD/ERS: {channels[i]}")
            ax.set_ylabel("Power Change (%)")
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='green', linestyle='-', linewidth=1.5, label='Cue')
            ax.legend(loc='upper right')

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"ERD/ERS Analysis (8-30 Hz) - {self.filename}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def csp(self):
        """
        Performs CSP analysis and plots spatial patterns.
        """
        if self.raw is None: return

        try:
            raw_csp = self.raw.copy().filter(8., 30., fir_design='firwin', verbose=False)
            epochs = mne.Epochs(raw_csp, self.events, self.event_id, tmin=0.5, tmax=3.5, 
                                baseline=None, preload=True, verbose=False)

            montage = mne.channels.make_standard_montage('standard_1020')
            epochs.set_montage(montage)

            csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
            
            labels = epochs.events[:, -1]
            data = epochs.get_data(copy=True)
            
            print("Fitting CSP...")
            csp.fit(data, labels)
            csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
            plt.show()

        except Exception as e:
            messagebox.showerror("CSP Error", str(e))


root = tk.Tk()
app = bci(root)
root.mainloop()