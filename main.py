import tkinter as tk
from tkinter import filedialog, messagebox
import mne
import numpy as np
import scipy.linalg
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ==========================================
# 1. THE MATH CLASS (FROM SCRATCH LOGIC)
# ==========================================
class BCIMath:
    """
    Implements BCI algorithms from scratch based on Pfurtscheller (1999)
    and standard CSP methodology.
    """
    
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        """
        Standard Butterworth bandpass filter using scipy.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        # filtfilt applies filter forward and backward (zero-phase distortion)
        y = filtfilt(b, a, data, axis=-1)
        return y

    @staticmethod
    def get_epochs_manual(data, events, event_id, fs, tmin, tmax):
        """
        Manually slices the continuous data array into epochs.
        Returns: (n_trials, n_channels, n_samples)
        """
        # Convert time to samples
        smin = int(tmin * fs)
        smax = int(tmax * fs)
        n_samples = smax - smin
        
        # Filter events for the specific ID
        target_events = events[events[:, 2] == event_id]
        
        epochs = []
        for evt in target_events:
            start_sample = evt[0] + smin
            end_sample = evt[0] + smax
            
            # Check boundary conditions
            if start_sample >= 0 and end_sample < data.shape[1]:
                epoch = data[:, start_sample:end_sample]
                epochs.append(epoch)
                
        return np.array(epochs)

    @staticmethod
    def compute_erd_ers(data, events, event_ids, fs, tmin, tmax, ref_tmin, ref_tmax):
        """
        Calculates ERD/ERS using the Band Power Method (Pfurtscheller 1999).
        
        Steps:
        1. Bandpass Filter (done before passing data usually, but we assume data is filtered)
        2. Slice Data
        3. Square (Power)
        4. Average across trials
        5. Normalize against Reference (R)
        """
        results = {}
        
        # Time vector for plotting
        n_samples = int((tmax - tmin) * fs)
        times = np.linspace(tmin, tmax, n_samples)
        
        # Indices for reference period inside the epoch
        ref_idx_start = int((ref_tmin - tmin) * fs)
        ref_idx_end = int((ref_tmax - tmin) * fs)

        for label, code in event_ids.items():
            # 1. Get Epochs (Trials x Channels x Time)
            epochs = BCIMath.get_epochs_manual(data, events, code, fs, tmin, tmax)
            
            if len(epochs) == 0:
                continue

            # 2. Squaring (Power samples)
            power = epochs ** 2
            
            # 3. Averaging across trials (Result: Channels x Time)
            avg_power = np.mean(power, axis=0)
            
            # 4. Calculate Reference (R)
            # Average power in the reference interval for each channel
            R = np.mean(avg_power[:, ref_idx_start:ref_idx_end], axis=1, keepdims=True)
            
            # 5. ERD/ERS Calculation: (A - R) / R * 100
            erd_ers = ((avg_power - R) / R) * 100
            
            results[label] = erd_ers
            
        return times, results

    @staticmethod
    def compute_csp_manual(data, events, event_ids, fs, tmin, tmax):
        """
        Calculates Common Spatial Patterns (CSP) from scratch.
        
        Math:
        1. Calculate Covariance Matrix for each class.
        2. Solve Generalized Eigenvalue Problem: C1 * w = lambda * (C1 + C2) * w
        3. Extract Spatial Patterns (A) from Filters (W): A = (W^-1).T
        """
        # 1. Extract Epochs for Class 1 and Class 2
        labels = list(event_ids.keys())
        epochs_c1 = BCIMath.get_epochs_manual(data, events, event_ids[labels[0]], fs, tmin, tmax)
        epochs_c2 = BCIMath.get_epochs_manual(data, events, event_ids[labels[1]], fs, tmin, tmax)
        
        # 2. Compute Average Covariance Matrices
        def compute_cov(epoch_data):
            # epoch_data: (Trials, Channels, Time)
            covs = []
            for trial in epoch_data:
                # Center the data (remove mean)
                trial_centered = trial - np.mean(trial, axis=1, keepdims=True)
                # Covariance: (X * X.T) / trace(X * X.T)
                c = np.dot(trial_centered, trial_centered.T)
                c = c / np.trace(c)
                covs.append(c)
            return np.mean(covs, axis=0)

        C1 = compute_cov(epochs_c1)
        C2 = compute_cov(epochs_c2)
        
        # 3. Generalized Eigenvalue Decomposition
        # We want to maximize variance for C1 relative to C1+C2 (or just C2)
        # scipy.linalg.eigh solves A x = lambda B x
        eigenvalues, eigenvectors = scipy.linalg.eigh(C1, C1 + C2)
        
        # 4. Sort Eigenvectors (Filters)
        # eigh returns them in ascending order.
        # We want the extremes (First and Last)
        ix = np.argsort(eigenvalues)[::-1] # Descending
        sorted_eigenvectors = eigenvectors[:, ix]
        
        # The columns of sorted_eigenvectors are the Spatial Filters (W)
        W = sorted_eigenvectors.T
        
        # 5. Compute Spatial Patterns (A)
        # The patterns are what we plot on the head.
        # A = (W^-1).T. Since W is orthogonal in CSP context, roughly A ~ W.T (or pinv)
        # Standard formula: Patterns = pinv(Filters).T
        patterns = scipy.linalg.pinv(W).T
        
        return patterns, labels

# ==========================================
# 2. THE GUI APP
# ==========================================
class BCIAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI EEG Analysis (Manual Implementation)")
        self.root.geometry("450x350")

        # Variables
        self.raw = None
        self.filename = None
        self.events = None
        self.event_id = None
        
        # UI Elements
        tk.Label(root, text="BCI Competition 2008 - Graz Data Analysis", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(root, text="Analysis implemented from scratch (NumPy/SciPy)", font=("Arial", 9, "italic")).pack(pady=0)
        
        self.btn_load = tk.Button(root, text="1. Load GDF File", command=self.load_data, width=30)
        self.btn_load.pack(pady=10)

        self.lbl_status = tk.Label(root, text="No file loaded", fg="gray")
        self.lbl_status.pack(pady=5)

        tk.Label(root, text="Analysis Methods", font=("Arial", 10, "bold")).pack(pady=10)

        self.btn_erd = tk.Button(root, text="2. Run ERD/ERS (Band Power)", command=self.run_erd_ers, state=tk.DISABLED, width=30)
        self.btn_erd.pack(pady=5)

        self.btn_csp = tk.Button(root, text="3. Run CSP (Spatial Patterns)", command=self.run_csp, state=tk.DISABLED, width=30)
        self.btn_csp.pack(pady=5)

    def load_data(self):
        """Loads GDF, picks C3/Cz/C4, renames channels."""
        file_path = filedialog.askopenfilename(filetypes=[("GDF files", "*.gdf")])
        if not file_path: return

        try:
            # Load Data
            self.raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            
            # Select Channels
            target_channels = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
            existing_chs = self.raw.ch_names
            to_pick = [ch for ch in target_channels if ch in existing_chs]
            
            if len(to_pick) != 3:
                messagebox.showwarning("Channel Warning", f"Expected 3 channels, found: {to_pick}")
            
            self.raw.pick_channels(to_pick)

            # Rename for MNE Plotting compatibility
            rename_dict = {ch: ch.replace('EEG:', '') for ch in to_pick}
            self.raw.rename_channels(rename_dict)
            
            # Set Montage (needed for CSP plotting later)
            montage = mne.channels.make_standard_montage('standard_1020')
            self.raw.set_montage(montage)

            # Extract Events
            events, event_id = mne.events_from_annotations(self.raw, verbose=False)
            
            # Map 769/770
            target_event_id = {}
            for key, val in event_id.items():
                if '769' in key: target_event_id['Left Hand'] = val
                elif '770' in key: target_event_id['Right Hand'] = val
            
            if not target_event_id:
                messagebox.showerror("Error", "Could not find Left/Right hand event markers.")
                return

            self.events = events
            self.event_id = target_event_id
            self.filename = file_path.split("/")[-1]
            
            self.lbl_status.config(text=f"Loaded: {self.filename}\nChannels: {self.raw.ch_names}", fg="green")
            self.btn_erd.config(state=tk.NORMAL)
            self.btn_csp.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_erd_ers(self):
        """Uses BCIMath class to calculate ERD/ERS from scratch."""
        if self.raw is None: return

        # 1. Get raw data array (Channels x Samples)
        # We multiply by 1e6 to convert Volts to Microvolts for better readability, 
        # though ERD is a ratio so units cancel out.
        raw_data = self.raw.get_data() * 1e6 
        fs = self.raw.info['sfreq']

        # 2. Filter (8-30 Hz) using Manual Butterworth
        filtered_data = BCIMath.butter_bandpass_filter(raw_data, 8.0, 30.0, fs, order=4)

        # 3. Compute ERD
        # Time window: -1.5s to 4.5s
        # Reference window: -1.0s to 0.0s
        times, erd_results = BCIMath.compute_erd_ers(
            filtered_data, self.events, self.event_id, fs, 
            tmin=-1.5, tmax=4.5, ref_tmin=-1.0, ref_tmax=0.0
        )

        # 4. Plotting
        self._plot_erd(times, erd_results)

    def _plot_erd(self, times, results):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        channels = self.raw.ch_names
        
        # results is a dict: {'Left Hand': array(3, time), 'Right Hand': array(3, time)}
        
        for i, ax in enumerate(axes):
            ax.plot(times, results['Left Hand'][i], label='Left Hand', color='blue')
            ax.plot(times, results['Right Hand'][i], label='Right Hand', color='red')
            
            ax.set_title(f"ERD/ERS (8-30Hz): {channels[i]}")
            ax.set_ylabel("Power Change (%)")
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='green', linestyle='-', linewidth=1.5, label='Cue')
            ax.legend(loc='upper right')

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"ERD/ERS Analysis (Manual Calculation) - {self.filename}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def run_csp(self):
        """Uses BCIMath class to calculate CSP patterns from scratch."""
        if self.raw is None: return

        # 1. Get Data & Filter
        raw_data = self.raw.get_data() * 1e6
        fs = self.raw.info['sfreq']
        filtered_data = BCIMath.butter_bandpass_filter(raw_data, 8.0, 30.0, fs, order=4)

        # 2. Compute CSP Patterns Manually
        # Active window: 0.5s to 3.5s
        patterns, labels = BCIMath.compute_csp_manual(
            filtered_data, self.events, self.event_id, fs, tmin=0.5, tmax=3.5
        )

        # 3. Plotting
        # We use MNE's plot_topomap because drawing heads from scratch is complex.
        # We need to visualize the patterns associated with the most discriminative filters.
        # Usually, the first component (index 0) and last component (index -1) are the most important.
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        info = self.raw.info
        
        # Component 1 (Largest Eigenvalue)
        mne.viz.plot_topomap(patterns[0, :], info, axes=axes[0], show=False, sphere=0.12)
        axes[0].set_title("CSP Component 1\n(Discriminates Class 1)")

        # Component 3 (Smallest Eigenvalue - Last one)
        # Since we have 3 channels, we have 3 components.
        mne.viz.plot_topomap(patterns[-1, :], info, axes=axes[1], show=False, sphere=0.12)
        axes[1].set_title("CSP Component 3\n(Discriminates Class 2)")

        plt.suptitle(f"CSP Patterns (Manual Calculation) - {self.filename}")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = BCIAnalysisApp(root)  
    root.mainloop()