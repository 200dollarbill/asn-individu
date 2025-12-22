import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import mne
import numpy as np
import scipy.linalg
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ==========================================
# 1. THE MATH CLASS (ALGORITHMS)
# ==========================================
class BCIMath:
    """
    Implements BCI algorithms (Filtering, CSP, Feature Extraction).
    """
    
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, axis=-1)
        return y

    @staticmethod
    def get_epochs_manual(data, events, event_id, fs, tmin, tmax):
        """Slices continuous data into epochs."""
        smin = int(tmin * fs)
        smax = int(tmax * fs)
        
        # Handle case where event_id is a list (for multiple triggers) or int
        if isinstance(event_id, int):
            target_events = events[events[:, 2] == event_id]
        else:
            # Filter for any event in the list
            mask = np.isin(events[:, 2], event_id)
            target_events = events[mask]

        epochs = []
        labels = []
        for evt in target_events:
            start_sample = evt[0] + smin
            end_sample = evt[0] + smax
            
            if start_sample >= 0 and end_sample < data.shape[1]:
                epoch = data[:, start_sample:end_sample]
                epochs.append(epoch)
                labels.append(evt[2])
                
        return np.array(epochs), np.array(labels)

    @staticmethod
    def compute_csp_filters(data, events, event_ids, fs, tmin, tmax):
        """
        Calculates CSP Filters (W).
        Returns: filters (W), eigenvalues
        """
        labels = list(event_ids.keys())
        # Extract data for both classes
        epochs_c1, _ = BCIMath.get_epochs_manual(data, events, event_ids[labels[0]], fs, tmin, tmax)
        epochs_c2, _ = BCIMath.get_epochs_manual(data, events, event_ids[labels[1]], fs, tmin, tmax)
        
        if len(epochs_c1) == 0 or len(epochs_c2) == 0:
            raise ValueError("Not enough epochs to train CSP.")

        # Covariance calculation
        def compute_cov(epoch_data):
            covs = []
            for trial in epoch_data:
                trial_centered = trial - np.mean(trial, axis=1, keepdims=True)
                c = np.dot(trial_centered, trial_centered.T)
                c = c / np.trace(c)
                covs.append(c)
            return np.mean(covs, axis=0)

        C1 = compute_cov(epochs_c1)
        C2 = compute_cov(epochs_c2)
        
        # Generalized Eigenvalue Decomposition
        eigenvalues, eigenvectors = scipy.linalg.eigh(C1, C1 + C2)
        
        # Sort descending
        ix = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, ix]
        
        # The columns are the Spatial Filters (W)
        W = sorted_eigenvectors
        return W

    @staticmethod
    def extract_log_var_features(epochs, W):
        """
        Projects epochs through CSP filters and calculates Log-Variance.
        
        X: (Trials, Channels, Samples)
        W: (Channels, Components)
        
        Z = W.T * X
        Feature = log(var(Z))
        """
        n_trials = epochs.shape[0]
        n_components = W.shape[1]
        features = np.zeros((n_trials, n_components))
        
        for i in range(n_trials):
            # Project: (Components x Samples) = (Components x Channels) @ (Channels x Samples)
            Z = np.dot(W.T, epochs[i])
            
            # Variance across time (axis 1)
            var_Z = np.var(Z, axis=1)
            
            # Log-transform (normalize distribution)
            features[i, :] = np.log(var_Z)
            
        return features

    @staticmethod
    def compute_erd_ers(data, events, event_ids, fs, tmin, tmax, ref_tmin, ref_tmax):
        # ... (Same as previous implementation) ...
        n_samples = int((tmax - tmin) * fs)
        times = np.linspace(tmin, tmax, n_samples)
        ref_idx_start = int((ref_tmin - tmin) * fs)
        ref_idx_end = int((ref_tmax - tmin) * fs)
        results = {}
        for label, code in event_ids.items():
            epochs, _ = BCIMath.get_epochs_manual(data, events, code, fs, tmin, tmax)
            if len(epochs) == 0: continue
            power = epochs ** 2
            avg_power = np.mean(power, axis=0)
            R = np.mean(avg_power[:, ref_idx_start:ref_idx_end], axis=1, keepdims=True)
            erd_ers = ((avg_power - R) / R) * 100
            results[label] = erd_ers
        return times, results

# ==========================================
# 2. THE GUI APP
# ==========================================
class BCIAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI EEG Analysis & Classification")
        self.root.geometry("500x550")

        # Variables
        self.raw = None
        self.filename = None
        self.events = None
        self.event_id = None
        self.fs = 250.0
        
        # Model Storage
        self.trained_W = None # CSP Filters
        self.trained_lda = None # LDA Model
        self.class_labels = None # ['Left', 'Right']

        # UI Layout
        tk.Label(root, text="BCI Competition 2008 - Graz Data B", font=("Arial", 14, "bold")).pack(pady=10)
        
        # --- Section 1: Data Loading ---
        frame_load = tk.LabelFrame(root, text="1. Data Loading", padx=10, pady=5)
        frame_load.pack(fill="x", padx=10, pady=5)
        
        self.btn_load = tk.Button(frame_load, text="Load GDF File", command=self.load_data)
        self.btn_load.pack(fill="x")
        self.lbl_status = tk.Label(frame_load, text="No file loaded", fg="gray")
        self.lbl_status.pack()

        # --- Section 2: Analysis ---
        frame_analysis = tk.LabelFrame(root, text="2. Analysis (Visual)", padx=10, pady=5)
        frame_analysis.pack(fill="x", padx=10, pady=5)

        self.btn_erd = tk.Button(frame_analysis, text="Run ERD/ERS Analysis", command=self.run_erd_ers, state=tk.DISABLED)
        self.btn_erd.pack(fill="x", pady=2)
        self.btn_csp = tk.Button(frame_analysis, text="Run CSP Pattern Plot", command=self.run_csp_plot, state=tk.DISABLED)
        self.btn_csp.pack(fill="x", pady=2)

        # --- Section 3: Classification ---
        frame_ml = tk.LabelFrame(root, text="3. Machine Learning (Train/Predict)", padx=10, pady=5)
        frame_ml.pack(fill="x", padx=10, pady=5)

        self.btn_train = tk.Button(frame_ml, text="Train Model (Current File)", command=self.train_model, state=tk.DISABLED, bg="#e1f5fe")
        self.btn_train.pack(fill="x", pady=2)
        
        self.btn_save = tk.Button(frame_ml, text="Save Trained Model (.dat)", command=self.save_model, state=tk.DISABLED)
        self.btn_save.pack(fill="x", pady=2)
        
        self.btn_load_model = tk.Button(frame_ml, text="Load Model (.dat)", command=self.load_model)
        self.btn_load_model.pack(fill="x", pady=2)

        tk.Label(frame_ml, text="--- Prediction ---").pack(pady=2)
        
        self.btn_predict = tk.Button(frame_ml, text="Predict Unknowns (Event 783)", command=self.predict_unknowns, state=tk.DISABLED, bg="#fff3e0")
        self.btn_predict.pack(fill="x", pady=2)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("GDF files", "*.gdf")])
        if not file_path: return

        try:
            # Load Data
            self.raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            
            # Pick Channels
            target_channels = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
            existing_chs = self.raw.ch_names
            to_pick = [ch for ch in target_channels if ch in existing_chs]
            
            if len(to_pick) != 3:
                messagebox.showwarning("Channel Warning", f"Expected 3 channels, found: {to_pick}")
            
            self.raw.pick_channels(to_pick)
            
            # Rename for MNE
            rename_dict = {ch: ch.replace('EEG:', '') for ch in to_pick}
            self.raw.rename_channels(rename_dict)
            montage = mne.channels.make_standard_montage('standard_1020')
            self.raw.set_montage(montage)

            # Events
            events, event_id = mne.events_from_annotations(self.raw, verbose=False)
            self.events = events
            self.fs = self.raw.info['sfreq']
            
            # Map standard Graz B codes
            # 769: Left, 770: Right, 783: Unknown/Cue
            self.event_id = {}
            for key, val in event_id.items():
                if '769' in key: self.event_id['Left Hand'] = val
                elif '770' in key: self.event_id['Right Hand'] = val
                elif '783' in key: self.event_id['Unknown'] = val

            self.filename = file_path.split("/")[-1]
            self.lbl_status.config(text=f"Loaded: {self.filename}\nEvents: {list(self.event_id.keys())}", fg="green")
            
            # Enable Buttons
            self.btn_erd.config(state=tk.NORMAL)
            self.btn_csp.config(state=tk.NORMAL)
            
            # Only enable Train if we have Left/Right labels
            if 'Left Hand' in self.event_id and 'Right Hand' in self.event_id:
                self.btn_train.config(state=tk.NORMAL)
            else:
                self.btn_train.config(state=tk.DISABLED)
                
            # Enable Predict if we have Unknown labels (or if user wants to force it)
            if 'Unknown' in self.event_id:
                self.btn_predict.config(state=tk.NORMAL)
            else:
                # Some eval files might not have 783, but we might want to predict anyway
                # For now, keep disabled unless 783 is found
                self.btn_predict.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_filtered_data(self):
        raw_data = self.raw.get_data() * 1e6
        return BCIMath.butter_bandpass_filter(raw_data, 8.0, 30.0, self.fs, order=4)

    # --- ANALYSIS METHODS ---
    def run_erd_ers(self):
        if self.raw is None: return
        data = self.get_filtered_data()
        # Only analyze labeled data
        labeled_ids = {k: v for k, v in self.event_id.items() if k in ['Left Hand', 'Right Hand']}
        
        times, results = BCIMath.compute_erd_ers(
            data, self.events, labeled_ids, self.fs, 
            tmin=-1.5, tmax=4.5, ref_tmin=-1.0, ref_tmax=0.0
        )
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        channels = self.raw.ch_names
        for i, ax in enumerate(axes):
            if 'Left Hand' in results: ax.plot(times, results['Left Hand'][i], 'b', label='Left')
            if 'Right Hand' in results: ax.plot(times, results['Right Hand'][i], 'r', label='Right')
            ax.set_title(f"ERD/ERS: {channels[i]}")
            ax.axvline(0, color='g', linestyle='--')
        plt.legend()
        plt.show()

    def run_csp_plot(self):
        if self.raw is None: return
        data = self.get_filtered_data()
        labeled_ids = {k: v for k, v in self.event_id.items() if k in ['Left Hand', 'Right Hand']}
        
        try:
            # Calculate Filters (W)
            W = BCIMath.compute_csp_filters(data, self.events, labeled_ids, self.fs, 0.5, 3.5)
            
            # Calculate Patterns (A = inv(W).T)
            patterns = scipy.linalg.pinv(W).T
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            info = self.raw.info
            mne.viz.plot_topomap(patterns[0, :], info, axes=axes[0], show=False, contours=0, sensors=True, names=self.raw.ch_names)
            axes[0].set_title("Component 1 (Left)")
            mne.viz.plot_topomap(patterns[-1, :], info, axes=axes[1], show=False, contours=0, sensors=True, names=self.raw.ch_names)
            axes[1].set_title("Component 3 (Right)")
            plt.show()
            
        except Exception as e:
            messagebox.showerror("CSP Error", str(e))

    # --- MACHINE LEARNING METHODS ---
    def train_model(self):
        """Trains CSP + LDA on the currently loaded labeled file."""
        if self.raw is None: return
        
        try:
            data = self.get_filtered_data()
            
            # 1. Prepare Data
            # We need Left (Class 0) and Right (Class 1)
            id_left = self.event_id['Left Hand']
            id_right = self.event_id['Right Hand']
            
            # Get Epochs (0.5s to 3.5s post cue)
            epochs_L, labels_L = BCIMath.get_epochs_manual(data, self.events, id_left, self.fs, 0.5, 3.5)
            epochs_R, labels_R = BCIMath.get_epochs_manual(data, self.events, id_right, self.fs, 0.5, 3.5)
            
            # Combine
            X_train = np.concatenate((epochs_L, epochs_R), axis=0)
            y_train = np.concatenate((np.zeros(len(epochs_L)), np.ones(len(epochs_R)))) # 0=Left, 1=Right
            
            # 2. Train CSP Filters (W)
            # We pass the dict to our helper to calculate W
            labeled_ids = {'Left': id_left, 'Right': id_right}
            W = BCIMath.compute_csp_filters(data, self.events, labeled_ids, self.fs, 0.5, 3.5)
            
            # 3. Extract Features (Log Variance of projected data)
            # We use all components (cols of W)
            features_train = BCIMath.extract_log_var_features(X_train, W)
            
            # 4. Train LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(features_train, y_train)
            
            # Store in memory
            self.trained_W = W
            self.trained_lda = lda
            self.class_labels = ['Left Hand', 'Right Hand']
            
            # Evaluate Accuracy on Training Set (Self-Check)
            acc = lda.score(features_train, y_train)
            
            messagebox.showinfo("Training Complete", 
                                f"Model Trained Successfully!\n\n"
                                f"Training Accuracy: {acc*100:.2f}%\n"
                                f"Trials: {len(y_train)}\n"
                                f"Features: Log-Variance of {W.shape[1]} CSP components.")
            
            self.btn_save.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    def save_model(self):
        if self.trained_lda is None: return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".dat", filetypes=[("DAT files", "*.dat")])
        if not file_path: return
        
        model_data = {
            'W': self.trained_W,
            'lda': self.trained_lda,
            'labels': self.class_labels
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        messagebox.showinfo("Saved", f"Model saved to {file_path}")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
        if not file_path: return
        
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_W = model_data['W']
            self.trained_lda = model_data['lda']
            self.class_labels = model_data['labels']
            
            messagebox.showinfo("Loaded", "Model loaded successfully.\nYou can now predict on evaluation files.")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def predict_unknowns(self):
        """Predicts class for event 783 (Unknown) in current file."""
        if self.trained_lda is None:
            messagebox.showwarning("No Model", "Please train or load a model first.")
            return
        
        if 'Unknown' not in self.event_id:
            messagebox.showwarning("No Data", "Current file does not have 'Unknown' (783) events.")
            return

        try:
            data = self.get_filtered_data()
            id_unknown = self.event_id['Unknown']
            
            # 1. Extract Unknown Epochs
            epochs_unk, _ = BCIMath.get_epochs_manual(data, self.events, id_unknown, self.fs, 0.5, 3.5)
            
            if len(epochs_unk) == 0:
                messagebox.showwarning("Empty", "No valid epochs found.")
                return

            # 2. Extract Features using the TRAINED Filters (W)
            features_unk = BCIMath.extract_log_var_features(epochs_unk, self.trained_W)
            
            # 3. Predict
            preds = self.trained_lda.predict(features_unk)
            
            # 4. Display Results
            count_left = np.sum(preds == 0)
            count_right = np.sum(preds == 1)
            
            # Visualize predictions
            plt.figure(figsize=(10, 4))
            plt.plot(preds, 'o-', label='Prediction (0=Left, 1=Right)')
            plt.yticks([0, 1], ['Left', 'Right'])
            plt.xlabel("Trial Number")
            plt.title(f"Predictions for {self.filename}\nLeft: {count_left}, Right: {count_right}")
            plt.grid(True, axis='y')
            plt.legend()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = BCIAnalysisApp(root)
    root.mainloop()