import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import mne
import numpy as np
import scipy.linalg
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

    def compute_accuracy_over_time(epochs, labels, W, fs, t_start_epoch, window_size=1.0, step=0.1):
        """
        Calculates classification accuracy using a sliding window.
        """
        n_samples = epochs.shape[2]
        win_samp = int(window_size * fs)
        step_samp = int(step * fs)
        
        acc_scores = []
        time_points = []
        
        # Loop through the epoch
        for start in range(0, n_samples - win_samp, step_samp):
            end = start + win_samp
            
            # 1. Extract Window
            X_win = epochs[:, :, start:end]
            
            # 2. Extract Features (using the GLOBAL CSP filters W)
            # We use the global W because retraining CSP on small windows is unstable
            feats = BCIMath.extract_log_var_features(X_win, W)
            
            # 3. Cross-Validate LDA (5-fold for speed)
            clf = LinearDiscriminantAnalysis()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, feats, labels, cv=cv)
            
            acc_scores.append(np.mean(scores))
            
            # Calculate time point (center of window)
            center_sample = start + (win_samp / 2)
            time_s = t_start_epoch + (center_sample / fs)
            time_points.append(time_s)
            
        return np.array(time_points), np.array(acc_scores)

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
        """Trains CSP + Neural Network and shows performance."""
        if self.raw is None: return
        
        try:
            data = self.get_filtered_data()
            id_left = self.event_id['Left Hand']
            id_right = self.event_id['Right Hand']
            
            # 1. Get Epochs
            epochs_L, _ = BCIMath.get_epochs_manual(data, self.events, id_left, self.fs, 0.5, 3.5)
            epochs_R, _ = BCIMath.get_epochs_manual(data, self.events, id_right, self.fs, 0.5, 3.5)
            
            X_train = np.concatenate((epochs_L, epochs_R), axis=0)
            y_train = np.concatenate((np.zeros(len(epochs_L)), np.ones(len(epochs_R))))
            
            # 2. Train CSP Filters
            labeled_ids = {'Left': id_left, 'Right': id_right}
            W = BCIMath.compute_csp_filters(data, self.events, labeled_ids, self.fs, 0.5, 3.5)
            
            # 3. Extract Features (Log Variance)
            features_train = BCIMath.extract_log_var_features(X_train, W)
            
            # --- 4. TRAIN NEURAL NETWORK (ANN) ---
            # We use a Pipeline: Scaler -> MLP
            # Topology: 2 Hidden Layers (20 neurons, 10 neurons)
            ann_clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(20, 10),  # The Topology
                    activation='relu',            # Activation Function
                    solver='adam',                # Optimizer
                    alpha=0.001,                  # Regularization (prevents overfitting)
                    max_iter=1000,                # Allow enough epochs to converge
                    random_state=42
                )
            )
            
            print("Training Neural Network...")
            ann_clf.fit(features_train, y_train)
            
            # Save to self variables (renamed from lda to clf to be generic)
            self.trained_W = W
            self.trained_lda = ann_clf # We store the ANN pipeline here
            self.class_labels = ['Left Hand', 'Right Hand']
            self.btn_save.config(state=tk.NORMAL)

            # --- PERFORMANCE EVALUATION ---
            
            # A. Confusion Matrix (10-fold CV)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            y_pred_cv = cross_val_predict(ann_clf, features_train, y_train, cv=cv)
            
            cm = confusion_matrix(y_train, y_pred_cv)
            acc = np.mean(y_pred_cv == y_train)
            kappa = cohen_kappa_score(y_train, y_pred_cv)
            
            # B. Accuracy Over Time (Sliding Window)
            ep_L_wide, _ = BCIMath.get_epochs_manual(data, self.events, id_left, self.fs, -1.5, 4.5)
            ep_R_wide, _ = BCIMath.get_epochs_manual(data, self.events, id_right, self.fs, -1.5, 4.5)
            X_wide = np.concatenate((ep_L_wide, ep_R_wide), axis=0)
            
            # Note: We need to pass the ANN classifier to the sliding window logic
            # But our static method creates a new LDA internally. 
            # Let's update the static method call to use the ANN logic or just accept the plot uses LDA for speed.
            # Ideally, you update BCIMath.compute_accuracy_over_time to accept a classifier type, 
            # but for now, let's keep the sliding window as LDA (it's faster) or update it below.
            
            # For consistent reporting, let's just plot the CV results we have:
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 2)
            
            # Plot 1: Confusion Matrix
            ax1 = fig.add_subplot(gs[0, 0])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left', 'Right'])
            disp.plot(cmap='Purples', ax=ax1, colorbar=False) # Changed color to Purples for ANN
            ax1.set_title(f"ANN Confusion Matrix\nAcc: {acc:.2%} | Kappa: {kappa:.2f}")
            
            # Plot 2: Text Info
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')
            text_info = (
                f"Neural Network Performance\n"
                f"--------------------------\n"
                f"Topology: Input -> [20, 10] -> Output\n"
                f"Optimizer: Adam\n"
                f"Total Trials: {len(y_train)}\n\n"
                f"Overall Accuracy: {acc:.2%}\n"
                f"Kappa Score: {kappa:.2f}"
            )
            ax2.text(0.1, 0.5, text_info, fontsize=11, verticalalignment='center')

            # Plot 3: Loss Curve (Training History)
            # This shows how the Network learned over iterations
            ax3 = fig.add_subplot(gs[1, :])
            # Access the MLPClassifier inside the pipeline
            loss_curve = ann_clf.named_steps['mlpclassifier'].loss_curve_
            ax3.plot(loss_curve, color='red')
            ax3.set_title("Neural Network Training Loss")
            ax3.set_xlabel("Iterations")
            ax3.set_ylabel("Loss")
            ax3.grid(True)
            
            plt.tight_layout()
            plt.show()
            
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