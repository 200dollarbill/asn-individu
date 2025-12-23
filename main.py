import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import mne
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.metrics import ConfusionMatrixDisplay

from erdcsp import BCIMath
from classifier import BCIClassifier

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
        
        # Classifier instance
        self.classifier = BCIClassifier()

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
            
            if 'Left Hand' in self.event_id and 'Right Hand' in self.event_id:
                self.btn_train.config(state=tk.NORMAL)
            else:
                self.btn_train.config(state=tk.DISABLED)
                
            if 'Unknown' in self.event_id:
                self.btn_predict.config(state=tk.NORMAL)
            else:
                self.btn_predict.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_filtered_data(self):
        raw_data = self.raw.get_data() * 1e6
        return BCIMath.butter_bandpass_filter(raw_data, 8.0, 30.0, self.fs, order=4)

    def run_erd_ers(self):
        if self.raw is None: return
        data = self.get_filtered_data()
        labeled_ids = {k: v for k, v in self.event_id.items() if k in ['Left Hand', 'Right Hand']}
        
        times, results = BCIMath.erders(
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
            W = BCIMath.gen_csp(data, self.events, labeled_ids, self.fs, 0.5, 3.5)
            patterns = scipy.linalg.pinv(W).T
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            info = self.raw.info
            mne.viz.plot_topomap(patterns[0, :], info, axes=axes[0], show=False, contours=0, sensors=True, names=self.raw.ch_names)
            axes[0].set_title("Component 1 (Left)")
            mne.viz.plot_topomap(patterns[-1, :], info, axes=axes[1], show=False, contours=0, sensors=True, names=self.raw.ch_names)
            axes[1].set_title("Component 3 (Right)")
            plt.show()
            
        except Exception as e:
            messagebox.showerror("CSP Error", str(e))

    def train_model(self):
        """Trains CSP + Neural Network and shows performance."""
        if self.raw is None: return
        
        try:
            data = self.get_filtered_data()
            labeled_ids = {k: v for k, v in self.event_id.items() if k in ['Left Hand', 'Right Hand']}
            
            # 1. Train Main Model
            acc, kappa, cm, loss_curve, total_trials = self.classifier.train_model(data, self.events, labeled_ids, self.fs)
            
            self.btn_save.config(state=tk.NORMAL)

            # 2. Run Sliding Window Analysis (New Feature)
            print("Running Sliding Window Analysis (this may take 10-20 seconds)...")
            t_course, acc_course = self.classifier.perform_sliding_window_analysis(data, self.events, labeled_ids, self.fs)

            # 3. Plotting Report
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2)
            
            # Plot 1: Confusion Matrix
            ax1 = fig.add_subplot(gs[0, 0])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left', 'Right'])
            disp.plot(cmap='Purples', ax=ax1, colorbar=False)
            ax1.set_title(f"ANN Confusion Matrix\nAcc: {acc:.2%} | Kappa: {kappa:.2f}")
            
            # Plot 2: Text Info
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')
            text_info = (
                f"Neural Network Performance\n"
                f"--------------------------\n"
                f"Topology: Input -> [20, 10] -> Output\n"
                f"Optimizer: Adam\n"
                f"Total Trials: {total_trials}\n\n"
                f"Overall Accuracy: {acc:.2%}\n"
                f"Kappa Score: {kappa:.2f}"
            )
            ax2.text(0.1, 0.5, text_info, fontsize=11, verticalalignment='center')

            # Plot 3: Loss Curve
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(loss_curve, color='red')
            ax3.set_title("Training Loss (Convergence)")
            ax3.set_xlabel("Iterations")
            ax3.set_ylabel("Loss")
            ax3.grid(True)

            # Plot 4: Accuracy Over Time (Sliding Window)
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

    def save_model(self):
        if not hasattr(self.classifier, 'trained_clf') or self.classifier.trained_clf is None:
            messagebox.showwarning("No Model", "No trained model to save.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".dat", filetypes=[("DAT files", "*.dat")])
        if not file_path: return
        
        try:
            self.classifier.save_model(file_path)
            messagebox.showinfo("Saved", f"Model saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
        if not file_path: return
        
        try:
            self.classifier.load_model(file_path)
            messagebox.showinfo("Loaded", "Model loaded successfully.\nYou can now predict on evaluation files.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def predict_unknowns(self):
        if not hasattr(self.classifier, 'trained_clf') or self.classifier.trained_clf is None:
            messagebox.showwarning("No Model", "Please train or load a model first.")
            return
        
        if 'Unknown' not in self.event_id:
            messagebox.showwarning("No Data", "Current file does not have 'Unknown' (783) events.")
            return

        try:
            data = self.get_filtered_data()
            id_unknown = self.event_id['Unknown']
            
            preds, count_left, count_right = self.classifier.forward_prop(data, self.events, id_unknown, self.fs)
            
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