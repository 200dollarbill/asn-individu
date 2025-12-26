import numpy as np
import scipy.linalg
import scipy.signal
from scipy.signal import butter, filtfilt
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pywt  # Requires: pip install PyWavelets

class ANALYSIS:
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
        smin = int(tmin * fs)
        smax = int(tmax * fs)

        if isinstance(event_id, int):
            target_events = events[events[:, 2] == event_id]
        else:
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

        print(len(epochs), "epochs extracted for event ID(s):", event_id)
        return np.array(epochs), np.array(labels)

    @staticmethod
    def cwt(epochs, fs, fmin=2, fmax=40, wavelet='cmor0.5-1.5'):
        evoked = np.mean(epochs, axis=0)
        freqs = np.linspace(fmin, fmax, 50)
        center_freq = pywt.central_frequency(wavelet)
        scales = center_freq * fs / freqs
        cwt_matrices = []
        for ch_idx in range(evoked.shape[0]):
            coefs, _ = pywt.cwt(evoked[ch_idx, :], scales, wavelet, sampling_period=1/fs)
            cwt_matrices.append(np.abs(coefs)**2) 
            
        return freqs, np.array(cwt_matrices)
    
    @staticmethod
    def gen_csp(data, events, event_ids, fs, tmin, tmax):
        labels = list(event_ids.keys())
        epochs_c1, _ = ANALYSIS.get_epochs_manual(data, events, event_ids[labels[0]], fs, tmin, tmax)
        epochs_c2, _ = ANALYSIS.get_epochs_manual(data, events, event_ids[labels[1]], fs, tmin, tmax)

        if len(epochs_c1) == 0 or len(epochs_c2) == 0:
            raise ValueError("Not enough epochs to train CSP.")

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

        eigenvalues, eigenvectors = scipy.linalg.eigh(C1, C1 + C2)
        ix = np.argsort(eigenvalues)[::-1]
        W = eigenvectors[:, ix]
        return W

    @staticmethod
    def calculate_acc(epochs, labels, W, fs, t_start_epoch, clf_template, window_size=1.0, step=0.1):
        """Calculates classification accuracy using a sliding window."""
        n_samples = epochs.shape[2]
        win_samp = int(window_size * fs)
        step_samp = int(step * fs)

        acc_scores = []
        time_points = []
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for start in range(0, n_samples - win_samp, step_samp):
            end = start + win_samp
            X_win = epochs[:, :, start:end]
            
            feats = ANALYSIS.extract_log_var_features(X_win, W)
            
            clf = clone(clf_template)
            scores = cross_val_score(clf, feats, labels, cv=cv)
            acc_scores.append(np.mean(scores))

            center_sample = start + (win_samp / 2)
            time_s = t_start_epoch + (center_sample / fs)
            time_points.append(time_s)

        return np.array(time_points), np.array(acc_scores)

    @staticmethod
    def extract_log_var_features(epochs, W):
        n_trials = epochs.shape[0]
        n_components = W.shape[1]
        features = np.zeros((n_trials, n_components))

        for i in range(n_trials):
            Z = np.dot(W.T, epochs[i])
            var_Z = np.var(Z, axis=1)
            features[i, :] = np.log(var_Z)

        return features

    @staticmethod
    def compute_erd_ers(data, events, event_ids, fs, tmin, tmax, ref_tmin, ref_tmax):
        n_samples = int((tmax - tmin) * fs)
        times = np.linspace(tmin, tmax, n_samples)
        ref_idx_start = int((ref_tmin - tmin) * fs)
        ref_idx_end = int((ref_tmax - tmin) * fs)
        results = {}
        for label, code in event_ids.items():
            epochs, _ = ANALYSIS.get_epochs_manual(data, events, code, fs, tmin, tmax)
            if len(epochs) == 0: continue
            power = epochs ** 2
            avg_power = np.mean(power, axis=0)
            R = np.mean(avg_power[:, ref_idx_start:ref_idx_end], axis=1, keepdims=True)
            erd_ers = ((avg_power - R) / R) * 100
            results[label] = erd_ers
        return times, results