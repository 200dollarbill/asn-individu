import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from erdcsp import BCIMath

class BCIClassifier:
    def __init__(self):
        self.trained_W = None
        self.trained_clf = None
        self.class_labels = None

    def train_model(self, data, events, event_ids, fs, tmin=0.5, tmax=3.5):
        """
        Trains CSP + Neural Network and returns performance metrics.
        """
        id_left = event_ids['Left Hand']
        id_right = event_ids['Right Hand']

        # 1. Get Epochs
        epochs_L, _ = BCIMath.get_epochs_manual(data, events, id_left, fs, tmin, tmax)
        epochs_R, _ = BCIMath.get_epochs_manual(data, events, id_right, fs, tmin, tmax)

        X_train = np.concatenate((epochs_L, epochs_R), axis=0)
        y_train = np.concatenate((np.zeros(len(epochs_L)), np.ones(len(epochs_R))))

        # 2. Train CSP Filters
        W = BCIMath.compute_csp_filters(data, events, event_ids, fs, tmin, tmax)

        # 3. Extract Features (Log Variance)
        features_train = BCIMath.extract_log_var_features(X_train, W)

        # 4. TRAIN NEURAL NETWORK (ANN)
        ann_clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(20, 10),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
        )

        print("Training Neural Network...")
        ann_clf.fit(features_train, y_train)

        # Save to instance
        self.trained_W = W
        self.trained_clf = ann_clf
        self.class_labels = ['Left Hand', 'Right Hand']

        # PERFORMANCE EVALUATION
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(ann_clf, features_train, y_train, cv=cv)

        cm = confusion_matrix(y_train, y_pred_cv)
        acc = np.mean(y_pred_cv == y_train)
        kappa = cohen_kappa_score(y_train, y_pred_cv)

        loss_curve = ann_clf.named_steps['mlpclassifier'].loss_curve_

        return acc, kappa, cm, loss_curve, len(y_train)

    def save_model(self, file_path):
        if self.trained_clf is None:
            raise ValueError("No trained model to save")

        model_data = {
            'W': self.trained_W,
            'clf': self.trained_clf,
            'labels': self.class_labels
        }

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        self.trained_W = model_data['W']
        self.trained_clf = model_data['clf']
        self.class_labels = model_data['labels']

    def predict_unknowns(self, data, events, event_id_unknown, fs, tmin=0.5, tmax=3.5):
        if self.trained_clf is None:
            raise ValueError("No model loaded")

        # 1. Extract Unknown Epochs
        epochs_unk, _ = BCIMath.get_epochs_manual(data, events, event_id_unknown, fs, tmin, tmax)

        if len(epochs_unk) == 0:
            raise ValueError("No valid epochs found")

        # 2. Extract Features
        features_unk = BCIMath.extract_log_var_features(epochs_unk, self.trained_W)

        # 3. Predict
        preds = self.trained_clf.predict(features_unk)

        count_left = np.sum(preds == 0)
        count_right = np.sum(preds == 1)

        return preds, count_left, count_right