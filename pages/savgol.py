import numpy as np
import matplotlib.pyplot as plt
from deps import handler
# window size index must be odd
def savitzky_golay(signal, window_size, poly_order):
       
    n_points = len(signal)
    half_window = window_size // 2
    smoothed_signal = np.zeros(n_points)
    x_local = np.arange(-half_window, half_window + 1)
    X = np.vander(x_local, N=poly_order + 1, increasing=True)
    X_transpose = X.T
    XTX = np.dot(X_transpose, X)
    XTX_inv = np.linalg.inv(XTX)
    projection_matrix = np.dot(XTX_inv, X_transpose)
    weights = projection_matrix[0, :]
    for i in range(half_window, n_points - half_window):
        y_window = signal[i - half_window : i + half_window + 1]
        
        smoothed_value = np.dot(weights, y_window)
        
        smoothed_signal[i] = smoothed_value
    smoothed_signal[:half_window] = signal[:half_window]
    smoothed_signal[-half_window:] = signal[-half_window:]
    
    return smoothed_signal

np.random.seed(42)
# y_pure = np.sin(x)
# y_noisy = y_pure + np.random.normal(0, 0.15, x.size)

glat = handler.load("g_lateralis")
x = glat.time
glat = glat.value

glat_old = glat

y_smooth = savitzky_golay(glat, window_size=11, poly_order=5)

plt.figure(figsize=(10, 6))
# plt.plot(x, glat, label='Noisy Signal', color='lightgray', linestyle='-', linewidth=1)
plt.plot(x, glat_old, label='Original Pure Signal', color='black', linestyle='--', alpha=0.5)
plt.plot(x, y_smooth, label='Custom Savitzky-Golay', color='red', linewidth=2)

plt.title("Custom Savitzky-Golay Implementation")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()