import numpy as np
from scipy import signal as sg

# credits to https://github.com/antimattercorrade/Pan_Tompkins_QRS_Detection

class Pan_Tompkins_QRS():
  """
  A class that implements the signal processing steps of the Pan-Tompkins algorithm.
  This includes band-pass filtering, differentiation, squaring, and moving window integration.
  """
  

  def band_pass_filter(self, signal):
    """
    Band Pass Filter
    :param signal: input signal as a NumPy array
    :return: processed signal

    Methodology/Explanation:
    A combination of a low-pass and high-pass filter to achieve a passband of 5-15 Hz.
    """
    # Create a copy of the input signal
    sig = signal.copy()
    
    # Low Pass Filter: y(n) = 2y(n-1) - y(n-2) + x(n) - 2x(n-6) + x(n-12)
    for index in range(len(signal)):
      if (index >= 1):
        sig[index] += 2 * sig[index-1]
      if (index >= 2):
        sig[index] -= sig[index-2]
      if (index >= 6):
        sig[index] -= 2 * signal[index-6]
      if (index >= 12):
        sig[index] += signal[index-12] 
	
    # Copy the result of the low pass filter
    result = sig.copy()

    # High Pass Filter: y(n) = 32x(n-16) - y(n-1) - x(n) + x(n-32)
    for index in range(len(signal)):
      result[index] = -1 * sig[index]
      if (index >= 1):
        result[index] -= result[index-1]
      if (index >= 16):
        result[index] += 32 * sig[index-16]
      if (index >= 32):
        result[index] += sig[index-32]

    # Normalize the result
    max_val = max(max(result), -min(result))
    if max_val > 0:
        result = result / max_val

    return result

  def derivative(self, signal, fs):
    """
    Derivative Filter 
    :param signal: input signal
    :param fs: sampling frequency
    :return: processed signal

    Methodology/Explanation:
    Approximates the derivative to find high-slope regions.
    y(n) = [-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)] / (8T), where T = 1/fs
    """
    result = np.zeros_like(signal)

    for index in range(2, len(signal) - 2):
      result[index] = ( -signal[index-2] - 2*signal[index-1] + 2*signal[index+1] + signal[index+2] ) * (fs / 8)
      
    return result

  def squaring(self, signal):
    """
    Squaring the Signal
    :param signal: input signal
    :return: processed signal

    Methodology/Explanation:
    Intensifies the derivative output and makes all values positive.
    y(n) = [x(n)]^2
    """
    return np.square(signal)

  def moving_window_integration(self, signal, fs):
    """
    Moving Window Integrator
    :param signal: input signal
    :param fs: sampling frequency
    :return: processed signal

    Methodology/Explanation:
    Integrates the signal over a 150ms window to get waveform feature information.
    """
    result = np.zeros_like(signal)
    win_size = round(0.150 * fs)
    
    sum_kernel = np.ones(win_size) / win_size
    result = np.convolve(signal, sum_kernel, mode='same')
    
    return result

  def solve(self, signal, fs):
    """
    Solver that combines all the processing steps.
    :param signal: input ECG signal as a NumPy array.
    :param fs: the sampling frequency of the signal.
    :return: A tuple containing (bpass, der, sqr, mwin) processed signals.
    """
    if not isinstance(signal, np.ndarray):
      raise TypeError("Input signal must be a NumPy ndarray.")

    # 1. Bandpass Filter
    bpass = self.band_pass_filter(signal.copy())

    # 2. Derivative Function
    der = self.derivative(bpass.copy(), fs)

    # 3. Squaring Function
    sqr = self.squaring(der.copy())

    # 4. Moving Window Integration Function
    mwin = self.moving_window_integration(sqr.copy(), fs)

    return bpass, der, sqr, mwin

class HeartRate():
  """
  A class that uses the processed signals from Pan_Tompkins_QRS to detect R-peaks
  and calculate heart rate.
  """
  def __init__(self, raw_signal, mwin_signal, bpass_signal, samp_freq):
    """
    :param raw_signal: ecg signal
    :param mwin_signal: window integration
    :param bpass_signal: bpf  out
    :param samp_freq: fs
    """
    self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for _ in range(6))
    self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2 = (0.0 for _ in range(4))
    self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (0.0 for _ in range(4))

    self.T_wave = False          
    self.m_win = mwin_signal
    self.b_pass = bpass_signal
    self.samp_freq = samp_freq
    self.signal = raw_signal
    self.win_150ms = round(0.15 * self.samp_freq)

    self.RR_Low_Limit = 0
    self.RR_High_Limit = 0
    self.RR_Missed_Limit = 0
    self.RR_Average1 = 0

  def approx_peak(self):
    slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')
    for i in range(round(0.5*self.samp_freq) + 1, len(slopes) - 1):
      if (slopes[i] > slopes[i-1]) and (slopes[i+1] < slopes[i]):
        self.peaks.append(i)  

  def adjust_rr_interval(self, ind):
    self.RR1 = np.diff(self.peaks[max(0, ind - 8) : ind + 1]) / self.samp_freq   
    self.RR_Average1 = np.mean(self.RR1)
    RR_Average2 = self.RR_Average1
      
    if ind >= 8:
      self.RR2 = [rr for rr in self.RR1 if self.RR_Low_Limit < rr < self.RR_High_Limit]
      if len(self.RR2) > 8:
          self.RR2 = self.RR2[-8:]
      if len(self.RR2) > 0:
          RR_Average2 = np.mean(self.RR2)    

    if len(self.RR2) > 7 or ind < 8:
      self.RR_Low_Limit = 0.92 * RR_Average2        
      self.RR_High_Limit = 1.16 * RR_Average2
      self.RR_Missed_Limit = 1.66 * RR_Average2

  def searchback(self, peak_val, RRn):
    if RRn > self.RR_Missed_Limit:
      sb_win = round(RRn * self.samp_freq)
      win_rr_mwin = self.m_win[max(0, peak_val - sb_win + 1) : peak_val + 1] 
      
      coord = np.where(win_rr_mwin > self.Threshold_I1)[0]
      if coord.size > 0:
        x_max_mwin_rel = coord[np.argmax(win_rr_mwin[coord])]
        x_max_mwin_abs = max(0, peak_val - sb_win + 1) + x_max_mwin_rel
        
        self.SPKI = 0.25 * self.m_win[x_max_mwin_abs] + 0.75 * self.SPKI                         
        
        win_rr_bpass = self.b_pass[max(0, x_max_mwin_abs - self.win_150ms) : min(len(self.b_pass), x_max_mwin_abs)]
        if win_rr_bpass.size > 0 and np.max(win_rr_bpass) > self.Threshold_F1:
          r_max_bpass_rel = np.argmax(win_rr_bpass)
          r_max_bpass_abs = max(0, x_max_mwin_abs - self.win_150ms) + r_max_bpass_rel
          
          if self.b_pass[r_max_bpass_abs] > self.Threshold_F2:                                                        
            self.SPKF = 0.25 * self.b_pass[r_max_bpass_abs] + 0.75 * self.SPKF                            
            self.r_locs.append(r_max_bpass_abs)

  def find_t_wave(self, peak_val, RRn, ind):
    if 0.20 < RRn < 0.36:
      prev_peak_val = self.peaks[ind-1]
      curr_slope = np.max(np.diff(self.m_win[max(0, peak_val - self.win_150ms//2) : peak_val + 1]))
      last_slope = np.max(np.diff(self.m_win[max(0, prev_peak_val - self.win_150ms//2) : prev_peak_val + 1]))
      if curr_slope < 0.5 * last_slope:  
        self.T_wave = True                             
        self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI 
        return

    if not self.T_wave:
      peak_idx_bpass = self.probable_peaks[ind]
      if peak_idx_bpass is not None and self.b_pass[peak_idx_bpass] > self.Threshold_F1:   
        self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI                                         
        self.SPKF = 0.125 * self.b_pass[peak_idx_bpass] + 0.875 * self.SPKF 
        self.r_locs.append(peak_idx_bpass)
      else:
        self.NPKF = 0.125 * self.b_pass[peak_idx_bpass] + 0.875 * self.NPKF

  def adjust_and_check_peaks(self, peak_val, ind):
    if self.m_win[peak_val] >= self.Threshold_I1:
      if ind > 0:
        RRn = (self.peaks[ind] - self.peaks[ind-1]) / self.samp_freq
        if self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit:
          self.Threshold_I1 /= 2
          self.Threshold_F1 /= 2
        self.searchback(peak_val, RRn)
        self.find_t_wave(peak_val, RRn, ind)
      else: # Learning Phase
        self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
        peak_idx_bpass = self.probable_peaks[ind]
        if peak_idx_bpass is not None and self.b_pass[peak_idx_bpass] > self.Threshold_F1:
          self.SPKF = 0.125 * self.b_pass[peak_idx_bpass] + 0.875 * self.SPKF
          self.r_locs.append(peak_idx_bpass)
        else:
          self.NPKF = 0.125 * self.b_pass[peak_idx_bpass] + 0.875 * self.NPKF
    else:
      peak_idx_bpass = self.probable_peaks[ind]
      self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
      if peak_idx_bpass is not None:
        self.NPKF = 0.125 * self.b_pass[peak_idx_bpass] + 0.875 * self.NPKF

  def update_thresholds(self):
    self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
    self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
    self.Threshold_I2 = 0.5 * self.Threshold_I1 
    self.Threshold_F2 = 0.5 * self.Threshold_F1
    self.T_wave = False 

  def ecg_searchback(self):
    self.r_locs = np.unique(np.array(self.r_locs).astype(int))
    win_200ms = round(0.2 * self.samp_freq)
    final_r_locs = []
    for r_val in self.r_locs:
      search_zone = np.arange(max(0, r_val - win_200ms), min(len(self.signal), r_val + win_200ms + 1))
      if search_zone.size > 0:
        final_r_locs.append(search_zone[np.argmax(self.signal[search_zone])])
    self.result = np.unique(final_r_locs)

  def find_r_peaks(self):
    self.approx_peak()
    for ind, peak_val in enumerate(self.peaks):
      win_150ms_range = np.arange(max(0, peak_val - self.win_150ms), min(peak_val + self.win_150ms, len(self.b_pass)))
      if win_150ms_range.size > 0:
        max_val_bpass = np.max(self.b_pass[win_150ms_range])
        x_coord = np.where(self.b_pass == max_val_bpass)[0]
        self.probable_peaks.append(x_coord[0] if x_coord.size > 0 else None)
      else:
        self.probable_peaks.append(None)
      
      if ind > 0:
        self.adjust_rr_interval(ind)
      
      self.adjust_and_check_peaks(peak_val, ind)
      self.update_thresholds()

    self.ecg_searchback()
    return self.result