import numpy as np
from itertools import product
import math
# from ft import STFT_LIB

class STFT_LIB:
    def __init__(self, segment_length, segment_length_padded, shift_length, window_function):
        """
        Initialize the STFT object with configuration parameters.
        
        Parameters
        ----------
        segment_length : int
            length of extracted segments
        segment_length_padded : int
            segment length with zero padding
        shift_length : int
            shift length 1 <= shift_length <= segment_length 
        window_function : scipy.signal.window object
            window from scipy.signal.windows
        """
        self.segment_length = segment_length
        self.segment_length_padded = segment_length_padded
        self.shift_length = shift_length
        self.window_function = window_function

    def window_nonzero(self):
        """Generate a window vector such that there will be no zeros at the 
        beginning or end of the vector.
        """
        zero_exist = 1
        zero_count = 0
        
        window_vector = self.window_function(self.segment_length + zero_count)

        while zero_exist:
                
            start = int( zero_count / 2 )
            stop = int( len(window_vector) - zero_count / 2)
            window_vector = window_vector[start : stop]

            zero_count = len(window_vector) - np.count_nonzero(window_vector)
            
            if zero_count > 0:
                window_vector = self.window_function(self.segment_length + zero_count)
            else:
                zero_exist = 0

        return window_vector

    def create_overlapping_segments(self, x):
        """Split signal along the first dimension into overlapping subarrays.
        """
        # input argument checks
        if type(x) is not np.ndarray:
            raise ValueError("x is not numpy array")
        if self.segment_length > x.shape[0]:
            raise ValueError("segment_length is greater than x.shape[0]")
        if self.shift_length <= 0:
            raise ValueError("shift_length <= 0")
        if self.shift_length > self.segment_length:
            raise ValueError("shift_length > segment_length")

        # squeeze x to get rid of extra dimensions
        x = np.squeeze(x)
        
        # start/stop positions
        start_list = np.arange(0, x.shape[0], self.shift_length)
        stop_list = start_list + self.segment_length

        # if last segments extend outside the array, remove them
        index = [i <= x.shape[0] for i in stop_list]
        start_list = start_list[index]
        stop_list = stop_list[index]

        # if last segment does not include end of the array, add a segment
        if stop_list[-1] != x.shape[0]:
            stop_list = np.append(stop_list, x.shape[0])
            start_list = np.append(start_list, x.shape[0] - self.segment_length)

        # Create list of subarrays using a list comprehension
        x_segments = [ x[start:stop,...] for start, stop in zip(start_list,
            stop_list)]

        # Now stack the subarrays
        x_segments = np.stack(x_segments, axis=1)

        return x_segments, start_list, stop_list

    def stft(self, x):
        """Transfer signal from time domain to frequency domain.
        """
        # input argument checks
        if type(x) is not np.ndarray:
            raise ValueError("x is not numpy array")
        if self.segment_length > x.shape[0]:
            raise ValueError("segment_length is greater than x.shape[0]")
        if self.shift_length <= 0:
            raise ValueError("shift_length <= 0")
        if self.shift_length > self.segment_length:
            raise ValueError("shift_length > segment_length")
        if self.segment_length_padded < self.segment_length:
            raise ValueError("segment_length_padded < segment_length")

        # create window_vector which is 1D
        window_vector = self.window_nonzero()
        
        # overlapping segments
        x_segments, start_list, stop_list = self.create_overlapping_segments(x)

        # create window_array which is the same size as x_segments
        window_array = np.ones(x_segments.shape)
        for i in range(window_array.shape[0]):
            window_array[i] = window_array[i] * window_vector[i]
        
        # apply window to signals
        x_segments = window_array * x_segments

        # take fft
        x_stft = np.fft.rfft(x_segments, n=self.segment_length_padded, axis=0)

        return x_stft, start_list, stop_list

    def istft(self, x_stft, start_list, stop_list, original_size, p):
        """Transfer signal from the frequency domain to the time domain.
        """
        # take inverse short-time Fourier transform
        x_segments = np.fft.irfft(x_stft, n=self.segment_length_padded, axis=0)
        x_segments = x_segments[0:self.segment_length]

        # generate window vector
        window_vector = self.window_nonzero()

        # create window_array which is the same size as x_segments
        # This is W^(p-1) in step 3 of Table 1 in source [1].
        window_array = np.ones(x_segments.shape)
        for i in range(window_array.shape[0]):
            window_array[i] = window_array[i] * window_vector[i]
        window_array = window_array ** (p-1)
        
        # apply window to segments
        x_segments = window_array * x_segments

        # find overlap-and-add vector for windowing operation
        # This is Dp in step 5 in Table 1 in source [1].
        window_overlap_add = np.zeros(original_size[0])
        number_segments = len(start_list)
        for i in range(number_segments):
            window_overlap_add[start_list[i]:stop_list[i]] += window_vector ** p
        window_overlap_add = window_overlap_add
        # invert this vector
        window_overlap_add = ( window_overlap_add ) ** -1

        # create output array
        x = np.zeros(original_size)
        # overlap and add segments
        for i, (start, stop) in enumerate( zip(start_list, stop_list) ):
            x[start:stop,...] += x_segments[:, i,...]

        # normalize x
        window_overlap_add_array = np.zeros(original_size)
        for i in range(x.shape[0]):
            window_overlap_add_array[i] = window_overlap_add[i]
        x = x * window_overlap_add_array

        return x
    

class STFTConfigurator:
    def __init__(self, signal_duration_seconds, sampling_rate, 
                 overlap_percentage, window_count, window_function):
        """
        signal_duration_seconds : float
        sampling_rate : int
        overlap_percentage : float
        window_count : int
        window_function : scipy.signal.window object
        """
        self.total_samples = int(signal_duration_seconds * sampling_rate)
        self.window_function = window_function
        
        self._calculate_parameters(overlap_percentage, window_count)

    def _calculate_parameters(self, overlap_percentage, window_count):
        overlap_decimal = overlap_percentage / 100.0

        denominator = (window_count - 1) * (1 - overlap_decimal) + 1
        segment_length_float = self.total_samples / denominator
        
        self.segment_length = int(round(segment_length_float))
        shift_length_float = self.segment_length * (1 - overlap_decimal)
        self.shift_length = int(round(shift_length_float))
        self.segment_length_padded = self._next_power_of_two(self.segment_length)

    def _next_power_of_two(self, n):
        if n == 0:
            return 1
        return 2**math.ceil(math.log2(n))

    def create_stft_instance(self):
        return STFT_LIB(
            segment_length=self.segment_length,
            segment_length_padded=self.segment_length_padded,
            shift_length=self.shift_length,
            window_function=self.window_function
        )