from logic.base_logic import BaseLogic
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations
import numpy as np
import time

class BlinkDetection(BaseLogic):
    def __init__(self, board, window_seconds=1, threshold=100, blink_duration=0.2, max_blink_duration=0.4, min_blink_interval=0.5):
        super().__init__(board)
        self.board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.max_sample_size = self.sampling_rate * window_seconds
        self.blink_duration = blink_duration
        self.max_blink_duration = max_blink_duration
        self.last_blink_time = 0
        self.blink_value = 0.0
        self.min_blink_interval = min_blink_interval
        self.blink_buffer = []
        self.max_buffer_size = 5  # Store last 5 blink times

    def detect_blinks(self, data):
        current_time = time.time()
        if current_time - self.last_blink_time < self.min_blink_interval:
            return 0

        blinks = 0
        for channel in self.eeg_channels:
            DataFilter.detrend(data[channel], DetrendOperations.LINEAR)
            
            signal_std = np.std(data[channel])
            dynamic_threshold = max(self.threshold, 3 * signal_std)
            
            peaks = np.where(data[channel] > dynamic_threshold)[0]
            
            if len(peaks) > 0:
                valid_peaks = [peaks[0]]
                for peak in peaks[1:]:
                    if peak - valid_peaks[-1] > self.sampling_rate * self.min_blink_interval:
                        valid_peaks.append(peak)
                
                if len(valid_peaks) > 0 and current_time - self.last_blink_time >= self.min_blink_interval:
                    if self.is_valid_blink(current_time):
                        blinks = 1
                        self.last_blink_time = current_time
                        self.blink_buffer.append(current_time)
                        if len(self.blink_buffer) > self.max_buffer_size:
                            self.blink_buffer.pop(0)
                        break

        return blinks

    def is_valid_blink(self, current_time):
        if len(self.blink_buffer) < 2:
            return True
        
        time_diffs = np.diff(self.blink_buffer + [current_time])
        if np.all(time_diffs < self.min_blink_interval * 1.5):
            return False
        
        return True

    def update_blink_value(self):
        current_time = time.time()
        time_since_blink = current_time - self.last_blink_time
        if time_since_blink < self.blink_duration:
            self.blink_value = 1.0 - (time_since_blink / self.blink_duration) ** 3
        elif time_since_blink < self.max_blink_duration:
            self.blink_value = max(0, 1.0 - ((time_since_blink - self.blink_duration) / (self.max_blink_duration - self.blink_duration)) ** 0.5)
        else:
            self.blink_value = 0.0

    def get_data_dict(self):
        data = self.board.get_current_board_data(self.max_sample_size)
        blink_count = self.detect_blinks(data)
        self.update_blink_value()
        return {
            "BlinkCount": blink_count,
            "BlinkValue": self.blink_value
        }
