from logic.base_logic import BaseLogic
from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, DetrendOperations
import numpy as np
import time

class BlinkDetection(BaseLogic):
    def __init__(self, board, window_seconds=1, threshold=100, blink_duration=0.5):
        super().__init__(board)
        self.board_id = board.get_board_id()
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.window_seconds = window_seconds
        self.threshold = threshold
        self.max_sample_size = self.sampling_rate * window_seconds
        self.blink_duration = blink_duration
        self.last_blink_time = 0
        self.blink_value = 0.0

    def detect_blinks(self, data):
        blinks = []
        for channel in self.eeg_channels:
            # Detrend the signal
            DataFilter.detrend(data[channel], DetrendOperations.LINEAR)
            
            # Simple threshold-based blink detection
            peaks = np.where(data[channel] > self.threshold)[0]
            
            # Merge nearby peaks
            blink_indices = []
            for peak in peaks:
                if not blink_indices or peak - blink_indices[-1] > self.sampling_rate * 0.1:  # 100ms between blinks
                    blink_indices.append(peak)
            
            blinks.append(len(blink_indices))
        
        return max(blinks)  # Return the maximum number of blinks detected across channels

    def update_blink_value(self):
        current_time = time.time()
        if current_time - self.last_blink_time < self.blink_duration:
            self.blink_value = 1.0 - (current_time - self.last_blink_time) / self.blink_duration
        else:
            self.blink_value = 0.0

    def get_data_dict(self):
        data = self.board.get_current_board_data(self.max_sample_size)
        blink_count = self.detect_blinks(data)
        if blink_count > 0:
            self.last_blink_time = time.time()
        self.update_blink_value()
        return {
            "BlinkCount": blink_count,
            "BlinkValue": self.blink_value
        }
