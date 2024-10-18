from pythonosc import dispatcher
from pythonosc import osc_server
import argparse
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QScrollArea, QGridLayout, QGroupBox, QProgressBar, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSize
import threading
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OSCHandler(QObject):
    message_received = pyqtSignal(str, object)

    def __call__(self, address, *args):
        self.message_received.emit(address, args)

class VisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = {}
        self.main_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.group_widgets = {}

        self.setup_ui()

    def setup_ui(self):
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)

    def update_value(self, address, value):
        if "/BFI/Info" in address or "biometrics/supported" in address.lower():
            return
        if isinstance(value, tuple) and len(value) == 1:
            value = value[0]
        
        simplified_address = "/".join(address.split("/")[4:])
        group_name = simplified_address.split("/")[0]

        if group_name == "NeuroFB":
            if "Focus" in simplified_address:
                group_name = "NeuroFB Focus"
            elif "Relax" in simplified_address:
                group_name = "NeuroFB Relax"

        if group_name not in self.group_widgets:
            group_box = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            group_box.setLayout(group_layout)
            row = len(self.group_widgets) // 3
            col = len(self.group_widgets) % 3
            self.grid_layout.addWidget(group_box, row, col)
            self.group_widgets[group_name] = (group_box, group_layout, {})

            group_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            group_box.setMinimumWidth(200)

        _, group_layout, value_widgets = self.group_widgets[group_name]
        if simplified_address not in value_widgets:
            if "Relax" in simplified_address or "Focus" in simplified_address:
                # Use QLabel for Relax and Focus values
                label = QLabel(f"{simplified_address.split('/')[-1]}: {value:.4f}")
                group_layout.addWidget(label)
                value_widgets[simplified_address] = label
            else:
                # Use QProgressBar for other values
                progress_bar = QProgressBar()
                progress_bar.setMinimum(0)
                progress_bar.setMaximum(100)
                progress_bar.setFormat(f"{simplified_address.split('/')[-1]}: %v")
                group_layout.addWidget(progress_bar)
                value_widgets[simplified_address] = progress_bar

        if isinstance(value, (int, float)):
            if "Relax" in simplified_address or "Focus" in simplified_address:
                # Update label text for Relax and Focus values
                value_widgets[simplified_address].setText(f"{simplified_address.split('/')[-1]}: {value:.4f}")
            elif "HeartBeatsPerMin" in simplified_address:
                max_value = 220  # Maximum realistic heart rate
                normalized_value = min(value, max_value)
                value_widgets[simplified_address].setMaximum(max_value)
                value_widgets[simplified_address].setValue(int(normalized_value))
            elif "HeartBeatsPerSecond" in simplified_address:
                max_value = 4  # Maximum realistic heart rate per second (240 bpm / 60)
                normalized_value = min(value, max_value)
                value_widgets[simplified_address].setMaximum(int(max_value * 100))
                value_widgets[simplified_address].setValue(int(normalized_value * 100))  # Multiply by 100 to show 2 decimal places
            else:
                normalized_value = int(min(max(value, 0), 1) * 100)
                value_widgets[simplified_address].setValue(normalized_value)

    def sizeHint(self):
        return QSize(800, 600)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BFI Monitor")
        self.setGeometry(100, 100, 1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.visualization = VisualizationWidget()
        self.main_layout.addWidget(self.visualization)

        self.status_label = QLabel("Waiting for OSC messages...")
        self.main_layout.addWidget(self.status_label)

    def update_data(self, address, value):
        self.visualization.update_value(address, value)
        self.status_label.setText(f"Last received: {address}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=9000, help="The port to listen on")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    osc_handler = OSCHandler()
    osc_handler.message_received.connect(window.update_data)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/avatar/parameters/*", osc_handler)

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    logger.info(f"Serving on {server.server_address}")

    # Run the OSC server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()

    sys.exit(app.exec())
