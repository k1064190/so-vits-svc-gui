import io
import json
import os
import sys
import wave
from functools import partial

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSlider, QComboBox, QCheckBox,
                             QTextEdit, QGroupBox, QFileDialog, QRadioButton, QInputDialog, QSizePolicy, QDesktopWidget,
                             QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt, QUrl, QTimer, QBuffer, QByteArray, QIODevice
import pylab
import pyaudio

import torch
import sounddevice as sd
import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore

from audio_player import AudioPlayer
from font_manager import FontManager
from inference_tab import InferenceTab
from ui_utils import calculateWindowSize
from widgets_visibility_manager import WidgetVisibilityManager

class VoiceChangerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("so-vits-svc-fork")
        self.setGeometry(*calculateWindowSize())

        self.available = True
        self.train = False

        '''
        Required components:
        - config_path
        - model_path
        - speaker
        - wave_input
        
        The program will automatically makes the following components:
        - wave_ppg
        - wave_hubert
        - wave_pitch
        
        Flowwork:
        - Load model
        - Load speaker
        - Load waveform (audio)
        - Makes ppg.npy with whisper
        - Makes vec.npy with HuBERT
        - Makes csv with f0 estimation
            - If pitch shift is not 0, shift the pitch
            - If checkbox is checked, you can pause here and modify f0 with graph manually
        - Run the model with the loaded components
        '''

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        inference_tab = InferenceTab()
        self.tabs.addTab(inference_tab, "Inference")

    def closeEvent(self, ev):
        # For every tab in tabs list, close the tab
        for i in range(self.tabs.count()):
            self.tabs.widget(i).close()
        ev.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    FontManager.set_global_font_size(app, 12, 14)
    window = VoiceChangerGUI()
    window.show()
    sys.exit(app.exec_())
