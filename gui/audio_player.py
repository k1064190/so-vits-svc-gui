import io
import wave
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtCore
import tempfile
import soundfile as sf
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg

class NumpyPlayer(QMediaPlayer):
    def __init__(self):
        super().__init__()

    def load_waveform(self, waveform, sample_rate=44100):
        waveform = self._prepare_waveform(waveform)
        temp_wav_path = self._save_temp_wav(waveform, sample_rate)
        self._set_media_content(temp_wav_path)
        self._reset_player()

    def _prepare_waveform(self, waveform):
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        return np.clip(waveform, -32768, 32767).astype(np.int16)

    def _save_temp_wav(self, waveform, sample_rate):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            sf.write(temp_wav.name, waveform, sample_rate)
            return temp_wav.name

    def _set_media_content(self, file_path):
        media_content = QMediaContent(QUrl.fromLocalFile(file_path))
        self.setMedia(media_content)

    def _reset_player(self):
        self.stop()
        self.setPosition(0)

class AudioPlayer(pg.PlotWidget):
    NOTIFY_INTERVAL = 50
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_HOP_SIZE = 512

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initialize_attributes()
        self._setup_plot()
        self._setup_player()

    def _initialize_attributes(self):
        self.available = False
        self.scatter = None
        self.dragPoint = None
        self.dragOffset = None
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.hop_size = self.DEFAULT_HOP_SIZE
        self.f0_period = self._calculate_f0_period()
        self.x_data = None
        self.y_data = None

    def _setup_plot(self):
        self.setBackground("w")
        self.linePen = pg.mkPen(color=(255, 0, 0), width=1)
        self.symbol = {"symbol": 'o', "size": 10, "brush": "b"}

    def _setup_player(self):
        self.player = NumpyPlayer()
        self.player_prev_position = self.player.position()
        self.player.setNotifyInterval(self.NOTIFY_INTERVAL)
        self.player.positionChanged.connect(self.update_plot)
        self.plotItem.scene().sigMouseClicked.connect(self.moveTime)

    def _calculate_f0_period(self):
        return self.hop_size / self.sample_rate * 1000

    def load_csv(self, csv_file):
        try:
            csv = pd.read_csv(csv_file, header=None)
            self.x_data = np.arange(len(csv[0]))
            self.y_data = csv[1].values
            self._plot_data()
            self.available = True
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def load_f0(self, f0, sample_rate=DEFAULT_SAMPLE_RATE, hop_size=DEFAULT_HOP_SIZE):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_period = self._calculate_f0_period()
        f0 = f0.squeeze()
        self.x_data = np.arange(len(f0))
        self.y_data = f0.cpu().numpy() if isinstance(f0, torch.Tensor) else f0
        self._plot_data()
        self.available = True
        self.adjust_view_box()

    def _plot_data(self):
        self.clear()
        self.scatter = pg.ScatterPlotItem(x=self.x_data, y=self.y_data, **self.symbol)
        self.addItem(self.scatter)
        self.line = pg.PlotDataItem(x=self.x_data, y=self.y_data, pen=self.linePen)
        self.addItem(self.line)
        self.playbar = self.addLine(x=0, pen=pg.mkPen(color=(0, 0, 0), width=2))
        self.autoRange()

    def adjust_view_box(self):
        if len(self.y_data) > 0:
            f0_min = max(50, np.min(self.y_data[self.y_data > 0]))
            f0_max = min(1100, np.max(self.y_data))
            x_range = len(self.x_data)
            y_range = f0_max - f0_min
            aspect_ratio = (x_range / y_range) * 0.2
            self.getViewBox().setAspectLocked(True, aspect_ratio)
            self.setXRange(0, x_range, padding=0.02)
            self.setYRange(f0_min, f0_max, padding=0.1)

    def load_wavefile(self, wavefile):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(wavefile)))
        self._reset_player()

    def load_audio(self, audio, sample_rate=DEFAULT_SAMPLE_RATE):
        self.player.load_waveform(audio, sample_rate)

    def toggle_play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.position() == self.player.duration():
                self.player.setPosition(0)
            self.player.play()

    def moveTime(self, ev):
        if self.sceneBoundingRect().contains(ev.scenePos()):
            mouse_point = self.plotItem.vb.mapSceneToView(ev.scenePos())
            time = int(mouse_point.x() * self.f0_period)
            time = max(0, min(time, self.player.duration()))
            self.player.setPosition(time)

    def mousePressEvent(self, ev):
        if not self.available:
            ev.ignore()
            return
        pos = self.plotItem.vb.mapSceneToView(ev.pos())
        points = self.scatter.pointsAt(pos)
        # Check if points are empty or not
        if len(points) > 0:
            self.dragPoint = points[0]
            self.dragStartPos = self.dragPoint.pos()
        elif ev.button() == QtCore.Qt.LeftButton:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if not self.available:
            ev.ignore()
            return
        if self.dragPoint is not None:
            if self.sceneBoundingRect().contains(ev.pos()):
                pos = self.plotItem.vb.mapSceneToView(ev.pos())
                self._update_point_position(self.dragPoint, pos.y())
            else:
                self._release_drag_point()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if not self.available:
            ev.ignore()
            return
        self._release_drag_point()
        super().mouseReleaseEvent(ev)

    def _release_drag_point(self):
        self.dragPoint = None
        self.dragStartPos = None

    def _update_point_position(self, point, new_y):
        index = point.index()
        self.y_data[index] = new_y
        self.scatter.setData(x=self.x_data, y=self.y_data)
        self.line.setData(x=self.x_data, y=self.y_data)

    def update_plot(self):
        if self.available:
            position = self.player.position()
            if position != self.player_prev_position:
                self.player_prev_position = position
                self.playbar.setValue(position // self.f0_period)

    def _reset_player(self):
        self.player.stop()
        self.player.setPosition(0)

    def get_f0(self):
        return torch.from_numpy(self.y_data).unsqueeze(0) if self.y_data is not None else None

    def clear_player(self):
        # reset datas and audio player, plot
        self.available = False
        self.x_data = None
        self.y_data = None
        self.clear()
        self.player.setMedia(QMediaContent())
        self._reset_player()


    # The following methods remain unchanged
    def wheelEvent(self, event):
        if self.available:
            super().wheelEvent(event)
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.available:
            super().mouseDoubleClickEvent(event)
        else:
            event.ignore()