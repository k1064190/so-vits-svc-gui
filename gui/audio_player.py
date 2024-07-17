import io
import wave

import numpy as np
import pandas as pd
from PyQt5 import QtCore

from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import pyqtgraph as pg


class NumpyPlayer(QMediaPlayer):
    def __init__(self):
        super().__init__()
        self.buffer = QBuffer()

    def numpy_to_wav_buffer(self, waveform, sample_rate=44100, sample_width=2):
        # Ensure the waveform is in the correct range (-1 to 1)
        waveform = np.clip(waveform, -1, 1)

        # Scale to 16-bit integers
        waveform = (waveform * 32767).astype(np.int16)

        # Create an in-memory binary stream
        byte_io = io.BytesIO()

        # Create a wave write object
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(waveform.tobytes())

        # Get the binary data
        wav_data = byte_io.getvalue()

        # Create a QByteArray from the WAV data
        qbyte_array = QByteArray(wav_data)

        # Set the QBuffer's data to our WAV data
        self.buffer.setData(qbyte_array)
        self.buffer.open(QIODevice.ReadOnly)

        return self.buffer

    def load_waveform(self, waveform, sample_rate=44100):
        wav_buffer = self.numpy_to_wav_buffer(waveform, sample_rate)
        media_content = QMediaContent(None, wav_buffer)
        self.setMedia(media_content)

        # clear the player state
        self.stop()
        self.setPosition(0)
class AudioPlayer(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.available = False
        self.scatter = None
        self.dragPoint = None
        self.dragOffset = None

        self.linePen = pg.mkPen(color=(255, 0, 0), width=1)
        self.symbol = {
            "symbol": 'o',
            "size": 10,
            "brush": "b",
        }

        # Load data
        self.setBackground("w")

        # Load audio
        self.player = NumpyPlayer()

        # Player parameters
        self.player_prev_position = self.player.position()

        # Player Event
        self.player.setNotifyInterval(50)  # 10ms
        self.player.positionChanged.connect(self.update_plot)
        # clicking empty space in the plot will move the position to that time
        self.plotItem.scene().sigMouseClicked.connect(self.moveTime)

        self.getViewBox().scaleBy((0.2, 1))

    def load_csv(self, csv):
        csv = pd.read_csv(csv, header=None)
        self.x_data = np.arange(0, len(csv[0]))  # unit = 10ms
        self.y_data = csv[1].values
        self.plot(
            self.x_data,
            self.y_data,
            pen=self.linePen,
            **self.symbol
        )

    def load_f0(self, f0):
        # waveform is a numpy array
        self.x_data = np.arange(0, len(f0))
        self.y_data = f0
        self.plot(
            self.x_data,
            self.y_data,
            pen=self.linePen,
            **self.symbol
        )

    def load_wavefile(self, wavefile):
        # Load the audio file
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(wavefile)))
        self.player.stop()
        self.player.setPosition(0)

    def load_audio(self, audio, sample_rate=44100):
        self.player.load_waveform(audio, sample_rate)


    def plot(self, x, y, pen, **kwargs):
        # clear the plot
        self.clear()

        self.scatter = pg.ScatterPlotItem(x=x, y=y, **kwargs)
        self.addItem(self.scatter)
        # self.scatter.sigClicked.connect(self.clicked)

        # Add a line connecting the points
        pen = 'r' if pen is None else pen
        self.line = pg.PlotDataItem(x=x, y=y, pen=pen)
        self.addItem(self.line)

        self.playbar = self.addLine(x=0, pen=pg.mkPen(color=(0, 0, 0), width=2))

        self.autoRange()

    def toggle_play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.position() == self.player.duration():
                self.player.setPosition(0)
            self.player.play()

    def moveTime(self, ev):
        vb = self.plotItem.vb
        scene_coords = ev.scenePos()
        if self.sceneBoundingRect().contains(scene_coords):
            mouse_point = vb.mapSceneToView(scene_coords)
            time = int(mouse_point.x() * 10)
            time = max(0, min(time, self.player.duration()))
            self.player.setPosition(time)

    def mousePressEvent(self, ev):
        if self.available:
            pos = self.plotItem.vb.mapSceneToView(ev.pos())
            points = self.scatter.pointsAt(pos)
            if len(points) > 0:
                self.dragPoint = points[0]
                self.dragStartPos = self.dragPoint.pos()
            elif ev.button() == QtCore.Qt.LeftButton:
                super().mousePressEvent(ev)
        else:
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self.available:
            if self.dragPoint is not None:
                if self.sceneBoundingRect().contains(ev.pos()):
                    pos = self.plotItem.vb.mapSceneToView(ev.pos())
                    new_y = pos.y()
                    self.updatePointPos(self.dragPoint, new_y)
                    self.updateLine()
                else:
                    # release the point
                    self.dragPoint = None
                    self.dragStartPos = None
                    super().mouseReleaseEvent(ev)
            else:
                super().mouseMoveEvent(ev)
        else:
            ev.ignore()
    def mouseReleaseEvent(self, ev):
        if self.available:
            self.dragPoint = None
            self.dragStartPos = None
            super().mouseReleaseEvent(ev)
        else:
            ev.ignore()

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

    def updatePointPos(self, point, new_y):
        index = point.index()
        self.y_data[index] = new_y
        self.scatter.setData(x=self.x_data, y=self.y_data)

    def updateLine(self):
        self.line.setData(x=self.x_data, y=self.y_data)


    def update_playbar(self, time):
        self.playbar.setValue(time)

    def update_plot(self):
        if self.available:
            position = self.player.position()
            if position != self.player_prev_position:
                self.player_prev_position = position
                self.update_playbar(position // 10)