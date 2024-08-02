import json
import os
from functools import partial
from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
import torchaudio
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QLabel,
    QComboBox, QPushButton, QLineEdit, QSlider, QRadioButton, QCheckBox,
    QFileDialog
)
from scipy.io import wavfile

from audio_player import AudioPlayer
from manager import *
from ui_utils import get_audio_devices, get_available_devices, load_json_file, save_json_file
from widgets_visibility_manager import WidgetVisibilityManager


class InferenceTab(QWidget):
    PRESET_FOLDER = "presets"
    VALUE_MULTIPLIER = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_data()
        self._setup_ui()
        self._setup_connections()
        self._initialize_widget_visibility()

    def _setup_data(self):
        self.inference_manager = infer_manager.InferManager()
        self.f0_manager = self.inference_manager.f0
        self.speech_encoder = self.inference_manager.speech_encoder
        self.svc = self.inference_manager.svc
        self.post_processing = self.inference_manager.post_processing

        self.input_devices, self.output_devices = get_audio_devices()
        self.devices = get_available_devices()
        self.presets = load_json_file(self.PRESET_FOLDER)

        self._initialize_arguments()

        self.f0_modification_widgets = []
        self.created_widgets = []

    def _initialize_arguments(self):
        self.path_arguments = {
            "model_path": None, "config_path": None, "cluster_model_path": None,
            "speaker": 0, "speech_encoder": None,
        }
        self.common_arguments = {
            "silence_threshold": 0, "cr_threshold": 0, "pitch_shift": 0,
            "noise_scale": 0, "pad_seconds": 0, "chunk_seconds": 0,
            "linear_gradient": 0, "linear_gradient_retain": 0,
            "cluster_infer_ratio": 0, "loudness_envelope_adjustment": 0,
            "retrieval": False, "f0_modification": False, "use_volume": False,
            "auto_predict_f0": False, "f0": 0, "svc": 0, "post_processing": 0,
        }
        self.realtime_arguments = {
            "realtime": False, "crossfade_seconds": 0, "block_seconds": 0,
            "additional_infer_before_seconds": 0, "additional_infer_after_seconds": 0,
        }
        self.run_arguments = {"f0_modification": False}
        self.file_arguments = {"input_audio": None, "output_audio": None}
        self.device_arguments = {"device": 0, "input_device": 0, "output_device": 0}

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.addLayout(self._create_left_column(), 1)
        main_layout.addLayout(self._create_right_column(), 1)

    def _create_left_column(self) -> QVBoxLayout:
        left_column = QVBoxLayout()
        self.paths_group = self._create_paths_group()
        self.common_group = self._create_common_group()
        left_column.addWidget(self.paths_group)
        left_column.addWidget(self.common_group, 1)
        return left_column

    def _create_right_column(self) -> QVBoxLayout:
        right_column = QVBoxLayout()
        self.file_group = self._create_file_group()
        self.run_arguments_group = self._create_run_arguments_group()
        self.realtime_group = self._create_realtime_group()
        self.record_group = self._create_record_group()

        right_column.addWidget(self.file_group, 1)
        right_column.addWidget(self.run_arguments_group)
        right_column.addWidget(self.realtime_group, 1)
        right_column.addWidget(self.record_group)
        right_column.addWidget(self._create_preset_group())
        return right_column

    def _create_paths_group(self) -> QGroupBox:
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.addWidget(
            self._create_path_input("Model path", "Select Model File", self.path_arguments, "model_path",
                                    directory="Model Files (*.pth)"))
        self.config = self._create_path_input("Config path", "Select Config File", self.path_arguments, "config_path",
                                              directory="Config Files (*.json)")
        paths_layout.addWidget(self.config)

        self.config_line_edit = self.config.layout().itemAt(1).widget()
        self.config_btn = self.config.layout().itemAt(2).widget()
        self.config_line_edit.returnPressed.connect(self._load_config)
        self.config_btn.clicked.connect(self._load_config)

        paths_layout.addWidget(
            self._create_path_input("Cluster model path (Optional)", "Select Cluster Model File", self.path_arguments,
                                    "cluster_model_path"))
        return paths_group

    def _load_config(self):
        config_path = self.config_line_edit.text()
        # check if config path is valid and it's json file
        if not config_path or not os.path.exists(config_path) or not config_path.endswith(".json"):
            return
        self.inference_manager.load_config(config_path)
        self.spk = self.inference_manager.spk2id.keys()
        speech_encoder = self.inference_manager.SE

        self._clear_combo_box(self.speaker_box, self.spk, 0)
        self.path_arguments["speaker"] = 0
        self.path_arguments["speech_encoder"] = speech_encoder

    def _create_common_group(self) -> QScrollArea:
        common_scroll_area = QScrollArea()
        common_scroll_area.setWidgetResizable(True)
        common_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        common_group = QGroupBox("Common")
        common_layout = QVBoxLayout(common_group)

        self._add_common_widgets(common_layout)

        common_scroll_area.setWidget(common_group)
        return common_scroll_area

    def _add_common_widgets(self, layout):
        self.svc_widget = self._create_combo_box("SVC", self.svc.svc_modes, self.common_arguments, "svc", 1)
        self.speaker_widget = self._create_combo_box("Speaker", [], self.path_arguments, "speaker", 1)
        self.speaker_box = self.speaker_widget.layout().itemAt(1).widget()
        layout.addWidget(self.speaker_widget)

        sliders = [
            ("Silence threshold", -35.0, -100.0, 0.0, 1.0, "silence_threshold"),
            ("Transpose(Pitch Shift)(12 = 1 Octave)", 0, -24, 24, 1, "pitch_shift"),
            ("Noise scale", 0.40, 0.0, 1.0, 0.01, "noise_scale"),
            ("Pad seconds", 0.50, 0.0, 1.0, 0.01, "pad_seconds"),
            ("Chunk seconds(No Clip When 0)", 0.50, 0.0, 25.0, 0.01, "chunk_seconds"),
            ("Linear Gradient", 0.0, 0.0, 1.0, 0.01, "linear_gradient"),
            ("Linear Gradient Retain", 0.75, 0.0, 1.0, 0.01, "linear_gradient_retain"),
            ("Loudness Envelope Adjustment", 1.0, 0.0, 1.0, 0.01, "loudness_envelope_adjustment"),
            ("Cluster Inference Ratio", 0.4, 0.0, 1.0, 0.01, "cluster_infer_ratio"),
            ("CR Threshold", 0.05, 0.0, 1.0, 0.01, "cr_threshold"),
        ]

        for slider_info in sliders:
            layout.addWidget(self._create_slider(*slider_info, self.common_arguments))

        checkboxes = [
            ("Feature Retrieval(Cluster required)", "retrieval"),
            ("Auto predict F0", "auto_predict_f0"),
            ("Use volume", "use_volume", True),
            ("Use multiple speakers(Not supported Yet)", "use_spk_mix", False),
        ]

        for checkbox_info in checkboxes:
            layout.addWidget(self._create_check_box(*checkbox_info, arguments_dict=self.common_arguments))

        layout.addWidget(self._create_combo_box("F0 method", self.f0_manager.f0_modes, self.common_arguments, "f0"))

        if self.devices:
            # If there is cuda or mps device, prioritize it
            idx = 0
            for i, device in enumerate(self.devices):
                if "cuda" in device[1] or "mps" in device[1]:
                    idx = i
                    break
            layout.addWidget(
                self._create_combo_box("Device", [device[0] for device in self.devices], self.device_arguments,
                                       "device", idx=idx))
        else:
            layout.addWidget(QLabel("No suitable devices found."))
            self.available = False

        self.realtime_checkbox = self._create_check_box("Realtime", "realtime", arguments_dict=self.realtime_arguments)
        layout.addWidget(self.realtime_checkbox)

    def _create_file_group(self) -> QGroupBox:
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(
            self._create_path_input("Input audio path", "Select Input Audio File", self.file_arguments, "input_audio",
                                    directory="Audio Files (*.wav *.mp3 *.flac)"))
        file_layout.addWidget(
            self._create_path_input("Output audio path", "Select Output Audio File", self.file_arguments,
                                    "output_audio", directory="Audio Files (*.wav)"))

        self.graph = AudioPlayer()
        file_layout.addWidget(self.graph)
        play_button = QPushButton("Play/Pause")
        file_layout.addWidget(play_button)
        play_button.clicked.connect(self.graph.toggle_play_pause)
        return file_group

    def _create_run_arguments_group(self) -> QGroupBox:
        run_arguments_group = QGroupBox("Run Arguments")
        run_arguments_layout = QHBoxLayout(run_arguments_group)
        f0_modification_widget = self._create_button("F0 Modification", self.f0_estimation)
        run_arguments_layout.addWidget(f0_modification_widget)
        self.apply_f0_widget = self._create_button("Apply F0 modification", self.apply_f0)
        run_arguments_layout.addWidget(self.apply_f0_widget)
        self.apply_f0_widget.setEnabled(False)
        button = self._create_button("Run", self.run)
        run_arguments_layout.addWidget(button)
        return run_arguments_group

    def _create_realtime_group(self) -> QGroupBox:
        realtime_group = QGroupBox("Realtime")
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.addWidget(
            self._create_slider("Crossfade seconds", 0.05, 0, 1, 0.01, "crossfade_seconds", self.realtime_arguments))
        realtime_layout.addWidget(
            self._create_slider("Block seconds", 0.35, 0, 1, 0.01,"block_seconds", self.realtime_arguments))
        realtime_layout.addWidget(
            self._create_combo_box("Input device", [device[0] for device in self.input_devices], self.device_arguments,
                                   "input_device"))
        realtime_layout.addWidget(self._create_combo_box("Output device", [device[0] for device in self.output_devices],
                                                         self.device_arguments, "output_device"))
        return realtime_group

    def _create_record_group(self) -> QGroupBox:
        record_group = QGroupBox("Record")
        record_layout = QHBoxLayout(record_group)
        record_button = self._create_button("Record", lambda: {})
        record_layout.addWidget(record_button)
        return record_group

    def _create_preset_group(self) -> QGroupBox:
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)

        preset_select_delete = QHBoxLayout()
        preset_select = self._create_combo_box("Select preset", list(self.presets.keys()))
        preset_combo_box = preset_select.layout().itemAt(1).widget()
        preset_select_delete.addWidget(preset_select, stretch=4)
        preset_delete = QPushButton("Delete")
        preset_delete.clicked.connect(lambda: self._delete_gui_preset(preset_combo_box))
        preset_select_delete.addWidget(preset_delete, stretch=1)

        preset_name_add = QHBoxLayout()
        preset_name_add.addWidget(QLabel("Preset name"))
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("Preset name")
        preset_name_add.addWidget(self.preset_name)
        preset_button = QPushButton("Save")
        preset_button.clicked.connect(lambda: self._save_gui_preset(self.preset_name.text(), preset_combo_box))
        preset_name_add.addWidget(preset_button)

        preset_layout.addLayout(preset_select_delete)
        preset_layout.addLayout(preset_name_add)

        preset_combo_box.currentIndexChanged.connect(lambda: self._load_gui_preset(preset_combo_box.currentText()))
        if preset_combo_box.count() > 0:
            self._load_gui_preset(preset_combo_box.currentText())

        return preset_group

    def _setup_connections(self) -> None:
        realtime_checkbox = self.realtime_checkbox.layout().itemAt(0).widget()
        realtime_checkbox.stateChanged.connect(self._toggle_realtime_mode)

    def _initialize_widget_visibility(self) -> None:
        self.realtime_group.hide()
        self.record_group.hide()

    def _toggle_realtime_mode(self) -> None:
        realtime_mode = self.realtime_arguments['realtime']
        widgets_to_hide = [self.file_group, self.run_arguments_group] if realtime_mode else [self.realtime_group,
                                                                                             self.record_group]
        widgets_to_show = [self.realtime_group, self.record_group] if realtime_mode else [self.file_group,
                                                                                          self.run_arguments_group]
        WidgetVisibilityManager.swap_widgets_visibility(widgets_to_hide, widgets_to_show)

    def _create_button(self, label: str, callback: Callable[[], Any]) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(callback)
        return button

    def _create_path_input(self, label: str, dialog_title: str, arguments_dict: Optional[Dict[str, Any]] = None,
                           var_name: Optional[str] = None, directory: str = "All Files (*)") -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel(label))
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        browse_button = QPushButton("Browse")
        layout.addWidget(browse_button)

        def update() -> None:
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = line_edit.text()

        def open_file_dialog() -> None:
            file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", directory)
            if file_path:
                line_edit.setText(file_path)
                update()

        browse_button.clicked.connect(open_file_dialog)
        if arguments_dict is not None and var_name is not None:
            line_edit.textChanged.connect(update)
            setattr(widget, "renew", lambda: line_edit.setText(arguments_dict.get(var_name, "")))

        self.created_widgets.append(widget)
        return widget

    def _create_combo_box(self, label: str, items: List[str], arguments_dict: Optional[Dict[str, Any]] = None,
                          var_name: Optional[str] = None, direction: int = 0, idx: int = 0) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout() if direction == 0 else QHBoxLayout()
        layout.addWidget(QLabel(label))
        combo_box = QComboBox()
        combo_box.addItems(items)
        combo_box.setCurrentIndex(idx)

        if arguments_dict is not None and var_name is not None:
            arguments_dict[var_name] = idx
            combo_box.currentIndexChanged.connect(
                lambda: arguments_dict.__setitem__(var_name, combo_box.currentIndex()))
            setattr(widget, "renew", lambda: combo_box.setCurrentIndex(arguments_dict.get(var_name, 0)))

        layout.addWidget(combo_box)
        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def _clear_combo_box(self, combo_box: QComboBox, items: List[str], name_or_idx: Optional[Union[str, int]]) -> None:
        combo_box.clear()
        combo_box.addItems(items)
        if name_or_idx is not None:
            combo_box.setCurrentIndex(name_or_idx if isinstance(name_or_idx, int) else combo_box.findText(name_or_idx))

    def _create_check_box(self, label: str, var_name: Optional[str] = None,
                          direction: int = 0, default: bool = False, arguments_dict: Optional[Dict[str, Any]] = None) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout() if direction == 0 else QVBoxLayout()
        check_box = QCheckBox(label)
        check_box.setChecked(default)

        def update() -> None:
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = check_box.isChecked()

        if arguments_dict is not None and var_name is not None:
            check_box.stateChanged.connect(update)
            update()
            setattr(widget, "renew", lambda: check_box.setChecked(arguments_dict.get(var_name, False)))

        layout.addWidget(check_box)
        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def _create_radio_button(self, label: str, items: List[str], arguments_dict: Optional[Dict[str, Any]] = None,
                             var_name: Optional[str] = None) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel(label))

        def update(idx) -> None:
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = idx

        update(0)
        for idx, item in enumerate(items):
            radio_button = QRadioButton(item)
            if arguments_dict is not None and var_name is not None:
                if arguments_dict[var_name] == idx:
                    radio_button.setChecked(True)
                radio_button.toggled.connect(lambda checked, idx=idx: update(idx) if checked else None)
            layout.addWidget(radio_button)

        if arguments_dict is not None and var_name is not None:
            setattr(widget, "renew",
                    lambda: layout.itemAt(arguments_dict.get(var_name, 0) + 1).widget().setChecked(True))

        self.created_widgets.append(widget)
        return widget

    def _create_slider(self, label: str, value: float, min: float = 0.0, max: float = 100.0, step: float = 0.5,
                    var_name: Optional[str] = None, arguments_dict: Optional[Dict[str, Any]] = None) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        qlabel = QLabel(f"{label}: {value}")
        layout.addWidget(qlabel)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min * self.VALUE_MULTIPLIER), int(max * self.VALUE_MULTIPLIER))
        slider.setSingleStep(int(step * self.VALUE_MULTIPLIER))
        slider.setValue(int(value * self.VALUE_MULTIPLIER))

        def update():
            value = slider.value() / self.VALUE_MULTIPLIER
            qlabel.setText(f"{label}: {value:.2f}")
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = value

        slider.valueChanged.connect(update)
        update()

        if arguments_dict is not None and var_name is not None:
            setattr(widget, "renew",
                    lambda: slider.setValue(int(arguments_dict.get(var_name, 0) * self.VALUE_MULTIPLIER)))

        layout.addWidget(slider)
        self.created_widgets.append(widget)
        return widget

    def _load_gui_preset(self, preset_name: str) -> None:
        preset = self.presets[preset_name]
        self.common_arguments.update(preset["common"])
        self.realtime_arguments.update(preset["realtime"])
        for widget in self.created_widgets:
            if hasattr(widget, "renew"):
                widget.renew()
        self.preset_name.setText(preset_name)

    def _save_gui_preset(self, preset_name: str, combo_box: Optional[QComboBox] = None) -> None:
        if not preset_name:
            return
        preset = {
            "common": self.common_arguments.copy(),
            "realtime": self.realtime_arguments.copy()
        }
        self.presets[preset_name] = preset
        save_json_file(self.PRESET_FOLDER, preset_name, preset)
        if combo_box is not None:
            self._clear_combo_box(combo_box, list(self.presets.keys()), preset_name)

    def _delete_gui_preset(self, combo_box: Optional[QComboBox] = None) -> None:
        preset_name = combo_box.currentText()
        idx = combo_box.currentIndex()
        del self.presets[preset_name]
        os.remove(f"{self.PRESET_FOLDER}/{preset_name}.json")
        self._clear_combo_box(combo_box, list(self.presets.keys()), max(0, idx - 1))

    def _check_availability(self) -> bool:
        if not self.path_arguments["model_path"] or not os.path.exists(self.path_arguments["model_path"]):
            print("Model path is required.")
            return False
        if not self.path_arguments["config_path"] or not self.inference_manager.config_loaded():
            print("Config path is required.")
            return False
        if not self.devices:
            print("No suitable devices found.")
            return False
        return True

    def _current_device(self):
        return self.devices[self.device_arguments["device"]][1]

    def f0_estimation(self):
        if not self._check_availability():
            return

        device = self._current_device()
        self.inference_manager.load_model(self.common_arguments, self.path_arguments, device)

        input_audio_path = self.file_arguments["input_audio"]
        f0, target_sr, hop_size = self.inference_manager.get_f0(input_audio_path)
        wav = self.inference_manager.f0_to_wav(f0)
        self.graph.load_f0(f0, target_sr, hop_size)
        self.graph.load_audio(wav, target_sr)
        self.apply_f0_widget.setEnabled(True)

    def apply_f0(self):
        device = self._current_device()
        new_f0 = self.graph.get_f0().to(device)
        wav = self.inference_manager.f0_to_wav(new_f0)
        self.graph.load_audio(wav, self.graph.sample_rate)

    def run(self):
        if not self._check_availability():
            return

        device = self._current_device()
        print(f"Common arguments: {self.common_arguments}")
        print(f"Path arguments: {self.path_arguments}")
        self.inference_manager.load_model(self.common_arguments, self.path_arguments, device)

        input_audio_path = self.file_arguments["input_audio"]
        f0 = self.graph.get_f0()

        if f0 is None:
            wav = self.inference_manager.infer(
                input_audio_path,
                self.path_arguments["speaker"],
                self.common_arguments["silence_threshold"],
                self.common_arguments["auto_predict_f0"],
                self.common_arguments["noise_scale"],
                self.common_arguments["use_spk_mix"],
                self.common_arguments["loudness_envelope_adjustment"]
            )
        else:
            f0 = f0.to(device)
            # TODO: Implement f0 modification
            # wav = self.inference_manager.infer_with_f0(input_audio_path, f0)

        output_audio_path = self.file_arguments["output_audio"]
        self._save_wav(wav, output_audio_path, sample_rate=self.inference_manager.target_sample, normalize=False)

    def _save_wav(self, audio_array, output_audio_path, sample_rate=44100, normalize=False):
        if normalize:
            audio_array = audio_array / np.max(np.abs(audio_array))

            # int16으로 변환 (WAV 파일 형식에 적합)
        audio_array_int = (audio_array * 32767).astype(np.int16)

        # WAV 파일로 저장
        wavfile.write(output_audio_path, sample_rate, audio_array_int)

    def closeEvent(self, ev: QCloseEvent) -> None:
        ev.accept()