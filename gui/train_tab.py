import json
import os
from functools import partial
from typing import Dict, Any, Optional, Callable, List, Union

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QLabel,
    QComboBox, QPushButton, QLineEdit, QSlider, QRadioButton, QCheckBox,
    QFileDialog
)

from audio_player import AudioPlayer
from manager import *
from ui_utils import get_audio_devices, get_available_devices, load_json_file, save_json_file
from widgets_visibility_manager import WidgetVisibilityManager


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_data()
        self.setup_ui()
        self.setup_connections()
        self.initialize_widget_visibility()

    def setup_data(self):
        self.f0_manager = f0.f0Manager()
        self.f0_modes = self.f0_manager.f0_modes
        self.speech_encoder = speech_encoder.SpeechEncoderManager()
        self.speech_encoder_modes = self.speech_encoder.speech_encoder_modes
        self.svc = svc.SVCManager()
        self.svc_modes = self.svc.svc_modes
        # self.svc_modes = ["so-vits"]
        self.post_processing = post_processing.PostProcessing()
        self.post_processing_modes = self.post_processing.post_processing_modes
        # self.post_processing_modes = ["NSF-HifiGAN", "shallow_diffusion"]
        self.input_devices, self.output_devices = get_audio_devices()
        self.devices = get_available_devices()
        self.preset_folder = "presets"
        self.presets = load_json_file(self.preset_folder)
        
        self.common_arguments = {
            "model_path": None,
            "config_path": None,
            "cluster_model_path": None,
            "speaker": None,
            "silence_threshold": 0,
            "pitch_shift": 0,
            "noise_scale": 0,
            "pad_seconds": 0,
            "chunk_seconds": 0,
            "max_chunk_seconds": 0,
            "absolute_thresh": False,
            "f0_modification": False,
            "auto_predict_f0": False,
            "f0": 0,
            "speech_encoder": [False] * len(self.speech_encoder_modes),
            "svc": 0,
            "post_processing": [False] * len(self.post_processing_modes),
        }
        self.realtime_arguments = {
            "realtime": False,
            "crossfade_seconds": 0,
            "block_seconds": 0,
            "additional_infer_before_seconds": 0,
            "additional_infer_after_seconds": 0,
        }
        self.retrieval_arguments = {
            "retrieval": False,
            "retrieval_ratio": 0.5,
            "n_retrieval_vectors": 3,
        }
        self.run_arguments = {"f0_modification": False}
        self.file_arguments = {"input_audio": None, "output_audio": None}
        self.device_arguments = {"device": 0, "input_device": 0, "output_device": 0}

        self.retrieval_widgets = []
        self.f0_modification_widgets = []
        self.created_widgets = []

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        main_layout.addLayout(self.create_left_column(), 1)
        main_layout.addLayout(self.create_right_column(), 1)

    def create_left_column(self) -> QVBoxLayout:
        left_column = QVBoxLayout()
        self.paths_group = self.create_paths_group()
        self.common_group = self.create_common_group()

        left_column.addWidget(self.paths_group)
        left_column.addWidget(self.common_group, 1)
        return left_column

    def create_right_column(self) -> QVBoxLayout:
        right_column = QVBoxLayout()
        self.file_group = self.create_file_group()
        self.run_arguments_group = self.create_run_arguments_group()
        self.realtime_group = self.create_realtime_group()
        self.record_group = self.create_record_group()

        right_column.addWidget(self.file_group, 1)
        right_column.addWidget(self.run_arguments_group)
        right_column.addWidget(self.realtime_group, 1)
        right_column.addWidget(self.record_group)
        right_column.addWidget(self.create_preset_group())
        return right_column

    def create_paths_group(self) -> QGroupBox:
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.addWidget(self.create_path_input("Model path", "Select Model File", self.common_arguments, "model_path"))
        paths_layout.addWidget(self.create_path_input("Config path", "Select Config File", self.common_arguments, "config_path"))
        paths_layout.addWidget(self.create_path_input("Cluster model path (Optional)", "Select Cluster Model File", self.common_arguments, "cluster_model_path"))
        return paths_group

    def create_common_group(self) -> QScrollArea:
        common_scroll_area = QScrollArea()
        common_scroll_area.setWidgetResizable(True)
        common_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        common_group = QGroupBox("Common")
        common_layout = QVBoxLayout(common_group)
        
        common_layout.addWidget(QLabel("Speaker"))
        common_layout.addWidget(QComboBox())
        common_layout.addWidget(self.create_slider("Silence threshold", -35.0, -100.0, 0.0, 1.0, self.common_arguments, "silence_threshold"))
        common_layout.addWidget(self.create_slider("Pitch", 12, -24, 36, 1, self.common_arguments, "pitch_shift"))
        common_layout.addWidget(self.create_slider("Noise scale", 0.40, 0.0, 1.0, 0.01, self.common_arguments, "noise_scale"))
        common_layout.addWidget(self.create_slider("Pad seconds", 0.10, 0.0, 1.0, 0.01, self.common_arguments, "pad_seconds"))
        common_layout.addWidget(self.create_slider("Chunk seconds", 0.50, 0.0, 1.0, 0.01, self.common_arguments, "chunk_seconds"))
        common_layout.addWidget(self.create_slider("Max chunk seconds", 40.0, 0.0, 200.0, 1, self.common_arguments, "max_chunk_seconds"))
        common_layout.addWidget(self.create_check_box("Absolute threshold", self.common_arguments, "absolute_thresh"))
        common_layout.addWidget(self.create_radio_button("F0 method", self.f0_modes, self.common_arguments, "f0"))
        
        common_layout.addWidget(self.create_retrieval_widgets())
        
        if self.devices:
            common_layout.addWidget(self.create_combo_box("Device", [device[0] for device in self.devices], self.device_arguments, "device"))
        else:
            common_layout.addWidget(QLabel("No suitable devices found."))
            self.available = False

        self.realtime_checkbox = self.create_check_box("Realtime", self.realtime_arguments, "realtime")
        common_layout.addWidget(self.realtime_checkbox)

        common_scroll_area.setWidget(common_group)
        return common_scroll_area

    def create_retrieval_widgets(self) -> QGroupBox:
        retrieval_widget = self.create_check_box("Retrieval", self.retrieval_arguments, "retrieval")
        retrieval_checkbox = retrieval_widget.layout().itemAt(0).widget()
        
        retrieval_ratio = self.create_slider("Retrieval ratio", 0.5, 0.0, 1.0, 0.01, self.retrieval_arguments, "retrieval_ratio")
        self.retrieval_widgets.append(retrieval_ratio)
        
        n_retrieval_vectors = self.create_slider("Number of retrieval vectors", 3, 1, 10, 1, self.retrieval_arguments, "n_retrieval_vectors")
        self.retrieval_widgets.append(n_retrieval_vectors)

        WidgetVisibilityManager.toggle_widgets_visibility(self.retrieval_widgets, True)
        retrieval_checkbox.stateChanged.connect(
            lambda state: WidgetVisibilityManager.toggle_widgets_visibility(self.retrieval_widgets, not state)
        )
        retrieval_checkbox.setChecked(False)

        retrieval_group = QGroupBox()
        retrieval_layout = QVBoxLayout(retrieval_group)
        retrieval_layout.addWidget(retrieval_widget)
        retrieval_layout.addWidget(retrieval_ratio)
        retrieval_layout.addWidget(n_retrieval_vectors)

        return retrieval_group

    def create_file_group(self) -> QGroupBox:
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(self.create_path_input("Input audio path", "Select Input Audio File", self.file_arguments, "input_audio"))
        file_layout.addWidget(self.create_path_input("Output audio path", "Select Output Audio File", self.file_arguments, "output_audio"))

        self.graph = AudioPlayer()
        file_layout.addWidget(self.graph)
        play_button = QPushButton("Play/Pause")
        file_layout.addWidget(play_button)
        play_button.clicked.connect(self.graph.toggle_play_pause)
        return file_group

    def create_run_arguments_group(self) -> QGroupBox:
        run_arguments_group = QGroupBox("Run Arguments")
        run_arguments_layout = QHBoxLayout(run_arguments_group)
        f0_modification_widget = self.create_check_box("F0 Modification")
        run_arguments_layout.addWidget(f0_modification_widget)
        button = self.create_button("Run", lambda: {})
        run_arguments_layout.addWidget(button)
        return run_arguments_group

    def create_realtime_group(self) -> QGroupBox:
        realtime_group = QGroupBox("Realtime")
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.addWidget(self.create_slider("Crossfade seconds", 0.05, 0, 1, 0.01, self.realtime_arguments, "crossfade_seconds"))
        realtime_layout.addWidget(self.create_slider("Block seconds", 0.35, 0, 1, 0.01, self.realtime_arguments, "block_seconds"))
        realtime_layout.addWidget(self.create_combo_box("Input device", [device[0] for device in self.input_devices], self.device_arguments, "input_device"))
        realtime_layout.addWidget(self.create_combo_box("Output device", [device[0] for device in self.output_devices], self.device_arguments, "output_device"))
        return realtime_group

    def create_record_group(self) -> QGroupBox:
        record_group = QGroupBox("Record")
        record_layout = QHBoxLayout(record_group)
        record_button = self.create_button("Record", lambda: {})
        record_layout.addWidget(record_button)
        return record_group

    def create_preset_group(self) -> QGroupBox:
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        preset_select_delete = QHBoxLayout()
        preset_select = self.create_combo_box("Select preset", list(self.presets.keys()))
        preset_combo_box = preset_select.layout().itemAt(1).widget()
        preset_combo_box.currentIndexChanged.connect(lambda: self.load_gui_preset(preset_combo_box.currentText()))
        preset_select_delete.addWidget(preset_select, stretch=4)
        preset_delete = QPushButton("Delete")
        preset_delete.clicked.connect(lambda: self.delete_current_preset(preset_combo_box))
        preset_select_delete.addWidget(preset_delete, stretch=1)

        preset_name_add = QHBoxLayout()
        preset_name_add.addWidget(QLabel("Preset name"))
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("Preset name")
        preset_name_add.addWidget(self.preset_name)
        preset_button = QPushButton("Save")
        preset_button.clicked.connect(lambda: self.save_gui_preset(self.preset_name.text(), preset_combo_box))
        preset_name_add.addWidget(preset_button)

        preset_layout.addLayout(preset_select_delete)
        preset_layout.addLayout(preset_name_add)
        self.load_gui_preset(preset_combo_box.currentText())
        return preset_group

    def setup_connections(self) -> None:
        realtime_checkbox = self.realtime_checkbox.layout().itemAt(0).widget()
        realtime_checkbox.stateChanged.connect(self.toggle_realtime_mode)

    def initialize_widget_visibility(self) -> None:
        # Initially hide realtime_group and record_group
        self.realtime_group.hide()
        self.record_group.hide()

    def toggle_realtime_mode(self) -> None:
        realtime_mode = self.realtime_arguments['realtime']

        if realtime_mode:
            widgets_to_hide = [self.file_group, self.run_arguments_group]
            widgets_to_show = [self.realtime_group, self.record_group]
        else:
            widgets_to_hide = [self.realtime_group, self.record_group]
            widgets_to_show = [self.file_group, self.run_arguments_group]

        WidgetVisibilityManager.swap_widgets_visibility(widgets_to_hide, widgets_to_show)

    def create_button(self, label: str, callback: Callable[[], Any]) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(callback)
        return button

    def create_path_input(self, label: str, dialog_title: str, arguments_dict: Optional[Dict[str, Any]] = None, var_name: Optional[str] = None) -> QWidget:
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        browse_button = QPushButton("Browse")
        layout.addWidget(browse_button)
        widget = QWidget()

        def open_file_dialog() -> None:
            file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", "All Files (*)")
            if file_path:
                line_edit.setText(file_path)

        browse_button.clicked.connect(open_file_dialog)
        if arguments_dict is not None and var_name is not None:
            line_edit.textChanged.connect(lambda: arguments_dict.__setitem__(var_name, line_edit.text()))
            def update_line_edit() -> None:
                text = arguments_dict[var_name]
                if text is not None:
                    line_edit.setText(text)
            setattr(widget, "renew", update_line_edit)

        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def create_combo_box(self, label: str, items: List[str], arguments_dict: Optional[Dict[str, Any]] = None, var_name: Optional[str] = None) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        combo_box = QComboBox()
        combo_box.addItems(items)
        if arguments_dict is not None and var_name is not None:
            combo_box.currentIndexChanged.connect(lambda: arguments_dict.__setitem__(var_name, combo_box.currentIndex()))
            arguments_dict.__setitem__(var_name, combo_box.currentIndex())
            def update_combo_box() -> None:
                combo_idx = arguments_dict[var_name]
                if combo_idx is not None:
                    combo_box.setCurrentIndex(combo_idx)
            setattr(widget, "renew", update_combo_box)
        layout.addWidget(combo_box)
        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def clear_combo_box(
            self,
            combo_box: QComboBox,
            items: List[str],
            name_or_idx: Optional[Union[str, int]]
    ) -> None:
        combo_box.clear()
        combo_box.addItems(items)
        if name_or_idx is not None:
            if isinstance(name_or_idx, int):
                combo_box.setCurrentIndex(name_or_idx)
            else:
                combo_box.setCurrentIndex(combo_box.findText(name_or_idx))

    def create_check_box(self, label: str, arguments_dict: Optional[Dict[str, Any]] = None, var_name: Optional[str] = None) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout()
        check_box = QCheckBox(label)
        if arguments_dict is not None and var_name is not None:
            check_box.stateChanged.connect(lambda: arguments_dict.__setitem__(var_name, check_box.isChecked()))
            arguments_dict.__setitem__(var_name, check_box.isChecked())
            def update_check_box() -> None:
                checked = arguments_dict[var_name]
                if checked is not None:
                    check_box.setChecked(checked)
            setattr(widget, "renew", update_check_box)
        layout.addWidget(check_box)
        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def create_radio_button(self, label: str, items: List[str], arguments_dict: Optional[Dict[str, Any]] = None, var_name: Optional[str] = None) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        for idx, item in enumerate(items):
            radio_button = QRadioButton(item)
            if arguments_dict is not None and var_name is not None:
                if arguments_dict[var_name] == idx:
                    radio_button.setChecked(True)
                radio_button.toggled.connect(partial(lambda idx, var_name, checked: arguments_dict.__setitem__(var_name, idx) if checked else None, idx, var_name))
            layout.addWidget(radio_button)
        widget.setLayout(layout)
        if arguments_dict is not None and var_name is not None:
            def update_radio_buttons() -> None:
                idx = arguments_dict[var_name]
                if idx is not None:
                    layout.itemAt(idx + 1).widget().setChecked(True)
            setattr(widget, "renew", update_radio_buttons)
        self.created_widgets.append(widget)
        return widget

    def create_slider(self, label: str, value: float, min: float = 0.0, max: float = 100.0, step: float = 0.5, arguments_dict: Optional[Dict[str, Any]] = None, var_name: Optional[str] = None) -> QWidget:
        widget = QWidget()
        min = int(min * 10)
        max = int(max * 10)
        value = int(value * 10)
        step = int(step * 10)
        layout = QVBoxLayout()
        qlabel = QLabel(f"{label}: {value / 10}")
        layout.addWidget(qlabel)
        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(max)
        slider.setMinimum(min)
        slider.setSingleStep(step)
        slider.setPageStep(step*2)
        slider.setValue(int(value))  # Assuming slider range 0-100
        def update():
            value = slider.value()
            qlabel.setText(f"{label}: {value / 10}")
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = value / 10
        slider.valueChanged.connect(update)
        update()
        if arguments_dict is not None and var_name is not None:
            def update_slider() -> None:
                slider_value = arguments_dict[var_name]
                if slider_value is not None:
                    slider.setValue(int(slider_value * 10))
                    qlabel.setText(f"{label}: {slider_value}")
            setattr(widget, "renew", update_slider)
        layout.addWidget(slider)
        widget.setLayout(layout)
        self.created_widgets.append(widget)
        return widget

    def load_gui_preset(self, preset_name: str) -> None:
        # Preset is dictionary with keys as the variable names
        preset = self.presets[preset_name]
        # common, realtime, retrieval arguments
        for key, value in preset["common"].items():
            self.common_arguments[key] = value
        for key, value in preset["realtime"].items():
            self.realtime_arguments[key] = value
        for key, value in preset["retrieval"].items():
            self.retrieval_arguments[key] = value
        # We should update sliders, checkboxes, etc. with the new values
        for widget in self.created_widgets:
            if hasattr(widget, "renew"):
                widget.renew()
        self.preset_name.setText(preset_name)


    def save_gui_preset(self, preset_name: str, combo_box: Optional[QComboBox] = None) -> None:
        if not preset_name:
            return
        preset = {}
        common = {}
        realtime = {}
        retrieval = {}
        # common, realtime, file, run_arguments, retrieval
        for key, value in self.common_arguments.items():
            common[key] = value
        for key, value in self.realtime_arguments.items():
            realtime[key] = value
        for key, value in self.retrieval_arguments.items():
            retrieval[key] = value
        preset["common"] = common
        preset["realtime"] = realtime
        preset["retrieval"] = retrieval
        self.presets[preset_name] = preset
        save_json_file(self.preset_folder, preset_name, preset)
        # combo_box_widget > layout > label, combo_box
        if combo_box is not None:
            self.clear_combo_box(combo_box, list(self.presets.keys()), preset_name)

    def save_gui_preset(self, preset_name: str, combo_box: Optional[QComboBox] = None) -> None:
        preset_name = combo_box.currentText()
        idx = combo_box.currentIndex()
        del self.presets[preset_name]
        # Delte the file, if exists
        os.remove(f"presets/{preset_name}.json")
        # Set the combo box to the previous index
        self.clear_combo_box(combo_box, list(self.presets.keys()), idx-1)

    def closeEvent(self, ev: QCloseEvent) -> None:
        ev.accept()