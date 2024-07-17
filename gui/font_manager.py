class FontManager:
    @staticmethod
    def set_global_font_size(app, small_size=12, large_size=16):
        app.setStyleSheet(f"""
            QLabel {{
                font-size: {small_size}px;
            }}
            QComboBox {{
                font-size: {small_size}px;
            }}
            QRadioButton {{
                font-size: {small_size}px;
            }}
            QCheckBox {{
                font-size: {small_size}px;
            }}
            QLineEdit {{
                font-size: {small_size}px;
            }}
            QGroupBox {{
                font-size: {large_size}px;
            }}
            QPushButton {{
                font-size: {large_size}px;
            }}
            QTabWidget {{
                font-size: {large_size}px;
            }}
        """)