from PyQt5.QtWidgets import QSizePolicy, QWidget

class WidgetVisibilityManager:
    @staticmethod
    def toggle_widgets_visibility(widgets, hide=True, parent_layout=None):
        for widget in widgets:
            if hide:
                widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                widget.hide()
            else:
                widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                widget.show()

        # 부모 레이아웃이 제공된 경우 레이아웃 업데이트
        if parent_layout:
            parent_layout.update()
            parent_layout.activate()

    @staticmethod
    def swap_widgets_visibility(widgets_to_hide, widgets_to_show):
        for widget in widgets_to_hide:
            # widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            widget.hide()
        for widget in widgets_to_show:
            # widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            widget.show()


    @staticmethod
    def find_top_level_widget(widget):
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QWidget) and parent.layout() is not None:
                return parent
            parent = parent.parent()
        return None