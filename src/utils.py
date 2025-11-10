from qtpy.QtWidgets import QMessageBox

def current_frame(viewer) -> int:
    """
    Return the current time/frame index from the napari viewer.
    Assumes the image is in shape (T, Y, X) so time is axis 0.
    """
    return int(viewer.dims.current_step[0])

def info(msg: str) -> None:
    """Show the informational dialog (confirm actions)."""
    m = QMessageBox()
    m.setIcon(QMessageBox.Information)
    m.setText(msg)
    m.exec_()

def warn(msg: str) -> None:
    """Show a warning dialog (error prompts)."""
    m = QMessageBox()
    m.setIcon(QMessageBox.Warning)
    m.setText(msg)
    m.exec_()