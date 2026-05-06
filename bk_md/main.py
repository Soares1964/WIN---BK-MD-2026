# ma_lab/main.py
import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def setup_environment():
    """Configura ambiente e logging na inicializacao"""
    os.makedirs('logs', exist_ok=True)
    try:
        from core.logging_config import setup_logging
        logger = setup_logging('ma_lab')
        logger.info('Sistema iniciado')
    except Exception as e:
        print(f"Logging nao disponivel: {e}")


def main():
    setup_environment()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
