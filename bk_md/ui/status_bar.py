# ma_lab/ui/status_bar.py
from PySide6.QtWidgets import QStatusBar, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont


class StatusBar(QStatusBar):
    """Barra de status personalizada"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configurações
        self.setStyleSheet("""
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                font-size: 12px;
                min-height: 30px;
            }
            QStatusBar::item {
                border: none;
            }
            QLabel {
                color: white;
            }
        """)
        
        # Indicador de status
        self.status_label = QLabel("✅ Pronto")
        self.status_label.setStyleSheet("padding: 2px 10px; font-weight: bold;")
        self.addWidget(self.status_label)
        
        # Separador
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #7f8c8d;")
        self.addPermanentWidget(sep1)
        
        # Informações do dataset
        self.data_label = QLabel("📁 Sem dados")
        self.data_label.setStyleSheet("padding: 2px 10px;")
        self.addPermanentWidget(self.data_label)
        
        # Separador
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #7f8c8d;")
        self.addPermanentWidget(sep2)
        
        # Progresso
        self.progress_label = QLabel("⏳ 0%")
        self.progress_label.setStyleSheet("padding: 2px 10px;")
        self.addPermanentWidget(self.progress_label)
        
        # Separador
        sep3 = QLabel("|")
        sep3.setStyleSheet("color: #7f8c8d;")
        self.addPermanentWidget(sep3)
        
        # Velocidade
        self.speed_label = QLabel("⚡ 0 sys/s")
        self.speed_label.setStyleSheet("padding: 2px 10px; color: #f1c40f;")
        self.addPermanentWidget(self.speed_label)
        
        # Separador
        sep4 = QLabel("|")
        sep4.setStyleSheet("color: #7f8c8d;")
        self.addPermanentWidget(sep4)
        
        # Melhor sistema
        self.best_label = QLabel("🏆 —")
        self.best_label.setStyleSheet("padding: 2px 10px; font-weight: bold; color: #f39c12;")
        self.best_label.setMinimumWidth(200)
        self.addPermanentWidget(self.best_label)
        
        # Timer para animações
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_dots = 0
        self.is_processing = False
    
    def show_message(self, message: str, timeout: int = 3000):
        """Mostra mensagem temporária"""
        self.showMessage(message, timeout)
    
    def update_data_info(self, filename: str, rows: int):
        """Atualiza informações do dataset"""
        if rows > 0:
            self.data_label.setText(f"📁 {filename} ({rows:,} linhas)")
        else:
            self.data_label.setText(f"📁 {filename}")
    
    def update_progress(self, progress: float, processed: int, total: int,
                       speed: float, remaining: float):
        """Atualiza informações de progresso"""
        if total > 0:
            self.progress_label.setText(f"⏳ {progress:.1f}% ({processed:,}/{total:,})")
        else:
            self.progress_label.setText(f"⏳ {progress:.1f}%")
            
        self.speed_label.setText(f"⚡ {speed:.0f} sys/s")
        
        if remaining > 0 and remaining < 86400:  # Menos de 24h
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            seconds = int(remaining % 60)
            
            if hours > 0:
                time_str = f"{hours}h{minutes:02d}m"
            else:
                time_str = f"{minutes:02d}:{seconds:02d}"
        else:
            time_str = "--:--"
        
        # Adiciona tempo restante como tooltip
        self.progress_label.setToolTip(f"Tempo restante: {time_str}")
        
        # Ativa animação se estiver processando
        if progress > 0 and progress < 100 and not self.is_processing:
            self.start_animation()
        elif progress >= 100:
            self.stop_animation()
    
    def show_best_system(self, best):
        """Mostra melhor sistema encontrado"""
        if best is not None:
            text = f"🏆 {best.fast_indicator}({best.fast_period}) × {best.slow_indicator}({best.slow_period})"
            self.best_label.setText(text)
            self.best_label.setToolTip(
                f"Sharpe: {best.sharpe:.2f} | PF: {best.profit_factor:.2f} | "
                f"DD: {best.max_dd:.1f}% | Ret: {best.total_return:.1f}%"
            )
        else:
            self.best_label.setText("🏆 —")
            self.best_label.setToolTip("")
    
    def set_status(self, status: str, is_error: bool = False):
        """Define status atual"""
        if is_error:
            self.status_label.setText(f"❌ {status}")
            self.status_label.setStyleSheet("padding: 2px 10px; color: #e74c3c; font-weight: bold;")
            self.stop_animation()
        else:
            self.status_label.setText(f"✅ {status}")
            self.status_label.setStyleSheet("padding: 2px 10px; color: #2ecc71; font-weight: bold;")
    
    def start_animation(self):
        """Inicia animação de processamento"""
        self.is_processing = True
        self.animation_timer.start(500)  # 500ms
    
    def stop_animation(self):
        """Para animação de processamento"""
        self.is_processing = False
        self.animation_timer.stop()
        self.status_label.setText("✅ Pronto")
    
    def update_animation(self):
        """Atualiza animação"""
        self.animation_dots = (self.animation_dots + 1) % 4
        dots = "." * self.animation_dots
        self.status_label.setText(f"⏳ Processando{dots}")