# ma_lab/ui/main_window.py
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QLabel, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon

from ui.tab_config import ConfigTab
from ui.tab_ranking import RankingTab
from ui.tab_heatmap import HeatmapTab
from ui.status_bar import StatusBar


class MainWindow(QMainWindow):
    """Janela principal do Laboratório de Médias Móveis"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Laboratório de Médias Móveis - Mini Índice (WIN)")
        self.resize(1600, 1000)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # Criar abas
        self.config_tab = ConfigTab(self)
        self.ranking_tab = RankingTab(self)
        self.heatmap_tab = HeatmapTab(self)
        
        # Adicionar abas
        self.tabs.addTab(self.config_tab, "⚙️ Configuração")
        self.tabs.addTab(self.ranking_tab, "📊 Ranking")
        self.tabs.addTab(self.heatmap_tab, "🔥 Heatmap")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)
        
        # Conectar sinais
        self.config_tab.optimization_started.connect(self.on_optimization_started)
        self.config_tab.optimization_progress.connect(self.on_optimization_progress)
        self.config_tab.optimization_finished.connect(self.on_optimization_finished)
        self.config_tab.results_ready.connect(self.on_results_ready)
        
        # Walk-Forward result (inicializado antes de qualquer callback)
        self.wf_result = None

        # Timer para atualizações
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # 10Hz
    
    def on_optimization_started(self):
        """Callback quando otimização inicia"""
        self.status_bar.show_message("Otimização iniciada...")
        self.tabs.setTabEnabled(1, False)  # Desabilita ranking
        self.tabs.setTabEnabled(2, False)  # Desabilita heatmap
    
    def on_optimization_progress(self, progress, processed, total, rate, remaining, best):
        """Callback para progresso da otimização"""
        self.status_bar.update_progress(progress, processed, total, rate, remaining)
        
        if best:
            self.status_bar.show_best_system(best)
    
    def on_optimization_finished(self, results):
        """Callback quando otimização termina"""
        self.status_bar.show_message("Otimização concluída!")
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)

        if self.wf_result is not None:
            metrics = self.wf_result.get_aggregated_metrics()
            status_msg = (
                f"WF: {self.wf_result.n_windows} janelas | "
                f"Sharpe OOS: {metrics.get('avg_sharpe', 0):.2f} | "
                f"Estabilidade: {metrics.get('stability_score', 0):.3f}"
            )
            self.status_bar.show_message(status_msg)
        elif results and len(results) > 0:
            msg = (
                f"Melhor: {results[0].fast_indicator}({results[0].fast_period}) x "
                f"{results[0].slow_indicator}({results[0].slow_period}) | "
                f"Sharpe: {results[0].sharpe:.2f} | PF: {results[0].profit_factor:.2f}"
            )
            self.status_bar.show_message(msg)
    
    def on_results_ready(self, data):
        """Callback quando resultados estão prontos"""
        if isinstance(data, tuple):
            # Walk-Forward mode: (df, wf_result)
            df, wf_result = data
            self.ranking_tab.update_data(df)
            self.heatmap_tab.update_data(df)
            # Store WF result for display
            self.wf_result = wf_result
        else:
            # Standard mode
            self.wf_result = None
            self.ranking_tab.update_data(data)
            self.heatmap_tab.update_data(data)
    
    def update_ui(self):
        """Atualizações periódicas da UI"""
        # Pode adicionar animações ou atualizações aqui
        pass
    
    def closeEvent(self, event):
        """Confirmar saída se otimização estiver rodando"""
        if hasattr(self.config_tab, 'engine') and self.config_tab.engine and self.config_tab.engine.is_running:
            reply = QMessageBox.question(
                self,
                "Confirmar Saída",
                "Otimização em andamento. Deseja realmente sair?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if hasattr(self.config_tab, 'engine') and self.config_tab.engine:
                    self.config_tab.engine.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()