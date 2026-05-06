# ma_lab/ui/tab_config.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QFileDialog, QHeaderView, QComboBox,
    QScrollArea, QFrame, QSplitter, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

import pandas as pd
import numpy as np
from datetime import datetime
import time

from core.data_loader import DataLoader
from core.optimizer_engine import OptimizationEngine, OptimizationConfig
from core.indicators import get_all_indicators, get_categories
from core.walkforward import WalkForwardEngine, WalkForwardConfig, WalkForwardResult
from utils.dataset_validator import validate_dataset


class OptimizationThread(QThread):
    """Thread para executar otimização em background"""

    progress = Signal(float, int, int, float, float, object)
    result = Signal(object)
    finished = Signal(object)
    wf_progress = Signal(float, int, int, float, float, object, str)

    def __init__(self, engine, config, wf_engine=None):
        super().__init__()
        self.engine = engine
        self.config = config
        self.wf_engine = wf_engine

    def run(self):
        if self.wf_engine is not None:
            # Walk-Forward mode
            wf_result = self.wf_engine.run(
                self.config,
                progress_callback=self._on_wf_progress
            )
            # Extract OOS results for ranking
            oos_list = wf_result.get_all_oos()
            # Store WF result for aggregated metrics
            self.wf_result = wf_result
            self.finished.emit(oos_list if oos_list else [])
        else:
            # Standard optimization mode
            results = self.engine.run(
                self.config,
                progress_callback=self._on_progress,
                result_callback=self._on_result
            )
            self.finished.emit(results)

    def _on_progress(self, progress, processed, total, rate, remaining, best):
        self.progress.emit(progress, processed, total, rate, remaining, best)

    def _on_result(self, result):
        self.result.emit(result)

    def _on_wf_progress(self, pct, window_num, total_windows, rate, remaining, best, best_str):
        self.wf_progress.emit(pct, window_num, total_windows, rate, remaining, best, best_str)


class ConfigTab(QWidget):
    """Aba de configuração e otimização"""
    
    # Sinais
    optimization_started = Signal()
    optimization_progress = Signal(float, int, int, float, float, object)
    optimization_finished = Signal(object)
    results_ready = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data_loader = DataLoader()
        self.engine = None
        self.optimization_thread = None
        self.current_results = None
        self.partial_results = []
        
        self.setup_ui()
        self.load_indicators()
    
    def setup_ui(self):
        """Configura a interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ==========================================================
        # TÍTULO
        # ==========================================================
        title = QLabel("⚙️ LABORATÓRIO DE OTIMIZAÇÃO DE MÉDIAS MÓVEIS - MINI ÍNDICE (WIN)")
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        """)
        main_layout.addWidget(title)
        
        # ==========================================================
        # DADOS
        # ==========================================================
        data_box = QGroupBox("📁 DADOS")
        data_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        data_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Carregar CSV")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.load_btn.clicked.connect(self.load_csv)
        
        self.file_label = QLabel("Arquivo: nenhum")
        self.file_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        self.stats_label = QLabel("Status: ⏳ Aguardando arquivo")
        self.stats_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(self.file_label, 2)
        data_layout.addWidget(self.stats_label, 2)
        
        data_box.setLayout(data_layout)
        main_layout.addWidget(data_box)
        
        # ==========================================================
        # SCROLL AREA PARA CONFIGURAÇÕES
        # ==========================================================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        # ==========================================================
        # TIMEFRAMES
        # ==========================================================
        tf_box = QGroupBox("⏱️ TIMEFRAMES PARA TESTE")
        tf_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        tf_layout = QHBoxLayout()
        self.tf_checks = []
        
        for i in range(1, 11):
            cb = QCheckBox(f"{i} min")
            if i <= 5:
                cb.setChecked(True)
            self.tf_checks.append(cb)
            tf_layout.addWidget(cb)
        
        tf_box.setLayout(tf_layout)
        scroll_layout.addWidget(tf_box)
        
        # ==========================================================
        # SELEÇÃO DE FILTROS
        # ==========================================================
        filters_box = QGroupBox("🔍 SELEÇÃO DE FILTROS DE SUAVIZAÇÃO")
        filters_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        filters_layout = QVBoxLayout()
        
        # Categorias de filtros
        self.category_widgets = {}
        categories = [
            "Clássicas", "Redução de Lag", "Adaptativas",
            "Filtros DSP", "Filtros Gaussianos", "Filtros Estatísticos",
            "Suavização Avançada"
        ]
        
        for category in categories:
            cat_box = QGroupBox(category)
            cat_box.setStyleSheet("QGroupBox { font-weight: normal; font-size: 12px; }")
            
            cat_layout = QGridLayout()
            cat_layout.setHorizontalSpacing(20)
            
            # Botões para selecionar todos/nenhum da categoria
            select_all = QPushButton("✓ Todos")
            select_all.setMaximumWidth(80)
            select_all.setStyleSheet("QPushButton { font-size: 11px; }")
            select_all.clicked.connect(lambda checked, c=category: self.select_category(c, True))
            
            select_none = QPushButton("✗ Nenhum")
            select_none.setMaximumWidth(80)
            select_none.setStyleSheet("QPushButton { font-size: 11px; }")
            select_none.clicked.connect(lambda checked, c=category: self.select_category(c, False))
            
            cat_layout.addWidget(select_all, 0, 0)
            cat_layout.addWidget(select_none, 0, 1)
            
            # Espaço para checkboxes
            self.category_widgets[category] = {
                'layout': cat_layout,
                'checkboxes': [],
                'row': 1,
                'col': 0
            }
            
            cat_box.setLayout(cat_layout)
            filters_layout.addWidget(cat_box)
        
        filters_box.setLayout(filters_layout)
        scroll_layout.addWidget(filters_box)
        
        # ==========================================================
        # PARÂMETROS DE PERÍODOS
        # ==========================================================
        param_box = QGroupBox("📊 PARÂMETROS DE PERÍODOS")
        param_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        param_layout = QHBoxLayout()
        
        # FAST
        fast_box = QGroupBox("FAST (Entrada)")
        fast_box.setStyleSheet("QGroupBox { font-weight: normal; }")
        fast_layout = QGridLayout()
        
        self.fast_min = QSpinBox()
        self.fast_min.setRange(2, 100)
        self.fast_min.setValue(5)
        self.fast_min.setStyleSheet("QSpinBox { padding: 3px; }")
        
        self.fast_max = QSpinBox()
        self.fast_max.setRange(3, 200)
        self.fast_max.setValue(50)
        self.fast_max.setStyleSheet("QSpinBox { padding: 3px; }")
        
        self.fast_step = QSpinBox()
        self.fast_step.setRange(1, 20)
        self.fast_step.setValue(1)
        self.fast_step.setStyleSheet("QSpinBox { padding: 3px; }")
        
        fast_layout.addWidget(QLabel("Mín:"), 0, 0)
        fast_layout.addWidget(self.fast_min, 0, 1)
        fast_layout.addWidget(QLabel("Máx:"), 1, 0)
        fast_layout.addWidget(self.fast_max, 1, 1)
        fast_layout.addWidget(QLabel("Step:"), 2, 0)
        fast_layout.addWidget(self.fast_step, 2, 1)
        
        fast_box.setLayout(fast_layout)
        
        # SLOW
        slow_box = QGroupBox("SLOW (Confirmação)")
        slow_box.setStyleSheet("QGroupBox { font-weight: normal; }")
        slow_layout = QGridLayout()
        
        self.slow_min = QSpinBox()
        self.slow_min.setRange(5, 200)
        self.slow_min.setValue(20)
        self.slow_min.setStyleSheet("QSpinBox { padding: 3px; }")
        
        self.slow_max = QSpinBox()
        self.slow_max.setRange(10, 500)
        self.slow_max.setValue(200)
        self.slow_max.setStyleSheet("QSpinBox { padding: 3px; }")
        
        self.slow_step = QSpinBox()
        self.slow_step.setRange(1, 50)
        self.slow_step.setValue(5)
        self.slow_step.setStyleSheet("QSpinBox { padding: 3px; }")
        
        slow_layout.addWidget(QLabel("Mín:"), 0, 0)
        slow_layout.addWidget(self.slow_min, 0, 1)
        slow_layout.addWidget(QLabel("Máx:"), 1, 0)
        slow_layout.addWidget(self.slow_max, 1, 1)
        slow_layout.addWidget(QLabel("Step:"), 2, 0)
        slow_layout.addWidget(self.slow_step, 2, 1)
        
        slow_box.setLayout(slow_layout)
        
        param_layout.addWidget(fast_box)
        param_layout.addWidget(slow_box)
        
        # Estimativa de combinações
        est_widget = QWidget()
        est_layout = QVBoxLayout(est_widget)
        
        est_label = QLabel("Total combinações estimadas:")
        est_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.est_value = QLabel("0")
        self.est_value.setAlignment(Qt.AlignCenter)
        self.est_value.setStyleSheet("""
            font-weight: bold; 
            color: #2980b9; 
            font-size: 18px;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
        """)
        
        est_layout.addWidget(est_label)
        est_layout.addWidget(self.est_value)
        
        param_layout.addWidget(est_widget)
        
        param_box.setLayout(param_layout)
        scroll_layout.addWidget(param_box)
        
        # ==========================================================
        # CONFIGURAÇÃO DE BACKTEST E FILTROS DE QUALIDADE
        # ==========================================================
        config_row = QHBoxLayout()
        
        # CONFIGURAÇÃO DE BACKTEST
        bt_box = QGroupBox("🔄 CONFIGURAÇÃO DE BACKTEST")
        bt_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        bt_layout = QGridLayout()
        bt_layout.setVerticalSpacing(10)
        
        # Execução
        bt_layout.addWidget(QLabel("Execução:"), 0, 0)
        self.exec_type = QComboBox()
        self.exec_type.addItems(["Vetorizado (ULTRA RÁPIDO)", "Tick a Tick"])
        self.exec_type.setStyleSheet("QComboBox { padding: 3px; }")
        bt_layout.addWidget(self.exec_type, 0, 1)
        
        # Entrada
        bt_layout.addWidget(QLabel("Entrada:"), 1, 0)
        self.entry_type = QComboBox()
        self.entry_type.addItems(["Cruzamento", "Filtro de inclinação"])
        self.entry_type.setStyleSheet("QComboBox { padding: 3px; }")
        bt_layout.addWidget(self.entry_type, 1, 1)
        
        # Custos
        bt_layout.addWidget(QLabel("Slippage:"), 2, 0)
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 10)
        self.slippage.setValue(1.0)
        self.slippage.setSingleStep(0.1)
        self.slippage.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        bt_layout.addWidget(self.slippage, 2, 1)
        
        bt_layout.addWidget(QLabel("Taxa:"), 3, 0)
        self.cost = QDoubleSpinBox()
        self.cost.setRange(0, 10)
        self.cost.setValue(0.5)
        self.cost.setSingleStep(0.1)
        self.cost.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        bt_layout.addWidget(self.cost, 3, 1)
        
        bt_box.setLayout(bt_layout)
        config_row.addWidget(bt_box)
        
        # FILTROS DE QUALIDADE
        quality_box = QGroupBox("🎯 FILTROS DE QUALIDADE")
        quality_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        quality_layout = QGridLayout()
        quality_layout.setVerticalSpacing(10)
        
        quality_layout.addWidget(QLabel("Mín Trades:"), 0, 0)
        self.min_trades = QSpinBox()
        self.min_trades.setRange(10, 10000)
        self.min_trades.setValue(100)
        self.min_trades.setStyleSheet("QSpinBox { padding: 3px; }")
        quality_layout.addWidget(self.min_trades, 0, 1)
        
        quality_layout.addWidget(QLabel("Mín Sharpe:"), 1, 0)
        self.min_sharpe = QDoubleSpinBox()
        self.min_sharpe.setRange(0, 5)
        self.min_sharpe.setValue(1.2)
        self.min_sharpe.setSingleStep(0.1)
        self.min_sharpe.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        quality_layout.addWidget(self.min_sharpe, 1, 1)
        
        quality_layout.addWidget(QLabel("Mín ProfitFact:"), 2, 0)
        self.min_pf = QDoubleSpinBox()
        self.min_pf.setRange(1, 10)
        self.min_pf.setValue(1.5)
        self.min_pf.setSingleStep(0.1)
        self.min_pf.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        quality_layout.addWidget(self.min_pf, 2, 1)
        
        quality_layout.addWidget(QLabel("Máx Drawdown:"), 3, 0)
        self.max_dd = QDoubleSpinBox()
        self.max_dd.setRange(1, 50)
        self.max_dd.setValue(8.0)
        self.max_dd.setSingleStep(0.5)
        self.max_dd.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        quality_layout.addWidget(self.max_dd, 3, 1)
        
        quality_box.setLayout(quality_layout)
        config_row.addWidget(quality_box)
        
        scroll_layout.addLayout(config_row)
        
        # ==========================================================
        # CONTROLE DE EXECUÇÃO
        # ==========================================================
        control_box = QGroupBox("⚡ CONTROLE DE EXECUÇÃO")
        control_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        control_layout = QGridLayout()
        control_layout.setHorizontalSpacing(30)
        
        control_layout.addWidget(QLabel("Threads CPU:"), 0, 0)
        self.threads = QSpinBox()
        self.threads.setRange(1, 32)
        self.threads.setValue(8)
        self.threads.setStyleSheet("QSpinBox { padding: 3px; }")
        control_layout.addWidget(self.threads, 0, 1)
        
        control_layout.addWidget(QLabel("Cache:"), 0, 2)
        self.use_cache = QCheckBox()
        self.use_cache.setChecked(True)
        control_layout.addWidget(self.use_cache, 0, 3)
        
        control_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(100, 50000)
        self.batch_size.setValue(5000)
        self.batch_size.setSingleStep(500)
        self.batch_size.setStyleSheet("QSpinBox { padding: 3px; }")
        control_layout.addWidget(self.batch_size, 1, 1)
        
        control_layout.addWidget(QLabel("Modo Debug:"), 1, 2)
        self.debug = QCheckBox()
        control_layout.addWidget(self.debug, 1, 3)
        
        control_box.setLayout(control_layout)
        scroll_layout.addWidget(control_box)

        # ==========================================================
        # STOP-LOSS E TAKE-PROFIT
        # ==========================================================
        sl_box = QGroupBox("🛑 STOP-LOSS E TAKE-PROFIT")
        sl_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        sl_layout = QGridLayout()
        sl_layout.setVerticalSpacing(8)
        sl_layout.setHorizontalSpacing(15)

        sl_layout.addWidget(QLabel("Stop-Loss:"), 0, 0)
        self.sl_type = QComboBox()
        self.sl_type.addItems(["Nenhum", "Fixo (pts)", "ATR múltiplo", "Trailing ATR"])
        self.sl_type.setStyleSheet("QComboBox { padding: 3px; }")
        sl_layout.addWidget(self.sl_type, 0, 1)

        self.sl_value = QDoubleSpinBox()
        self.sl_value.setRange(0.1, 500)
        self.sl_value.setValue(50.0)
        self.sl_value.setSingleStep(5)
        self.sl_value.setDecimals(1)
        self.sl_value.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        sl_layout.addWidget(self.sl_value, 0, 2)

        sl_layout.addWidget(QLabel("Take-Profit:"), 1, 0)
        self.tp_type = QComboBox()
        self.tp_type.addItems(["Nenhum", "Fixo (pts)", "ATR múltiplo"])
        self.tp_type.setStyleSheet("QComboBox { padding: 3px; }")
        sl_layout.addWidget(self.tp_type, 1, 1)

        self.tp_value = QDoubleSpinBox()
        self.tp_value.setRange(0.1, 1000)
        self.tp_value.setValue(150.0)
        self.tp_value.setSingleStep(10)
        self.tp_value.setDecimals(1)
        self.tp_value.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        sl_layout.addWidget(self.tp_value, 1, 2)

        sl_layout.addWidget(QLabel("Período ATR:"), 2, 0)
        self.atr_period = QSpinBox()
        self.atr_period.setRange(5, 100)
        self.atr_period.setValue(14)
        self.atr_period.setStyleSheet("QSpinBox { padding: 3px; }")
        sl_layout.addWidget(self.atr_period, 2, 1)

        sl_box.setLayout(sl_layout)
        scroll_layout.addWidget(sl_box)

        # ==========================================================
        # POSITION SIZING
        # ==========================================================
        sizing_box = QGroupBox("📐 POSITION SIZING")
        sizing_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        sizing_layout = QGridLayout()
        sizing_layout.setVerticalSpacing(8)
        sizing_layout.setHorizontalSpacing(15)

        sizing_layout.addWidget(QLabel("Método:"), 0, 0)
        self.sizing_method = QComboBox()
        self.sizing_method.addItems(["Fixo (1 contrato)", "Fracionário Fixo", "Kelly Ótimo"])
        self.sizing_method.setStyleSheet("QComboBox { padding: 3px; }")
        sizing_layout.addWidget(self.sizing_method, 0, 1)

        sizing_layout.addWidget(QLabel("Risco por trade:"), 1, 0)
        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10)
        self.risk_per_trade.setValue(1.0)
        self.risk_per_trade.setSingleStep(0.25)
        self.risk_per_trade.setDecimals(2)
        self.risk_per_trade.setSuffix("%")
        self.risk_per_trade.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        sizing_layout.addWidget(self.risk_per_trade, 1, 1)

        sizing_layout.addWidget(QLabel("Kelly fracão:"), 2, 0)
        self.kelly_fraction = QDoubleSpinBox()
        self.kelly_fraction.setRange(0.05, 1.0)
        self.kelly_fraction.setValue(0.25)
        self.kelly_fraction.setSingleStep(0.05)
        self.kelly_fraction.setDecimals(2)
        self.kelly_fraction.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        sizing_layout.addWidget(self.kelly_fraction, 2, 1)

        sizing_layout.addWidget(QLabel("Aloc. máxima:"), 3, 0)
        self.max_alloc_pct = QDoubleSpinBox()
        self.max_alloc_pct.setRange(1, 100)
        self.max_alloc_pct.setValue(20)
        self.max_alloc_pct.setSingleStep(5)
        self.max_alloc_pct.setSuffix("%")
        self.max_alloc_pct.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        sizing_layout.addWidget(self.max_alloc_pct, 3, 1)

        sizing_box.setLayout(sizing_layout)
        scroll_layout.addWidget(sizing_box)

        # ==========================================================
        # WALK-FORWARD OPTIMIZATION
        # ==========================================================
        wf_box = QGroupBox("🔄 WALK-FORWARD OPTIMIZATION")
        wf_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        wf_layout = QGridLayout()
        wf_layout.setVerticalSpacing(8)
        wf_layout.setHorizontalSpacing(15)

        self.wf_enabled = QCheckBox("Ativar Walk-Forward")
        self.wf_enabled.setStyleSheet("font-weight: bold; color: #27ae60;")
        wf_layout.addWidget(self.wf_enabled, 0, 0, 1, 3)

        wf_layout.addWidget(QLabel("Janelas:"), 1, 0)
        self.wf_n_windows = QSpinBox()
        self.wf_n_windows.setRange(2, 10)
        self.wf_n_windows.setValue(3)
        self.wf_n_windows.setStyleSheet("QSpinBox { padding: 3px; }")
        wf_layout.addWidget(self.wf_n_windows, 1, 1)

        wf_layout.addWidget(QLabel("Treino %:"), 2, 0)
        self.wf_train_pct = QDoubleSpinBox()
        self.wf_train_pct.setRange(0.2, 0.8)
        self.wf_train_pct.setValue(0.6)
        self.wf_train_pct.setSingleStep(0.05)
        self.wf_train_pct.setDecimals(2)
        self.wf_train_pct.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        wf_layout.addWidget(self.wf_train_pct, 2, 1)

        wf_layout.addWidget(QLabel("Teste %:"), 3, 0)
        self.wf_test_pct = QDoubleSpinBox()
        self.wf_test_pct.setRange(0.1, 0.4)
        self.wf_test_pct.setValue(0.2)
        self.wf_test_pct.setSingleStep(0.05)
        self.wf_test_pct.setDecimals(2)
        self.wf_test_pct.setStyleSheet("QDoubleSpinBox { padding: 3px; }")
        wf_layout.addWidget(self.wf_test_pct, 3, 1)

        wf_layout.addWidget(QLabel("Modo:"), 4, 0)
        self.wf_cv_mode = QComboBox()
        self.wf_cv_mode.addItems(["rolling", "expanding"])
        self.wf_cv_mode.setStyleSheet("QComboBox { padding: 3px; }")
        wf_layout.addWidget(self.wf_cv_mode, 4, 1)

        wf_layout.addWidget(QLabel("Top-N:"), 5, 0)
        self.wf_top_n = QSpinBox()
        self.wf_top_n.setRange(1, 20)
        self.wf_top_n.setValue(5)
        self.wf_top_n.setStyleSheet("QSpinBox { padding: 3px; }")
        wf_layout.addWidget(self.wf_top_n, 5, 1)

        wf_box.setLayout(wf_layout)
        scroll_layout.addWidget(wf_box)

        # ==========================================================
        # VALIDAÇÃO ESTATÍSTICA
        # ==========================================================
        stat_box = QGroupBox("📊 VALIDAÇÃO ESTATÍSTICA")
        stat_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        stat_layout = QGridLayout()
        stat_layout.setVerticalSpacing(8)
        stat_layout.setHorizontalSpacing(15)

        self.stat_enabled = QCheckBox("Ativar validação estatística (pós-otimização)")
        self.stat_enabled.setStyleSheet("font-weight: bold; color: #8e44ad;")
        stat_layout.addWidget(self.stat_enabled, 0, 0, 1, 4)

        stat_layout.addWidget(QLabel("Top-N sistemas:"), 1, 0)
        self.stat_top_n = QSpinBox()
        self.stat_top_n.setRange(5, 100)
        self.stat_top_n.setValue(10)
        self.stat_top_n.setStyleSheet("QSpinBox { padding: 3px; }")
        stat_layout.addWidget(self.stat_top_n, 1, 1)

        stat_layout.addWidget(QLabel("MC cenários:"), 1, 2)
        self.stat_n_monte_carlo = QSpinBox()
        self.stat_n_monte_carlo.setRange(100, 5000)
        self.stat_n_monte_carlo.setValue(1000)
        self.stat_n_monte_carlo.setSingleStep(100)
        self.stat_n_monte_carlo.setStyleSheet("QSpinBox { padding: 3px; }")
        stat_layout.addWidget(self.stat_n_monte_carlo, 1, 3)

        stat_layout.addWidget(QLabel("Bootstrap:"), 2, 0)
        self.stat_n_bootstrap = QSpinBox()
        self.stat_n_bootstrap.setRange(100, 5000)
        self.stat_n_bootstrap.setValue(1000)
        self.stat_n_bootstrap.setSingleStep(100)
        self.stat_n_bootstrap.setStyleSheet("QSpinBox { padding: 3px; }")
        stat_layout.addWidget(self.stat_n_bootstrap, 2, 1)

        stat_layout.addWidget(QLabel("Null shuffles:"), 2, 2)
        self.stat_n_null = QSpinBox()
        self.stat_n_null.setRange(50, 1000)
        self.stat_n_null.setValue(200)
        self.stat_n_null.setSingleStep(50)
        self.stat_n_null.setStyleSheet("QSpinBox { padding: 3px; }")
        stat_layout.addWidget(self.stat_n_null, 2, 3)

        stat_box.setLayout(stat_layout)
        scroll_layout.addWidget(stat_box)

        # ==========================================================
        # REALISMO TRANSACIONAL
        # ==========================================================
        tc_box = QGroupBox("🛠 REALISMO TRANSACIONAL")
        tc_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        tc_layout = QGridLayout()
        tc_layout.setVerticalSpacing(8)
        tc_layout.setHorizontalSpacing(15)

        self.tc_enabled = QCheckBox("Ativar custos transacionais realistas")
        self.tc_enabled.setStyleSheet("font-weight: bold; color: #c0392b;")
        tc_layout.addWidget(self.tc_enabled, 0, 0, 1, 6)

        # -- Spread Variável --
        tc_layout.addWidget(QLabel("Spread variável:"), 1, 0)
        self.tc_spread_variable = QCheckBox()
        tc_layout.addWidget(self.tc_spread_variable, 1, 1)
        tc_layout.addWidget(QLabel("Base (pts):"), 1, 2)
        self.tc_spread_base_pts = QDoubleSpinBox()
        self.tc_spread_base_pts.setRange(0, 20)
        self.tc_spread_base_pts.setSingleStep(0.1)
        self.tc_spread_base_pts.setValue(0.5)
        tc_layout.addWidget(self.tc_spread_base_pts, 1, 3)
        tc_layout.addWidget(QLabel("Mult. ATR:"), 1, 4)
        self.tc_spread_vol_mult = QDoubleSpinBox()
        self.tc_spread_vol_mult.setRange(0, 2)
        self.tc_spread_vol_mult.setSingleStep(0.01)
        self.tc_spread_vol_mult.setDecimals(3)
        self.tc_spread_vol_mult.setValue(0.05)
        tc_layout.addWidget(self.tc_spread_vol_mult, 1, 5)
        tc_layout.addWidget(QLabel("Máx (pts):"), 2, 2)
        self.tc_spread_max_pts = QDoubleSpinBox()
        self.tc_spread_max_pts.setRange(0, 100)
        self.tc_spread_max_pts.setSingleStep(0.5)
        self.tc_spread_max_pts.setValue(5.0)
        tc_layout.addWidget(self.tc_spread_max_pts, 2, 3)

        # -- Slippage Variável --
        tc_layout.addWidget(QLabel("Slippage variável:"), 3, 0)
        self.tc_slippage_variable = QCheckBox()
        tc_layout.addWidget(self.tc_slippage_variable, 3, 1)
        tc_layout.addWidget(QLabel("Base (pts):"), 3, 2)
        self.tc_slippage_base_pts = QDoubleSpinBox()
        self.tc_slippage_base_pts.setRange(0, 20)
        self.tc_slippage_base_pts.setSingleStep(0.1)
        self.tc_slippage_base_pts.setValue(0.5)
        tc_layout.addWidget(self.tc_slippage_base_pts, 3, 3)
        tc_layout.addWidget(QLabel("Mult. ATR:"), 3, 4)
        self.tc_slippage_vol_mult = QDoubleSpinBox()
        self.tc_slippage_vol_mult.setRange(0, 2)
        self.tc_slippage_vol_mult.setSingleStep(0.01)
        self.tc_slippage_vol_mult.setDecimals(3)
        self.tc_slippage_vol_mult.setValue(0.10)
        tc_layout.addWidget(self.tc_slippage_vol_mult, 3, 5)
        tc_layout.addWidget(QLabel("Máx (pts):"), 4, 2)
        self.tc_slippage_max_pts = QDoubleSpinBox()
        self.tc_slippage_max_pts.setRange(0, 100)
        self.tc_slippage_max_pts.setSingleStep(0.5)
        self.tc_slippage_max_pts.setValue(10.0)
        tc_layout.addWidget(self.tc_slippage_max_pts, 4, 3)

        # -- Corretagem e IR --
        tc_layout.addWidget(QLabel("Corretagem (pts/trade):"), 5, 0, 1, 2)
        self.tc_brokerage = QDoubleSpinBox()
        self.tc_brokerage.setRange(0, 20)
        self.tc_brokerage.setSingleStep(0.1)
        self.tc_brokerage.setDecimals(3)
        self.tc_brokerage.setValue(0.0)
        tc_layout.addWidget(self.tc_brokerage, 5, 2)
        tc_layout.addWidget(QLabel("Exchange (pts/trade):"), 5, 3)
        self.tc_exchange_fee = QDoubleSpinBox()
        self.tc_exchange_fee.setRange(0, 20)
        self.tc_exchange_fee.setSingleStep(0.1)
        self.tc_exchange_fee.setDecimals(3)
        self.tc_exchange_fee.setValue(0.0)
        tc_layout.addWidget(self.tc_exchange_fee, 5, 4)

        tc_layout.addWidget(QLabel("IR sobre lucro:"), 6, 0, 1, 2)
        self.tc_ir_tax_enabled = QCheckBox()
        tc_layout.addWidget(self.tc_ir_tax_enabled, 6, 2)
        tc_layout.addWidget(QLabel("Alíquota:"), 6, 3)
        self.tc_ir_tax_rate = QDoubleSpinBox()
        self.tc_ir_tax_rate.setRange(0, 0.50)
        self.tc_ir_tax_rate.setSingleStep(0.01)
        self.tc_ir_tax_rate.setDecimals(2)
        self.tc_ir_tax_rate.setValue(0.20)
        tc_layout.addWidget(self.tc_ir_tax_rate, 6, 4)

        # -- Liquidez --
        tc_layout.addWidget(QLabel("Restrição de liquidez:"), 7, 0, 1, 2)
        self.tc_liquidity_enabled = QCheckBox()
        tc_layout.addWidget(self.tc_liquidity_enabled, 7, 2)
        tc_layout.addWidget(QLabel("Min ATR ratio:"), 7, 3)
        self.tc_liquidity_min_atr_ratio = QDoubleSpinBox()
        self.tc_liquidity_min_atr_ratio.setRange(0.05, 1.0)
        self.tc_liquidity_min_atr_ratio.setSingleStep(0.05)
        self.tc_liquidity_min_atr_ratio.setDecimals(2)
        self.tc_liquidity_min_atr_ratio.setValue(0.3)
        tc_layout.addWidget(self.tc_liquidity_min_atr_ratio, 7, 4)

        # -- Gaps --
        tc_layout.addWidget(QLabel("Gaps entre sessões:"), 8, 0, 1, 2)
        self.tc_gap_enabled = QCheckBox()
        tc_layout.addWidget(self.tc_gap_enabled, 8, 2)
        tc_layout.addWidget(QLabel("Máx gap (min):"), 8, 3)
        self.tc_gap_max_minutes = QSpinBox()
        self.tc_gap_max_minutes.setRange(1, 240)
        self.tc_gap_max_minutes.setValue(30)
        tc_layout.addWidget(self.tc_gap_max_minutes, 8, 4)

        tc_box.setLayout(tc_layout)
        scroll_layout.addWidget(tc_box)

        # Master toggle: enable/disable TC children
        self._tc_widgets = [
            self.tc_spread_variable, self.tc_spread_base_pts,
            self.tc_spread_vol_mult, self.tc_spread_max_pts,
            self.tc_slippage_variable, self.tc_slippage_base_pts,
            self.tc_slippage_vol_mult, self.tc_slippage_max_pts,
            self.tc_brokerage, self.tc_exchange_fee,
            self.tc_ir_tax_enabled, self.tc_ir_tax_rate,
            self.tc_liquidity_enabled, self.tc_liquidity_min_atr_ratio,
            self.tc_gap_enabled, self.tc_gap_max_minutes,
        ]
        # ==========================================================
        # GESTÃO DE PORTFOLIO
        # ==========================================================
        pf_box = QGroupBox("📊 GESTÃO DE PORTFOLIO")
        pf_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        pf_layout = QGridLayout()
        pf_layout.setVerticalSpacing(8)
        pf_layout.setHorizontalSpacing(15)

        self.pf_enabled = QCheckBox("Ativar gestão de portfolio (pós-otimização)")
        self.pf_enabled.setStyleSheet("font-weight: bold; color: #16a085;")
        pf_layout.addWidget(self.pf_enabled, 0, 0, 1, 4)

        pf_layout.addWidget(QLabel("Alocação:"), 1, 0)
        self.pf_allocation_method = QComboBox()
        self.pf_allocation_method.addItems(["equal", "markowitz", "risk_parity"])
        self.pf_allocation_method.setStyleSheet("QComboBox { padding: 3px; }")
        pf_layout.addWidget(self.pf_allocation_method, 1, 1)

        pf_layout.addWidget(QLabel("Top-N sistemas:"), 1, 2)
        self.pf_top_n = QSpinBox()
        self.pf_top_n.setRange(2, 50)
        self.pf_top_n.setValue(10)
        self.pf_top_n.setStyleSheet("QSpinBox { padding: 3px; }")
        pf_layout.addWidget(self.pf_top_n, 1, 3)

        pf_layout.addWidget(QLabel("Máx correlação:"), 2, 0)
        self.pf_max_correlation = QDoubleSpinBox()
        self.pf_max_correlation.setRange(0.5, 1.0)
        self.pf_max_correlation.setSingleStep(0.05)
        self.pf_max_correlation.setDecimals(2)
        self.pf_max_correlation.setValue(0.95)
        pf_layout.addWidget(self.pf_max_correlation, 2, 1)

        # -- Rebalanceamento --
        self.pf_rebalance_enabled = QCheckBox("Rebalanceamento periódico")
        pf_layout.addWidget(self.pf_rebalance_enabled, 3, 0, 1, 2)
        pf_layout.addWidget(QLabel("Freq (barras):"), 3, 2)
        self.pf_rebalance_freq_bars = QSpinBox()
        self.pf_rebalance_freq_bars.setRange(100, 10000)
        self.pf_rebalance_freq_bars.setSingleStep(100)
        self.pf_rebalance_freq_bars.setValue(1000)
        pf_layout.addWidget(self.pf_rebalance_freq_bars, 3, 3)

        # -- Drawdown Management --
        self.pf_dd_management_enabled = QCheckBox("Gerenciamento de Drawdown")
        pf_layout.addWidget(self.pf_dd_management_enabled, 4, 0, 1, 2)
        pf_layout.addWidget(QLabel("Threshold %:"), 4, 2)
        self.pf_dd_threshold = QDoubleSpinBox()
        self.pf_dd_threshold.setRange(1, 50)
        self.pf_dd_threshold.setSingleStep(1)
        self.pf_dd_threshold.setDecimals(1)
        self.pf_dd_threshold.setValue(10.0)
        pf_layout.addWidget(self.pf_dd_threshold, 4, 3)

        pf_layout.addWidget(QLabel("Redução:"), 5, 2)
        self.pf_dd_reduction = QDoubleSpinBox()
        self.pf_dd_reduction.setRange(0.1, 1.0)
        self.pf_dd_reduction.setSingleStep(0.1)
        self.pf_dd_reduction.setDecimals(2)
        self.pf_dd_reduction.setValue(0.5)
        pf_layout.addWidget(self.pf_dd_reduction, 5, 3)

        pf_box.setLayout(pf_layout)
        scroll_layout.addWidget(pf_box)

        # Master toggle: enable/disable PF children
        self._pf_widgets = [
            self.pf_allocation_method, self.pf_top_n,
            self.pf_max_correlation, self.pf_rebalance_enabled,
            self.pf_rebalance_freq_bars, self.pf_dd_management_enabled,
            self.pf_dd_threshold, self.pf_dd_reduction,
        ]
        self.pf_enabled.toggled.connect(self._toggle_pf_widgets)

        self.tc_enabled.toggled.connect(self._toggle_tc_widgets)

        # ==========================================================
        # AÇÕES
        # ==========================================================
        actions_box = QGroupBox("🎮 AÇÕES")
        actions_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)
        
        self.start_btn = QPushButton("▶ INICIAR OTIMIZAÇÃO")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.start_btn.clicked.connect(self.start_optimization)
        
        self.pause_btn = QPushButton("⏸ PAUSAR")
        self.pause_btn.setMinimumHeight(50)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #f1c40f;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.pause_btn.clicked.connect(self.pause_optimization)
        self.pause_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹ PARAR")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        
        self.save_btn = QPushButton("💾 SALVAR CONFIG")
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.save_btn.clicked.connect(self.save_config)
        
        self.load_btn_config = QPushButton("📂 CARREGAR CONFIG")
        self.load_btn_config.setMinimumHeight(40)
        self.load_btn_config.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
        """)
        self.load_btn_config.clicked.connect(self.load_config)
        
        actions_layout.addWidget(self.start_btn)
        actions_layout.addWidget(self.pause_btn)
        actions_layout.addWidget(self.stop_btn)
        actions_layout.addWidget(self.save_btn)
        actions_layout.addWidget(self.load_btn_config)
        
        actions_box.setLayout(actions_layout)
        scroll_layout.addWidget(actions_box)
        
        # ==========================================================
        # PROGRESSO DA OTIMIZAÇÃO
        # ==========================================================
        progress_box = QGroupBox("📈 PROGRESSO DA OTIMIZAÇÃO")
        progress_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        
        progress_layout = QVBoxLayout()
        
        # Barra de progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Informações de progresso
        info_layout = QGridLayout()
        info_layout.setHorizontalSpacing(20)
        
        info_layout.addWidget(QLabel("Sistemas processados:"), 0, 0)
        self.processed_label = QLabel("0")
        self.processed_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        info_layout.addWidget(self.processed_label, 0, 1)
        
        info_layout.addWidget(QLabel("Total estimado:"), 0, 2)
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        info_layout.addWidget(self.total_label, 0, 3)
        
        info_layout.addWidget(QLabel("Velocidade:"), 1, 0)
        self.speed_label = QLabel("0 sistemas/s")
        self.speed_label.setStyleSheet("font-weight: bold; color: #e67e22;")
        info_layout.addWidget(self.speed_label, 1, 1)
        
        info_layout.addWidget(QLabel("Tempo restante:"), 1, 2)
        self.remaining_label = QLabel("--:--:--")
        self.remaining_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        info_layout.addWidget(self.remaining_label, 1, 3)
        
        progress_layout.addLayout(info_layout)
        
        # Sistema atual
        current_frame = QFrame()
        current_frame.setFrameStyle(QFrame.Box)
        current_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa; 
                padding: 5px; 
                border-radius: 3px;
                margin-top: 5px;
            }
        """)
        current_layout = QHBoxLayout(current_frame)
        
        current_layout.addWidget(QLabel("📌 Sistema atual:"))
        self.current_system = QLabel("—")
        self.current_system.setStyleSheet("font-weight: bold; color: #7f8c8d;")
        current_layout.addWidget(self.current_system)
        current_layout.addStretch()
        
        progress_layout.addWidget(current_frame)
        
        # Melhor sistema
        best_frame = QFrame()
        best_frame.setFrameStyle(QFrame.Box)
        best_frame.setStyleSheet("""
            QFrame {
                background-color: #f1c40f; 
                padding: 10px; 
                border: 2px solid #f39c12;
                border-radius: 5px;
                margin-top: 10px;
            }
        """)
        best_layout = QVBoxLayout(best_frame)
        
        best_title = QLabel("🏆 MELHOR SISTEMA ATUAL")
        best_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        best_layout.addWidget(best_title)
        
        self.best_system = QLabel("—")
        self.best_system.setStyleSheet("font-weight: bold; font-size: 16px; color: #2c3e50;")
        best_layout.addWidget(self.best_system)
        
        self.best_metrics = QLabel("Sharpe: — | PF: — | DD: —% | Ret: —%")
        self.best_metrics.setStyleSheet("color: #2c3e50;")
        best_layout.addWidget(self.best_metrics)
        
        progress_layout.addWidget(best_frame)
        
        progress_box.setLayout(progress_layout)
        scroll_layout.addWidget(progress_box)
        
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        self.setLayout(main_layout)
        
        # Conectar sinais para atualizar estimativa
        self.fast_min.valueChanged.connect(self.update_estimate)
        self.fast_max.valueChanged.connect(self.update_estimate)
        self.fast_step.valueChanged.connect(self.update_estimate)
        self.slow_min.valueChanged.connect(self.update_estimate)
        self.slow_max.valueChanged.connect(self.update_estimate)
        self.slow_step.valueChanged.connect(self.update_estimate)
    
    def load_indicators(self):
        """Carrega os indicadores na interface"""
        indicators = get_all_indicators()
        
        # Organiza por categoria
        for name, cls in indicators.items():
            category = cls.category
            
            if category in self.category_widgets:
                cat = self.category_widgets[category]
                cb = QCheckBox(f"{name}")
                cb.setToolTip(cls.description)
                cb.setChecked(True)  # Todos selecionados por padrão
                
                cat['layout'].addWidget(cb, cat['row'], cat['col'])
                
                cat['checkboxes'].append({
                    'name': name,
                    'widget': cb
                })
                
                cat['col'] += 1
                if cat['col'] >= 4:
                    cat['col'] = 0
                    cat['row'] += 1
    
    def select_category(self, category: str, select: bool):
        """Seleciona ou desmarca todos os indicadores de uma categoria"""
        if category in self.category_widgets:
            for item in self.category_widgets[category]['checkboxes']:
                item['widget'].setChecked(select)
            self.update_estimate()
    
    def get_selected_indicators(self) -> list:
        """Retorna lista de indicadores selecionados"""
        selected = []
        
        for category, cat in self.category_widgets.items():
            for item in cat['checkboxes']:
                if item['widget'].isChecked():
                    selected.append(item['name'])
        
        return selected
    
    def update_estimate(self):
        """Atualiza estimativa de combinações"""
        indicators = self.get_selected_indicators()
        fast_count = len(indicators)
        slow_count = len(indicators)
        
        fast_periods = len(range(
            self.fast_min.value(),
            self.fast_max.value() + 1,
            self.fast_step.value()
        ))
        
        slow_periods = len(range(
            self.slow_min.value(),
            self.slow_max.value() + 1,
            self.slow_step.value()
        ))
        
        tfs = sum(1 for cb in self.tf_checks if cb.isChecked())
        
        total = fast_count * slow_count * fast_periods * slow_periods * tfs
        
        # Ajusta para combinações inválidas (fast >= slow) - aproximadamente 50%
        total = int(total * 0.5)
        
        self.est_value.setText(f"{total:,}")
    
    def load_csv(self):
        """Carrega arquivo CSV"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo CSV",
            "",
            "Arquivos CSV (*.csv);;Todos os arquivos (*.*)"
        )
        
        if path:
            try:
                filename = path.split('/')[-1].split('\\')[-1]
                self.file_label.setText(f"Arquivo: {filename}")
                df = self.data_loader.load_csv(path)
                
                is_valid, message = self.data_loader.validate()
                
                if is_valid:
                    stats = self.data_loader.get_stats()
                    self.stats_label.setText(
                        f"Status: ✔ {stats['linhas']:,} linhas | "
                        f"✔ {len(stats['timeframes_disponiveis'])} timeframes | "
                        f"✔ Validado"
                    )
                    self.stats_label.setStyleSheet(
                        "padding: 5px; background-color: #d4edda; color: #155724; border-radius: 3px;"
                    )
                    self.start_btn.setEnabled(True)
                else:
                    self.stats_label.setText(f"Status: ❌ {message}")
                    self.stats_label.setStyleSheet(
                        "padding: 5px; background-color: #f8d7da; color: #721c24; border-radius: 3px;"
                    )
                    self.start_btn.setEnabled(False)
                
                self.update_estimate()
                
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao carregar arquivo:\n{str(e)}")
    
    def get_config(self) -> OptimizationConfig:
        """Cria objeto de configuração a partir da UI"""
        sl_map = {"Nenhum": 0, "Fixo (pts)": 1, "ATR múltiplo": 2, "Trailing ATR": 3}
        tp_map = {"Nenhum": 0, "Fixo (pts)": 1, "ATR múltiplo": 2}
        sizing_map = {"Fixo (1 contrato)": 0, "Fracionário Fixo": 1, "Kelly Ótimo": 2}

        return OptimizationConfig(
            fast_indicators=self.get_selected_indicators(),
            slow_indicators=self.get_selected_indicators(),
            fast_min=self.fast_min.value(),
            fast_max=self.fast_max.value(),
            fast_step=self.fast_step.value(),
            slow_min=self.slow_min.value(),
            slow_max=self.slow_max.value(),
            slow_step=self.slow_step.value(),
            timeframes=[i+1 for i, cb in enumerate(self.tf_checks) if cb.isChecked()],
            min_trades=self.min_trades.value(),
            min_sharpe=self.min_sharpe.value(),
            min_profit_factor=self.min_pf.value(),
            max_dd=self.max_dd.value(),
            slippage=self.slippage.value(),
            cost=self.cost.value(),
            batch_size=self.batch_size.value(),
            n_jobs=self.threads.value(),
            use_cache=self.use_cache.isChecked(),
            debug=self.debug.isChecked(),
            # Stop-Loss / Take-Profit
            sl_type=sl_map.get(self.sl_type.currentText(), 0),
            sl_value=self.sl_value.value(),
            tp_type=tp_map.get(self.tp_type.currentText(), 0),
            tp_value=self.tp_value.value(),
            atr_period=self.atr_period.value(),
            # Position Sizing
            sizing_method=sizing_map.get(self.sizing_method.currentText(), 0),
            risk_per_trade=self.risk_per_trade.value() / 100.0,
            kelly_fraction=self.kelly_fraction.value(),
            max_alloc_pct=self.max_alloc_pct.value() / 100.0,
            # Walk-Forward
            use_walkforward=self.wf_enabled.isChecked(),
            wf_n_windows=self.wf_n_windows.value(),
            wf_train_pct=self.wf_train_pct.value(),
            wf_test_pct=self.wf_test_pct.value(),
            wf_cv_mode=self.wf_cv_mode.currentText(),
            wf_top_n=self.wf_top_n.value(),
            # Statistical Validation
            run_statistical_validation=self.stat_enabled.isChecked(),
            stat_top_n=self.stat_top_n.value(),
            stat_n_monte_carlo=self.stat_n_monte_carlo.value(),
            stat_n_bootstrap=self.stat_n_bootstrap.value(),
            stat_n_null_hypothesis=self.stat_n_null.value(),
            # Realismo Transacional
            tc_enabled=self.tc_enabled.isChecked(),
            tc_spread_variable=self.tc_spread_variable.isChecked(),
            tc_spread_base_pts=self.tc_spread_base_pts.value(),
            tc_spread_vol_mult=self.tc_spread_vol_mult.value(),
            tc_spread_max_pts=self.tc_spread_max_pts.value(),
            tc_slippage_variable=self.tc_slippage_variable.isChecked(),
            tc_slippage_base_pts=self.tc_slippage_base_pts.value(),
            tc_slippage_vol_mult=self.tc_slippage_vol_mult.value(),
            tc_slippage_max_pts=self.tc_slippage_max_pts.value(),
            tc_brokerage_per_trade=self.tc_brokerage.value(),
            tc_exchange_fee=self.tc_exchange_fee.value(),
            tc_ir_tax_enabled=self.tc_ir_tax_enabled.isChecked(),
            tc_ir_tax_rate=self.tc_ir_tax_rate.value(),
            tc_liquidity_enabled=self.tc_liquidity_enabled.isChecked(),
            tc_liquidity_min_atr_ratio=self.tc_liquidity_min_atr_ratio.value(),
            tc_gap_enabled=self.tc_gap_enabled.isChecked(),
            tc_gap_max_minutes=self.tc_gap_max_minutes.value(),
            # Portfolio Management
            pf_enabled=self.pf_enabled.isChecked(),
            pf_allocation_method=self.pf_allocation_method.currentText(),
            pf_rebalance_enabled=self.pf_rebalance_enabled.isChecked(),
            pf_rebalance_freq_bars=self.pf_rebalance_freq_bars.value(),
            pf_dd_management_enabled=self.pf_dd_management_enabled.isChecked(),
            pf_dd_threshold=self.pf_dd_threshold.value(),
            pf_dd_reduction=self.pf_dd_reduction.value(),
            pf_top_n=self.pf_top_n.value(),
            pf_max_correlation=self.pf_max_correlation.value(),
        )
    
    def start_optimization(self):
        """Inicia otimização"""
        if self.data_loader.original_df is None:
            QMessageBox.warning(self, "Aviso", "Carregue um arquivo CSV primeiro!")
            return
        
        config = self.get_config()
        
        if len(config.fast_indicators) == 0:
            QMessageBox.warning(self, "Aviso", "Selecione pelo menos um indicador!")
            return
        
        if len(config.timeframes) == 0:
            QMessageBox.warning(self, "Aviso", "Selecione pelo menos um timeframe!")
            return

        # Limpa resultados parciais anteriores
        self.partial_results = []
        self.wf_result = None  # Store WF result for aggregated metrics

        # Cria engine(s)
        use_wf = self.wf_enabled.isChecked()

        if use_wf:
            # Walk-Forward mode
            from core.walkforward import WalkForwardEngine, WalkForwardConfig
            self.engine = None
            self.wf_engine = WalkForwardEngine(self.data_loader.timeframe_dfs)
            total_systems = self.wf_engine.estimate_total(config)
            self.total_label.setText(f"{total_systems:,} (WF)")
        else:
            # Standard mode
            self.wf_engine = None
            self.engine = OptimizationEngine(self.data_loader.timeframe_dfs)
            total_systems = self.engine.estimate_total(config)
            self.total_label.setText(f"{total_systems:,}")

        # Cria thread
        self.optimization_thread = OptimizationThread(
            self.engine, config, wf_engine=self.wf_engine
        )
        self.optimization_thread.progress.connect(self.on_progress)
        self.optimization_thread.wf_progress.connect(self.on_wf_progress)
        self.optimization_thread.result.connect(self.on_partial_result)
        self.optimization_thread.finished.connect(self.on_finished)

        # Atualiza UI
        self.start_btn.setEnabled(False)
        # Pause nao disponivel em modo WF
        self.pause_btn.setEnabled(not use_wf)
        self.stop_btn.setEnabled(True)
        self.load_btn.setEnabled(False)

        self.progress_bar.setValue(0)
        self.processed_label.setText("0")
        self.speed_label.setText("0 sistemas/s")
        self.remaining_label.setText("--:--:--")
        self.current_system.setText("—")
        self.best_system.setText("—")
        self.best_metrics.setText("Sharpe: — | PF: — | DD: —% | Ret: —%")

        # Corrigido: emite sinal ANTES de iniciar a thread (evita race condition)
        self.optimization_started.emit()

        # Depois inicia a thread
        self.optimization_thread.start()
    
    def pause_optimization(self):
        """Pausa otimização"""
        if self.engine:
            if self.engine.is_paused:
                self.engine.resume()
                self.pause_btn.setText("⏸ PAUSAR")
                self.pause_btn.setStyleSheet(self.pause_btn.styleSheet().replace("#f1c40f", "#f39c12"))
            else:
                self.engine.pause()
                self.pause_btn.setText("▶ CONTINUAR")
                self.pause_btn.setStyleSheet(self.pause_btn.styleSheet().replace("#f39c12", "#f1c40f"))
    
    def stop_optimization(self):
        """Para otimização"""
        if self.engine:
            self.engine.stop()

        if self.optimization_thread and self.optimization_thread.isRunning():
            self.optimization_thread.quit()
            self.optimization_thread.wait()

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.pause_btn.setText("⏸ PAUSAR")
    
    def on_progress(self, progress, processed, total, rate, remaining, best):
        """Atualiza UI com progresso"""
        self.progress_bar.setValue(int(progress))
        self.processed_label.setText(f"{processed:,}")
        self.speed_label.setText(f"{rate:.0f} sistemas/s")
        
        if remaining > 0:
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            seconds = int(remaining % 60)
            self.remaining_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.remaining_label.setText("--:--:--")
        
        if best:
            self.best_system.setText(f"{best.fast_indicator}({best.fast_period}) × {best.slow_indicator}({best.slow_period})")
            self.best_metrics.setText(
                f"Sharpe: {best.sharpe:.2f} | PF: {best.profit_factor:.2f} | "
                f"DD: {best.max_dd:.1f}% | Ret: {best.total_return:.1f}%"
            )
        
        self.optimization_progress.emit(progress, processed, total, rate, remaining, best)
    
    def on_wf_progress(self, pct, window_num, total_windows, rate, remaining, best, best_str):
        """Atualiza UI com progresso do Walk-Forward"""
        self.progress_bar.setValue(int(pct))
        self.processed_label.setText(f"Janela {window_num}/{total_windows}")
        self.speed_label.setText(f"{rate:.0f} janelas/s")

        if remaining > 0:
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            seconds = int(remaining % 60)
            self.remaining_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.remaining_label.setText("--:--:--")

        if best:
            self.best_system.setText(
                f"[WF Janela {window_num}] {best.fast_indicator}({best.fast_period}) × {best.slow_indicator}({best.slow_period})"
            )
            self.best_metrics.setText(
                f"Sharpe: {best.sharpe:.2f} | PF: {best.profit_factor:.2f} | "
                f"DD: {best.max_dd:.1f}% | Ret: {best.total_return:.1f}%"
            )

        self.optimization_progress.emit(pct, window_num, total_windows, rate, remaining, best)

    def on_partial_result(self, result):
        """Recebe resultados parciais"""
        if result:
            self.partial_results.append(result)
            if hasattr(result, 'to_dict'):
                self.current_results = result.to_dict()
    
    def on_finished(self, results):
        """Otimização finalizada"""
        # Atualiza UI
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self.pause_btn.setText("⏸ PAUSAR")

        is_wf = self.wf_enabled.isChecked()
        wf_result = getattr(self.optimization_thread, 'wf_result', None) if is_wf else None

        if results:
            df = pd.DataFrame([r.to_dict() for r in results])
            self.current_results = df

            if is_wf and wf_result:
                # WF mode: emit results with WF aggregated metrics
                self.results_ready.emit((df, wf_result))
            else:
                self.results_ready.emit(df)

            if is_wf and wf_result:
                metrics = wf_result.get_aggregated_metrics()
                wf_summary = (
                    f"Walk-Forward: {wf_result.n_windows} janelas, "
                    f"{metrics.get('n_total_oos', 0)} sistemas OOS\n"
                    f"Sharpe médio OOS: {metrics.get('avg_sharpe', 0):.2f} "
                    f"(σ={metrics.get('std_sharpe', 0):.2f}) | "
                    f"PF médio: {metrics.get('avg_pf', 0):.2f} | "
                    f"Estabilidade: {metrics.get('stability_score', 0):.3f}"
                )
                QMessageBox.information(
                    self,
                    "Walk-Forward Concluído",
                    f"✅ {wf_result.n_windows} janelas processadas!\n\n{wf_summary}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Otimização Concluída",
                    f"✅ {len(results)} sistemas aprovados nos filtros!\n\n"
                    f"Total processado: {self.processed_label.text()} sistemas\n"
                    f"Taxa de aprovação: {len(results)/max(int(self.processed_label.text().replace(',','').split()[0]),1)*100:.1f}%"
                )
        else:
            QMessageBox.warning(
                self,
                "Otimização Concluída",
                "Nenhum sistema passou nos filtros de qualidade.\n"
                "Tente ajustar os parâmetros ou filtros."
            )

        self.optimization_finished.emit(results if results else [])
    
    def save_config(self):
        """Salva configuração atual"""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar Configuração",
            "",
            "Arquivos JSON (*.json)"
        )
        
        if path:
            config = {
                'fast_min': self.fast_min.value(),
                'fast_max': self.fast_max.value(),
                'fast_step': self.fast_step.value(),
                'slow_min': self.slow_min.value(),
                'slow_max': self.slow_max.value(),
                'slow_step': self.slow_step.value(),
                'timeframes': [i+1 for i, cb in enumerate(self.tf_checks) if cb.isChecked()],
                'min_trades': self.min_trades.value(),
                'min_sharpe': self.min_sharpe.value(),
                'min_pf': self.min_pf.value(),
                'max_dd': self.max_dd.value(),
                'slippage': self.slippage.value(),
                'cost': self.cost.value(),
                'threads': self.threads.value(),
                'batch_size': self.batch_size.value(),
                'use_cache': self.use_cache.isChecked(),
                # Stop-Loss / TP
                'sl_type': self.sl_type.currentText(),
                'sl_value': self.sl_value.value(),
                'tp_type': self.tp_type.currentText(),
                'tp_value': self.tp_value.value(),
                'atr_period': self.atr_period.value(),
                # Position Sizing
                'sizing_method': self.sizing_method.currentText(),
                'risk_per_trade': self.risk_per_trade.value(),
                'kelly_fraction': self.kelly_fraction.value(),
                'max_alloc_pct': self.max_alloc_pct.value(),
                # Walk-Forward
                'wf_enabled': self.wf_enabled.isChecked(),
                'wf_n_windows': self.wf_n_windows.value(),
                'wf_train_pct': self.wf_train_pct.value(),
                'wf_test_pct': self.wf_test_pct.value(),
                'wf_cv_mode': self.wf_cv_mode.currentText(),
                'wf_top_n': self.wf_top_n.value(),
                # Statistical Validation
                'stat_enabled': self.stat_enabled.isChecked(),
                'stat_top_n': self.stat_top_n.value(),
                'stat_n_monte_carlo': self.stat_n_monte_carlo.value(),
                'stat_n_bootstrap': self.stat_n_bootstrap.value(),
                'stat_n_null': self.stat_n_null.value(),
                # Realismo Transacional
                'tc_enabled': self.tc_enabled.isChecked(),
                'tc_spread_variable': self.tc_spread_variable.isChecked(),
                'tc_spread_base_pts': self.tc_spread_base_pts.value(),
                'tc_spread_vol_mult': self.tc_spread_vol_mult.value(),
                'tc_spread_max_pts': self.tc_spread_max_pts.value(),
                'tc_slippage_variable': self.tc_slippage_variable.isChecked(),
                'tc_slippage_base_pts': self.tc_slippage_base_pts.value(),
                'tc_slippage_vol_mult': self.tc_slippage_vol_mult.value(),
                'tc_slippage_max_pts': self.tc_slippage_max_pts.value(),
                'tc_brokerage': self.tc_brokerage.value(),
                'tc_exchange_fee': self.tc_exchange_fee.value(),
                'tc_ir_tax_enabled': self.tc_ir_tax_enabled.isChecked(),
                'tc_ir_tax_rate': self.tc_ir_tax_rate.value(),
                'tc_liquidity_enabled': self.tc_liquidity_enabled.isChecked(),
                'tc_liquidity_min_atr_ratio': self.tc_liquidity_min_atr_ratio.value(),
                'tc_gap_enabled': self.tc_gap_enabled.isChecked(),
                'tc_gap_max_minutes': self.tc_gap_max_minutes.value(),
                # Portfolio Management
                'pf_enabled': self.pf_enabled.isChecked(),
                'pf_allocation_method': self.pf_allocation_method.currentText(),
                'pf_rebalance_enabled': self.pf_rebalance_enabled.isChecked(),
                'pf_rebalance_freq_bars': self.pf_rebalance_freq_bars.value(),
                'pf_dd_management_enabled': self.pf_dd_management_enabled.isChecked(),
                'pf_dd_threshold': self.pf_dd_threshold.value(),
                'pf_dd_reduction': self.pf_dd_reduction.value(),
                'pf_top_n': self.pf_top_n.value(),
                'pf_max_correlation': self.pf_max_correlation.value(),
            }
            
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Sucesso", "Configuração salva com sucesso!")
    
    def load_config(self):
        """Carrega configuração salva"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Carregar Configuração",
            "",
            "Arquivos JSON (*.json)"
        )
        
        if path:
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                self.fast_min.setValue(config.get('fast_min', 5))
                self.fast_max.setValue(config.get('fast_max', 50))
                self.fast_step.setValue(config.get('fast_step', 1))
                self.slow_min.setValue(config.get('slow_min', 20))
                self.slow_max.setValue(config.get('slow_max', 200))
                self.slow_step.setValue(config.get('slow_step', 5))
                
                # Timeframes
                tfs = config.get('timeframes', [1,2,3,4,5])
                for i, cb in enumerate(self.tf_checks):
                    cb.setChecked((i+1) in tfs)
                
                self.min_trades.setValue(config.get('min_trades', 100))
                self.min_sharpe.setValue(config.get('min_sharpe', 1.2))
                self.min_pf.setValue(config.get('min_pf', 1.5))
                self.max_dd.setValue(config.get('max_dd', 8.0))
                self.slippage.setValue(config.get('slippage', 1.0))
                self.cost.setValue(config.get('cost', 0.5))
                self.threads.setValue(config.get('threads', 8))
                self.batch_size.setValue(config.get('batch_size', 5000))
                self.use_cache.setChecked(config.get('use_cache', True))

                # Stop-Loss / TP
                sl_type = config.get('sl_type', 'Nenhum')
                idx = self.sl_type.findText(sl_type)
                if idx >= 0: self.sl_type.setCurrentIndex(idx)
                self.sl_value.setValue(config.get('sl_value', 50.0))
                tp_type = config.get('tp_type', 'Nenhum')
                idx = self.tp_type.findText(tp_type)
                if idx >= 0: self.tp_type.setCurrentIndex(idx)
                self.tp_value.setValue(config.get('tp_value', 150.0))
                self.atr_period.setValue(config.get('atr_period', 14))

                # Position Sizing
                sizing_method = config.get('sizing_method', 'Fixo (1 contrato)')
                idx = self.sizing_method.findText(sizing_method)
                if idx >= 0: self.sizing_method.setCurrentIndex(idx)
                self.risk_per_trade.setValue(config.get('risk_per_trade', 1.0))
                self.kelly_fraction.setValue(config.get('kelly_fraction', 0.25))
                self.max_alloc_pct.setValue(config.get('max_alloc_pct', 20))

                # Walk-Forward
                self.wf_enabled.setChecked(config.get('wf_enabled', False))
                self.wf_n_windows.setValue(config.get('wf_n_windows', 3))
                self.wf_train_pct.setValue(config.get('wf_train_pct', 0.6))
                self.wf_test_pct.setValue(config.get('wf_test_pct', 0.2))
                cv_mode = config.get('wf_cv_mode', 'rolling')
                idx = self.wf_cv_mode.findText(cv_mode)
                if idx >= 0: self.wf_cv_mode.setCurrentIndex(idx)
                self.wf_top_n.setValue(config.get('wf_top_n', 5))

                # Statistical Validation
                self.stat_enabled.setChecked(config.get('stat_enabled', False))
                self.stat_top_n.setValue(config.get('stat_top_n', 10))
                self.stat_n_monte_carlo.setValue(config.get('stat_n_monte_carlo', 1000))
                self.stat_n_bootstrap.setValue(config.get('stat_n_bootstrap', 1000))
                self.stat_n_null.setValue(config.get('stat_n_null', 200))

                # Realismo Transacional
                self.tc_enabled.setChecked(config.get('tc_enabled', False))
                self.tc_spread_variable.setChecked(config.get('tc_spread_variable', False))
                self.tc_spread_base_pts.setValue(config.get('tc_spread_base_pts', 0.5))
                self.tc_spread_vol_mult.setValue(config.get('tc_spread_vol_mult', 0.05))
                self.tc_spread_max_pts.setValue(config.get('tc_spread_max_pts', 5.0))
                self.tc_slippage_variable.setChecked(config.get('tc_slippage_variable', False))
                self.tc_slippage_base_pts.setValue(config.get('tc_slippage_base_pts', 0.5))
                self.tc_slippage_vol_mult.setValue(config.get('tc_slippage_vol_mult', 0.10))
                self.tc_slippage_max_pts.setValue(config.get('tc_slippage_max_pts', 10.0))
                self.tc_brokerage.setValue(config.get('tc_brokerage', 0.0))
                self.tc_exchange_fee.setValue(config.get('tc_exchange_fee', 0.0))
                self.tc_ir_tax_enabled.setChecked(config.get('tc_ir_tax_enabled', False))
                self.tc_ir_tax_rate.setValue(config.get('tc_ir_tax_rate', 0.20))
                self.tc_liquidity_enabled.setChecked(config.get('tc_liquidity_enabled', False))
                self.tc_liquidity_min_atr_ratio.setValue(config.get('tc_liquidity_min_atr_ratio', 0.3))
                self.tc_gap_enabled.setChecked(config.get('tc_gap_enabled', False))
                self.tc_gap_max_minutes.setValue(config.get('tc_gap_max_minutes', 30))

                # Portfolio Management
                self.pf_enabled.setChecked(config.get('pf_enabled', False))
                method = config.get('pf_allocation_method', 'equal')
                idx = self.pf_allocation_method.findText(method)
                if idx >= 0: self.pf_allocation_method.setCurrentIndex(idx)
                self.pf_rebalance_enabled.setChecked(config.get('pf_rebalance_enabled', False))
                self.pf_rebalance_freq_bars.setValue(config.get('pf_rebalance_freq_bars', 1000))
                self.pf_dd_management_enabled.setChecked(config.get('pf_dd_management_enabled', False))
                self.pf_dd_threshold.setValue(config.get('pf_dd_threshold', 10.0))
                self.pf_dd_reduction.setValue(config.get('pf_dd_reduction', 0.5))
                self.pf_top_n.setValue(config.get('pf_top_n', 10))
                self.pf_max_correlation.setValue(config.get('pf_max_correlation', 0.95))

                QMessageBox.information(self, "Sucesso", "Configuração carregada com sucesso!")

            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao carregar configuração:\n{str(e)}")

    def _toggle_tc_widgets(self, enabled: bool):
        """Enable/disable TC child widgets based on master switch"""
        for widget in self._tc_widgets:
            widget.setEnabled(enabled)

    def _toggle_pf_widgets(self, enabled: bool):
        """Enable/disable PF child widgets based on master switch"""
        for widget in self._pf_widgets:
            widget.setEnabled(enabled)