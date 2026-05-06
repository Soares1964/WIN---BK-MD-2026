# ma_lab/ui/tab_ranking.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QComboBox, QSpinBox, QCheckBox,
    QFrame, QGridLayout, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

import pandas as pd
import numpy as np


class RankingTab(QWidget):
    """Aba de ranking e análise de sistemas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data = None
        self.filtered_data = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface"""
        main_layout = QVBoxLayout()
        
        # ==========================================================
        # TÍTULO
        # ==========================================================
        title = QLabel("📊 QUADRO 2 – RANKING & ANÁLISE DE SISTEMAS")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            padding: 8px;
            background-color: #34495e;
            color: white;
            border-radius: 5px;
        """)
        main_layout.addWidget(title)
        
        # ==========================================================
        # FILTROS DE VISUALIZAÇÃO
        # ==========================================================
        filters_box = QGroupBox("🔍 FILTROS DE VISUALIZAÇÃO")
        filters_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        filters_layout = QGridLayout()
        
        # Métrica
        filters_layout.addWidget(QLabel("Métrica:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Sharpe", "Profit Factor", "Win Rate", "Score Global"])
        self.metric_combo.currentTextChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.metric_combo, 0, 1)
        
        # Período
        filters_layout.addWidget(QLabel("Período:"), 0, 2)
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Todos", "Último mês", "Últimos 3 meses", "Último ano"])
        filters_layout.addWidget(self.period_combo, 0, 3)
        
        # Timeframe
        filters_layout.addWidget(QLabel("Timeframe:"), 1, 0)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["Todos", "1 min", "2 min", "3 min", "4 min", "5 min",
                                "6 min", "7 min", "8 min", "9 min", "10 min"])
        self.tf_combo.currentTextChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.tf_combo, 1, 1)
        
        # Top N
        filters_layout.addWidget(QLabel("Top N:"), 1, 2)
        self.top_n = QSpinBox()
        self.top_n.setRange(10, 1000)
        self.top_n.setValue(50)
        self.top_n.setSingleStep(10)
        self.top_n.valueChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.top_n, 1, 3)
        
        # Checkboxes de filtro
        self.filter_sharpe = QCheckBox("Sharpe > 1.2")
        self.filter_sharpe.setChecked(True)
        self.filter_sharpe.stateChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.filter_sharpe, 2, 0)
        
        self.filter_pf = QCheckBox("PF > 1.5")
        self.filter_pf.setChecked(True)
        self.filter_pf.stateChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.filter_pf, 2, 1)
        
        self.filter_dd = QCheckBox("DD < 10%")
        self.filter_dd.setChecked(True)
        self.filter_dd.stateChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.filter_dd, 2, 2)
        
        self.filter_trades = QCheckBox("Trades > 100")
        self.filter_trades.setChecked(True)
        self.filter_trades.stateChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.filter_trades, 2, 3)
        
        # Botão de refresh
        refresh_btn = QPushButton("🔄 Atualizar")
        refresh_btn.clicked.connect(self.apply_filters)
        filters_layout.addWidget(refresh_btn, 3, 0, 1, 4)
        
        filters_box.setLayout(filters_layout)
        main_layout.addWidget(filters_box)
        
        # Checkbox para filtrar só OOS (Walk-Forward)
        self.oos_only = QCheckBox("Mostrar apenas resultados OOS (Walk-Forward)")
        self.oos_only.stateChanged.connect(self.apply_filters)
        filters_layout.addWidget(self.oos_only, 2, 4)

        # ==========================================================
        # RANKING PRINCIPAL
        # ==========================================================
        ranking_box = QGroupBox("🏆 RANKING PRINCIPAL")
        ranking_box.setStyleSheet("QGroupBox { font-weight: bold; }")

        ranking_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "#", "FAST", "SLOW", "TF", "Sharpe", "PF", "Win%", "DD%", "Trades", "Ret%",
            "Janela", "OOS", "Tr.Score"
        ])
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        
        ranking_layout.addWidget(self.table)
        ranking_box.setLayout(ranking_layout)
        main_layout.addWidget(ranking_box)
        
        # ==========================================================
        # DISTRIBUIÇÃO DE PERFORMANCE
        # ==========================================================
        dist_box = QGroupBox("📊 DISTRIBUIÇÃO DE PERFORMANCE")
        dist_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        dist_layout = QVBoxLayout()
        
        # Placeholder para gráfico
        self.dist_label = QLabel("Gráfico de distribuição será exibido aqui")
        self.dist_label.setAlignment(Qt.AlignCenter)
        self.dist_label.setMinimumHeight(100)
        self.dist_label.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7;")
        
        dist_layout.addWidget(self.dist_label)
        dist_box.setLayout(dist_layout)
        main_layout.addWidget(dist_box)
        
        # ==========================================================
        # ANÁLISE POR FILTRO (AGREGADO)
        # ==========================================================
        agg_box = QGroupBox("📈 ANÁLISE POR FILTRO (AGREGADO)")
        agg_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        agg_layout = QVBoxLayout()
        
        self.agg_table = QTableWidget()
        self.agg_table.setColumnCount(6)
        self.agg_table.setHorizontalHeaderLabels([
            "Filtro", "Avg Sharpe", "Win Rate", "Ocorrências", "Score", "★"
        ])
        
        self.agg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.agg_table.setAlternatingRowColors(True)
        
        agg_layout.addWidget(self.agg_table)
        agg_box.setLayout(agg_layout)
        main_layout.addWidget(agg_box)
        
        # ==========================================================
        # INSIGHTS AUTOMÁTICOS
        # ==========================================================
        insights_box = QGroupBox("💡 INSIGHTS AUTOMÁTICOS")
        insights_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        insights_layout = QVBoxLayout()
        
        self.insights_label = QLabel(
            "✔ Aguardando dados para gerar insights...\n"
            "✔ Execute uma otimização para ver análises automáticas"
        )
        self.insights_label.setStyleSheet("""
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 10px;
            border-radius: 5px;
        """)
        
        insights_layout.addWidget(self.insights_label)
        insights_box.setLayout(insights_layout)
        main_layout.addWidget(insights_box)
        
        self.setLayout(main_layout)
    
    def update_data(self, df):
        """Atualiza dados da tabela"""
        self.data = df
        self.apply_filters()
    
    def apply_filters(self):
        """Aplica filtros aos dados"""
        if self.data is None or self.data.empty:
            return
        
        filtered = self.data.copy()
        
        # Filtro por timeframe
        tf_text = self.tf_combo.currentText()
        if tf_text != "Todos":
            tf_value = int(tf_text.split()[0])
            filtered = filtered[filtered['tf'] == tf_value]
        
        # Filtro OOS-only (Walk-Forward)
        if self.oos_only.isChecked() and 'is_oos' in filtered.columns:
            filtered = filtered[filtered['is_oos'] == True]

        # Filtros de qualidade
        if self.filter_sharpe.isChecked():
            filtered = filtered[filtered['sharpe'] >= 1.2]

        if self.filter_pf.isChecked():
            filtered = filtered[filtered['pf'] >= 1.5]

        if self.filter_dd.isChecked():
            filtered = filtered[filtered['dd'] <= 10.0]

        if self.filter_trades.isChecked():
            filtered = filtered[filtered['trades'] >= 100]
        
        # Calcula score
        filtered['score'] = (
            filtered['sharpe'] * 3 +
            filtered['pf'] * 2.5 +
            filtered['win'] * 0.25 -
            filtered['dd'] * 0.3
        )
        
        # Ordena por métrica selecionada
        metric_map = {
            "Sharpe": "sharpe",
            "Profit Factor": "pf",
            "Win Rate": "win",
            "Score Global": "score"
        }
        
        sort_col = metric_map.get(self.metric_combo.currentText(), "score")
        filtered = filtered.sort_values(sort_col, ascending=False)
        
        # Limita ao top N
        filtered = filtered.head(self.top_n.value())
        
        self.filtered_data = filtered
        self.populate_table(filtered)
        self.populate_agg_table(filtered)
        self.update_insights(filtered)
        self.update_distribution(filtered)
    
    def populate_table(self, df):
        """Preenche tabela principal"""
        # Detect if statistical validation data is present
        has_stats = 'dsr' in df.columns and not df['dsr'].isna().all()
        base_cols = 13
        stat_cols = 5 if has_stats else 0
        total_cols = base_cols + stat_cols

        self.table.setColumnCount(total_cols)
        if has_stats:
            self.table.setHorizontalHeaderLabels([
                "#", "FAST", "SLOW", "TF", "Sharpe", "PF", "Win%", "DD%", "Trades", "Ret%",
                "Janela", "OOS", "Tr.Score",
                "DSR", "P(Boot)", "P(Null)", "MC Sharpe", "Robust"
            ])
        else:
            self.table.setHorizontalHeaderLabels([
                "#", "FAST", "SLOW", "TF", "Sharpe", "PF", "Win%", "DD%", "Trades", "Ret%",
                "Janela", "OOS", "Tr.Score"
            ])

        self.table.setRowCount(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            # Rank
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, rank_item)
            
            # FAST
            fast_item = QTableWidgetItem(f"{row['fast']}({row['fp']})")
            fast_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, fast_item)
            
            # SLOW
            slow_item = QTableWidgetItem(f"{row['slow']}({row['sp']})")
            slow_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, slow_item)
            
            # TF
            tf_item = QTableWidgetItem(f"{row['tf']} min")
            tf_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 3, tf_item)
            
            # Sharpe
            sharpe_item = QTableWidgetItem(f"{row['sharpe']:.2f}")
            sharpe_item.setTextAlignment(Qt.AlignCenter)
            self.colorize_item(sharpe_item, row['sharpe'], 1.0, 2.0)
            self.table.setItem(i, 4, sharpe_item)
            
            # PF
            pf_item = QTableWidgetItem(f"{row['pf']:.2f}")
            pf_item.setTextAlignment(Qt.AlignCenter)
            self.colorize_item(pf_item, row['pf'], 1.0, 2.5)
            self.table.setItem(i, 5, pf_item)
            
            # Win%
            win_item = QTableWidgetItem(f"{row['win']:.1f}%")
            win_item.setTextAlignment(Qt.AlignCenter)
            self.colorize_item(win_item, row['win'], 40, 70)
            self.table.setItem(i, 6, win_item)
            
            # DD%
            dd_item = QTableWidgetItem(f"{row['dd']:.1f}%")
            dd_item.setTextAlignment(Qt.AlignCenter)
            self.colorize_item(dd_item, row['dd'], 10, 3, reverse=True)
            self.table.setItem(i, 7, dd_item)
            
            # Trades
            trades_item = QTableWidgetItem(f"{int(row['trades']):,}")
            trades_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 8, trades_item)
            
            # Ret%
            ret_item = QTableWidgetItem(f"{row['ret']:.1f}%")
            ret_item.setTextAlignment(Qt.AlignCenter)
            self.colorize_item(ret_item, row['ret'], 50, 200)
            self.table.setItem(i, 9, ret_item)

            # Janela (10)
            window_val = row.get('window_id', '—')
            win_item = QTableWidgetItem(str(window_val) if window_val >= 0 else '—')
            win_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 10, win_item)

            # OOS (11)
            oos_val = row.get('is_oos', False)
            oos_item = QTableWidgetItem('✓ OOS' if oos_val else 'IS')
            oos_item.setTextAlignment(Qt.AlignCenter)
            if oos_val:
                oos_item.setBackground(QColor(200, 255, 200))
            self.table.setItem(i, 11, oos_item)

            # Train Score (12)
            ts_val = row.get('train_score', 0)
            ts_item = QTableWidgetItem(f"{ts_val:.3f}" if 'train_score' in row.index else '—')
            ts_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 12, ts_item)

            # Statistical validation columns (13-17), only if present
            if has_stats:
                # DSR (13)
                dsr_val = row.get('dsr', 0)
                dsr_item = QTableWidgetItem(f"{dsr_val:.3f}")
                dsr_item.setTextAlignment(Qt.AlignCenter)
                self.colorize_item(dsr_item, dsr_val, 0.5, 0.95)
                self.table.setItem(i, 13, dsr_item)

                # Bootstrap P-value (14)
                boot_p = row.get('boot_sharpe_p_value', 1.0)
                boot_item = QTableWidgetItem(f"{boot_p:.4f}")
                boot_item.setTextAlignment(Qt.AlignCenter)
                self.colorize_item(boot_item, 1.0 - boot_p, 0.5, 0.95)
                self.table.setItem(i, 14, boot_item)

                # Null P-value (15)
                null_p = row.get('null_p_value', 1.0)
                null_item = QTableWidgetItem(f"{null_p:.4f}")
                null_item.setTextAlignment(Qt.AlignCenter)
                self.colorize_item(null_item, 1.0 - null_p, 0.5, 0.95)
                self.table.setItem(i, 15, null_item)

                # MC Sharpe CI range (16)
                mc_low = row.get('mc_sharpe_ci_lower', 0)
                mc_high = row.get('mc_sharpe_ci_upper', 0)
                mc_item = QTableWidgetItem(f"[{mc_low:.2f}, {mc_high:.2f}]")
                mc_item.setTextAlignment(Qt.AlignCenter)
                self.colorize_item(mc_item, (mc_low + mc_high) / 2, 0.5, 2.0)
                self.table.setItem(i, 16, mc_item)

                # Robustness (17)
                robust = row.get('sens_robustness', 0)
                rob_item = QTableWidgetItem(f"{robust:.3f}")
                rob_item.setTextAlignment(Qt.AlignCenter)
                self.colorize_item(rob_item, robust, 0.3, 0.8)
                self.table.setItem(i, 17, rob_item)
    
    def populate_agg_table(self, df):
        """Preenche tabela agregada por filtro"""
        if df.empty:
            self.agg_table.setRowCount(0)
            return
        
        # Agrega por FAST
        agg = df.groupby('fast').agg({
            'sharpe': 'mean',
            'win': 'mean',
            'fast': 'count',
            'score': 'mean'
        }).rename(columns={'fast': 'count'})
        
        # Ordena por score
        agg = agg.sort_values('score', ascending=False)
        
        self.agg_table.setRowCount(len(agg))
        
        for i, (filtro, row) in enumerate(agg.iterrows()):
            # Filtro
            filtro_item = QTableWidgetItem(filtro)
            filtro_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 0, filtro_item)
            
            # Avg Sharpe
            sharpe_item = QTableWidgetItem(f"{row['sharpe']:.2f}")
            sharpe_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 1, sharpe_item)
            
            # Win Rate
            win_item = QTableWidgetItem(f"{row['win']:.1f}%")
            win_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 2, win_item)
            
            # Ocorrências
            count_item = QTableWidgetItem(f"{int(row['count']):,}")
            count_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 3, count_item)
            
            # Score
            score_item = QTableWidgetItem(f"{row['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 4, score_item)
            
            # Estrelas
            stars = "★" * min(5, max(1, int(row['score'] / 2)))
            star_item = QTableWidgetItem(stars)
            star_item.setTextAlignment(Qt.AlignCenter)
            self.agg_table.setItem(i, 5, star_item)
    
    def update_insights(self, df):
        """Atualiza insights automáticos"""
        if df.empty:
            self.insights_label.setText(
                "✔ Aguardando dados para gerar insights...\n"
                "✔ Execute uma otimização para ver análises automáticas"
            )
            return
        
        insights = []
        
        # Top filtros
        top_fast = df.groupby('fast')['score'].mean().nlargest(3)
        if not top_fast.empty:
            insights.append(f"✔ Top FAST: {', '.join([f'{f}({s:.2f})' for f, s in top_fast.items()])}")
        
        # Melhor timeframe
        best_tf = df.groupby('tf')['score'].mean()
        if not best_tf.empty:
            best_tf_val = best_tf.idxmax()
            insights.append(f"✔ Melhor timeframe: {best_tf_val} min")
        
        # Combinações especiais
        adaptativas = ['KAMA', 'VIDYA', 'FRAMA', 'MAMA']
        lowlag = ['HMA', 'ZLEMA', 'DEMA', 'TEMA']
        
        adapt_count = df[df['fast'].isin(adaptativas) | df['slow'].isin(adaptativas)].shape[0]
        if adapt_count > 0:
            insights.append(f"✔ Filtros adaptativos aparecem em {adapt_count} sistemas")
        
        # Média de Sharpe
        insights.append(f"✔ Sharpe médio: {df['sharpe'].mean():.2f}")
        
        # Melhor combinação
        best = df.iloc[0]
        insights.append(
            f"✔ Melhor: {best['fast']}({best['fp']}) × {best['slow']}({best['sp']}) "
            f"→ Sharpe {best['sharpe']:.2f}"
        )
        
        self.insights_label.setText("\n".join(insights))
    
    def update_distribution(self, df):
        """Atualiza distribuição de performance"""
        if df.empty:
            return
        
        # Calcula distribuição de Sharpe
        sharpe_vals = df['sharpe'].values
        
        # Cria representação ASCII
        bins = np.linspace(0.8, 2.2, 8)
        hist, _ = np.histogram(sharpe_vals, bins=bins)
        
        if hist.max() > 0:
            hist = hist / hist.max() * 20  # Normaliza para 20 caracteres
        
        dist_text = "Distribuição de Sharpe:\n\n"
        
        for i, count in enumerate(hist):
            bin_start = bins[i]
            bin_end = bins[i+1]
            bar = "█" * int(count)
            dist_text += f"{bin_start:.1f}-{bin_end:.1f}: {bar}\n"
        
        self.dist_label.setText(dist_text)
    
    def colorize_item(self, item, value, min_val, max_val, reverse=False):
        """Aplica cor ao item baseado no valor"""
        if max_val == min_val:
            norm = 0.5
        else:
            if reverse:
                norm = 1 - (value - min_val) / (max_val - min_val)
            else:
                norm = (value - min_val) / (max_val - min_val)
        
        norm = max(0, min(1, norm))
        
        if norm < 0.33:
            color = QColor(255, 200, 200)  # Vermelho claro
        elif norm < 0.66:
            color = QColor(255, 255, 200)  # Amarelo claro
        else:
            color = QColor(200, 255, 200)  # Verde claro
        
        item.setBackground(color)