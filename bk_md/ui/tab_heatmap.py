# ma_lab/ui/tab_heatmap.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QComboBox, QCheckBox, QGridLayout,
    QFrame, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont

import pandas as pd
import numpy as np


class HeatmapTab(QWidget):
    """Aba de heatmap de cruzamentos"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data = None
        self.indicators = [
            "SMA", "EMA", "HMA", "KAMA", "FRAMA",
            "MAMA", "Kalman", "GaussianMA", "SuperSmoother"
        ]
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ==========================================================
        # TÍTULO
        # ==========================================================
        title = QLabel("🔥 QUADRO 3 – HEATMAP DE CRUZAMENTOS DE MÉDIAS MÓVEIS")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            padding: 8px;
            background-color: #e67e22;
            color: white;
            border-radius: 5px;
        """)
        main_layout.addWidget(title)
        
        # ==========================================================
        # LEGENDA
        # ==========================================================
        legend_box = QGroupBox("📌 LEGENDA")
        legend_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        legend_layout = QHBoxLayout()
        
        legend_colors = [
            ("<1.0 = ruim", "#ffcccc"),
            ("1.0-1.3 = neutro", "#ffffcc"),
            ("1.3-1.6 = bom", "#ccffcc"),
            ("1.6-2.0 = forte", "#99cc99"),
            (">2.0 = excelente", "#66cc66")
        ]
        
        for text, color in legend_colors:
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setStyleSheet(f"background-color: {color}; padding: 5px; border-radius: 3px;")
            
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            
            layout = QHBoxLayout(frame)
            layout.setContentsMargins(5, 2, 5, 2)
            layout.addWidget(label)
            
            legend_layout.addWidget(frame)
        
        legend_box.setLayout(legend_layout)
        main_layout.addWidget(legend_box)
        
        # ==========================================================
        # CONTROLES
        # ==========================================================
        controls_box = QGroupBox("⚙️ CONTROLES AVANÇADOS")
        controls_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        controls_layout = QGridLayout()
        controls_layout.setHorizontalSpacing(15)
        controls_layout.setVerticalSpacing(10)
        
        # Métrica
        controls_layout.addWidget(QLabel("Métrica:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Sharpe", "Profit Factor", "Win Rate", "Retorno"])
        self.metric_combo.currentTextChanged.connect(self.update_heatmap)
        self.metric_combo.setMinimumWidth(120)
        controls_layout.addWidget(self.metric_combo, 0, 1)
        
        # Normalização
        controls_layout.addWidget(QLabel("Normalização:"), 0, 2)
        self.norm_check = QCheckBox("Z-score")
        self.norm_check.setChecked(True)
        self.norm_check.stateChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.norm_check, 0, 3)
        
        # Escala
        controls_layout.addWidget(QLabel("Escala:"), 1, 0)
        self.log_check = QCheckBox("Log")
        self.log_check.stateChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.log_check, 1, 1)
        
        # Remover outliers
        controls_layout.addWidget(QLabel("Outliers:"), 1, 2)
        self.outlier_check = QCheckBox("Remover")
        self.outlier_check.setChecked(True)
        self.outlier_check.stateChanged.connect(self.update_heatmap)
        controls_layout.addWidget(self.outlier_check, 1, 3)
        
        # Timeframe
        controls_layout.addWidget(QLabel("Timeframe:"), 2, 0)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["Todos", "1 min", "2 min", "3 min", "4 min", "5 min",
                                "6 min", "7 min", "8 min", "9 min", "10 min"])
        self.tf_combo.currentTextChanged.connect(self.update_heatmap)
        self.tf_combo.setMinimumWidth(120)
        controls_layout.addWidget(self.tf_combo, 2, 1)
        
        # Botão de atualizar
        refresh_btn = QPushButton("🔄 Atualizar Heatmap")
        refresh_btn.setMinimumHeight(35)
        refresh_btn.setStyleSheet("""
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
        refresh_btn.clicked.connect(self.update_heatmap)
        controls_layout.addWidget(refresh_btn, 2, 2, 1, 2)
        
        controls_box.setLayout(controls_layout)
        main_layout.addWidget(controls_box)
        
        # ==========================================================
        # HEATMAP TABLE
        # ==========================================================
        heatmap_box = QGroupBox("📊 MATRIZ DE CRUZAMENTOS")
        heatmap_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        heatmap_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        n = len(self.indicators)
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        
        self.table.setHorizontalHeaderLabels(self.indicators)
        self.table.setVerticalHeaderLabels(self.indicators)
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Configurar altura mínima
        self.table.setMinimumHeight(400)
        
        heatmap_layout.addWidget(self.table)
        heatmap_box.setLayout(heatmap_layout)
        main_layout.addWidget(heatmap_box)
        
        # ==========================================================
        # TOP ZONAS
        # ==========================================================
        top_box = QGroupBox("🔥 TOP ZONAS (AUTOMÁTICO)")
        top_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        top_layout = QVBoxLayout()
        
        self.top_label = QLabel(
            "🔥 Aguardando dados...\n"
            "Execute uma otimização para ver os hotspots"
        )
        self.top_label.setStyleSheet("""
            background-color: #fff3cd;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
        """)
        self.top_label.setMinimumHeight(80)
        
        top_layout.addWidget(self.top_label)
        top_box.setLayout(top_layout)
        main_layout.addWidget(top_box)
        
        # ==========================================================
        # PADRÕES IDENTIFICADOS
        # ==========================================================
        patterns_box = QGroupBox("📈 PADRÕES IDENTIFICADOS")
        patterns_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        patterns_layout = QVBoxLayout()
        
        self.patterns_label = QLabel(
            "✔ Aguardando análise...\n"
            "✔ Os padrões aparecerão após a otimização"
        )
        self.patterns_label.setStyleSheet("""
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
        """)
        self.patterns_label.setMinimumHeight(80)
        
        patterns_layout.addWidget(self.patterns_label)
        patterns_box.setLayout(patterns_layout)
        main_layout.addWidget(patterns_box)
        
        self.setLayout(main_layout)
        
        # Carrega exemplo inicial
        self.load_example()
    
    def update_data(self, df):
        """Atualiza dados do heatmap"""
        self.data = df
        self.update_heatmap()
    
    def update_heatmap(self):
        """Atualiza o heatmap com os dados atuais"""
        if self.data is None or self.data.empty:
            return
        
        try:
            # Filtra por timeframe se necessário
            df = self.data.copy()
            tf_text = self.tf_combo.currentText()
            if tf_text != "Todos":
                tf_value = int(tf_text.split()[0])
                df = df[df['tf'] == tf_value]
            
            if df.empty:
                return
            
            # Mapeia métrica
            metric_map = {
                "Sharpe": "sharpe",
                "Profit Factor": "pf",
                "Win Rate": "win",
                "Retorno": "ret"
            }
            metric = metric_map.get(self.metric_combo.currentText(), "sharpe")
            
            # Cria pivot table
            pivot = df.pivot_table(
                values=metric,
                index='fast',
                columns='slow',
                aggfunc='mean',
                fill_value=0  # Preenche NaN com 0
            )
            
            # Obtém matriz de valores
            matrix = pivot.values.astype(float)
            
            # Trata NaN e infinitos
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Aplica transformações apenas se a matriz não for toda zero
            if np.any(matrix != 0):
                if self.log_check.isChecked():
                    matrix = np.log1p(np.abs(matrix) + 1) * np.sign(matrix)
                
                if self.norm_check.isChecked() and matrix.std() > 0:
                    matrix = (matrix - matrix.mean()) / (matrix.std() + 1e-10)
                
                if self.outlier_check.isChecked():
                    q1, q3 = np.percentile(matrix, [25, 75])
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    matrix = np.clip(matrix, lower, upper)
            
            # Garante que não há NaN após transformações
            matrix = np.nan_to_num(matrix, nan=0.0)
            
            self.populate_heatmap(pivot.index.tolist(), pivot.columns.tolist(), matrix)
            self.update_top_zones(df, metric)
            self.update_patterns(df)
            
        except Exception as e:
            print(f"Erro ao atualizar heatmap: {e}")
            import traceback
            traceback.print_exc()
    
    def populate_heatmap(self, rows, cols, matrix):
        """Preenche a tabela com cores"""
        try:
            n = len(rows)
            self.table.setRowCount(n)
            self.table.setColumnCount(n)
            self.table.setHorizontalHeaderLabels(cols)
            self.table.setVerticalHeaderLabels(rows)
            
            # Encontra valores min e max para escala de cores
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()
            else:
                min_val, max_val = -1, 1
            
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    
                    # Trata NaN
                    if np.isnan(val):
                        val = 0.0
                    
                    # Formata o valor
                    if abs(val) < 0.01:
                        text = "0.00"
                    else:
                        text = f"{val:.2f}"
                    
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    
                    # Aplica cor
                    color = self.get_color(val, min_val, max_val)
                    item.setBackground(color)
                    
                    self.table.setItem(i, j, item)
                    
        except Exception as e:
            print(f"Erro ao popular heatmap: {e}")
    
    def get_color(self, value, min_val, max_val):
        """Retorna cor baseada no valor"""
        try:
            # Trata NaN
            if np.isnan(value):
                return QColor(200, 200, 200)  # Cinza para NaN
            
            # Se min e max são iguais
            if max_val == min_val:
                norm = 0.5
            else:
                norm = (value - min_val) / (max_val - min_val)
                norm = max(0.0, min(1.0, norm))  # Garante entre 0 e 1
            
            # Vermelho (baixo) -> Amarelo (médio) -> Verde (alto)
            if norm < 0.33:
                # Vermelho para amarelo
                r = 255
                g = int(255 * (norm / 0.33))
                b = 0
            elif norm < 0.66:
                # Amarelo para verde claro
                progress = (norm - 0.33) / 0.33
                r = int(255 * (1 - progress))
                g = 255
                b = 0
            else:
                # Verde claro para verde escuro
                progress = (norm - 0.66) / 0.34
                r = 0
                g = int(255 * (1 - progress * 0.5))  # Diminui gradualmente
                b = int(100 * progress)  # Aumenta azul levemente
            
            # Garante valores inteiros válidos
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            
            return QColor(r, g, b)
            
        except Exception as e:
            print(f"Erro ao calcular cor: {e}")
            return QColor(200, 200, 200)  # Cinza em caso de erro
    
    def update_top_zones(self, df, metric):
        """Atualiza top zonas"""
        try:
            if df.empty:
                return
            
            top = df.nlargest(5, metric)[['fast', 'slow', metric]]
            
            text = "🔥 TOP ZONAS:\n\n"
            for i, (_, row) in enumerate(top.iterrows(), 1):
                text += f"{i}. {row['fast']} × {row['slow']} → {metric} {row[metric]:.2f}\n"
            
            self.top_label.setText(text)
            
        except Exception as e:
            print(f"Erro ao atualizar top zonas: {e}")
    
    def update_patterns(self, df):
        """Atualiza padrões identificados"""
        try:
            if df.empty:
                return
            
            patterns = []
            
            # Melhor coluna
            col_means = df.groupby('slow')['sharpe'].mean()
            if not col_means.empty:
                best_col = col_means.idxmax()
                patterns.append(f"✔ Coluna {best_col} tem alta consistência")
            
            # Melhor linha
            row_means = df.groupby('fast')['sharpe'].mean()
            if not row_means.empty:
                best_row = row_means.idxmax()
                patterns.append(f"✔ Linha {best_row} domina cruzamentos")
            
            # Hotspot adaptativo × low-lag
            adaptativas = ['KAMA', 'VIDYA', 'FRAMA', 'MAMA']
            lowlag = ['HMA', 'ZLEMA', 'DEMA', 'TEMA']
            
            hotspot = df[df['fast'].isin(adaptativas) & df['slow'].isin(lowlag)]
            if not hotspot.empty:
                avg_sharpe = hotspot['sharpe'].mean()
                patterns.append(f"✔ Região (Adaptativo × LowLag) é hotspot: Sharpe médio {avg_sharpe:.2f}")
            
            # SMA/EMA performance
            simple = df[df['fast'].isin(['SMA', 'EMA']) | df['slow'].isin(['SMA', 'EMA'])]
            if not simple.empty:
                avg_sharpe = simple['sharpe'].mean()
                patterns.append(f"✔ SMA/EMA têm performance: Sharpe médio {avg_sharpe:.2f}")
            
            if patterns:
                self.patterns_label.setText("\n".join(patterns))
            else:
                self.patterns_label.setText("✔ Análise em andamento...")
                
        except Exception as e:
            print(f"Erro ao atualizar padrões: {e}")
    
    def load_example(self):
        """Carrega dados de exemplo"""
        try:
            # Cria dados de exemplo
            np.random.seed(42)
            
            data = []
            for fast in self.indicators:
                for slow in self.indicators:
                    if fast == slow:
                        # Diagonal principal com valor zero
                        data.append({
                            'fast': fast,
                            'slow': slow,
                            'sharpe': 0.0,
                            'pf': 0.0,
                            'win': 0.0,
                            'ret': 0.0,
                            'tf': 5
                        })
                    else:
                        # Valores aleatórios para combinações diferentes
                        data.append({
                            'fast': fast,
                            'slow': slow,
                            'sharpe': np.random.uniform(0.8, 2.0),
                            'pf': np.random.uniform(1.0, 2.5),
                            'win': np.random.uniform(40, 70),
                            'ret': np.random.uniform(50, 200),
                            'tf': np.random.randint(1, 11)
                        })
            
            self.data = pd.DataFrame(data)
            self.update_heatmap()
            
        except Exception as e:
            print(f"Erro ao carregar exemplo: {e}")