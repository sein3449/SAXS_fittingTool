import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from scipy import stats
import os

class GISAXSAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("GISAXS Radius of Gyration Calculator - Copyright Reserved 2025.08 ver POSTECH Sein Chung")
        self.root.geometry("1600x1000")
        
        # Data storage for multiple files
        self.file_data = {}
        self.current_file = None
        self.span_selector = None
        
        # Create GUI elements
        self.create_widgets()
        self.add_copyright()
        
    def create_widgets(self):
        # Main frame with three columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left panel for file management and parameters
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.grid_propagate(False)
        
        # Middle panel for plots
        middle_frame = ttk.Frame(main_frame, width=600)
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        middle_frame.grid_propagate(False)
        
        # Right panel for results table
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_frame.grid_propagate(False)
        
        # Configure main frame columns
        main_frame.columnconfigure(0, weight=2)  # Left panel
        main_frame.columnconfigure(1, weight=3)  # Middle panel (plots)
        main_frame.columnconfigure(2, weight=2)  # Right panel
        main_frame.rowconfigure(0, weight=1)
        
        self.setup_left_panel(left_frame)
        self.setup_middle_panel(middle_frame)
        self.setup_right_panel(right_frame)
        
    def setup_left_panel(self, parent):
        # File management section
        file_frame = ttk.LabelFrame(parent, text="File Management", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File loading buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Load CSV File(s)", command=self.load_csv_files).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_selected_files).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(button_frame, text="Clear All", command=self.clear_all_files).pack(side=tk.LEFT, padx=(10, 0))
        
        # File list with checkboxes
        list_frame = ttk.LabelFrame(file_frame, text="Loaded Files", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create treeview for file list with checkboxes
        self.file_tree = ttk.Treeview(list_frame, columns=('filename', 'points', 'range'), show='tree headings', height=6)
        self.file_tree.heading('#0', text='✓')
        self.file_tree.heading('filename', text='Filename')
        self.file_tree.heading('points', text='Points')
        self.file_tree.heading('range', text='Q Range (Å⁻¹)')
        
        self.file_tree.column('#0', width=30)
        self.file_tree.column('filename', width=150)
        self.file_tree.column('points', width=60)
        self.file_tree.column('range', width=100)
        
        scrollbar_files = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar_files.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_files.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.file_tree.bind('<Double-1>', self.toggle_file_visibility)
        self.file_tree.bind('<Button-1>', self.on_file_select)
        
        # Current file selection
        current_frame = ttk.Frame(file_frame)
        current_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(current_frame, text="Current file for analysis:").pack(side=tk.LEFT)
        self.current_file_label = ttk.Label(current_frame, text="None selected", foreground="blue")
        self.current_file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Parameters section
        param_frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Particle type and qRg settings
        particle_frame = ttk.Frame(param_frame)
        particle_frame.pack(fill=tk.X)
        
        ttk.Label(particle_frame, text="Particle type:").grid(row=0, column=0, sticky=tk.W)
        self.particle_type = tk.StringVar(value="Globular")
        particle_combo = ttk.Combobox(particle_frame, textvariable=self.particle_type, 
                                     values=["Globular", "Rod-like", "Disc-like", "Flexible/Disordered"], width=15)
        particle_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        particle_combo.bind("<<ComboboxSelected>>", self.update_qrg_max)
        
        ttk.Label(particle_frame, text="Max qRg:").grid(row=0, column=2, sticky=tk.W)
        self.qrg_max_var = tk.DoubleVar(value=1.3)
        qrg_entry = ttk.Entry(particle_frame, textvariable=self.qrg_max_var, width=8)
        qrg_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Low-q fitting ratio
        lowq_frame = ttk.Frame(param_frame)
        lowq_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(lowq_frame, text="Low-q fitting ratio:").grid(row=0, column=0, sticky=tk.W)
        self.lowq_ratio_var = tk.DoubleVar(value=0.3)
        lowq_entry = ttk.Entry(lowq_frame, textvariable=self.lowq_ratio_var, width=8)
        lowq_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(lowq_frame, text="(0.1-0.8)").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        # Guinier equation info
        info_frame = ttk.LabelFrame(param_frame, text="Partial Low-q Fitting Algorithm", padding="5")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=8, width=45, font=('Arial', 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.update_guinier_info()
        
        # Manual range selection
        range_frame = ttk.LabelFrame(param_frame, text="Fitting Range Selection", padding="5")
        range_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.q_min_var = tk.DoubleVar()
        self.q_max_var = tk.DoubleVar()
        
        range_entry_frame = ttk.Frame(range_frame)
        range_entry_frame.pack(fill=tk.X)
        
        ttk.Label(range_entry_frame, text="Q min:").grid(row=0, column=0, sticky=tk.W)
        q_min_entry = ttk.Entry(range_entry_frame, textvariable=self.q_min_var, width=12)
        q_min_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(range_entry_frame, text="Q max:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        q_max_entry = ttk.Entry(range_entry_frame, textvariable=self.q_max_var, width=12)
        q_max_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        button_range_frame = ttk.Frame(range_frame)
        button_range_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 드래그 선택 버튼
        self.drag_button = ttk.Button(button_range_frame, text="🖱️ Enable Drag Selection", 
                                     command=self.toggle_range_selection)
        self.drag_button.pack(side=tk.LEFT)
        ttk.Button(button_range_frame, text="Reset", command=self.reset_range).pack(side=tk.LEFT, padx=(10, 0))
        
        self.range_status = ttk.Label(range_frame, text="Select file & enable drag mode", font=('Arial', 8))
        self.range_status.pack(pady=5)
        
        # Analysis button
        ttk.Button(param_frame, text="Calculate Rg for Current File", command=self.calculate_rg).pack(pady=15)
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="Current Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, height=8, width=45, font=('Courier', 9))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def setup_middle_panel(self, parent):
        # Plot section
        plot_frame = ttk.LabelFrame(parent, text="Data Visualization", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 여백 조정 - Guinier plot 제목 공간 확보
        self.fig.subplots_adjust(top=0.95, bottom=0.22, left=0.12, right=0.95, hspace=0.30)
        
    def setup_right_panel(self, parent):
        # Results table section
        table_frame = ttk.LabelFrame(parent, text="Rg Analysis Results Summary", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        control_frame = ttk.Frame(table_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Export to CSV", command=self.export_results).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Clear Results", command=self.clear_results_table).pack(side=tk.LEFT, padx=(10, 0))
        
        # Create frame for treeview and scrollbars
        tree_frame = ttk.Frame(table_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for results table
        self.results_tree = ttk.Treeview(tree_frame, 
                                        columns=('filename', 'rg', 'rg_error', 'i0', 'r_squared', 'qrg_max', 'constraint', 'quality'), 
                                        show='headings', height=15)
        
        # Define headings
        self.results_tree.heading('filename', text='Filename')
        self.results_tree.heading('rg', text='Rg (Å)')
        self.results_tree.heading('rg_error', text='Rg Error')
        self.results_tree.heading('i0', text='I(0)')
        self.results_tree.heading('r_squared', text='R²')
        self.results_tree.heading('qrg_max', text='qRg_max')
        self.results_tree.heading('constraint', text='Constraint')
        self.results_tree.heading('quality', text='Quality')
        
        # Define column widths
        self.results_tree.column('filename', width=120)
        self.results_tree.column('rg', width=80)
        self.results_tree.column('rg_error', width=80)
        self.results_tree.column('i0', width=80)
        self.results_tree.column('r_squared', width=60)
        self.results_tree.column('qrg_max', width=80)
        self.results_tree.column('constraint', width=80)
        self.results_tree.column('quality', width=80)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack tree and scrollbars
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(table_frame, text="Summary Statistics", padding="5")
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.summary_text = tk.Text(summary_frame, height=6, width=45, font=('Courier', 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
    def add_copyright(self):
        """Add copyright notice at the bottom"""
        copyright_frame = ttk.Frame(self.root)
        copyright_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        copyright_label = ttk.Label(copyright_frame, 
                                   text="Copyright Reserved 2025.08 ver POSTECH Sein Chung", 
                                   font=('Arial', 8), foreground='gray')
        copyright_label.pack()
        
        self.root.rowconfigure(0, weight=1)
        
    def update_qrg_max(self, event=None):
        """Update qRg max value based on particle type"""
        particle_type = self.particle_type.get()
        if particle_type == "Globular":
            self.qrg_max_var.set(1.3)
        elif particle_type == "Rod-like":
            self.qrg_max_var.set(1.0)
        elif particle_type == "Disc-like":
            self.qrg_max_var.set(1.7)
        else:  # Flexible/Disordered
            self.qrg_max_var.set(1.1)
        
        self.update_guinier_info()
    
    def update_guinier_info(self):
        """Update Guinier equation information with partial fitting explanation"""
        self.info_text.delete(1.0, tk.END)
        
        particle_type = self.particle_type.get()
        lowq_ratio = self.lowq_ratio_var.get()
        
        info_text = f"""Partial Low-q Fitting Algorithm
{'='*35}

Strategy: Use only LOW-Q portion for fitting
• Selected range: User specified
• Fitting range: {lowq_ratio*100:.0f}% of front portion
• Focus: Early linear behavior only

Why Partial Fitting?
• Guinier approximation valid at very low q
• Early region = most reliable data
• Avoids higher-q deviations
• Better R² and constraint satisfaction

Current Type: {particle_type}
qRg Limit: ≤ {self.qrg_max_var.get()}

Algorithm Steps:
1. Take selected q range (user defined)
2. Use only first {lowq_ratio*100:.0f}% of this range
3. Fit Guinier equation to this portion
4. Extract Rg from slope
5. Validate qRg constraint

Benefits:
✓ Higher R² values
✓ More reliable fits  
✓ Better constraint satisfaction
✓ Avoids overfitting issues
✓ Focuses on true Guinier region

Equation: I(q) = I₀ × exp(-Rg²q²/3)
ln(I) = ln(I₀) - (Rg²/3)q²
"""
        
        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state=tk.DISABLED)
    
    def load_csv_files(self):
        """Load multiple CSV files"""
        file_paths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        for file_path in file_paths:
            if file_path not in self.file_data:
                try:
                    # Read CSV file
                    data = pd.read_csv(file_path)
                    
                    if data.shape[1] < 2:
                        messagebox.showerror("Error", f"File {os.path.basename(file_path)} must have at least 2 columns")
                        continue
                    
                    # Process data
                    q_data = data.iloc[:, 0].values
                    intensity_data = data.iloc[:, 1].values
                    
                    # Remove invalid data
                    valid_mask = (np.isfinite(q_data) & 
                                 np.isfinite(intensity_data) & 
                                 (intensity_data > 0) &
                                 (q_data > 0))
                    
                    q_data = q_data[valid_mask]
                    intensity_data = intensity_data[valid_mask]
                    
                    # Sort by q value
                    sort_idx = np.argsort(q_data)
                    q_data = q_data[sort_idx]
                    intensity_data = intensity_data[sort_idx]
                    
                    # Store data
                    file_info = {
                        'q_data': q_data,
                        'intensity_data': intensity_data,
                        'visible': True,
                        'color': plt.cm.tab10(len(self.file_data) % 10),
                        'q_min': q_data[0],
                        'q_max': q_data[min(len(q_data)-1, 20)],
                        'results': None
                    }
                    
                    self.file_data[file_path] = file_info
                    
                    # Add to tree view
                    filename = os.path.basename(file_path)
                    q_range = f"{q_data[0]:.3f} - {q_data[-1]:.3f}"
                    self.file_tree.insert('', 'end', iid=file_path, text='✓', 
                                         values=(filename, len(q_data), q_range))
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {os.path.basename(file_path)}: {str(e)}")
        
        self.update_plots()
    
    def remove_selected_files(self):
        """Remove selected files from the list"""
        selected_items = self.file_tree.selection()
        for item in selected_items:
            if item in self.file_data:
                del self.file_data[item]
            self.file_tree.delete(item)
            if self.results_tree.exists(item):
                self.results_tree.delete(item)
        
        if self.current_file not in self.file_data:
            self.current_file = None
            self.current_file_label.config(text="None selected")
        
        self.update_plots()
        self.update_summary_stats()
    
    def clear_all_files(self):
        """Clear all loaded files"""
        self.file_data.clear()
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.current_file = None
        self.current_file_label.config(text="None selected")
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()
        self.update_summary_stats()
    
    def toggle_file_visibility(self, event):
        """Toggle visibility of a file by double-clicking"""
        item = self.file_tree.identify('item', event.x, event.y)
        if item and item in self.file_data:
            self.file_data[item]['visible'] = not self.file_data[item]['visible']
            symbol = '✓' if self.file_data[item]['visible'] else '✗'
            self.file_tree.set(item, '#0', symbol)
            self.update_plots()
    
    def on_file_select(self, event):
        """Handle file selection for analysis"""
        item = self.file_tree.identify('item', event.x, event.y)
        if item and item in self.file_data:
            self.current_file = item
            filename = os.path.basename(item)
            self.current_file_label.config(text=filename)
            
            file_info = self.file_data[item]
            self.q_min_var.set(file_info['q_min'])
            self.q_max_var.set(file_info['q_max'])
            
            self.range_status.config(text=f"Selected: {filename}")
            
            if file_info['results']:
                self.display_current_results(file_info['results'])
    
    def toggle_range_selection(self):
        """토글 방식으로 드래그 선택 모드 활성화/비활성화"""
        if not self.current_file or self.current_file not in self.file_data:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        if self.span_selector is None:
            self.enable_range_selection()
        else:
            self.disable_range_selection()
    
    def enable_range_selection(self):
        """드래그 선택 모드 활성화"""
        try:
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
                self.span_selector = None
            
            self.span_selector = SpanSelector(
                self.ax1, self.on_range_select, 'horizontal',
                useblit=False,
                props=dict(alpha=0.3, facecolor='red', edgecolor='darkred'),
                button=1,
                minspan=0.001
            )
            
            self.drag_button.config(text="🚫 Disable Drag Mode")
            self.range_status.config(text="DRAG MODE: Click and drag on the plot to select range", 
                                   foreground="red", font=('Arial', 9, 'bold'))
            
            self.canvas.draw_idle()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to enable drag selection: {str(e)}")
    
    def disable_range_selection(self):
        """드래그 선택 모드 비활성화"""
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
            self.span_selector = None
        
        self.drag_button.config(text="🖱️ Enable Drag Selection")
        self.range_status.config(text="Drag mode disabled", 
                               foreground="black", font=('Arial', 8, 'normal'))
    
    def on_range_select(self, qmin, qmax):
        """범위 선택 콜백"""
        try:
            if qmax <= qmin:
                return
            
            self.q_min_var.set(round(qmin, 6))
            self.q_max_var.set(round(qmax, 6))
            
            if self.current_file:
                self.file_data[self.current_file]['q_min'] = qmin
                self.file_data[self.current_file]['q_max'] = qmax
            
            self.range_status.config(text=f"Range selected: {qmin:.4f} - {qmax:.4f} Å⁻¹",
                                   foreground="green")
            
            self.update_plots()
            self.disable_range_selection()
            
        except Exception as e:
            print(f"Error in range selection: {e}")
    
    def reset_range(self):
        """Reset the fitting range for current file"""
        if not self.current_file or self.current_file not in self.file_data:
            return
        
        file_info = self.file_data[self.current_file]
        q_data = file_info['q_data']
        
        default_q_max = q_data[min(len(q_data)-1, 20)]
        self.q_min_var.set(q_data[0])
        self.q_max_var.set(default_q_max)
        
        file_info['q_min'] = q_data[0]
        file_info['q_max'] = default_q_max
        
        self.disable_range_selection()
        
        self.range_status.config(text="Range reset to default", foreground="black")
        self.update_plots()
    
    def partial_lowq_guinier_fit(self, q_data, intensity_data, q_min, q_max, max_qrg, lowq_ratio=0.3):
        """
        핵심 개선: 선택 범위의 앞쪽 일부만 사용하여 Guinier fitting
        전체 범위를 맞추려 하지 않고, low-q 부분만 확실하게 피팅
        """
        
        # 1단계: 사용자 선택 범위 내의 데이터 추출
        range_mask = (q_data >= q_min) & (q_data <= q_max)
        q_selected = q_data[range_mask]
        i_selected = intensity_data[range_mask]
        
        if len(q_selected) < 4:
            raise ValueError("Not enough data points in selected range (need at least 4)")
        
        # 2단계: 선택 범위의 앞쪽 일부만 사용 (lowq_ratio로 조절)
        lowq_ratio = max(0.1, min(0.8, lowq_ratio))  # 비율을 0.1-0.8로 제한
        n_points_to_use = max(4, int(len(q_selected) * lowq_ratio))
        
        # 앞쪽 부분만 추출
        q_fit = q_selected[:n_points_to_use]
        i_fit = i_selected[:n_points_to_use]
        
        # 3단계: qRg 제한 조건에 맞게 범위 추가 조정
        # 임시로 Rg를 추정하여 qRg가 제한을 초과하지 않도록 조정
        for attempt_points in range(n_points_to_use, max(3, n_points_to_use//2), -1):
            q_temp = q_fit[:attempt_points]
            i_temp = i_fit[:attempt_points]
            
            if len(q_temp) < 3:
                break
                
            try:
                # Guinier 피팅
                q_squared = q_temp ** 2
                log_i = np.log(i_temp)
                
                coeffs = np.polyfit(q_squared, log_i, 1)
                slope, intercept = coeffs
                
                if slope >= 0:
                    continue
                
                # 임시 Rg 계산
                temp_rg = np.sqrt(-3 * slope)
                
                # qRg 제약 확인
                qrg_max_actual = q_temp[-1] * temp_rg
                
                if qrg_max_actual <= max_qrg:
                    # 제약 조건을 만족하므로 이 범위 사용
                    q_fit = q_temp
                    i_fit = i_temp
                    break
                    
            except:
                continue
        
        # 4단계: 최종 Guinier fitting
        q_squared = q_fit ** 2
        log_i = np.log(i_fit)
        
        # 선형 회귀
        coeffs = np.polyfit(q_squared, log_i, 1)
        slope, intercept = coeffs
        
        if slope >= 0:
            raise ValueError("Positive slope detected - not consistent with Guinier behavior")
        
        # Rg와 I0 계산
        rg = np.sqrt(-3 * slope)
        i0 = np.exp(intercept)
        
        # R-squared 계산
        log_i_pred = intercept + slope * q_squared
        ss_res = np.sum((log_i - log_i_pred) ** 2)
        ss_tot = np.sum((log_i - np.mean(log_i)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 최종 qRg 검증
        qrg_max_actual = q_fit[-1] * rg
        
        # 실제 fitting에 사용된 q 범위
        actual_q_min = q_fit[0]
        actual_q_max = q_fit[-1]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'rg': rg,
            'i0': i0,
            'r_squared': r_squared,
            'qrg_max_actual': qrg_max_actual,
            'fit_range': (actual_q_min, actual_q_max),
            'n_points': len(q_fit),
            'lowq_ratio_used': len(q_fit) / len(q_selected)
        }
    
    def calculate_rg(self):
        """Partial Low-q Fitting을 사용한 간소화된 Guinier 분석"""
        if not self.current_file or self.current_file not in self.file_data:
            messagebox.showwarning("Warning", "Please select a file first")
            return
        
        try:
            file_info = self.file_data[self.current_file]
            q_data = file_info['q_data']
            intensity_data = file_info['intensity_data']
            
            q_min = self.q_min_var.get()
            q_max = self.q_max_var.get()
            max_qrg = self.qrg_max_var.get()
            lowq_ratio = self.lowq_ratio_var.get()
            
            if q_min >= q_max:
                raise ValueError("Q min must be less than Q max")
            
            # Partial Low-q Fitting 수행
            fit_results = self.partial_lowq_guinier_fit(
                q_data, intensity_data, q_min, q_max, max_qrg, lowq_ratio
            )
            
            # 결과 저장
            results = {
                'rg': fit_results['rg'],
                'i0': fit_results['i0'],
                'r_squared': fit_results['r_squared'],
                'slope': fit_results['slope'],
                'intercept': fit_results['intercept'],
                'fit_range': fit_results['fit_range'],
                'original_range': (q_min, q_max),
                'qrg_max_actual': fit_results['qrg_max_actual'],
                'n_points': fit_results['n_points'],
                'particle_type': self.particle_type.get(),
                'qrg_limit': max_qrg,
                'fitting_method': 'Partial Low-q',
                'lowq_ratio_used': fit_results['lowq_ratio_used']
            }
            
            file_info['results'] = results
            
            # 오차 추정
            actual_q_min, actual_q_max = fit_results['fit_range']
            range_mask = (q_data >= actual_q_min) & (q_data <= actual_q_max)
            q_fit = q_data[range_mask]
            i_fit = intensity_data[range_mask]
            
            log_i = np.log(i_fit)
            q_squared = q_fit ** 2
            log_i_pred = fit_results['intercept'] + fit_results['slope'] * q_squared
            residuals = log_i - log_i_pred
            mse = np.mean(residuals**2)
            rg_error = fit_results['rg'] * np.sqrt(mse / len(q_fit))
            
            # 결과 업데이트
            self.update_results_table(results, rg_error)
            self.display_current_results(results, rg_error)
            self.update_plots()
            self.update_summary_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
    
    def update_plots(self):
        """Update both plots with current data"""
        self.ax1.clear()
        self.ax2.clear()
        
        if not self.file_data:
            self.canvas.draw()
            return
        
        # Plot raw data
        for file_path, file_info in self.file_data.items():
            if file_info['visible']:
                q_data = file_info['q_data']
                intensity_data = file_info['intensity_data']
                color = file_info['color']
                filename = os.path.basename(file_path)
                
                self.ax1.loglog(q_data, intensity_data, '-', alpha=0.8, 
                               color=color, label=filename, linewidth=1.5)
                
                # Highlight ranges for current file
                if file_path == self.current_file:
                    q_min = file_info['q_min']
                    q_max = file_info['q_max']
                    
                    # Show selected range
                    range_mask = (q_data >= q_min) & (q_data <= q_max)
                    if np.any(range_mask):
                        self.ax1.loglog(q_data[range_mask], intensity_data[range_mask], 
                                       's', markersize=2, color='orange', alpha=0.7,
                                       label=f'{filename} (selected range)')
                    
                    # Show actual fitting range if results exist
                    if file_info['results']:
                        fit_q_min, fit_q_max = file_info['results']['fit_range']
                        fit_mask = (q_data >= fit_q_min) & (q_data <= fit_q_max)
                        if np.any(fit_mask):
                            self.ax1.loglog(q_data[fit_mask], intensity_data[fit_mask], 
                                           'o', markersize=4, color='red', alpha=0.9,
                                           markeredgecolor='darkred', markeredgewidth=1,
                                           label=f'{filename} (low-q fit)')
        
        self.ax1.set_xlabel('q (Å⁻¹)', fontsize=12)
        self.ax1.set_ylabel('Intensity', fontsize=12)
        self.ax1.set_title('Raw GISAXS Data - Partial Low-q Fitting', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        
        # Legend를 하단에 배치
        visible_files = [os.path.basename(f) for f, info in self.file_data.items() if info['visible']]
        if visible_files:
            ncol = min(len(visible_files) + 1, 3)  # +1 for fit range
            self.ax1.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                           framealpha=0.9, fancybox=True, shadow=True,
                           fontsize=9, ncol=ncol)
        
        # Plot Guinier plots if results exist
        has_results = False
        for file_path, file_info in self.file_data.items():
            if file_info['visible'] and file_info['results']:
                self.plot_guinier_result(file_path, file_info)
                has_results = True
        
        if not has_results:
            self.ax2.text(0.5, 0.5, 'No Guinier fits calculated yet\n\nSelect file and click "Calculate Rg"\nAlgorithm will use only low-q portion', 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=self.ax2.transAxes, fontsize=12, alpha=0.7)
            self.ax2.set_xlabel('q² (Å⁻²)', fontsize=12)
            self.ax2.set_ylabel('Intensity', fontsize=12)
        
        # Guinier plot 제목을 아래쪽으로 이동
        if has_results:
            self.ax2.text(0.5, -0.12, 'Partial Low-q Guinier Plot (Log-Log Scale)', 
                         horizontalalignment='center', verticalalignment='top',
                         transform=self.ax2.transAxes, fontsize=14, fontweight='bold')
        
        self.fig.subplots_adjust(top=0.95, bottom=0.22, left=0.12, right=0.95, hspace=0.30)
        self.canvas.draw()
    
    def display_current_results(self, results, rg_error=0):
        """Display enhanced calculation results"""
        self.results_text.delete(1.0, tk.END)
        
        filename = os.path.basename(self.current_file)
        constraint_satisfied = results['qrg_max_actual'] <= results['qrg_limit']
        
        result_text = f"""Partial Low-q Fit Results: {filename}
{'='*35}

Fitting Strategy: {results.get('fitting_method', 'Unknown')}
• Used ratio: {results.get('lowq_ratio_used', 0)*100:.1f}% of selected range
• Rg: {results['rg']:.2f} ± {rg_error:.2f} Å
• I(0): {results['i0']:.2e}
• R²: {results['r_squared']:.4f}
• Data points: {results['n_points']}

Range Information:
• Selected range: {results.get('original_range', results['fit_range'])[0]:.4f} - {results.get('original_range', results['fit_range'])[1]:.4f} Å⁻¹
• Actual fit range: {results['fit_range'][0]:.4f} - {results['fit_range'][1]:.4f} Å⁻¹

Constraint Validation:
• qRg_max: {results['qrg_max_actual']:.3f}
• Limit ({results['particle_type']}): {results['qrg_limit']:.1f}
• Status: {'✅ SATISFIED' if constraint_satisfied else '❌ VIOLATED'}

Quality Assessment:
• R² Quality: {'Excellent' if results['r_squared'] > 0.99 else 'Good' if results['r_squared'] > 0.95 else 'Fair' if results['r_squared'] > 0.90 else 'Poor'}
• Fitting reliability: {'High' if constraint_satisfied and results['r_squared'] > 0.95 else 'Medium' if constraint_satisfied else 'Low'}

Physical Properties:
• Characteristic size: ~{results['rg']:.1f} Å
• Equivalent sphere radius: ~{results['rg'] * np.sqrt(5/3):.1f} Å

Algorithm Benefits:
✓ Focuses on most reliable low-q data
✓ Avoids higher-q deviations
✓ Better constraint satisfaction
✓ Higher R² values expected
✓ More stable fitting results
"""
        
        self.results_text.insert(tk.END, result_text)
    
    def plot_guinier_result(self, file_path, file_info):
        """Enhanced Guinier plot with partial fitting visualization"""
        results = file_info['results']
        if not results:
            return
        
        q_data = file_info['q_data']
        intensity_data = file_info['intensity_data']
        color = file_info['color']
        filename = os.path.basename(file_path)
        
        # Get actual fit range data
        q_min, q_max = results['fit_range']
        range_mask = (q_data >= q_min) & (q_data <= q_max)
        q_fit_data = q_data[range_mask]
        i_fit_data = intensity_data[range_mask]
        
        # Plot all data points (faded)
        self.ax2.loglog(q_data**2, intensity_data, '.', alpha=0.3, 
                       color=color, markersize=1)
        
        # Emphasize the partial fitting region
        self.ax2.loglog(q_fit_data**2, i_fit_data, 'o', markersize=6, 
                       color=color, markeredgecolor='black', markeredgewidth=1,
                       label=f'{filename} (low-q fit)')
        
        # Enhanced fit line
        q_squared_fit = np.logspace(np.log10(q_fit_data[0]**2), 
                                   np.log10(q_fit_data[-1]**2), 100)
        i_fit_line = np.exp(results['intercept'] + results['slope'] * q_squared_fit)
        
        self.ax2.loglog(q_squared_fit, i_fit_line, '-', linewidth=4, 
                       color=color, alpha=0.9,
                       label=f'{filename} (Rg={results["rg"]:.2f}Å, R²={results["r_squared"]:.3f})')
        
        # Add qRg limit line for current file
        if file_path == self.current_file:
            if results['rg'] > 0:
                qrg_limit = results['qrg_limit'] / results['rg']
                xlim = self.ax2.get_xlim()
                if qrg_limit**2 >= xlim[0] and qrg_limit**2 <= xlim[1]:
                    self.ax2.axvline(qrg_limit**2, color='orange', linestyle=':', 
                                   alpha=0.8, linewidth=3,
                                   label=f'qRg={results["qrg_limit"]:.1f} limit')
        
        self.ax2.set_xlabel('q² (Å⁻²)', fontsize=12)
        self.ax2.set_ylabel('Intensity', fontsize=12)
        self.ax2.grid(True, alpha=0.3)
        
        # Legend를 하단에 배치
        handles, labels = self.ax2.get_legend_handles_labels()
        if handles:
            ncol = min(len(handles), 2)
            self.ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center',
                           framealpha=0.9, fancybox=True, shadow=True,
                           fontsize=9, ncol=ncol)
    
    def update_results_table(self, results, rg_error):
        """Update the results table with new calculation"""
        filename = os.path.basename(self.current_file)
        constraint_satisfied = results['qrg_max_actual'] <= results['qrg_limit']
        
        if self.results_tree.exists(self.current_file):
            self.results_tree.delete(self.current_file)
        
        quality = 'Excellent' if results['r_squared'] > 0.99 else 'Good' if results['r_squared'] > 0.95 else 'Fair' if results['r_squared'] > 0.90 else 'Poor'
        constraint_status = 'OK' if constraint_satisfied else 'VIOLATED'
        
        self.results_tree.insert('', 'end', iid=self.current_file,
                                values=(
                                    filename,
                                    f"{results['rg']:.2f}",
                                    f"±{rg_error:.2f}",
                                    f"{results['i0']:.2e}",
                                    f"{results['r_squared']:.3f}",
                                    f"{results['qrg_max_actual']:.3f}",
                                    constraint_status,
                                    quality
                                ))
    
    def update_summary_stats(self):
        """Update summary statistics"""
        self.summary_text.delete(1.0, tk.END)
        
        if not any(info.get('results') for info in self.file_data.values()):
            self.summary_text.insert(tk.END, "No analysis results available yet.")
            return
        
        results_list = []
        for file_info in self.file_data.values():
            if file_info.get('results'):
                results_list.append(file_info['results'])
        
        if not results_list:
            return
        
        rg_values = [r['rg'] for r in results_list]
        r_squared_values = [r['r_squared'] for r in results_list]
        qrg_max_values = [r['qrg_max_actual'] for r in results_list]
        
        constraint_violations = sum(1 for r in results_list if r['qrg_max_actual'] > r['qrg_limit'])
        excellent_quality = sum(1 for r in results_list if r['r_squared'] > 0.99)
        good_quality = sum(1 for r in results_list if r['r_squared'] > 0.95)
        
        # Average low-q ratio used
        avg_lowq_ratio = np.mean([r.get('lowq_ratio_used', 0) for r in results_list])
        
        summary = f"""Partial Low-q Fit Summary ({len(results_list)} files)
{'='*35}

Rg Statistics:
• Mean Rg: {np.mean(rg_values):.2f} ± {np.std(rg_values):.2f} Å
• Range: {np.min(rg_values):.2f} - {np.max(rg_values):.2f} Å

Quality Metrics:
• Mean R²: {np.mean(r_squared_values):.3f}
• Excellent fits (R²>0.99): {excellent_quality}/{len(results_list)}
• Good fits (R²>0.95): {good_quality}/{len(results_list)}
• Constraint violations: {constraint_violations}/{len(results_list)}

Fitting Statistics:
• Avg low-q ratio used: {avg_lowq_ratio*100:.1f}%
• All fits: Partial Low-q method
• Focus: Early Guinier region only

qRg Statistics:
• Mean qRg_max: {np.mean(qrg_max_values):.3f}
• Range: {np.min(qrg_max_values):.3f} - {np.max(qrg_max_values):.3f}

Algorithm Success:
✓ Simplified and reliable approach
✓ Better constraint satisfaction expected  
✓ Higher R² values from focused fitting
"""
        
        self.summary_text.insert(tk.END, summary)
    
    def export_results(self):
        """Export results table to CSV"""
        if not self.results_tree.get_children():
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save results as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = []
                for item in self.results_tree.get_children():
                    values = self.results_tree.item(item, 'values')
                    data.append(values)
                
                columns = ['Filename', 'Rg (Å)', 'Rg Error', 'I(0)', 'R²', 'qRg_max', 'Constraint', 'Quality']
                df = pd.DataFrame(data, columns=columns)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def clear_results_table(self):
        """Clear all results from the table"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        for file_info in self.file_data.values():
            file_info['results'] = None
        
        self.update_plots()
        self.update_summary_stats()

def main():
    root = tk.Tk()
    app = GISAXSAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
