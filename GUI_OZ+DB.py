import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class SAXSFittingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("SAXS OZ + DB + Power Law + Background Fitting")
        self.master.geometry("1450x1050")

        self.datasets = {}
        self.fit_results = {}
        self.qmin_global = None
        self.qmax_global = None
        self.drag_start = None
        self.selection_span = None
        self.max_fit_attempts = 15
        self.file_vars = {}
        self.checkboxes = {}

        self.build_gui()

    def build_gui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(left_frame, text="Load CSV Files", command=self.load_files).pack(fill=tk.X, pady=5)
        
        file_frame = ttk.LabelFrame(left_frame, text="Files (check to display)")
        file_frame.pack(fill=tk.Y, expand=True, pady=5)
        
        self.check_canvas = tk.Canvas(file_frame, width=380, height=200)
        self.check_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.check_frame = ttk.Frame(self.check_canvas)
        self.check_scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.check_canvas.yview)
        self.check_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.check_canvas.configure(yscrollcommand=self.check_scrollbar.set)
        self.check_canvas.create_window((0,0), window=self.check_frame, anchor="nw")
        self.check_frame.bind("<Configure>", lambda e: self.check_canvas.configure(scrollregion=self.check_canvas.bbox("all")))

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        ttk.Label(left_frame, text="Global q range (drag in plot or type):").pack(pady=5)
        qrange_frame = ttk.Frame(left_frame)
        qrange_frame.pack(fill=tk.X)
        ttk.Label(qrange_frame, text="q min:").grid(row=0, column=0, sticky=tk.W)
        self.qmin_var = tk.StringVar(value="0.01")
        ttk.Entry(qrange_frame, textvariable=self.qmin_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(qrange_frame, text="q max:").grid(row=1, column=0, sticky=tk.W)
        self.qmax_var = tk.StringVar(value="0.1")
        ttk.Entry(qrange_frame, textvariable=self.qmax_var, width=12).grid(row=1, column=1, padx=5)

        ttk.Label(left_frame, text="Fitting parameters (OZ + DB + Power Law + BG):").pack(pady=(10,5))
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill=tk.X)
        
        ttk.Label(param_frame, text="I_OZ(0):").grid(row=0, column=0, sticky=tk.W)
        self.OZ0_var = tk.StringVar(value="1.0")
        ttk.Entry(param_frame, textvariable=self.OZ0_var, width=10).grid(row=0, column=1, padx=2)
        ttk.Label(param_frame, text="ξ_OZ (nm):").grid(row=0, column=2, sticky=tk.W)
        self.xi_OZ_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.xi_OZ_var, width=10).grid(row=0, column=3, padx=2)
        
        ttk.Label(param_frame, text="I_DB(0):").grid(row=1, column=0, sticky=tk.W)
        self.DB0_var = tk.StringVar(value="0.5")
        ttk.Entry(param_frame, textvariable=self.DB0_var, width=10).grid(row=1, column=1, padx=2)
        ttk.Label(param_frame, text="ξ_DB (nm):").grid(row=1, column=2, sticky=tk.W)
        self.xi_DB_var = tk.StringVar(value="5.0")
        ttk.Entry(param_frame, textvariable=self.xi_DB_var, width=10).grid(row=1, column=3, padx=2)
        
        ttk.Label(param_frame, text="A (power):").grid(row=2, column=0, sticky=tk.W)
        self.A_var = tk.StringVar(value="1e-5")
        ttk.Entry(param_frame, textvariable=self.A_var, width=10).grid(row=2, column=1, padx=2)
        ttk.Label(param_frame, text="n (exp):").grid(row=2, column=2, sticky=tk.W)
        self.n_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.n_var, width=10).grid(row=2, column=3, padx=2)
        
        ttk.Label(param_frame, text="Background:").grid(row=3, column=0, sticky=tk.W)
        self.bg_var = tk.StringVar(value="0.001")
        ttk.Entry(param_frame, textvariable=self.bg_var, width=10).grid(row=3, column=1, padx=2)

        ttk.Label(left_frame, text="Parameter limits:").pack(pady=(10,5))
        limit_frame = ttk.Frame(left_frame)
        limit_frame.pack(fill=tk.X)
        
        ttk.Label(limit_frame, text="ξ_OZ min:").grid(row=0, column=0, sticky=tk.W)
        self.xi_OZ_min_var = tk.StringVar(value="0.1")
        ttk.Entry(limit_frame, textvariable=self.xi_OZ_min_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(limit_frame, text="max:").grid(row=0, column=2, sticky=tk.W)
        self.xi_OZ_max_var = tk.StringVar(value="20")
        ttk.Entry(limit_frame, textvariable=self.xi_OZ_max_var, width=8).grid(row=0, column=3, padx=2)
        
        ttk.Label(limit_frame, text="ξ_DB min:").grid(row=1, column=0, sticky=tk.W)
        self.xi_DB_min_var = tk.StringVar(value="0.5")
        ttk.Entry(limit_frame, textvariable=self.xi_DB_min_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(limit_frame, text="max:").grid(row=1, column=2, sticky=tk.W)
        self.xi_DB_max_var = tk.StringVar(value="100")
        ttk.Entry(limit_frame, textvariable=self.xi_DB_max_var, width=8).grid(row=1, column=3, padx=2)
        
        ttk.Label(limit_frame, text="n min:").grid(row=2, column=0, sticky=tk.W)
        self.n_min_var = tk.StringVar(value="1.0")
        ttk.Entry(limit_frame, textvariable=self.n_min_var, width=8).grid(row=2, column=1, padx=2)
        ttk.Label(limit_frame, text="max:").grid(row=2, column=2, sticky=tk.W)
        self.n_max_var = tk.StringVar(value="5.0")
        ttk.Entry(limit_frame, textvariable=self.n_max_var, width=8).grid(row=2, column=3, padx=2)

        fitbtn_frame = ttk.Frame(left_frame)
        fitbtn_frame.pack(fill=tk.X, pady=15)
        ttk.Button(fitbtn_frame, text="Fit Selected Files", command=self.fit_selected_files).pack(fill=tk.X, pady=3)
        ttk.Button(fitbtn_frame, text="Save Results & Curves", command=self.save_results).pack(fill=tk.X, pady=3)
        ttk.Button(fitbtn_frame, text="Reset Fitting Inputs", command=self.reset_fitting_inputs).pack(fill=tk.X, pady=3)

        result_frame = ttk.LabelFrame(left_frame, text="Fit Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("File", "I_OZ(0)", "ξ_OZ", "I_DB(0)", "ξ_DB", "A", "n", "BG", "R²")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=8)
        col_widths = [45, 65, 50, 65, 50, 60, 40, 50, 40]
        for i, col in enumerate(columns):
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=col_widths[i])
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar=ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, (self.ax_data, self.ax_bar) = plt.subplots(2,1,figsize=(10,10))
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(right_frame)
        toolbar_frame.pack(fill=tk.X)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)


    def reset_fitting_inputs(self):
        """
        피팅 관련 모든 입력값을 기본 초기값으로 복원하며,
        피팅 결과 테이블과 그래프도 초기화
        """
        # 기본 초기값 지정 (필요시 변경 가능)
        self.OZ0_var.set("1.0")
        self.xi_OZ_var.set("2.0")
        self.DB0_var.set("0.5")
        self.xi_DB_var.set("5.0")
        self.A_var.set("1e-5")
        self.n_var.set("2.0")
        self.bg_var.set("0.001")

        self.xi_OZ_min_var.set("0.1")
        self.xi_OZ_max_var.set("20")
        self.xi_DB_min_var.set("0.5")
        self.xi_DB_max_var.set("100")
        self.n_min_var.set("1.0")
        self.n_max_var.set("5.0")

        self.qmin_var.set("0.01")
        self.qmax_var.set("0.1")

        # 피팅 결과 초기화
        self.fit_results.clear()
        self.result_tree.delete(*self.result_tree.get_children())
        self.ax_data.clear()
        self.ax_bar.clear()
        self.ax_data.set_title("Fitting inputs reset. No results.")
        self.canvas.draw()

    def select_all(self):
        for var in self.file_vars.values(): 
            var.set(1)
        self.update_data_display()

    def deselect_all(self):
        for var in self.file_vars.values(): 
            var.set(0)
        self.update_data_display()

    def load_files(self):
        files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not files: return
        self.datasets.clear()
        self.file_vars.clear()
        self.checkboxes.clear()
        for widget in self.check_frame.winfo_children(): widget.destroy()
        self.fit_results.clear()
        self.result_tree.delete(*self.result_tree.get_children())
        
        for file in files:
            try:
                df = pd.read_csv(file)
                q = df.iloc[:,0].values
                intensity = df.iloc[:,1].values
                mask = ~(np.isnan(q) | np.isnan(intensity))
                q, intensity = q[mask], intensity[mask]
                filename = os.path.basename(file)
                self.datasets[filename] = (q, intensity)
                var = tk.IntVar(value=0)
                cb = tk.Checkbutton(
                    self.check_frame, 
                    text=filename, 
                    variable=var,
                    command=self.update_data_display
                )
                cb.pack(anchor='w')
                self.file_vars[filename] = var
                self.checkboxes[filename] = cb
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file}: {e}")
        messagebox.showinfo("Success", f"Loaded {len(self.datasets)} files")
        self.update_data_display()

    def update_data_display(self):
        """선택된 파일들의 raw data를 그래프에 표시"""
        selected_files = [f for f, v in self.file_vars.items() if v.get() == 1]
        
        self.ax_data.clear()
        
        if not selected_files:
            self.ax_data.set_title("No files selected")
            self.ax_data.set_xlabel('q (nm⁻¹)')
            self.ax_data.set_ylabel('Intensity (a.u.)')
            self.canvas.draw()
            return
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))
        for idx, filename in enumerate(selected_files):
            if filename in self.datasets:
                q, intensity = self.datasets[filename]
                color = colors[idx % len(colors)]
                self.ax_data.loglog(
                    q, intensity, 'o-', 
                    color=color, markersize=3, linewidth=1, alpha=0.7,
                    label=filename
                )
                
                if idx == 0 and self.qmin_global is None:
                    self.qmin_var.set(f"{np.min(q):.4f}")
                    self.qmax_var.set(f"{np.max(q):.4f}")
        
        self.ax_data.set_xlabel('q (nm⁻¹)', fontsize=12)
        self.ax_data.set_ylabel('Intensity (a.u.)', fontsize=12)
        self.ax_data.set_title(f'Raw Data ({len(selected_files)} files selected)', fontsize=13)
        self.ax_data.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        self.ax_data.grid(True, which='both', ls='--', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    # 4항 모델: OZ + DB + Power Law + BG
    def full_model(self, q, OZ0, xi_OZ, DB0, xi_DB, A, n, bg):
        return (self.oz_term(q, OZ0, xi_OZ) + 
                self.db_term(q, DB0, xi_DB) + 
                self.power_law_term(q, A, n) + bg)
    
    def oz_term(self, q, OZ0, xi_OZ):
        return OZ0 / (1 + (xi_OZ**2) * q**2)
    
    def db_term(self, q, DB0, xi_DB):
        return DB0 / ((1 + (xi_DB**2) * q**2) ** 2)
    
    def power_law_term(self, q, A, n):
        return A * q**(-n)

    def fit_single_with_bounds(self, q_fit, intensity_fit, p0,
        xi_OZ_min, xi_OZ_max, xi_DB_min, xi_DB_max, n_min, n_max):
        attempt = 0
        params_now = p0[:]
        popt = None
        while attempt < self.max_fit_attempts:
            try:
                popt, _ = curve_fit(
                    self.full_model,
                    q_fit, intensity_fit,
                    p0=params_now,
                    maxfev=15000,
                    bounds=(
                        [0, xi_OZ_min, 0, xi_DB_min, 0, n_min, 0],
                        [np.inf, xi_OZ_max, np.inf, xi_DB_max, np.inf, n_max, np.inf]
                    )
                )
                xi_OZ_fit, xi_DB_fit, n_fit = popt[1], popt[3], popt[5]
                if (xi_OZ_min <= xi_OZ_fit <= xi_OZ_max and 
                    xi_DB_min <= xi_DB_fit <= xi_DB_max and 
                    n_min <= n_fit <= n_max):
                    return popt
                else:
                    params_now[1] = (params_now[1] + (xi_OZ_min + xi_OZ_max)/2) / 2
                    params_now[3] = (params_now[3] + (xi_DB_min + xi_DB_max)/2) / 2
                    params_now[5] = (params_now[5] + (n_min + n_max)/2) / 2
                    attempt += 1
            except Exception:
                attempt += 1
                params_now[1] = p0[1]
                params_now[3] = p0[3]
                params_now[5] = p0[5]
        return popt

    def fit_selected_files(self):
        selected_files = [f for f, v in self.file_vars.items() if v.get() == 1]
        if not selected_files:
            messagebox.showwarning("Warning", "No files selected for fitting")
            return
        self.run_fitting(files=selected_files)

    def run_fitting(self, files):
        try:
            qmin = float(self.qmin_var.get())
            qmax = float(self.qmax_var.get())
            if qmin >= qmax:
                raise ValueError("q min must be less than q max")
            OZ0_init = float(self.OZ0_var.get())
            xi_OZ_init = float(self.xi_OZ_var.get())
            DB0_init = float(self.DB0_var.get())
            xi_DB_init = float(self.xi_DB_var.get())
            A_init = float(self.A_var.get())
            n_init = float(self.n_var.get())
            bg_init = float(self.bg_var.get())
            
            xi_OZ_min = float(self.xi_OZ_min_var.get())
            xi_OZ_max = float(self.xi_OZ_max_var.get())
            xi_DB_min = float(self.xi_DB_min_var.get())
            xi_DB_max = float(self.xi_DB_max_var.get())
            n_min = float(self.n_min_var.get())
            n_max = float(self.n_max_var.get())
            
            if (xi_OZ_min >= xi_OZ_max or xi_DB_min >= xi_DB_max or n_min >= n_max or 
                xi_OZ_min <= 0 or xi_DB_min <= 0 or n_min <= 0):
                raise ValueError("Invalid parameter ranges")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input parameters: {e}")
            return

        self.qmin_global = qmin
        self.qmax_global = qmax
        self.fit_results.clear()
        self.result_tree.delete(*self.result_tree.get_children())
        self.ax_data.clear()

        colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
        for idx, filename in enumerate(files):
            q, intensity = self.datasets[filename]
            mask = (q >= qmin) & (q <= qmax)
            q_fit = q[mask]
            intensity_fit = intensity[mask]
            if len(q_fit) < 15:
                messagebox.showwarning("Warning", f"Not enough points in range for {filename}")
                continue
            p0 = [OZ0_init, xi_OZ_init, DB0_init, xi_DB_init, A_init, n_init, bg_init]
            popt = self.fit_single_with_bounds(
                q_fit, intensity_fit, p0,
                xi_OZ_min, xi_OZ_max, xi_DB_min, xi_DB_max, n_min, n_max)
            if popt is None:
                messagebox.showerror("Error", f"Fitting failed for {filename}.")
                continue
            OZ0_fit, xi_OZ_fit, DB0_fit, xi_DB_fit, A_fit, n_fit, bg_fit = popt
            y_pred = self.full_model(q_fit, *popt)
            ss_res = np.sum((intensity_fit - y_pred) ** 2)
            ss_tot = np.sum((intensity_fit - np.mean(intensity_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            self.fit_results[filename] = {
                'OZ0': OZ0_fit, 'xi_OZ': xi_OZ_fit, 'DB0': DB0_fit,
                'xi_DB': xi_DB_fit, 'A': A_fit, 'n': n_fit, 'bg': bg_fit, 'r2': r2,
                'q_fit': q_fit, 'intensity_fit': intensity_fit,
                'popt': popt
            }
            color = colors[idx % len(colors)]
            # 전체 데이터 (연한 색)
            self.ax_data.loglog(q, intensity, 'o', color=color, markersize=3, alpha=0.3, zorder=1, label=f"{filename} (data)")
            # 피팅 구간 데이터 (진한 색)
            self.ax_data.loglog(q_fit, intensity_fit, 'o', color=color, markersize=4, alpha=0.8, zorder=2)
            # 피팅곡선 (가장 진하게)
            q_smooth = np.logspace(np.log10(qmin), np.log10(qmax), 300)
            fit_curve = self.full_model(q_smooth, *popt)
            self.ax_data.plot(q_smooth, fit_curve, '-', color=color, linewidth=3, alpha=0.95, zorder=10, 
                            label=f"{filename} fit (R²={r2:.3f})")
        
        self.ax_data.set_xlabel('q (nm⁻¹)', fontsize=14)
        self.ax_data.set_ylabel('Intensity (a.u.)', fontsize=14)
        self.ax_data.set_title(f'Fitting Results (q ∈ [{qmin:.4f}, {qmax:.4f}] nm⁻¹)', fontsize=15)
        self.ax_data.grid(True, which='both', ls='--', alpha=0.5)
        self.ax_data.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, ncol=1, frameon=True, fancybox=True)
        self.update_result_table()
        self.update_bar_chart()
        self.fig.tight_layout()
        self.canvas.draw()

    def update_result_table(self):
        self.result_tree.delete(*self.result_tree.get_children())
        for filename, res in self.fit_results.items():
            self.result_tree.insert('', tk.END, values=(
                filename[:12]+"..." if len(filename) > 15 else filename,
                f"{res['OZ0']:.2e}",
                f"{res['xi_OZ']:.2f}",
                f"{res['DB0']:.2e}",
                f"{res['xi_DB']:.2f}",
                f"{res['A']:.2e}",
                f"{res['n']:.2f}",
                f"{res['bg']:.2e}",
                f"{res['r2']:.3f}"
            ))

    def update_bar_chart(self):
        self.ax_bar.clear()
        if not self.fit_results:
            self.canvas.draw()
            return
        files = list(self.fit_results.keys())
        xi_ozs = [self.fit_results[f]['xi_OZ'] for f in files]
        xi_dbs = [self.fit_results[f]['xi_DB'] for f in files]
        ns = [self.fit_results[f]['n'] for f in files]
        
        w = 0.25
        ind = np.arange(len(files))
        self.ax_bar.bar(ind, xi_ozs, width=w, label='ξ_OZ (nm)', color='tab:blue')
        self.ax_bar.bar(ind + w, xi_dbs, width=w, label='ξ_DB (nm)', color='tab:orange')
        self.ax_bar.bar(ind + 2*w, ns, width=w, label='n (exponent)', color='tab:green')
        self.ax_bar.set_xticks(ind + w)
        self.ax_bar.set_xticklabels([f[:10]+"..." if len(f) > 12 else f for f in files], rotation=45, ha='right')
        self.ax_bar.set_ylabel('Parameter values')
        self.ax_bar.set_title('Key Parameters Comparison')
        self.ax_bar.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def save_results(self):
        if not self.fit_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        folder_path = filedialog.askdirectory(title="Select folder to save results")
        if not folder_path:
            return
        try:
            params_data = []
            for filename, res in self.fit_results.items():
                params_data.append({
                    'Filename': filename,
                    'I_OZ(0)': res['OZ0'],
                    'xi_OZ_nm': res['xi_OZ'],
                    'I_DB(0)': res['DB0'],
                    'xi_DB_nm': res['xi_DB'],
                    'A_power_law': res['A'],
                    'n_exponent': res['n'],
                    'Background': res['bg'],
                    'R_squared': res['r2'],
                    'q_min': self.qmin_global,
                    'q_max': self.qmax_global
                })
            params_df = pd.DataFrame(params_data)
            params_file = os.path.join(folder_path, 'fitting_parameters.csv')
            params_df.to_csv(params_file, index=False)

            for filename, res in self.fit_results.items():
                q_fit = res['q_fit']
                popt = res['popt']
                intensity_fit = res['intensity_fit']
                fit_curve = self.full_model(q_fit, *popt)
                
                # 전체 피팅곡선+실험점
                curve_data = pd.DataFrame({
                    'q_nm-1': q_fit,
                    'intensity_exp': intensity_fit,
                    'intensity_fit': fit_curve,
                    'residuals': intensity_fit - fit_curve
                })
                curve_file = os.path.join(folder_path, f'fit_curve_{filename[:-4]}.csv')
                curve_data.to_csv(curve_file, index=False)
                
                # 각 항 곡선만 저장 (Origin용)
                oz_only = self.oz_term(q_fit, popt[0], popt[1])
                db_only = self.db_term(q_fit, popt[2], popt[3])
                power_only = self.power_law_term(q_fit, popt[4], popt[5])
                bg_only = np.full_like(q_fit, popt[6])
                
                pd.DataFrame({'q_nm-1': q_fit, 'intensity_curve': oz_only}).to_csv(
                    os.path.join(folder_path, f'oz_curve_{filename[:-4]}.csv'), index=False)
                pd.DataFrame({'q_nm-1': q_fit, 'intensity_curve': db_only}).to_csv(
                    os.path.join(folder_path, f'db_curve_{filename[:-4]}.csv'), index=False)
                pd.DataFrame({'q_nm-1': q_fit, 'intensity_curve': power_only}).to_csv(
                    os.path.join(folder_path, f'power_curve_{filename[:-4]}.csv'), index=False)
                pd.DataFrame({'q_nm-1': q_fit, 'intensity_curve': bg_only}).to_csv(
                    os.path.join(folder_path, f'bg_curve_{filename[:-4]}.csv'), index=False)
                    
            plot_file = os.path.join(folder_path, 'all_fits_plot.png')
            self.fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success",
                f"Results saved to {folder_path}:\n"
                "- fitting_parameters.csv\n"
                "- fit_curve_[filename].csv (exp,fit,residual)\n"
                "- oz_curve_[filename].csv\n"
                "- db_curve_[filename].csv\n"
                "- power_curve_[filename].csv\n"
                "- bg_curve_[filename].csv\n"
                "- all_fits_plot.png")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_data and event.button == 1:
            self.drag_start = event.xdata
    def on_mouse_motion(self, event):
        if self.drag_start is not None and event.inaxes == self.ax_data and event.xdata is not None:
            if self.selection_span is not None: self.selection_span.remove()
            qmin = min(self.drag_start, event.xdata)
            qmax = max(self.drag_start, event.xdata)
            self.selection_span = self.ax_data.axvspan(qmin, qmax, alpha=0.2, color='yellow')
            self.canvas.draw_idle()
    def on_mouse_release(self, event):
        if self.drag_start is not None and event.inaxes == self.ax_data and event.button == 1 and event.xdata is not None:
            drag_end = event.xdata
            qmin = min(self.drag_start, drag_end)
            qmax = max(self.drag_start, drag_end)
            if abs(qmax - qmin) > 1e-6:
                self.qmin_var.set(f"{qmin:.4f}")
                self.qmax_var.set(f"{qmax:.4f}")
            if self.selection_span is not None:
                self.selection_span.remove()
                self.selection_span = None
            self.drag_start = None
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SAXSFittingApp(root)
    root.mainloop()
