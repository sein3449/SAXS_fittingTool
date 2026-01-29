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
        self.master.title("SAXS Ornstein-Zernike Fitting with Selective Fit - Copyright 2025 POSTECH Chem. Eng. Sein Chung")
        self.master.geometry("1400x950")

        self.datasets = {}  # filename -> (q, intensity)
        self.fit_results = {}
        self.qmin_global = None
        self.qmax_global = None
        self.drag_start = None
        self.selection_span = None
        self.max_fit_attempts = 15

        # 체크박스 상태 저장 dict 파일명 -> tk.IntVar()
        self.file_vars = {}

        self.build_gui()

    def build_gui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        copyright_label = ttk.Label(left_frame, text="© 2025 S. Chung", 
                           font=('Arial', 7), foreground='darkgray')
        copyright_label.pack(side=tk.BOTTOM, pady=5)

        ttk.Button(left_frame, text="Load CSV Files", command=self.load_files).pack(fill=tk.X, pady=5)

        # 체크박스 스크롤 가능 영역
        self.check_canvas = tk.Canvas(left_frame, width=350, height=200)
        self.check_canvas.pack(fill=tk.Y)
        self.check_frame = ttk.Frame(self.check_canvas)
        self.check_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.check_canvas.yview)
        self.check_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.check_canvas.configure(yscrollcommand=self.check_scrollbar.set)
        self.check_canvas.create_window((0,0), window=self.check_frame, anchor="nw")
        self.check_frame.bind("<Configure>", lambda e: self.check_canvas.configure(scrollregion=self.check_canvas.bbox("all")))
        
        # 전체 선택/해제 버튼
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # q 범위
        ttk.Label(left_frame, text="Global q range (drag in plot or type):").pack(pady=5)
        qrange_frame = ttk.Frame(left_frame)
        qrange_frame.pack(fill=tk.X)
        ttk.Label(qrange_frame, text="q min:").grid(row=0, column=0, sticky=tk.W)
        self.qmin_var = tk.StringVar(value="0.01")
        ttk.Entry(qrange_frame, textvariable=self.qmin_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(qrange_frame, text="q max:").grid(row=1, column=0, sticky=tk.W)
        self.qmax_var = tk.StringVar(value="0.1")
        ttk.Entry(qrange_frame, textvariable=self.qmax_var, width=12).grid(row=1, column=1, padx=5)

        # 파라미터
        ttk.Label(left_frame, text="Fitting parameters:").pack(pady=(10,5))
        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill=tk.X)
        ttk.Label(param_frame, text="I_L(0):").grid(row=0, column=0, sticky=tk.W)
        self.IL0_var = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.IL0_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="ξ (nm):").grid(row=1, column=0, sticky=tk.W)
        self.xi_var = tk.StringVar(value="3.0")
        ttk.Entry(param_frame, textvariable=self.xi_var, width=12).grid(row=1, column=1, padx=5)
        ttk.Label(param_frame, text="A (power):").grid(row=2, column=0, sticky=tk.W)
        self.A_var = tk.StringVar(value="1e-5")
        ttk.Entry(param_frame, textvariable=self.A_var, width=12).grid(row=2, column=1, padx=5)
        ttk.Label(param_frame, text="n (exp):").grid(row=3, column=0, sticky=tk.W)
        self.n_var = tk.StringVar(value="2.0")
        ttk.Entry(param_frame, textvariable=self.n_var, width=12).grid(row=3, column=1, padx=5)
        ttk.Label(param_frame, text="Background:").grid(row=4, column=0, sticky=tk.W)
        self.bg_var = tk.StringVar(value="0.001")
        ttk.Entry(param_frame, textvariable=self.bg_var, width=12).grid(row=4, column=1, padx=5)

        # ξ 제한
        ttk.Label(left_frame, text="Correlation length ξ limits:").pack(pady=(10,5))
        limit_frame = ttk.Frame(left_frame)
        limit_frame.pack(fill=tk.X)
        ttk.Label(limit_frame, text="ξ min:").grid(row=0, column=0, sticky=tk.W)
        self.xi_min_var = tk.StringVar(value="0.5")
        ttk.Entry(limit_frame, textvariable=self.xi_min_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(limit_frame, text="ξ max:").grid(row=1, column=0, sticky=tk.W)
        self.xi_max_var = tk.StringVar(value="10")
        ttk.Entry(limit_frame, textvariable=self.xi_max_var, width=12).grid(row=1, column=1, padx=5)

        # 버튼
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=15)
        ttk.Button(btn_frame, text="Fit Selected Files", command=self.fit_selected_files).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Fit All Files", command=self.fit_all_files).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Save Results & Curves", command=self.save_results).pack(fill=tk.X, pady=3)

        # 결과 테이블
        result_frame = ttk.LabelFrame(left_frame, text="Fit Results")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("File", "I_L(0)", "ξ", "A", "n", "BG", "R²")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=10)
        col_widths = [60, 70, 60, 70, 50, 60, 50]
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

    def select_all(self):
        for var in self.file_vars.values():
            var.set(1)

    def deselect_all(self):
        for var in self.file_vars.values():
            var.set(0)

    def load_files(self):
        files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if not files:
            return
        self.datasets.clear()
        self.file_vars.clear()
        for widget in self.check_frame.winfo_children():
            widget.destroy()
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
                var = tk.IntVar(value=1)
                cb = tk.Checkbutton(self.check_frame, text=filename, variable=var)
                cb.pack(anchor='w')
                self.file_vars[filename] = var
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file}: {e}")
        messagebox.showinfo("Success", f"Loaded {len(self.datasets)} files")
        self.update_bar_chart()

    def ornstein_zernike_model(self, q, IL0, xi, A, n, bg):
        return IL0 / (1 + (q * xi)**2) + A * q**(-n) + bg

    def fit_single_with_xi_bounds(self, q_fit, intensity_fit, p0, xi_min, xi_max):
        attempt = 0
        params_now = p0[:]
        while attempt < self.max_fit_attempts:
            try:
                popt, _ = curve_fit(
                    self.ornstein_zernike_model,
                    q_fit, intensity_fit,
                    p0=params_now,
                    maxfev=10000,
                    bounds=([0, xi_min, 0, 0, 0], [np.inf, xi_max, np.inf, 10, np.inf])
                )
                xi_fit = popt[1]
                if xi_min <= xi_fit <= xi_max:
                    return popt
                else:
                    xi_mid = (xi_min + xi_max) / 2
                    params_now[1] = (params_now[1] + xi_mid) / 2
                    attempt += 1
            except Exception:
                attempt += 1
                params_now[1] = p0[1]
        return popt if 'popt' in locals() else None

    def fit_selected_files(self):
        if not self.datasets:
            messagebox.showwarning("Warning", "No data loaded")
            return
        selected_files = [f for f, v in self.file_vars.items() if v.get() == 1]
        if not selected_files:
            messagebox.showwarning("Warning", "No files selected for fitting")
            return
        self.run_fitting(files=selected_files)

    def fit_all_files(self):
        if not self.datasets:
            messagebox.showwarning("Warning", "No data loaded")
            return
        self.run_fitting(files=list(self.datasets.keys()))

    def run_fitting(self, files):
        try:
            qmin = float(self.qmin_var.get())
            qmax = float(self.qmax_var.get())
            if qmin >= qmax:
                raise ValueError("q min must be less than q max")

            xi_min = float(self.xi_min_var.get())
            xi_max = float(self.xi_max_var.get())
            if xi_min >= xi_max or xi_min <= 0:
                raise ValueError("Invalid ξ range")

            IL0_init = float(self.IL0_var.get())
            xi_init = float(self.xi_var.get())
            A_init = float(self.A_var.get())
            n_init = float(self.n_var.get())
            bg_init = float(self.bg_var.get())
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
            if len(q_fit) < 10:
                messagebox.showwarning("Warning", f"Not enough points in range for {filename}")
                continue
            p0 = [IL0_init, xi_init, A_init, n_init, bg_init]
            popt = self.fit_single_with_xi_bounds(q_fit, intensity_fit, p0, xi_min, xi_max)
            if popt is None:
                messagebox.showerror("Error", f"Fitting failed for {filename}.")
                continue
            IL0_fit, xi_fit, A_fit, n_fit, bg_fit = popt
            y_pred = self.ornstein_zernike_model(q_fit, *popt)
            ss_res = np.sum((intensity_fit - y_pred) ** 2)
            ss_tot = np.sum((intensity_fit - np.mean(intensity_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            self.fit_results[filename] = {
                'IL0': IL0_fit, 'xi': xi_fit, 'A': A_fit,
                'n': n_fit, 'bg': bg_fit, 'r2': r2,
                'q_fit': q_fit, 'intensity_fit': intensity_fit,
                'popt': popt
            }
            color = colors[idx]
            self.ax_data.loglog(q, intensity, 'o', color=color, markersize=3, alpha=0.4, label=f'{filename} (data)')
            q_smooth = np.logspace(np.log10(qmin), np.log10(qmax), 200)
            fit_curve = self.ornstein_zernike_model(q_smooth, *popt)
            self.ax_data.loglog(q_smooth, fit_curve, '-', color=color, linewidth=2, label=f'{filename} (ξ={xi_fit:.3f})')

        self.ax_data.set_xlabel('q (nm⁻¹)')
        self.ax_data.set_ylabel('Intensity (a.u.)')
        self.ax_data.set_title(f'Fitting Results (q ∈ [{qmin:.4f}, {qmax:.4f}] nm⁻¹)')
        self.ax_data.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
        self.ax_data.grid(True, which='both', ls='--', alpha=0.5)
        self.update_result_table()
        self.update_bar_chart()
        self.fig.tight_layout()
        self.canvas.draw()

    def update_result_table(self):
        self.result_tree.delete(*self.result_tree.get_children())
        for filename, res in self.fit_results.items():
            self.result_tree.insert('', tk.END, values=(
                filename[:15]+"..." if len(filename) > 15 else filename,
                f"{res['IL0']:.3e}",
                f"{res['xi']:.3f}",
                f"{res['A']:.2e}",
                f"{res['n']:.2f}",
                f"{res['bg']:.3e}",
                f"{res['r2']:.4f}"
            ))

    def update_bar_chart(self):
        self.ax_bar.clear()
        if not self.fit_results:
            self.canvas.draw()
            return
        files = list(self.fit_results.keys())
        xis = [self.fit_results[f]['xi'] for f in files]
        colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
        bars = self.ax_bar.bar(range(len(files)), xis, color=colors)
        self.ax_bar.set_xticks(range(len(files)))
        self.ax_bar.set_xticklabels([f[:12] + "..." if len(f) > 12 else f for f in files], rotation=45, ha='right')
        self.ax_bar.set_ylabel('Correlation Length ξ (nm⁻¹)')
        self.ax_bar.set_title('Correlation Length Comparison')
        if xis:
            ymax = max(xis) * 1.2
            self.ax_bar.set_ylim(0, ymax)
            for bar, xi in zip(bars, xis):
                self.ax_bar.text(bar.get_x() + bar.get_width()/2, xi + ymax*0.03, f"{xi:.3f}", ha='center')
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
                    'I_L(0)': res['IL0'],
                    'xi_nm-1': res['xi'],
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
                intensity_fit = res['intensity_fit']
                fit_curve = self.ornstein_zernike_model(q_fit, *res['popt'])
                curve_data = pd.DataFrame({
                    'q_nm-1': q_fit,
                    'intensity_exp': intensity_fit,
                    'intensity_fit': fit_curve,
                    'residuals': intensity_fit - fit_curve
                })
                curve_file = os.path.join(folder_path, f'fit_curve_{filename[:-4]}.csv')
                curve_data.to_csv(curve_file, index=False)
            plot_file = os.path.join(folder_path, 'all_fits_plot.png')
            self.fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success",
                f"Results saved to {folder_path}:\n"
                "- fitting_parameters.csv\n"
                "- fit_curve_[filename].csv per file\n"
                "- all_fits_plot.png")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")

    def on_mouse_press(self, event):
        if event.inaxes == self.ax_data and event.button == 1:
            self.drag_start = event.xdata

    def on_mouse_motion(self, event):
        if self.drag_start is not None and event.inaxes == self.ax_data and event.xdata is not None:
            if self.selection_span is not None:
                self.selection_span.remove()
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
