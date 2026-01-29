import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import itertools

class AdvancedBroadPeakFitter:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced SAXS Broad Peak Fitter (Angstrom Units) - Copyright 2025 Sein Chung")
        self.root.geometry("1300x850")

        # Data & State
        self.file_paths = []
        self.current_data = None # (q, I)
        self.current_filename = ""
        self.results = {} 

        # --- Tabs ---
        self.tab_control = ttk.Notebook(root)
        self.tab_main = ttk.Frame(self.tab_control)
        self.tab_guide = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_main, text='Main Analysis')
        self.tab_control.add(self.tab_guide, text='Usage Guide (Help)')
        self.tab_control.pack(expand=1, fill="both")

        # --- Tab 1: Main Analysis Layout ---
        # Left Panel: Controls
        left_panel = tk.Frame(self.tab_main, width=350, bg="#f5f5f5")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 1. File Loader
        tk.Label(left_panel, text="1. Load Data", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w", pady=(5,0))
        tk.Button(left_panel, text="Load .dat Files", command=self.load_files, bg="#e0e0e0", height=2).pack(fill=tk.X, pady=5)
        
        self.listbox = tk.Listbox(left_panel, height=10, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.X, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # 2. Fitting Range
        tk.Label(left_panel, text="2. Fitting Range (A^-1)", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w", pady=(15,0))
        range_frame = tk.Frame(left_panel, bg="#f5f5f5")
        range_frame.pack(fill=tk.X)
        
        tk.Label(range_frame, text="Min:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.entry_qmin = tk.Entry(range_frame, width=8)
        self.entry_qmin.insert(0, "0.008") # Adjusted for A^-1
        self.entry_qmin.pack(side=tk.LEFT, padx=5)
        
        tk.Label(range_frame, text="Max:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.entry_qmax = tk.Entry(range_frame, width=8)
        self.entry_qmax.insert(0, "0.15") # Adjusted for A^-1
        self.entry_qmax.pack(side=tk.LEFT, padx=5)

        # 3. Fitting Options
        tk.Label(left_panel, text="3. Fitting Strategy", font=("Arial", 11, "bold"), bg="#f5f5f5").pack(anchor="w", pady=(15,0))
        self.auto_iter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(left_panel, text="Auto-Iteration (Find Best R²)", variable=self.auto_iter_var, bg="#f5f5f5", font=("Arial", 10)).pack(anchor="w")

        # Manual Params (Still visible for reference or manual mode)
        param_frame = tk.LabelFrame(left_panel, text="Initial Guess (Manual Mode)", bg="#f5f5f5")
        param_frame.pack(fill=tk.X, pady=5)
        
        self.entries = {}
        # Default Guesses adapted for A^-1 scale
        # q_max ~ 0.04 A^-1, A0, A1 scales might change depending on data unit
        labels = ['A0', 'n', 'A1', 'L', 'q_max', 'm', 'A2']
        defaults = ['1e-5', '3.0', '1e-5', '50.0', '0.04', '2.0', '0.0']
        
        for i, (lbl, val) in enumerate(zip(labels, defaults)):
            tk.Label(param_frame, text=lbl, bg="#f5f5f5").grid(row=i//2, column=(i%2)*2, sticky="e", padx=2)
            ent = tk.Entry(param_frame, width=8)
            ent.insert(0, val)
            ent.grid(row=i//2, column=(i%2)*2+1, padx=2, pady=2)
            self.entries[lbl] = ent

        # Run Button
        tk.Button(left_panel, text="RUN FITTING", command=self.run_fitting, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), height=2).pack(fill=tk.X, pady=15)
        
        # Result Text
        self.txt_result = tk.Text(left_panel, height=10, width=40, font=("Courier New", 9))
        self.txt_result.pack(fill=tk.X)
        
        tk.Button(left_panel, text="Save Results (CSV)", command=self.save_results, bg="#2196F3", fg="white").pack(fill=tk.X, pady=5)

        # Right Panel: Plot
        right_panel = tk.Frame(self.tab_main)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Tab 2: Usage Guide ---
        self.create_guide_tab()

    def create_guide_tab(self):
        guide_text = """
        === Advanced SAXS Broad Peak Fitter User Guide ===
        
        [ 모델 수식 ]
        I(q) = A0/q^n + A1 / (1 + (L * |q - q_max|)^m) + A2
        
        * 단위 주의: 이 프로그램은 x축 단위가 [Angstrom^-1] 인 데이터를 가정합니다.
        * n: Low-q exponent (프랙탈 차원 관련)
        * m: High-q exponent (상호작용 스케일)
        * L: Correlation length (Angstrom 단위)
        
        [ 사용 방법 ]
        
        1. 데이터 로드 (Load Data)
           - '.dat' 파일을 불러옵니다. (1열: q [A^-1], 2열: Intensity)
           - 데이터 포맷은 공백 또는 콤마로 구분된 텍스트 파일이어야 합니다.
           
        2. 피팅 범위 설정 (Fitting Range)
           - Broad Peak가 잘 보이는 q 영역을 지정합니다.
           - 기본값: 0.008 ~ 0.15 A^-1 (PEDOT:PSS 일반적 범위)
           - 노이즈가 심한 영역은 제외하는 것이 좋습니다.
           
        3. 피팅 실행 (Run Fitting)
           - 'Auto-Iteration' 체크 시 (추천):
             프로그램이 q_max, n, m, L의 다양한 초기값 조합을 자동으로 시도합니다.
             가장 높은 R^2 (결정계수)를 가진 결과를 자동으로 선택하여 보여줍니다.
             초기값에 민감한 비선형 피팅의 단점을 보완해줍니다.
           - 체크 해제 시:
             'Initial Guess' 패널에 입력된 단일 초기값만 사용하여 1회 피팅합니다.
             
        4. 결과 확인 및 저장
           - 파란색 점: 원본 데이터, 빨간색 선: 피팅 곡선
           - 텍스트 창에 추출된 파라미터(n, m, L 등)와 R^2 값이 표시됩니다.
           - 'Save Results' 버튼으로 모든 파일의 분석 결과를 CSV로 저장합니다.
           
        (Copyright 2025 Sein Chung)
        """
        lbl = tk.Label(self.tab_guide, text=guide_text, justify=tk.LEFT, font=("Courier", 11), padx=20, pady=20, bg="white", relief="sunken")
        lbl.pack(fill="both", expand=True)

    def empirical_model(self, q, A0, n, A1, L, q_max, m, A2):
        # Prevent singular or complex values
        # q > 0 assumed due to range filter
        term1 = A0 / (q**n)
        term2 = A1 / (1 + (L * np.abs(q - q_max))**m)
        return term1 + term2 + A2

    def parse_dat_file(self, filepath):
        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python', on_bad_lines='skip')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] < 2:
                df = pd.read_csv(filepath, sep=',', comment='#', header=None, engine='python', on_bad_lines='skip')
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                if df.shape[1] < 2: return None, None
            return df.iloc[:, 0].values, df.iloc[:, 1].values
        except:
            return None, None

    def load_files(self):
        files = filedialog.askopenfilenames(title="Select .dat files", filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if files:
            self.file_paths = list(files)
            self.listbox.delete(0, tk.END)
            self.results = {}
            for f in self.file_paths:
                self.listbox.insert(tk.END, os.path.basename(f))
            self.listbox.selection_set(0)
            self.on_file_select(None)

    def on_file_select(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        
        index = selection[0]
        filepath = self.file_paths[index]
        self.current_filename = os.path.basename(filepath)
        
        q, I = self.parse_dat_file(filepath)
        if q is not None:
            self.current_data = (q, I)
            # Just plot raw data initially
            self.plot_data(q, I)

    def plot_data(self, q, I, fit_I=None, fit_label="Best Fit"):
        self.ax.clear()
        self.ax.loglog(q, I, 'ko', markersize=3, alpha=0.4, label='Data')
        
        if fit_I is not None:
            self.ax.loglog(q, fit_I, 'r-', linewidth=2.5, label=fit_label)
            
        self.ax.set_title(f"Data: {self.current_filename}")
        self.ax.set_xlabel("q ($\AA^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(True, which="both", alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

    def run_fitting(self):
        if self.current_data is None: return
        
        q_raw, I_raw = self.current_data
        
        # 1. Range Filter
        try:
            q_min = float(self.entry_qmin.get())
            q_max = float(self.entry_qmax.get())
        except: return

        mask = (q_raw >= q_min) & (q_raw <= q_max)
        q_fit = q_raw[mask]
        I_fit = I_raw[mask]
        
        if len(q_fit) < 10:
            messagebox.showerror("Error", "Not enough points in fitting range.")
            return

        # 2. Setup Guess List
        guess_list = []
        
        if self.auto_iter_var.get():
            # Generate Grid of Initial Guesses
            # Core idea: vary q_max (peak pos), n (low slope), L (width)
            # A0, A1 usually depend on intensity scale, estimatable from data
            
            # Estimate scale
            i_mid = np.median(I_fit)
            
            # Grids
            q_peaks = np.linspace(q_min * 1.5, q_max * 0.8, 3) # 3 positions
            ns = [2.0, 3.0, 4.0]
            Ls = [10, 50, 100]
            
            for qp, n_val, L_val in itertools.product(q_peaks, ns, Ls):
                # Rough guesses for A0, A1 based on scale
                guess = [i_mid/10, n_val, i_mid, L_val, qp, 2.0, 0.0]
                guess_list.append(guess)
        else:
            # Manual Mode: Single guess from UI
            try:
                p0 = [float(self.entries[k].get()) for k in ['A0', 'n', 'A1', 'L', 'q_max', 'm', 'A2']]
                guess_list.append(p0)
            except:
                messagebox.showerror("Error", "Invalid manual parameters.")
                return

        # 3. Iterative Fitting
        best_r2 = -np.inf
        best_popt = None
        
        # Bounds: A0, n, A1, L, q_max, m, A2
        # q_max should be within range ideally
        lower = [0, 0, 0, 0, q_min, 0, 0]
        upper = [np.inf, 6, np.inf, np.inf, q_max, 10, np.inf]
        
        for p0 in guess_list:
            try:
                popt, _ = curve_fit(self.empirical_model, q_fit, I_fit, p0=p0, bounds=(lower, upper), maxfev=2000)
                
                # Eval R2
                I_pred = self.empirical_model(q_fit, *popt)
                r2 = r2_score(I_fit, I_pred) # Note: r2_score on log data might be better for log-log plots but standard R2 on linear is acceptable
                
                # Check log-log R2 for better visual match metric
                r2_log = r2_score(np.log10(I_fit), np.log10(I_pred))
                
                if r2_log > best_r2:
                    best_r2 = r2_log
                    best_popt = popt
                    
            except RuntimeError:
                continue # Skip failed fits

        # 4. Finalize
        if best_popt is not None:
            A0, n, A1, L, q_peak, m, A2 = best_popt
            
            # Display Results
            res_str = (f"=== Best Fit (R²_log: {best_r2:.4f}) ===\n"
                       f"n  (Low-q) : {n:.4f}\n"
                       f"m  (High-q): {m:.4f}\n"
                       f"L  (Corr.) : {L:.4f} A\n"
                       f"q_max      : {q_peak:.4f} A^-1\n"
                       f"A0: {A0:.2e}, A1: {A1:.2e}, A2: {A2:.2e}")
            
            self.txt_result.delete(1.0, tk.END)
            self.txt_result.insert(tk.END, res_str)
            
            # Plot
            fit_curve = self.empirical_model(q_fit, *best_popt)
            self.plot_data(q_fit, I_fit, fit_curve, f"Fit (R²={best_r2:.3f})")
            
            # Save
            self.results[self.current_filename] = {
                'R2_log': best_r2,
                'n': n, 'm': m, 'L_Angstrom': L, 'q_max_A-1': q_peak,
                'A0': A0, 'A1': A1, 'A2': A2
            }
        else:
            messagebox.showwarning("Fit Failed", "Could not converge with any initial guess.")

    def save_results(self):
        if not self.results:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if save_path:
            data = []
            for fname, val in self.results.items():
                row = {'Filename': fname}
                row.update(val)
                data.append(row)
            pd.DataFrame(data).to_csv(save_path, index=False)
            messagebox.showinfo("Saved", "Results saved.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBroadPeakFitter(root)
    root.mainloop()
