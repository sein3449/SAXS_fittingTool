import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.stats import linregress
import os

class SAXSAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAXS Decay Exponent Extractor - Copyright 2025 Sein Chung")
        self.root.geometry("1200x800")

        # Data storage
        self.file_paths = []
        self.current_data = None
        # results 구조: {filename: {'exponent': val, 'r_squared': val, 'q_range': str}}
        self.results = {}  
        self.current_file_index = -1

        # --- Layout ---
        left_panel = tk.Frame(root, width=300, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Buttons
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="1. Load .dat Files", command=self.load_files, height=2, bg="#e1e1e1").pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="2. Save Results (CSV)", command=self.save_results, height=2, bg="#d1e7dd").pack(fill=tk.X, pady=2)

        # File Listbox
        tk.Label(left_panel, text="File List:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
        self.listbox = tk.Listbox(left_panel, selectmode=tk.SINGLE, height=30)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Result Display
        self.lbl_result = tk.Label(left_panel, text="Select a region to fit", font=("Arial", 12), fg="blue", justify=tk.LEFT)
        self.lbl_result.pack(pady=10)

        # Copyright Label
        tk.Label(left_panel, text="Copyright 2025 Sein Chung", font=("Arial", 9), fg="gray").pack(side=tk.BOTTOM, pady=5)

        # Right Panel: Plot
        right_panel = tk.Frame(root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.span = SpanSelector(
            self.ax, self.on_select, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='red'), interactive=True, drag_from_anywhere=True
        )

    def load_files(self):
        files = filedialog.askopenfilenames(title="Select SAXS .dat files", filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if files:
            self.file_paths = list(files)
            self.listbox.delete(0, tk.END)
            self.results = {}
            
            for f in self.file_paths:
                filename = os.path.basename(f)
                self.listbox.insert(tk.END, filename)
                self.results[filename] = {'exponent': np.nan, 'r_squared': np.nan, 'q_range': "Not selected"}
            
            self.listbox.selection_set(0)
            self.on_file_select(None)

    def parse_dat_file(self, filepath):
        """
        데이터 파싱: 전체를 읽고 앞의 2개 열만 추출
        """
        try:
            # 1. 공백 구분자로 읽기 시도
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python', on_bad_lines='skip')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            # 2. 컬럼 부족 시 콤마 구분자로 재시도
            if df.shape[1] < 2:
                df = pd.read_csv(filepath, sep=',', comment='#', header=None, engine='python', on_bad_lines='skip')
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                if df.shape[1] < 2:
                    raise ValueError(f"Found only {df.shape[1]} columns. Need at least 2 (q, I). Check delimiters.")

            # 3. 첫 2개 컬럼만 추출 (q, I)
            q = df.iloc[:, 0].values
            I = df.iloc[:, 1].values
            return q, I
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read {os.path.basename(filepath)}\n{e}")
            return None, None

    def on_file_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return

        index = selection[0]
        self.current_file_index = index
        filepath = self.file_paths[index]
        filename = os.path.basename(filepath)

        q, I = self.parse_dat_file(filepath)
        if q is not None:
            self.current_data = (q, I)
            self.plot_data(q, I, filename)

    def plot_data(self, q, I, title):
        self.ax.clear()
        self.ax.loglog(q, I, 'o-', markersize=2, label='Data', color='black', alpha=0.6)
        
        filename = os.path.basename(self.file_paths[self.current_file_index])
        if filename in self.results and not np.isnan(self.results[filename]['exponent']):
            res = self.results[filename]
            self.lbl_result.config(text=f"File: {filename}\nDecay Exponent: {res['exponent']:.3f}\nR²: {res['r_squared']:.4f}")
        else:
            self.lbl_result.config(text="Select a region to fit (Drag on graph)", fg="blue")

        self.ax.set_title(f"SAXS Data: {title}")
        # 단위 변경 반영: Angstrom^-1
        self.ax.set_xlabel("q ($\AA^{-1}$)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # 가이드라인 수정:
        # 1 nm^-1 = 0.1 A^-1 이므로
        # 0.7 ~ 1.1 nm^-1 -> 0.07 ~ 0.11 A^-1
        target_min = 0.07
        target_max = 0.11
        
        # 만약 데이터 자체가 이미 nm^-1 스케일이 아니고, 사용자가 숫자 0.7~1.1을 그대로 A^-1 영역으로 보고 싶다면
        # 아래 두 줄의 주석을 해제하고 위 두 줄을 주석 처리하세요.
        # target_min = 0.7
        # target_max = 1.1

        self.ax.axvspan(target_min, target_max, color='yellow', alpha=0.1, label=f'Target Range ({target_min}-{target_max})')
        self.ax.legend()
        self.canvas.draw()

    def on_select(self, q_min, q_max):
        if self.current_data is None:
            return

        q, I = self.current_data
        mask = (q >= q_min) & (q <= q_max)
        q_sub = q[mask]
        I_sub = I[mask]

        if len(q_sub) < 3:
            self.lbl_result.config(text="Not enough points selected!")
            return

        try:
            valid_idx = (q_sub > 0) & (I_sub > 0)
            q_fit = q_sub[valid_idx]
            I_fit = I_sub[valid_idx]

            if len(q_fit) < 3:
                self.lbl_result.config(text="Invalid data in selection")
                return

            log_q = np.log10(q_fit)
            log_I = np.log10(I_fit)
            
            slope, intercept, r_value, p_value, std_err = linregress(log_q, log_I)
            decay_exponent = -slope
            r_squared = r_value**2

            filename = os.path.basename(self.file_paths[self.current_file_index])
            self.results[filename] = {
                'exponent': decay_exponent,
                'r_squared': r_squared,
                'q_range': f"{q_min:.4f} - {q_max:.4f}"
            }

            self.lbl_result.config(
                text=f"Selected Range: {q_min:.4f} - {q_max:.4f} $\AA^{{-1}}$\n"
                     f"Decay Exponent (-Slope): {decay_exponent:.4f}\n"
                     f"R²: {r_squared:.4f}",
                fg="red"
            )

            fit_I = (q_fit**slope) * (10**intercept)
            for line in self.ax.get_lines():
                if line.get_label() == 'Fit':
                    line.remove()
            
            self.ax.plot(q_fit, fit_I, 'r-', linewidth=2, label='Fit')
            self.ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Fitting error: {e}")

    def save_results(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            data = []
            for filename, res in self.results.items():
                data.append({
                    'Filename': filename,
                    'Decay Exponent': res['exponent'],
                    'R_squared': res['r_squared'],
                    'Q_Range_A-1': res['q_range']  # 컬럼명 단위 수정
                })
            
            df = pd.DataFrame(data)
            try:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Results saved to:\n{save_path}\n\n(Copyright 2025 Sein Chung)")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SAXSAnalyzerGUI(root)
    root.mainloop()
