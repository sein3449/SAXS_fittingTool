import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
import os
import re

class ComprehensiveGelAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Comprehensive Gel SAXS Analyzer - Copyright 2025 Sein Chung")
        self.root.geometry("1450x980")

        # Variables
        self.files_data = [] 
        self.current_idx = -1
        self.current_q_range = None
        
        # --- Top Panel (Settings) ---
        top_frame = tk.Frame(root, bg="#f5f5f5", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Strain Params
        tk.Label(top_frame, text="Ref (0%):", bg="#f5f5f5").pack(side=tk.LEFT, padx=5)
        self.entry_base = tk.Entry(top_frame, width=8)
        self.entry_base.insert(0, "47000")
        self.entry_base.pack(side=tk.LEFT)
        
        tk.Label(top_frame, text="L0:", bg="#f5f5f5").pack(side=tk.LEFT, padx=5)
        self.entry_L0 = tk.Entry(top_frame, width=8)
        self.entry_L0.insert(0, "10000")
        self.entry_L0.pack(side=tk.LEFT)
        
        # Model
        tk.Label(top_frame, text="| Model:", bg="#f5f5f5", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=15)
        self.model_var = tk.StringVar(value="Generalized OZ")
        self.combo_model = ttk.Combobox(top_frame, textvariable=self.model_var, 
                                        values=["Standard OZ (m=2)", "Generalized OZ (Var m)"], 
                                        state="readonly", width=22)
        self.combo_model.pack(side=tk.LEFT, padx=5)
        self.combo_model.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # m Limit
        tk.Label(top_frame, text=" [m Limit] Min:", bg="#f5f5f5").pack(side=tk.LEFT, padx=5)
        self.entry_m_min = tk.Entry(top_frame, width=4)
        self.entry_m_min.insert(0, "1.0")
        self.entry_m_min.pack(side=tk.LEFT)
        tk.Label(top_frame, text="Max:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.entry_m_max = tk.Entry(top_frame, width=4)
        self.entry_m_max.insert(0, "4.5")
        self.entry_m_max.pack(side=tk.LEFT)

        # --- Main View ---
        paned = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Left Panel
        left_panel = tk.Frame(paned, width=450, bg="white")
        paned.add(left_panel)
        
        tk.Button(left_panel, text="1. Load Files", command=self.load_files, bg="#dff9fb", height=2).pack(fill=tk.X, padx=5, pady=5)
        
        # Treeview
        cols = ("Strain", "Filename", "Xi", "m")
        self.tree = ttk.Treeview(left_panel, columns=cols, show='headings', height=15)
        self.tree.heading("Strain", text="Strain %")
        self.tree.heading("Filename", text="File")
        self.tree.heading("Xi", text="ξ (nm)")
        self.tree.heading("m", text="m")
        self.tree.column("Strain", width=60)
        self.tree.column("Filename", width=120)
        self.tree.column("Xi", width=60)
        self.tree.column("m", width=50)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5)
        self.tree.bind('<<TreeviewSelect>>', self.on_file_select)
        
        # Detailed Info Panel (New Feature)
        info_frame = tk.LabelFrame(left_panel, text="Detailed Fitting Info", bg="white", font=("Arial", 10, "bold"), fg="#333")
        info_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.lbl_equation = tk.Label(info_frame, text="Equation: -", bg="white", anchor="w", fg="blue")
        self.lbl_equation.pack(fill=tk.X, padx=5, pady=2)
        
        self.lbl_range = tk.Label(info_frame, text="Fit Range: -", bg="white", anchor="w")
        self.lbl_range.pack(fill=tk.X, padx=5, pady=2)
        
        self.lbl_params = tk.Label(info_frame, text="Params: -", bg="white", anchor="w", justify="left")
        self.lbl_params.pack(fill=tk.X, padx=5, pady=2)

        # Feedback Panel
        fb_frame = tk.LabelFrame(left_panel, text="Range Quality (q_max * ξ)", bg="white", font=("Arial", 10, "bold"), fg="#333")
        fb_frame.pack(fill=tk.X, padx=5, pady=5)
        self.lbl_qxi = tk.Label(fb_frame, text="-", font=("Arial", 11, "bold"), bg="white")
        self.lbl_qxi.pack(pady=5)
        
        self.btn_batch = tk.Button(left_panel, text="2. Apply Range to ALL", command=self.batch_fit, bg="#ffeaa7", state="disabled")
        self.btn_batch.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(left_panel, text="3. Show Graph", command=self.show_graph).pack(fill=tk.X, padx=5, pady=2)
        tk.Button(left_panel, text="Save Detailed CSV", command=self.save_csv, bg="#a5d6a7").pack(fill=tk.X, padx=5, pady=5)

        # Right Panel
        right_panel = tk.Frame(paned, bg="white")
        paned.add(right_panel)

        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right_panel)
        
        self.span = SpanSelector(self.ax, self.on_drag_done, 'horizontal', useblit=True,
                                 props=dict(alpha=0.2, facecolor='green'), interactive=True, drag_from_anywhere=True)

    # --- Models ---
    def standard_oz(self, q, I0, xi, bkg):
        return I0 / (1 + (xi * q)**2) + bkg

    def generalized_oz(self, q, I0, xi, m, bkg):
        return I0 / (1 + (xi * q)**m) + bkg

    # --- File Parsing ---
    def parse_strain(self, filename):
        try:
            base = float(self.entry_base.get())
            L0 = float(self.entry_L0.get())
            numbers = re.findall(r'\d+', filename)
            if not numbers: return 0.0
            cand = [float(n) for n in numbers]
            best = min(cand, key=lambda x: abs(x - base))
            return ((best - base) / L0) * 100.0
        except: return 0.0

    def parse_file_content(self, filepath):
        encodings = ['utf-8', 'cp949', 'latin-1']
        data = []
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    for line in f:
                        line = line.split('#')[0].strip()
                        if not line: continue
                        line = re.sub(r'[,;\t]', ' ', line)
                        parts = line.split()
                        if len(parts) < 2: continue
                        try:
                            data.append([float(parts[0]), float(parts[1])])
                        except: continue
                if len(data) > 5: break
            except: continue
        
        if not data: return None, None
        arr = np.array(data)
        arr = arr[arr[:, 0].argsort()]
        return arr[:, 0], arr[:, 1]

    def load_files(self):
        self.root.update()
        files = filedialog.askopenfilenames(title="Select .dat Files")
        if not files: return
        
        self.files_data = []
        for f in files:
            fname = os.path.basename(f)
            q, I = self.parse_file_content(f)
            if q is not None:
                strain = self.parse_strain(fname)
                self.files_data.append({
                    'filename': fname, 'path': f, 'strain': strain,
                    'q': q, 'I': I, 
                    'xi': 0.0, 'm': 2.0, 'I0': 0.0, 'bkg': 0.0,
                    'fit_qmin': 0.0, 'fit_qmax': 0.0, 'model_used': '-'
                })
        
        if not self.files_data:
            messagebox.showerror("Error", "No valid data loaded.")
            return

        self.files_data.sort(key=lambda x: x['strain'])
        self.update_tree()
        self.current_idx = 0
        self.tree.selection_set(0)
        self.plot_file(0)
        self.btn_batch.config(state="normal")

    def update_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, d in enumerate(self.files_data):
            self.tree.insert("", "end", iid=i, values=(f"{d['strain']:.1f}", d['filename'], "-", "-"))

    def on_file_select(self, event):
        sel = self.tree.selection()
        if sel:
            self.current_idx = int(sel[0])
            self.plot_file(self.current_idx)
            self.update_info_panel(self.files_data[self.current_idx])

    def update_info_panel(self, d):
        """Display fit info on GUI"""
        if d['model_used'] == '-':
            self.lbl_equation.config(text="Equation: Not fitted yet")
            self.lbl_range.config(text="Fit Range: -")
            self.lbl_params.config(text="Params: -")
            self.lbl_qxi.config(text="-", fg="black")
        else:
            eq_text = "I(q) = I0 / (1 + (ξq)^m) + bkg" if "Generalized" in d['model_used'] else "I(q) = I0 / (1 + (ξq)^2) + bkg"
            self.lbl_equation.config(text=f"Equation: {d['model_used']}")
            self.lbl_range.config(text=f"Fit Range: {d['fit_qmin']:.4f} ~ {d['fit_qmax']:.4f} nm^-1")
            
            param_str = f"ξ = {d['xi']:.3f} nm\nm = {d['m']:.3f}\nI0 = {d['I0']:.2e}\nBkg = {d['bkg']:.2e}"
            self.lbl_params.config(text=param_str)
            
            # Feedback
            qxi = d['fit_qmax'] * d['xi']
            self.lbl_qxi.config(text=f"q_max · ξ = {qxi:.2f}")
            if qxi < 1.5: self.lbl_qxi.config(fg="green")
            elif qxi < 2.5: self.lbl_qxi.config(fg="#FBC02D")
            else: self.lbl_qxi.config(fg="red")

    def on_model_change(self, event):
        if "Standard" in self.model_var.get():
            self.entry_m_min.config(state="disabled")
            self.entry_m_max.config(state="disabled")
        else:
            self.entry_m_min.config(state="normal")
            self.entry_m_max.config(state="normal")
            
        if self.current_idx >= 0 and self.current_q_range:
            self.do_fit(self.current_idx, *self.current_q_range, draw=True)

    def plot_file(self, idx, fit_q=None, fit_I=None):
        self.ax.clear()
        d = self.files_data[idx]
        self.ax.loglog(d['q'], d['I'], 'k.', markersize=2, alpha=0.4, label=f"Strain {d['strain']:.1f}%")
        
        if fit_q is not None:
            m_val = d.get('m', 2.0)
            label = f"Fit (ξ={d['xi']:.1f}nm, m={m_val:.2f})"
            self.ax.loglog(fit_q, fit_I, 'r-', linewidth=2.5, label=label)
            
        self.ax.set_title(f"File: {d['filename']}")
        self.ax.set_xlabel("q (nm^-1)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, which="both", alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

    def on_drag_done(self, qmin, qmax):
        self.current_q_range = (qmin, qmax)
        self.do_fit(self.current_idx, qmin, qmax, draw=True)

    def do_fit(self, idx, qmin, qmax, draw=False):
        d = self.files_data[idx]
        mask = (d['q'] >= qmin) & (d['q'] <= qmax)
        q_sub = d['q'][mask]
        I_sub = d['I'][mask]
        
        if len(q_sub) < 5: return

        model_name = self.model_var.get()
        
        try:
            if "Standard" in model_name:
                p0 = [np.max(I_sub), 10.0, np.min(I_sub)]
                popt, _ = curve_fit(self.standard_oz, q_sub, I_sub, p0=p0, bounds=(0, np.inf), maxfev=5000)
                I0, xi, bkg = popt
                m = 2.0
                fit_curve = self.standard_oz(q_sub, *popt)
            else:
                try:
                    m_min = float(self.entry_m_min.get())
                    m_max = float(self.entry_m_max.get())
                except: m_min, m_max = 1.0, 4.5

                m_guess = (m_min + m_max) / 2
                p0 = [np.max(I_sub), 10.0, m_guess, np.min(I_sub)]
                lower = [0, 0, m_min, 0]
                upper = [np.inf, np.inf, m_max, np.inf]
                
                popt, _ = curve_fit(self.generalized_oz, q_sub, I_sub, p0=p0, bounds=(lower, upper), maxfev=5000)
                I0, xi, m, bkg = popt
                fit_curve = self.generalized_oz(q_sub, *popt)

            # Store Results
            d.update({
                'xi': xi, 'm': m, 'I0': I0, 'bkg': bkg,
                'fit_qmin': qmin, 'fit_qmax': qmax, 'model_used': model_name
            })
            
            self.tree.item(idx, values=(f"{d['strain']:.1f}", d['filename'], f"{xi:.2f}", f"{m:.2f}"))
            self.update_info_panel(d)
            
            if draw:
                self.plot_file(idx, q_sub, fit_curve)

        except Exception as e:
            print(f"Fit Fail: {e}")

    def batch_fit(self):
        if not self.current_q_range: return
        for i in range(len(self.files_data)):
            self.do_fit(i, *self.current_q_range)
        self.plot_file(self.current_idx)
        messagebox.showinfo("Done", "Batch fitting complete.")

    def show_graph(self):
        if not self.files_data: return
        strains = [d['strain'] for d in self.files_data]
        xis = [d['xi'] for d in self.files_data]
        ms = [d['m'] for d in self.files_data]
        
        top = tk.Toplevel(self.root)
        top.geometry("600x700")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8))
        
        ax1.plot(strains, xis, 'bo-', label='Correlation Length ξ')
        ax1.set_ylabel("ξ (nm)")
        ax1.set_title("Correlation Length vs Strain")
        ax1.grid(True)
        
        ax2.plot(strains, ms, 'rs-', label='Exponent m')
        ax2.set_xlabel("Strain (%)")
        ax2.set_ylabel("Exponent m")
        ax2.set_title("Exponent m vs Strain")
        ax2.grid(True)
        
        cv = FigureCanvasTkAgg(fig, top)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_csv(self):
        if not self.files_data: return
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            # Create DataFrame with all detailed info
            df = pd.DataFrame(self.files_data)
            # Select and Rename columns for clarity
            columns = ['filename', 'strain', 'xi', 'm', 'I0', 'bkg', 'fit_qmin', 'fit_qmax', 'model_used']
            out_df = df[columns].copy()
            out_df.columns = ['Filename', 'Strain_Percent', 'Xi_nm', 'Exponent_m', 'I0', 'Background', 'Fit_Q_Min', 'Fit_Q_Max', 'Model_Equation']
            
            out_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", "Detailed results saved to CSV.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ComprehensiveGelAnalyzer(root)
    root.mainloop()
