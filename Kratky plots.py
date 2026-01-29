import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import linregress
import os

class KratkyAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kratky Plot Analyzer (Persistence Length) - Copyright 2025 Sein Chung")
        self.root.geometry("1200x850")

        # Data storage
        self.file_paths = []
        self.current_data = None  # (q, I)
        self.results = {}  # {filename: {'q_star': val, 'a_length': val, 'decay_exp': val}}
        self.current_file_index = -1

        # --- Layout ---
        # Left Panel: Controls & List
        left_panel = tk.Frame(root, width=350, bg="#f8f9fa")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Buttons
        tk.Label(left_panel, text="Step 1: Load Data", font=("Arial", 10, "bold"), bg="#f8f9fa").pack(anchor="w", pady=(5, 0))
        tk.Button(left_panel, text="Load .dat Files", command=self.load_files, height=2, bg="#e2e6ea").pack(fill=tk.X, pady=5)
        
        tk.Label(left_panel, text="Step 3: Save Results", font=("Arial", 10, "bold"), bg="#f8f9fa").pack(anchor="w", pady=(15, 0))
        tk.Button(left_panel, text="Save Results (CSV)", command=self.save_results, height=2, bg="#d1e7dd").pack(fill=tk.X, pady=5)

        # File List
        tk.Label(left_panel, text="File List:", font=("Arial", 10, "bold"), bg="#f8f9fa").pack(anchor="w", pady=(20, 0))
        self.listbox = tk.Listbox(left_panel, selectmode=tk.SINGLE, height=20)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Info Panel
        self.info_frame = tk.Frame(left_panel, bg="white", relief="groove", bd=2)
        self.info_frame.pack(fill=tk.X, pady=10, ipadx=5, ipady=5)
        
        self.lbl_file = tk.Label(self.info_frame, text="File: -", font=("Arial", 10, "bold"), bg="white", anchor="w")
        self.lbl_file.pack(fill=tk.X)
        
        self.lbl_decay = tk.Label(self.info_frame, text="Decay Exp (High-Q): -", font=("Arial", 10), bg="white", anchor="w")
        self.lbl_decay.pack(fill=tk.X)
        
        self.lbl_warning = tk.Label(self.info_frame, text="", font=("Arial", 9, "bold"), fg="red", bg="white", wraplength=280)
        self.lbl_warning.pack(fill=tk.X)
        
        tk.Label(self.info_frame, text="----------------", bg="white").pack()
        
        self.lbl_q_star = tk.Label(self.info_frame, text="q* (Crossover): -", font=("Arial", 11, "bold"), fg="blue", bg="white", anchor="w")
        self.lbl_q_star.pack(fill=tk.X)
        
        self.lbl_a_length = tk.Label(self.info_frame, text="a (Persist. Length): -", font=("Arial", 11, "bold"), fg="green", bg="white", anchor="w")
        self.lbl_a_length.pack(fill=tk.X)
        
        tk.Label(left_panel, text="Copyright 2025 Sein Chung", font=("Arial", 8), fg="gray", bg="#f8f9fa").pack(side=tk.BOTTOM)

        # Right Panel: Plot
        right_panel = tk.Frame(root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Click Event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def parse_dat_file(self, filepath):
        try:
            # Flexible parser (space or comma)
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python', on_bad_lines='skip')
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            if df.shape[1] < 2:
                df = pd.read_csv(filepath, sep=',', comment='#', header=None, engine='python', on_bad_lines='skip')
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                if df.shape[1] < 2: return None, None

            q = df.iloc[:, 0].values
            I = df.iloc[:, 1].values
            return q, I
        except:
            return None, None

    def calculate_decay_exponent(self, q, I):
        """Check slope in high-q range (0.7 ~ 1.1 nm^-1)"""
        # Note: Assuming input q is in nm^-1 based on user context
        mask = (q >= 0.7) & (q <= 1.1)
        if np.sum(mask) < 3:
            return None
        
        q_fit = q[mask]
        I_fit = I[mask]
        
        # Log-Log fit
        slope, _, _, _, _ = linregress(np.log10(q_fit), np.log10(I_fit))
        return -slope  # Decay exponent

    def load_files(self):
        files = filedialog.askopenfilenames(title="Select SAXS .dat files", filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
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
        
        self.current_file_index = selection[0]
        filepath = self.file_paths[self.current_file_index]
        filename = os.path.basename(filepath)
        
        q, I = self.parse_dat_file(filepath)
        if q is not None:
            self.current_data = (q, I)
            
            # 1. Calculate Decay Exponent for validation
            decay = self.calculate_decay_exponent(q, I)
            
            # Update Info Panel
            self.lbl_file.config(text=f"File: {filename}")
            if decay:
                self.lbl_decay.config(text=f"Decay Exp (High-Q): {decay:.2f}")
                if decay >= 2.0:
                    self.lbl_warning.config(text="⚠ Warning: Decay exponent ≥ 2.0\nRod-like assumption may be invalid!")
                else:
                    self.lbl_warning.config(text="✔ Condition met (Exp < 2.0)", fg="green")
            else:
                self.lbl_decay.config(text="Decay Exp: N/A (Range not found)")
                self.lbl_warning.config(text="")

            # Load previous results if any
            if filename in self.results:
                res = self.results[filename]
                self.lbl_q_star.config(text=f"q* (Crossover): {res['q_star']:.4f}")
                self.lbl_a_length.config(text=f"a (Persist. Length): {res['a_length']:.2f} nm")
            else:
                self.lbl_q_star.config(text="q* (Crossover): Click graph")
                self.lbl_a_length.config(text="a (Persist. Length): -")

            self.plot_kratky(q, I, filename)

    def plot_kratky(self, q, I, title):
        self.ax.clear()
        
        # Kratky transformation: I * q^2
        # Note: If q is nm^-1, then I*q^2 unit depends on I unit.
        kratky_val = I * (q**2)
        
        # Plot I*q^2 vs q (Log scale on X is common for wide range)
        self.ax.semilogx(q, kratky_val, 'b-', linewidth=1.5, label='Kratky Data')
        
        # Add visual guide for picked point
        filename = os.path.basename(self.file_paths[self.current_file_index])
        if filename in self.results:
            q_star = self.results[filename]['q_star']
            # Find closest y value for visual marker
            idx = (np.abs(q - q_star)).argmin()
            y_val = kratky_val[idx]
            self.ax.plot(q_star, y_val, 'ro', markersize=8, label=f'q*={q_star:.3f}')
            self.ax.axvline(x=q_star, color='r', linestyle='--', alpha=0.5)

        self.ax.set_title(f"Kratky Plot: {title}", fontsize=12)
        self.ax.set_xlabel("q ($nm^{-1}$)", fontsize=10)
        self.ax.set_ylabel("$I(q) \cdot q^2$ (a.u.)", fontsize=10)
        self.ax.grid(True, which="both", ls="-", alpha=0.2)
        self.ax.legend()
        
        # Guide text
        self.ax.text(0.02, 0.95, "Click on the crossover point (upturn)", transform=self.ax.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        self.canvas.draw()

    def on_click(self, event):
        """Handle mouse click on the plot to select q*"""
        if event.inaxes != self.ax or self.current_data is None:
            return
            
        # Get clicked x-coordinate (q*)
        q_star_click = event.xdata
        
        # Snap to nearest actual data point (Optional but better for precision)
        q, I = self.current_data
        idx = (np.abs(q - q_star_click)).argmin()
        q_star_snap = q[idx]
        
        # Calculate Persistence Length a
        # Formula: a = 1.91 / q*
        if q_star_snap == 0: return
        a_length = 1.91 / q_star_snap
        
        # Save Result
        filename = os.path.basename(self.file_paths[self.current_file_index])
        
        # Re-calc decay for storage
        decay = self.calculate_decay_exponent(q, I)
        
        self.results[filename] = {
            'q_star': q_star_snap,
            'a_length': a_length,
            'decay_exp': decay if decay else np.nan
        }
        
        # Update UI
        self.lbl_q_star.config(text=f"q* (Crossover): {q_star_snap:.4f} $nm^{{-1}}$")
        self.lbl_a_length.config(text=f"a (Persist. Length): {a_length:.2f} nm")
        
        # Re-draw plot with marker
        self.plot_kratky(q, I, filename)

    def save_results(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            data = []
            for fname, res in self.results.items():
                data.append({
                    'Filename': fname,
                    'Decay_Exponent_HighQ': res['decay_exp'],
                    'q_star_nm-1': res['q_star'],
                    'Persistence_Length_a_nm': res['a_length']
                })
            
            df = pd.DataFrame(data)
            try:
                df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Results saved to:\n{save_path}\n\n(Copyright 2025 Sein Chung)")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = KratkyAnalyzerGUI(root)
    root.mainloop()
