import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit, least_squares, differential_evolution
import platform
import warnings
import os

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

# ========= 한글 폰트 자동 설정 =========
def setup_korean_font():
    system = platform.system()
    if system == "Windows":
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'Dotum', 'Gulim']
    elif system == "Darwin":
        font_candidates = ['AppleGothic', 'NanumGothic', 'Arial Unicode MS']
    else:
        font_candidates = ['NanumGothic', 'DejaVu Sans', 'Liberation Sans']
    
    available = [f.name for f in fm.fontManager.ttflist]
    korean = None
    for fn in font_candidates:
        if fn in available:
            korean = fn
            break
    
    if korean is None:
        fallback = [f.name for f in fm.fontManager.ttflist
                   if any(k in f.name.lower() for k in ['gothic','nanum','malgun','dotum'])]
        korean = fallback[0] if fallback else 'DejaVu Sans'
    
    plt.rcParams['font.family'] = korean
    plt.rcParams['axes.unicode_minus'] = False
    return korean

setup_korean_font()

# ========= 모델들 =========
def dab_component(q, A1, xi):
    """DAB (Debye-Anderson-Brumberger) 성분"""
    return A1 / (1.0 + (q*xi)**2)**2

def form_factor_sphere(q, Rg):
    """간단 구형 형태인자 근사"""
    return np.exp(-(q*Rg)**2 / 3.0)

def structure_factor_fractal(q, eta, D):
    """간단 프랙탈 구조인자 근사"""
    return (1.0 + (q*eta)**2)**(-0.5*D)

def fractal_component(q, A2, Rg, eta, D):
    """프랙탈 성분 (형태인자 × 구조인자)"""
    return A2 * form_factor_sphere(q, Rg) * structure_factor_fractal(q, eta, D)

def full_model(q, A1, xi, A2, Rg, eta, D, bg):
    """전체 모델: DAB + 프랙탈 + 배경"""
    I_dab = dab_component(q, A1, xi)
    I_frac = fractal_component(q, A2, Rg, eta, D)
    return I_dab + I_frac + bg

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

# ========= GUI 클래스 =========
class GISAXSFullGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GISAXS 통합 피팅 도구 (DAB + Fractal×FormFactor)")
        self.root.geometry("1600x900")
        
        # 데이터
        self.q_all = None
        self.I_all = None
        self.file_path = None
        
        # 선택 범위
        self.q_sel = None
        self.I_sel = None
        self.sel_range = None
        self.span = None
        
        # 파라미터 상태
        self.params = {
            'A1': {'value': 1e3, 'min': 0.0, 'max': 1e12, 'fixed': False},
            'xi': {'value': 2.0, 'min': 0.01, 'max': 1e3, 'fixed': False},
            'A2': {'value': 1e2, 'min': 0.0, 'max': 1e12, 'fixed': False},
            'Rg': {'value': 3.0, 'min': 0.05, 'max': 1e3, 'fixed': False},
            'eta': {'value': 5.0, 'min': 0.05, 'max': 1e3, 'fixed': False},
            'D': {'value': 2.2, 'min': 1.0, 'max': 3.0, 'fixed': False},
            'bg': {'value': 0.0, 'min': 0.0, 'max': 1e12, 'fixed': False},
        }
        
        # 피팅 결과
        self.last_fit = None
        self.method = tk.StringVar(value='curve_fit')
        self.robust_loss = tk.StringVar(value='soft_l1')
        self.robust_fscale = tk.StringVar(value='1.0')
        
        # 레이아웃
        self._build_left_frame()
        self._build_center_plot()
        self._build_right_controls()
        self._update_param_entries()

    # ---------- UI ----------
    def _build_left_frame(self):
        # 좌측 전체 프레임
        left_main_frame = tk.Frame(self.root)
        left_main_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 1) 데이터 불러오기 & 범위 선택
        frm_data = ttk.LabelFrame(left_main_frame, text="1) 데이터 불러오기 & 범위 선택")
        frm_data.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(frm_data, text="CSV 불러오기", command=self.load_csv).pack(fill=tk.X, pady=5)
        
        self.lbl_file = ttk.Label(frm_data, text="파일: -")
        self.lbl_file.pack(fill=tk.X, pady=2)
        
        ttk.Separator(frm_data, orient='horizontal').pack(fill=tk.X, pady=8)
        
        ttk.Label(frm_data, text="범위 선택 방법").pack(anchor='w')
        ttk.Label(frm_data, text="- 그래프에서 드래그로 q 최소/최대 선택\n- 선택 후 Enter 또는 '범위 확정'").pack(anchor='w')
        
        ttk.Button(frm_data, text="범위 확정", command=self.confirm_selection).pack(fill=tk.X, pady=5)
        ttk.Button(frm_data, text="범위 초기화", command=self.clear_selection).pack(fill=tk.X, pady=5)
        
        self.lbl_range = ttk.Label(frm_data, text="선택 범위: -")
        self.lbl_range.pack(fill=tk.X, pady=2)
        
        ttk.Separator(frm_data, orient='horizontal').pack(fill=tk.X, pady=8)
        
        ttk.Label(frm_data, text="저장").pack(anchor='w')
        ttk.Button(frm_data, text="최종 파라미터 CSV 저장", command=self.save_params_csv).pack(fill=tk.X, pady=4)
        ttk.Button(frm_data, text="Raw/Fit 곡선 CSV 저장", command=self.save_curves_csv).pack(fill=tk.X, pady=4)
        
        # 파라미터 설명 프레임
        frm_explanation = ttk.LabelFrame(left_main_frame, text="파라미터 의미 설명")
        frm_explanation.pack(fill=tk.BOTH, expand=True)
        
        explanation_text = """DAB 성분 (무정형 2상 구조):
• A1: DAB 세기 스케일 - 전자밀도 대비와 체적분율에 비례
• xi (ξ): 상관길이 (Å) - 두 무정형 상이 교대되는 평균 거리 스케일

Fractal 성분 (응집체/네트워크 구조):
• A2: 프랙탈 세기 스케일 - 응집체의 수밀도와 대비에 비례
• Rg: 회전반경 (Å) - 1차 입자 또는 응집체의 유효 크기
• eta (η): 프랙탈 상관길이 (Å) - 프랙탈 거동이 유효한 길이 범위
• D: 질량 프랙탈 차원 (1-3) - 응집체의 조밀도 (값이 클수록 조밀)

기타:
• bg: 배경 산란 - 비상관 배경 신호 (형광, 기기 노이즈 등)

피팅 해석 가이드:
- DAB 성분은 고분자 매트릭스의 무정형 상분리를 나타냄
- Fractal 성분은 수용체/소분자의 응집 네트워크를 설명
- xi 값이 클수록 거친 상분리, 작을수록 미세한 혼합
- D 값이 1에 가까우면 선형/막대형(극도로 확장된 1차원 구조) 구조, 2에 가까우면 가우시안 사슬(이상적인 무작위 코일) 구조, 3에 가까우면 조밀한 응집체(구형에 가까운 조밀한 3차원 응집) 구조
- eta는 프랙탈 특성이 나타나는 최대 길이 스케일을 의미"""
        
        text_widget = tk.Text(frm_explanation, wrap=tk.WORD, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(frm_explanation, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        text_widget.insert(tk.END, explanation_text.strip())
        text_widget.config(state=tk.DISABLED)  # 읽기 전용으로 설정

    def _build_center_plot(self):
        frm = ttk.LabelFrame(self.root, text="데이터와 피팅 결과")
        frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(8.5, 7))
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel('q (Å-1)')
        self.ax.set_ylabel('Intensity (a.u.)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Raw 데이터 (검정) / 선택(회색) / Total Fit(빨강) / DAB(초록) / Fractal(파랑)')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _build_right_controls(self):
        frm = ttk.LabelFrame(self.root, text="2) 파라미터 설정 & 피팅 실행")
        frm.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # 파라미터 그리드
        grid = ttk.Frame(frm)
        grid.pack(fill=tk.X, pady=5)
        
        headers = ["Param", "Value", "Min", "Max", "Fix"]
        for j, h in enumerate(headers):
            ttk.Label(grid, text=h).grid(row=0, column=j, padx=4, pady=2, sticky='w')
        
        self.param_vars = {}
        row = 1
        for name in ['A1','xi','A2','Rg','eta','D','bg']:
            v = self.params[name]
            val = tk.StringVar(value=str(v['value']))
            mn = tk.StringVar(value=str(v['min']))
            mx = tk.StringVar(value=str(v['max']))
            fx = tk.BooleanVar(value=v['fixed'])
            
            self.param_vars[name] = {'val': val, 'min': mn, 'max': mx, 'fix': fx}
            
            ttk.Label(grid, text=name).grid(row=row, column=0, padx=4, pady=2, sticky='w')
            ttk.Entry(grid, textvariable=val, width=10).grid(row=row, column=1, padx=2)
            ttk.Entry(grid, textvariable=mn, width=10).grid(row=row, column=2, padx=2)
            ttk.Entry(grid, textvariable=mx, width=10).grid(row=row, column=3, padx=2)
            ttk.Checkbutton(grid, variable=fx, command=self._on_fix_toggle).grid(row=row, column=4, padx=2)
            row += 1
        
        ttk.Separator(frm, orient='horizontal').pack(fill=tk.X, pady=8)
        
        # 피팅 방법
        ttk.Label(frm, text="피팅 방법").pack(anchor='w')
        ttk.Combobox(frm, textvariable=self.method,
                    values=['curve_fit','robust_ls','diff_evo'],
                    state='readonly', width=15).pack(anchor='w', pady=4)
        ttk.Label(frm, text="- curve_fit: TRF bounded LS\n- robust_ls: soft_l1/huber 등\n- diff_evo: 전역탐색(느림)").pack(anchor='w')
        
        # robust_ls 옵션
        robust_box = ttk.LabelFrame(frm, text="robust_ls 옵션")
        robust_box.pack(fill=tk.X, pady=6)
        
        ttk.Label(robust_box, text="loss").grid(row=0,column=0, padx=4, pady=2, sticky='w')
        ttk.Combobox(robust_box, textvariable=self.robust_loss,
                    values=['soft_l1','huber','cauchy','arctan'],
                    state='readonly', width=10).grid(row=0,column=1,padx=2)
        
        ttk.Label(robust_box, text="f_scale").grid(row=1,column=0, padx=4, pady=2, sticky='w')
        ttk.Entry(robust_box, textvariable=self.robust_fscale, width=12).grid(row=1,column=1,padx=2)
        
        ttk.Separator(frm, orient='horizontal').pack(fill=tk.X, pady=8)
        
        # 피팅 실행 버튼
        ttk.Button(frm, text="피팅 실행", command=self.run_fit).pack(fill=tk.X, pady=4)
        ttk.Button(frm, text="현재 파라미터로 곡선만 표시", command=self.plot_only).pack(fill=tk.X, pady=4)
        
        ttk.Separator(frm, orient='horizontal').pack(fill=tk.X, pady=8)
        
        # 결과 표시
        self.lbl_status = ttk.Label(frm, text="상태: -", foreground='blue')
        self.lbl_status.pack(fill=tk.X, pady=4)
        
        self.lbl_r2 = ttk.Label(frm, text="R²: -")
        self.lbl_r2.pack(fill=tk.X, pady=2)
        
        self.lbl_params = ttk.Label(frm, text="A1: -, xi: -, A2: -, Rg: -, eta: -, D: -, bg: -")
        self.lbl_params.pack(fill=tk.X, pady=2)

    # ---------- 유틸 ----------
    def _update_param_entries(self):
        for name, d in self.param_vars.items():
            try:
                self.params[name]['value'] = float(d['val'].get())
                self.params[name]['min'] = float(d['min'].get())
                self.params[name]['max'] = float(d['max'].get())
                self.params[name]['fixed'] = bool(d['fix'].get())
            except ValueError:
                pass

    def _set_params_from_dict(self, param_dict):
        # 피팅 결과를 우측 패널(Value)에 반영하고 내부 상태 업데이트
        for name, val in param_dict.items():
            if name in self.param_vars and 'val' in self.param_vars[name]:
                try:
                    self.param_vars[name]['val'].set(str(float(val)))
                except Exception:
                    self.param_vars[name]['val'].set(str(val))
        self._update_param_entries()

    def _on_fix_toggle(self):
        self._update_param_entries()

    def _on_key_press(self, event):
        if event.key == 'enter':
            self.confirm_selection()

    # ---------- 데이터 로드 & 범위 ----------
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="GISAXS CSV 선택",
            filetypes=[("CSV Files","*.csv"),("All Files","*.*")]
        )
        
        if not path:
            return
        
        try:
            df = pd.read_csv(path, header=None)
            q = df.iloc[:,0].values.astype(float)
            I = df.iloc[:,1].values.astype(float)
            
            m = np.isfinite(q) & np.isfinite(I)
            q = q[m]; I = I[m]
            idx = np.argsort(q)
            
            self.q_all = q[idx]
            self.I_all = I[idx]
            self.file_path = path
            
            self.lbl_file.config(text=f"파일: {os.path.basename(path)} | 포인트: {len(q)}")
            
            self.ax.cla()
            self.ax.set_xscale('log'); self.ax.set_yscale('log'); self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('q (Å-1)'); self.ax.set_ylabel('Intensity (a.u.)')
            self.ax.set_title('Raw 데이터 (검정) / 선택(회색) / Total Fit(빨강) / DAB(초록) / Fractal(파랑)')
            self.ax.plot(self.q_all, self.I_all, 'k.-', ms=3, lw=1, alpha=0.8, label='Raw')
            self.ax.legend()
            self.canvas.draw()
            
            if self.span:
                try:
                    self.span.disconnect_events()
                except Exception:
                    pass
                self.span = None
            
            def onselect(xmin, xmax):
                self.sel_range = (min(xmin,xmax), max(xmin,xmax))
                self.lbl_range.config(text=f"선택 범위: q = {self.sel_range[0]:.4g} ~ {self.sel_range[1]:.4g}")
            
            self.span = SpanSelector(self.ax, onselect, 'horizontal',
                                   useblit=True, interactive=True,
                                   props=dict(alpha=0.2, facecolor='red'))
            
            self.lbl_status.config(text="상태: 데이터 로드 완료. 범위를 선택하세요.", foreground='green')
            
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패: {e}")

    def confirm_selection(self):
        if self.q_all is None or self.sel_range is None:
            messagebox.showwarning("알림", "데이터와 선택 범위를 먼저 준비하세요.")
            return
        
        qmin, qmax = self.sel_range
        mask = (self.q_all >= qmin) & (self.q_all <= qmax)
        
        if not np.any(mask):
            messagebox.showwarning("알림", "선택 범위 내 데이터가 없습니다.")
            return
        
        self.q_sel = self.q_all[mask]
        self.I_sel = self.I_all[mask]
        
        self.lbl_status.config(text=f"상태: 범위 확정 (N={len(self.q_sel)}). 피팅 실행 가능.", foreground='green')
        
        self.ax.cla()
        self.ax.set_xscale('log'); self.ax.set_yscale('log'); self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('q (Å-1)'); self.ax.set_ylabel('Intensity (a.u.)')
        self.ax.set_title('Raw 데이터 (검정) / 선택(회색) / Total Fit(빨강) / DAB(초록) / Fractal(파랑)')
        self.ax.plot(self.q_all, self.I_all, 'k.-', ms=3, lw=1, alpha=0.5, label='Raw')
        self.ax.plot(self.q_sel, self.I_sel, 'o', color='gray', ms=4, alpha=0.9, label='선택')
        self.ax.legend()
        self.canvas.draw()

    def clear_selection(self):
        self.sel_range = None
        self.q_sel = None
        self.I_sel = None
        self.lbl_range.config(text="선택 범위: -")
        
        if self.q_all is not None:
            self.ax.cla()
            self.ax.set_xscale('log'); self.ax.set_yscale('log'); self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('q (Å-1)'); self.ax.set_ylabel('Intensity (a.u.)')
            self.ax.set_title('Raw 데이터 (검정) / 선택(회색) / Total Fit(빨강) / DAB(초록) / Fractal(파랑)')
            self.ax.plot(self.q_all, self.I_all, 'k.-', ms=3, lw=1, alpha=0.8, label='Raw')
            self.ax.legend()
            self.canvas.draw()

    # ---------- 피팅 ----------
    def _build_bounds_and_p0(self):
        self._update_param_entries()
        
        q = self.q_sel if self.q_sel is not None else self.q_all
        I = self.I_sel if self.I_sel is not None else self.I_all
        
        if q is None or I is None:
            return None
        
        # bg 초기 추정: 하위 10% 분위수
        try:
            bg_est = float(np.percentile(I, 10))
        except Exception:
            bg_est = self.params['bg']['value']
        
        if not self.param_vars['bg']['val'].get():
            self.param_vars['bg']['val'].set(str(bg_est))
            self._update_param_entries()
        
        pnames = ['A1','xi','A2','Rg','eta','D','bg']
        p0, lb, ub, fixed_mask = [], [], [], []
        
        for n in pnames:
            v = self.params[n]
            vmin, vmax = v['min'], v['max']
            vval = v['value']
            
            if vval <= vmin:
                vval = vmin*1.1 if vmin != 0 else 1e-12
            if vval >= vmax:
                vval = vmax*0.9
            
            p0.append(vval); lb.append(vmin); ub.append(vmax); fixed_mask.append(v['fixed'])
        
        return {'q': q, 'I': I, 'p0': np.array(p0, float), 'lb': np.array(lb, float),
                'ub': np.array(ub, float), 'fixed': np.array(fixed_mask, bool),
                'names': pnames}

    def _apply_fixed(self, vec, lb, ub, fixed_mask):
        free_idx = np.where(~fixed_mask)
        return vec[free_idx], lb[free_idx], ub[free_idx], free_idx

    def _reconstruct_full(self, free_params, full_template, free_idx):
        full = full_template.copy()
        full[free_idx] = free_params
        return full

    def run_fit(self):
        setup = self._build_bounds_and_p0()
        if setup is None:
            messagebox.showwarning("알림", "데이터/범위를 먼저 준비하세요.")
            return
        
        q, I = setup['q'], setup['I']
        p0, lb, ub, fixed_mask, names = setup['p0'], setup['lb'], setup['ub'], setup['fixed'], setup['names']
        
        p0_free, lb_free, ub_free, free_idx = self._apply_fixed(p0, lb, ub, fixed_mask)
        
        if p0_free.size == 0:
            y_pred = full_model(q, *p0)
            R2 = r_squared(I, y_pred)
            self.last_fit = dict(zip(names, p0)); self.last_fit['R2'] = R2
            self._set_params_from_dict({k: v for k, v in zip(names, p0)})
            self._plot_fit(q, I, p0)
            self._update_result_labels()
            return
        
        method = self.method.get()
        
        try:
            if method == 'curve_fit':
                def model_free(qx, *theta_free):
                    theta = self._reconstruct_full(np.array(theta_free), p0.copy(), free_idx)
                    return full_model(qx, *theta)
                
                popt_free, pcov = curve_fit(model_free, q, I, p0=p0_free,
                                          bounds=(lb_free, ub_free), maxfev=30000, method='trf')
                popt = self._reconstruct_full(popt_free, p0.copy(), free_idx)
                
            elif method == 'robust_ls':
                loss = self.robust_loss.get()
                try:
                    fscale = float(self.robust_fscale.get())
                except ValueError:
                    fscale = 1.0
                
                def residual(theta_free):
                    theta = self._reconstruct_full(theta_free, p0.copy(), free_idx)
                    return I - full_model(q, *theta)
                
                res = least_squares(residual, p0_free, bounds=(lb_free, ub_free),
                                  loss=loss, f_scale=fscale, max_nfev=30000)
                popt = self._reconstruct_full(res.x, p0.copy(), free_idx)
                
            elif method == 'diff_evo':
                bounds_de = list(zip(lb_free, ub_free))
                
                def objective(theta_free):
                    theta = self._reconstruct_full(theta_free, p0.copy(), free_idx)
                    pred = full_model(q, *theta)
                    return np.mean((I - pred)**2)
                
                res = differential_evolution(objective, bounds_de, maxiter=600, polish=True, seed=42)
                theta_free_best = res.x
                
                def residual(theta_free):
                    theta = self._reconstruct_full(theta_free, p0.copy(), free_idx)
                    return I - full_model(q, *theta)
                
                res2 = least_squares(residual, theta_free_best, bounds=(lb_free, ub_free),
                                   loss='soft_l1', f_scale=1.0, max_nfev=20000)
                popt = self._reconstruct_full(res2.x, p0.copy(), free_idx)
                
            else:
                messagebox.showwarning("알림", "알 수 없는 방법입니다.")
                return
            
            y_pred = full_model(q, *popt)
            R2 = r_squared(I, y_pred)
            
            self.last_fit = dict(zip(names, popt)); self.last_fit['R2'] = R2
            
            # 최적 파라미터를 우측 입력(Value)에 자동 반영
            self._set_params_from_dict({k: v for k, v in zip(names, popt)})
            
            self._plot_fit(q, I, popt)
            self._update_result_labels()
            self.lbl_status.config(text="상태: 피팅 완료", foreground='green')
            
        except Exception as e:
            messagebox.showerror("오류", f"피팅 실패: {e}")

    def plot_only(self):
        setup = self._build_bounds_and_p0()
        if setup is None:
            messagebox.showwarning("알림", "데이터/범위를 먼저 준비하세요.")
            return
        
        q, I, p = setup['q'], setup['I'], setup['p0']
        self._plot_fit(q, I, p, is_fit=False)
        
        R2 = r_squared(I, full_model(q, *p))
        names = ['A1','xi','A2','Rg','eta','D','bg']
        self.last_fit = dict(zip(names, p)); self.last_fit['R2'] = R2
        self._update_result_labels()
        
        # 현재 표시 파라미터(p0)를 패널에 반영
        self._set_params_from_dict({k: v for k, v in zip(names, p)})

    def _plot_fit(self, q, I, theta, is_fit=True):
        self.ax.cla()
        self.ax.set_xscale('log'); self.ax.set_yscale('log'); self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('q (Å-1)'); self.ax.set_ylabel('Intensity (a.u.)')
        title = 'Raw(검정)/선택(회색)/' + ('피팅' if is_fit else '예측') + ' - Total(빨강)/DAB(초록)/Fractal(파랑)'
        self.ax.set_title(title)
        
        # Raw 데이터
        if self.q_all is not None and self.I_all is not None:
            self.ax.plot(self.q_all, self.I_all, 'k.-', ms=3, lw=1, alpha=0.5, label='Raw')
        
        # 선택된 데이터
        if self.q_sel is not None and self.I_sel is not None and len(self.q_sel) > 0:
            self.ax.plot(self.q_sel, self.I_sel, 'o', color='gray', ms=4, alpha=0.9, label='선택')
        
        # 피팅 곡선들
        q_fit = q
        A1, xi, A2, Rg, eta, D, bg = theta
        
        # 각 성분 계산
        I_dab = dab_component(q_fit, A1, xi)
        I_fractal = fractal_component(q_fit, A2, Rg, eta, D)
        I_total = I_dab + I_fractal + bg
        
        # 플롯
        self.ax.plot(q_fit, I_total, 'r-', lw=2.5, label='Total Fit', alpha=0.9)
        self.ax.plot(q_fit, I_dab + bg, 'g-', lw=2, label='DAB', alpha=0.8)
        self.ax.plot(q_fit, I_fractal + bg, 'b-', lw=2, label='Fractal', alpha=0.8)
        
        # R² 표시
        R2 = r_squared(I, I_total)
        if np.isfinite(R2):
            self.ax.text(0.02, 0.95, f"R² = {R2:.4f}", transform=self.ax.transAxes,
                        ha='left', va='top', color='darkred', fontweight='bold')
        
        self.ax.legend()
        self.canvas.draw()

    def _update_result_labels(self):
        if self.last_fit is None:
            self.lbl_r2.config(text="R²: -")
            self.lbl_params.config(text="A1: -, xi: -, A2: -, Rg: -, eta: -, D: -, bg: -")
            return
        
        txt = (f"A1: {self.last_fit['A1']:.3e}, xi: {self.last_fit['xi']:.4f} Å, "
               f"A2: {self.last_fit['A2']:.3e}, Rg: {self.last_fit['Rg']:.3f} Å, "
               f"eta: {self.last_fit['eta']:.3f} Å, D: {self.last_fit['D']:.3f}, "
               f"bg: {self.last_fit['bg']:.3e}")
        
        self.lbl_r2.config(text=f"R²: {self.last_fit['R2']:.6f}")
        self.lbl_params.config(text=txt)

    # ---------- 저장 ----------
    def save_params_csv(self):
        if self.last_fit is None:
            messagebox.showwarning("알림", "저장할 피팅 결과가 없습니다.")
            return
        
        base = "GISAXS_full_fit_params.csv" if self.file_path is None else os.path.splitext(self.file_path)[0] + "_GISAXS_full_fit_params.csv"
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(base),
                                          filetypes=[("CSV Files","*.csv")])
        if not out:
            return
        
        order = ['A1','xi','A2','Rg','eta','D','bg','R2']
        rec = {k: self.last_fit.get(k, np.nan) for k in order}
        df = pd.DataFrame([rec])
        df.to_csv(out, index=False, encoding='utf-8-sig')
        messagebox.showinfo("완료", f"파라미터 저장: {out}")

    def save_curves_csv(self):
        if self.q_all is None or self.I_all is None:
            messagebox.showwarning("알림", "저장할 데이터가 없습니다.")
            return
        
        base = "GISAXS_full_fit_curves.csv" if self.file_path is None else os.path.splitext(self.file_path)[0] + "_GISAXS_full_fit_curves.csv"
        out = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=os.path.basename(base),
                                          filetypes=[("CSV Files","*.csv")])
        if not out:
            return
        
        if self.last_fit is not None:
            A1 = self.last_fit['A1']; xi = self.last_fit['xi']; A2 = self.last_fit['A2']
            Rg = self.last_fit['Rg']; eta = self.last_fit['eta']; D = self.last_fit['D']; bg = self.last_fit['bg']
        else:
            self._update_param_entries()
            A1 = self.params['A1']['value']; xi = self.params['xi']['value']
            A2 = self.params['A2']['value']; Rg = self.params['Rg']['value']
            eta = self.params['eta']['value']; D = self.params['D']['value']
            bg = self.params['bg']['value']
        
        # 각 성분별로 계산
        I_fit_total = full_model(self.q_all, A1, xi, A2, Rg, eta, D, bg)
        I_dab = dab_component(self.q_all, A1, xi) + bg
        I_fractal = fractal_component(self.q_all, A2, Rg, eta, D) + bg
        
        df = pd.DataFrame({
            'q_all': self.q_all,
            'I_all': self.I_all,
            'I_fit_total': I_fit_total,
            'I_dab': I_dab,
            'I_fractal': I_fractal
        })
        
        if self.q_sel is not None and self.I_sel is not None:
            dsel = pd.DataFrame({'q_sel': self.q_sel, 'I_sel': self.I_sel})
            if len(dsel) < len(df):
                pad = pd.DataFrame({'q_sel':[np.nan]*(len(df)-len(dsel)),
                                  'I_sel':[np.nan]*(len(df)-len(dsel))})
                dsel = pd.concat([dsel, pad], ignore_index=True)
            df = pd.concat([df, dsel], axis=1)
        
        df.to_csv(out, index=False, encoding='utf-8-sig')
        messagebox.showinfo("완료", f"곡선 저장: {out}")

# ========= 실행 =========
if __name__ == "__main__":
    root = tk.Tk()
    app = GISAXSFullGUI(root)
    root.mainloop()
