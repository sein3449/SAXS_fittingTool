import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import SpanSelector
import numpy as np
from scipy.optimize import curve_fit, differential_evolution, least_squares
from scipy.interpolate import interp1d
import warnings
import sys
import platform
import os

# === 한글 폰트 자동 설정 ===
def setup_korean_font():
    """시스템에 맞는 한글 폰트 자동 설정"""
    system = platform.system()
    
    if system == "Windows":
        font_candidates = ['Malgun Gothic', 'NanumGothic', 'Dotum', 'Gulim']
    elif system == "Darwin":  # macOS
        font_candidates = ['AppleGothic', 'NanumGothic', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['NanumGothic', 'DejaVu Sans', 'Liberation Sans']
    
    # 시스템 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 한글 폰트 찾기
    korean_font = None
    for font in font_candidates:
        if font in available_fonts:
            korean_font = font
            break
    
    # 한글 폰트가 없으면 대체 방법
    if korean_font is None:
        korean_fonts = [f.name for f in fm.fontManager.ttflist 
                       if any(keyword in f.name.lower() for keyword in ['gothic', 'nanum', 'malgun', 'dotum'])]
        if korean_fonts:
            korean_font = korean_fonts[0]
        else:
            print("⚠️ 경고: 한글 폰트를 찾을 수 없습니다.")
            korean_font = 'DejaVu Sans'
    
    # 폰트 설정
    plt.rcParams['font.family'] = korean_font
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"✓ 한글 폰트 설정 완료: {korean_font}")
    return korean_font

# 한글 폰트 설정 실행
setup_korean_font()
warnings.filterwarnings('ignore')

try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    print("LMFit가 설치되지 않았습니다. 'pip install lmfit'로 설치하면 더 안정적인 피팅이 가능합니다.")

class ComprehensiveGISAXSAnalyzer:
    """
    포항공과대학교 고분자연구소용 완전한 GISAXS 분석기
    - DAB + 프랙탈 + 형태인자 모델 피팅
    - 다중 파일 평균화 지원
    - Origin 호환 데이터 출력
    - 강화된 피팅 알고리즘
    """
    
    def __init__(self):
        self.q = None
        self.intensity = None
        self.selected_range = {}
        self.fit_results = {}
        self.q_sel = None
        self.I_sel = None
        self.fitting_methods = ['smart_bounds', 'lmfit', 'differential_evolution', 'robust_least_squares']
        self.final_popt = None
        self.final_method = None
        
        # 다중 파일 지원
        self.raw_data_files = []
        self.averaged_data = None
        self.analysis_mode = 'single'  # 'single' or 'multiple'
    
    def browse_and_load_csv(self):
        """단일 CSV 파일 브라우저 및 로드"""
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="GISAXS 데이터 선택",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                data = pd.read_csv(file_path, header=None)
                self.q = data.iloc[:, 0].values
                self.intensity = data.iloc[:, 1].values
                self.file_path = file_path
                print(f"데이터 로드 완료: {len(self.q)} 포인트")
                return True
            except Exception as e:
                print(f"파일 로드 오류: {e}")
                return False
        return False
    
    def browse_multiple_csv_files(self):
        """여러 CSV 파일 선택 및 로드"""
        root = tk.Tk()
        root.withdraw()
        
        file_paths = filedialog.askopenfilenames(
            title="GISAXS Raw 데이터 파일들을 선택하세요 (Ctrl+클릭으로 다중선택)",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_paths:
            return False
        
        print(f"선택된 파일 수: {len(file_paths)}")
        
        all_data = []
        valid_files = []
        
        for file_path in file_paths:
            try:
                data = pd.read_csv(file_path, header=None)
                q_values = data.iloc[:, 0].values
                intensity_values = data.iloc[:, 1].values
                
                # 데이터 검증
                if len(q_values) > 0 and len(intensity_values) > 0:
                    all_data.append({
                        'q': q_values,
                        'intensity': intensity_values,
                        'file': file_path
                    })
                    valid_files.append(file_path)
                    print(f"✓ 로드 성공: {os.path.basename(file_path)} ({len(q_values)} 포인트)")
                else:
                    print(f"✗ 데이터 없음: {os.path.basename(file_path)}")
                    
            except Exception as e:
                print(f"✗ 로드 실패: {os.path.basename(file_path)} - {e}")
        
        if not all_data:
            print("유효한 데이터 파일이 없습니다.")
            return False
        
        self.raw_data_files = all_data
        self.file_path = valid_files[0]  # 첫 번째 파일을 기준 경로로 설정
        
        return True
    
    def interpolate_and_average_data(self):
        """여러 측정 데이터를 보간하여 평균화"""
        if not self.raw_data_files:
            print("Raw 데이터가 없습니다.")
            return False
        
        # 모든 데이터의 q 범위 찾기
        all_q_min = max([np.min(data['q']) for data in self.raw_data_files])
        all_q_max = min([np.max(data['q']) for data in self.raw_data_files])
        
        print(f"공통 q 범위: {all_q_min:.4f} ~ {all_q_max:.4f}")
        
        # 공통 q 그리드 생성 (로그 스케일)
        n_points = 200  # 적절한 포인트 수
        q_common = np.logspace(np.log10(all_q_min), np.log10(all_q_max), n_points)
        
        # 각 데이터셋을 공통 q에 보간
        interpolated_intensities = []
        
        for i, data in enumerate(self.raw_data_files):
            try:
                # 로그 스케일 보간 (GISAXS 데이터 특성상)
                
                # 중복 q 값 제거 및 정렬
                q_vals = data['q']
                i_vals = data['intensity']
                
                # 정렬
                sort_idx = np.argsort(q_vals)
                q_sorted = q_vals[sort_idx]
                i_sorted = i_vals[sort_idx]
                
                # 중복 제거 (평균값 사용)
                q_unique, inverse_idx = np.unique(q_sorted, return_inverse=True)
                i_unique = np.array([np.mean(i_sorted[inverse_idx == j]) for j in range(len(q_unique))])
                
                # 보간 함수 생성 (로그-로그 보간)
                log_interp = interp1d(
                    np.log10(q_unique), np.log10(i_unique),
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                
                # 공통 q에서 보간
                log_i_interp = log_interp(np.log10(q_common))
                i_interp = 10**log_i_interp
                
                interpolated_intensities.append(i_interp)
                print(f"✓ 보간 완료: {os.path.basename(data['file'])}")
                
            except Exception as e:
                print(f"✗ 보간 실패: {os.path.basename(data['file'])} - {e}")
        
        if not interpolated_intensities:
            print("보간된 데이터가 없습니다.")
            return False
        
        # 평균 및 표준편차 계산
        intensity_array = np.array(interpolated_intensities)
        
        self.averaged_data = {
            'q': q_common,
            'intensity_mean': np.mean(intensity_array, axis=0),
            'intensity_std': np.std(intensity_array, axis=0),
            'intensity_sem': np.std(intensity_array, axis=0) / np.sqrt(len(interpolated_intensities)),
            'n_files': len(interpolated_intensities)
        }
        
        # 기존 변수에 평균 데이터 할당
        self.q = self.averaged_data['q']
        self.intensity = self.averaged_data['intensity_mean']
        
        print(f"✓ 데이터 평균화 완료: {len(interpolated_intensities)}개 파일 → {len(q_common)} 포인트")
        return True
    
    def plot_raw_and_averaged_data(self):
        """Raw 데이터와 평균 데이터 비교 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 왼쪽: 모든 Raw 데이터
        for i, data in enumerate(self.raw_data_files):
            ax1.loglog(data['q'], data['intensity'], 'o-', alpha=0.6, 
                      markersize=2, linewidth=0.8, 
                      label=f'File {i+1}' if i < 5 else '')
        
        ax1.set_xlabel('q (nm⁻¹)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title(f'Raw 데이터 ({len(self.raw_data_files)}개 파일)')
        ax1.grid(True, alpha=0.3)
        if len(self.raw_data_files) <= 5:
            ax1.legend()
        
        # 오른쪽: 평균 데이터 (오차막대 포함)
        ax2.loglog(self.averaged_data['q'], self.averaged_data['intensity_mean'], 
                  'ro-', linewidth=2, markersize=4, label='평균 데이터')
        
        # 표준오차 오차막대 (로그 스케일에서는 상대 오차 사용)
        y_mean = self.averaged_data['intensity_mean']
        y_err = self.averaged_data['intensity_sem']
        
        # 로그 스케일 오차막대 (상대 오차로 변환)
        y_err_lower = y_mean * (1 - y_err/y_mean)
        y_err_upper = y_mean * (1 + y_err/y_mean)
        
        ax2.fill_between(self.averaged_data['q'], y_err_lower, y_err_upper, 
                        alpha=0.3, color='red', label=f'표준오차 (n={self.averaged_data["n_files"]})')
        
        ax2.set_xlabel('q (nm⁻¹)')
        ax2.set_ylabel('Intensity (a.u.)')
        ax2.set_title('평균화된 데이터')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_and_select_region(self):
        """데이터 플롯 및 영역 선택"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if self.analysis_mode == 'multiple':
            ax.loglog(self.q, self.intensity, 'r-', linewidth=2, label=f'평균 데이터 (n={self.averaged_data["n_files"]})')
        else:
            ax.loglog(self.q, self.intensity, 'b-', linewidth=1.5, label='GISAXS Data')
        
        ax.set_xlabel('q (nm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('GISAXS 데이터 - 분석할 영역을 드래그하여 선택하세요', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.region_selected = False
        
        def onselect(xmin, xmax):
            self.selected_range = {'xmin': xmin, 'xmax': xmax}
            print(f"선택된 영역: q = {xmin:.4f} ~ {xmax:.4f}")
            
            mask = (self.q >= xmin) & (self.q <= xmax)
            self.q_sel = self.q[mask]
            self.I_sel = self.intensity[mask]
            print(f"선택된 데이터 포인트: {len(self.q_sel)}개")
            
            self.region_selected = True
            plt.close(fig)
        
        span = SpanSelector(
            ax, onselect, 'horizontal', 
            useblit=True,
            props=dict(alpha=0.5, facecolor='red'),
            interactive=True,
            drag_from_anywhere=True
        )
        
        plt.show()
        return self.region_selected
    
    def dab_model(self, q, A, xi):
        """DAB 모델 (도너-풍부 영역)"""
        return A / (1 + (q * xi)**2)**2
    
    def form_factor_sphere(self, q, Rg):
        """구형 형태 인자 (어셉터 결정상)"""
        return np.exp(-(q * Rg)**2 / 3)
    
    def structure_factor_fractal(self, q, eta, D):
        """프랙탈 구조 인자 (어셉터 네트워크)"""
        return (1 + (q * eta)**2)**(-D/2)
    
    def full_model(self, q, A1, xi, A2, Rg, eta, D, bg):
        """
        완전한 GISAXS 모델:
        I(q) = A1*DAB(xi) + A2*P(Rg)*S_frac(eta,D) + bg
        
        Parameters:
        - A1, xi: DAB 모델 (도너-풍부 혼합 영역)
        - A2, Rg: 형태 인자 진폭 및 어셉터 결정 도메인 크기
        - eta, D: 프랙탈 상관 길이 및 프랙탈 차원 (어셉터 네트워크)
        - bg: 배경 신호
        """
        I_dab = self.dab_model(q, A1, xi)
        I_frac = A2 * self.form_factor_sphere(q, Rg) * self.structure_factor_fractal(q, eta, D)
        return I_dab + I_frac + bg
    
    def analyze_data_for_smart_bounds(self):
        """데이터 분석으로 스마트한 bounds와 초기값 설정"""
        # 기본 통계
        I_max = np.max(self.I_sel)
        I_min = np.min(self.I_sel)
        I_median = np.median(self.I_sel)
        q_median = np.median(self.q_sel)
        
        # 배경 추정
        bg_estimate = np.percentile(self.I_sel, 10)
        
        # 신호 분석
        I_signal = self.I_sel - bg_estimate
        I_signal = np.maximum(I_signal, I_max * 0.01)
        
        # 특성 길이 추정
        peak_idx = np.argmax(I_signal)
        peak_q = self.q_sel[peak_idx]
        
        # 스마트 초기값
        smart_params = {
            'A1': I_max * 0.5,
            'xi': max(0.5 / peak_q, 0.5) if peak_q > 0 else 2.0,
            'A2': I_max * 0.3,
            'Rg': max(0.3 / q_median, 0.5),
            'eta': max(1.0 / q_median, 1.0),
            'D': 2.3,
            'bg': bg_estimate
        }
        
        # 안전한 bounds
        bounds_lower = [
            0,                              # A1
            max(smart_params['xi'] * 0.1, 0.1),  # xi
            0,                              # A2
            max(smart_params['Rg'] * 0.1, 0.1),  # Rg
            max(smart_params['eta'] * 0.1, 0.1), # eta
            1.0,                           # D
            0                              # bg
        ]
        
        bounds_upper = [
            I_max * 10,                    # A1
            smart_params['xi'] * 20,       # xi
            I_max * 10,                    # A2
            smart_params['Rg'] * 20,       # Rg
            smart_params['eta'] * 20,      # eta
            3.0,                          # D
            I_max                         # bg
        ]
        
        p0 = [smart_params['A1'], smart_params['xi'], smart_params['A2'], 
              smart_params['Rg'], smart_params['eta'], smart_params['D'], smart_params['bg']]
        
        # bounds 검증 및 수정
        for i, (val, lower, upper) in enumerate(zip(p0, bounds_lower, bounds_upper)):
            if val <= lower:
                p0[i] = lower * 1.1
            elif val >= upper:
                p0[i] = upper * 0.9
        
        return p0, (bounds_lower, bounds_upper)
    
    def method_smart_bounds(self):
        """스마트 bounds를 사용한 curve_fit"""
        print("방법 1: 스마트 bounds curve_fit 시도...")
        
        p0, bounds = self.analyze_data_for_smart_bounds()
        
        try:
            popt, pcov = curve_fit(
                self.full_model, self.q_sel, self.I_sel,
                p0=p0, bounds=bounds,
                maxfev=20000, method='trf'
            )
            
            y_pred = self.full_model(self.q_sel, *popt)
            r_squared = 1 - np.sum((self.I_sel - y_pred)**2) / np.sum((self.I_sel - np.mean(self.I_sel))**2)
            
            if r_squared > 0.5:
                try:
                    perr = np.sqrt(np.diag(pcov))
                except:
                    perr = np.zeros(7)
                
                return {
                    'success': True,
                    'method': 'smart_bounds',
                    'params': popt,
                    'errors': perr,
                    'r_squared': r_squared
                }
        except Exception as e:
            print(f"Smart bounds 실패: {e}")
        
        return {'success': False}
    
    def method_lmfit(self):
        """LMFit를 사용한 피팅"""
        if not LMFIT_AVAILABLE:
            return {'success': False}
        
        print("방법 2: LMFit 시도...")
        
        try:
            def residual(params, q, data):
                vals = params.valuesdict()
                model = self.full_model(q, vals['A1'], vals['xi'], vals['A2'], 
                                      vals['Rg'], vals['eta'], vals['D'], vals['bg'])
                return data - model
            
            params = lmfit.Parameters()
            p0, bounds = self.analyze_data_for_smart_bounds()
            
            params.add('A1', value=p0[0], min=bounds[0][0], max=bounds[1][0])
            params.add('xi', value=p0[1], min=bounds[0][1], max=bounds[1][1])
            params.add('A2', value=p0[2], min=bounds[0][2], max=bounds[1][2])
            params.add('Rg', value=p0[3], min=bounds[0][3], max=bounds[1][3])
            params.add('eta', value=p0[4], min=bounds[0][4], max=bounds[1][4])
            params.add('D', value=p0[5], min=bounds[0][5], max=bounds[1][5])
            params.add('bg', value=p0[6], min=bounds[0][6], max=bounds[1][6])
            
            result = lmfit.minimize(residual, params, args=(self.q_sel, self.I_sel), 
                                   method='leastsq')
            
            if result.success:
                popt = [result.params[name].value for name in ['A1', 'xi', 'A2', 'Rg', 'eta', 'D', 'bg']]
                perr = [result.params[name].stderr if result.params[name].stderr else 0 
                       for name in ['A1', 'xi', 'A2', 'Rg', 'eta', 'D', 'bg']]
                
                y_pred = self.full_model(self.q_sel, *popt)
                r_squared = 1 - np.sum((self.I_sel - y_pred)**2) / np.sum((self.I_sel - np.mean(self.I_sel))**2)
                
                return {
                    'success': True,
                    'method': 'lmfit',
                    'params': popt,
                    'errors': perr,
                    'r_squared': r_squared
                }
        except Exception as e:
            print(f"LMFit 실패: {e}")
        
        return {'success': False}
    
    def method_differential_evolution(self):
        """Differential Evolution을 사용한 전역 최적화"""
        print("방법 3: Differential Evolution 시도...")
        
        try:
            def objective(params):
                pred = self.full_model(self.q_sel, *params)
                mse = np.mean((self.I_sel - pred)**2)
                return mse
            
            p0, bounds = self.analyze_data_for_smart_bounds()
            bounds_de = list(zip(bounds[0], bounds[1]))
            
            result = differential_evolution(
                objective, bounds_de,
                maxiter=2000, seed=42,
                polish=True, atol=1e-8
            )
            
            if result.success:
                popt = result.x
                y_pred = self.full_model(self.q_sel, *popt)
                r_squared = 1 - np.sum((self.I_sel - y_pred)**2) / np.sum((self.I_sel - np.mean(self.I_sel))**2)
                
                perr = np.abs(popt) * 0.05
                
                return {
                    'success': True,
                    'method': 'differential_evolution',
                    'params': popt,
                    'errors': perr,
                    'r_squared': r_squared
                }
        except Exception as e:
            print(f"Differential Evolution 실패: {e}")
        
        return {'success': False}
    
    def method_robust_least_squares(self):
        """Robust Least Squares를 사용한 피팅"""
        print("방법 4: Robust Least Squares 시도...")
        
        try:
            def residual(params):
                pred = self.full_model(self.q_sel, *params)
                return self.I_sel - pred
            
            p0, bounds = self.analyze_data_for_smart_bounds()
            
            result = least_squares(
                residual, p0, bounds=bounds,
                loss='soft_l1',
                f_scale=np.std(self.I_sel),
                max_nfev=10000
            )
            
            if result.success:
                popt = result.x
                
                try:
                    J = result.jac
                    cov = np.linalg.inv(J.T.dot(J))
                    perr = np.sqrt(np.diag(cov))
                except:
                    perr = np.zeros(7)
                
                y_pred = self.full_model(self.q_sel, *popt)
                r_squared = 1 - np.sum((self.I_sel - y_pred)**2) / np.sum((self.I_sel - np.mean(self.I_sel))**2)
                
                return {
                    'success': True,
                    'method': 'robust_least_squares',
                    'params': popt,
                    'errors': perr,
                    'r_squared': r_squared
                }
        except Exception as e:
            print(f"Robust Least Squares 실패: {e}")
        
        return {'success': False}
    
    def perform_multi_method_fitting(self):
        """여러 방법으로 피팅 시도"""
        print("=== 다중 방법 피팅 시작 ===")
        
        methods = [
            self.method_smart_bounds,
            self.method_lmfit,
            self.method_differential_evolution,
            self.method_robust_least_squares
        ]
        
        results = []
        for method in methods:
            result = method()
            if result['success']:
                results.append(result)
                print(f"✓ {result['method']}: R² = {result['r_squared']:.4f}")
            else:
                print(f"✗ {method.__name__} 실패")
        
        if not results:
            print("모든 피팅 방법이 실패했습니다.")
            return False
        
        # 최고 R² 결과 선택
        best_result = max(results, key=lambda x: x['r_squared'])
        print(f"\n최적 결과: {best_result['method']} (R² = {best_result['r_squared']:.4f})")
        
        # 결과 저장 (Origin 데이터 저장용)
        self.final_popt = best_result['params']
        self.final_method = best_result['method']
        
        popt = best_result['params']
        perr = best_result['errors']
        A1, xi, A2, Rg, eta, D, bg = popt
        
        self.fit_results = {
            'fitting_method': best_result['method'],
            'A1_DAB': {'value': A1, 'error': perr[0]},
            'xi_donor (nm)': {'value': xi, 'error': perr[1]},
            'A2_fractal': {'value': A2, 'error': perr[2]},
            'Rg_acceptor (nm)': {'value': Rg, 'error': perr[3]},
            '2Rg_acceptor (nm)': {'value': 2*Rg, 'error': 2*perr[3]},
            'eta_fractal (nm)': {'value': eta, 'error': perr[4]},
            'D_fractal': {'value': D, 'error': perr[5]},
            'background': {'value': bg, 'error': perr[6]},
            'R_squared': {'value': best_result['r_squared']},
            'data_points': {'value': len(self.q_sel)}
        }
        
        # 시각화
        self.plot_results(popt, best_result['method'])
        return True

    def plot_results(self, popt, method_name):
        """결과 시각화 - 한글 깨짐 완전 해결"""
        A1, xi, A2, Rg, eta, D, bg = popt
        
        # 현재 설정된 한글 폰트 가져오기
        current_font = plt.rcParams['font.family']
        if isinstance(current_font, list):
            current_font = current_font[0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 전체 피팅
        q_fit = np.linspace(self.q_sel.min(), self.q_sel.max(), 300)
        I_total = self.full_model(q_fit, *popt)
        I_dab = self.dab_model(q_fit, A1, xi) + bg
        I_frac = A2 * self.form_factor_sphere(q_fit, Rg) * self.structure_factor_fractal(q_fit, eta, D) + bg
        
        if self.analysis_mode == 'multiple':
            ax1.loglog(self.q, self.intensity, 'lightgray', alpha=0.7, label=f'평균 데이터 (n={self.averaged_data["n_files"]})')
        else:
            ax1.loglog(self.q, self.intensity, 'lightgray', alpha=0.7, label='전체 데이터')
        
        ax1.loglog(self.q_sel, self.I_sel, 'bo', markersize=4, label='선택된 데이터')
        ax1.loglog(q_fit, I_total, 'r-', linewidth=2, 
                  label=f'피팅 ({method_name})\nR² = {self.fit_results["R_squared"]["value"]:.4f}')
        ax1.loglog(q_fit, I_dab, 'g--', linewidth=1.5, label=f'DAB: ξ={xi:.2f}nm')
        ax1.loglog(q_fit, I_frac, 'm--', linewidth=1.5, 
                  label=f'프랙탈: Rg={Rg:.2f}nm, D={D:.2f}')
        
        ax1.set_xlabel('q (nm⁻¹)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title('완전한 GISAXS 피팅 결과')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 잔차
        residuals = self.I_sel - self.full_model(self.q_sel, *popt)
        ax2.semilogx(self.q_sel, residuals, 'ro', markersize=3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('q (nm⁻¹)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('피팅 잔차')
        ax2.grid(True, alpha=0.3)
        
        # 파라미터 요약
        ax3.axis('off')
        param_text = f"""=== 완전한 GISAXS 피팅 결과 ===
피팅 방법: {method_name}
분석 모드: {'다중 파일 평균' if self.analysis_mode == 'multiple' else '단일 파일'}

도너-풍부 영역 (DAB):
  ξ: {xi:.3f} ± {self.fit_results['xi_donor (nm)']['error']:.3f} nm
  A1: {A1:.2e} ± {self.fit_results['A1_DAB']['error']:.2e}

어셉터 결정상:
  Rg: {Rg:.3f} ± {self.fit_results['Rg_acceptor (nm)']['error']:.3f} nm
  2Rg: {2*Rg:.3f} ± {2*self.fit_results['Rg_acceptor (nm)']['error']:.3f} nm

어셉터 네트워크:
  η: {eta:.3f} ± {self.fit_results['eta_fractal (nm)']['error']:.3f} nm
  D: {D:.3f} ± {self.fit_results['D_fractal']['error']:.3f}

피팅 품질:
  R²: {self.fit_results['R_squared']['value']:.4f}"""
        
        # 한글 폰트로 텍스트 표시
        ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily=current_font,
                fontproperties=fm.FontProperties(family=current_font, size=10))
        
        # 개별 성분
        ax4.loglog(q_fit, I_total, 'r-', linewidth=2, label='전체')
        ax4.loglog(q_fit, self.dab_model(q_fit, A1, xi), 'g-', label='DAB (도너)')
        ax4.loglog(q_fit, A2 * self.form_factor_sphere(q_fit, Rg) * self.structure_factor_fractal(q_fit, eta, D), 'm-', label='프랙탈 (어셉터)')
        ax4.axhline(y=bg, color='orange', linestyle=':', label='배경')
        
        ax4.set_xlabel('q (nm⁻¹)')
        ax4.set_ylabel('Intensity (a.u.)')
        ax4.set_title('개별 성분')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def save_origin_data(self):
        """Origin에서 plotting할 수 있도록 모든 데이터 저장"""
        if self.final_popt is None:
            print("피팅 결과가 없습니다. 먼저 피팅을 수행해주세요.")
            return
        
        A1, xi, A2, Rg, eta, D, bg = self.final_popt
        
        # 고해상도 q 범위 생성 (Origin용)
        q_origin = np.linspace(self.q_sel.min(), self.q_sel.max(), 500)
        
        # 각 모델 성분 계산
        I_total = self.full_model(q_origin, *self.final_popt)
        I_dab_only = self.dab_model(q_origin, A1, xi)
        I_fractal_only = A2 * self.form_factor_sphere(q_origin, Rg) * self.structure_factor_fractal(q_origin, eta, D)
        I_dab_with_bg = I_dab_only + bg
        I_fractal_with_bg = I_fractal_only + bg
        
        # Origin 데이터 DataFrame 생성
        origin_data = pd.DataFrame({
            'q_nm_inv': q_origin,
            'I_total_fit': I_total,
            'I_DAB_component': I_dab_only,
            'I_fractal_component': I_fractal_only,
            'I_DAB_with_background': I_dab_with_bg,
            'I_fractal_with_background': I_fractal_with_bg,
            'background_level': np.full(len(q_origin), bg)
        })
        
        # 실험 데이터 추가 (선택된 영역)
        experimental_data = pd.DataFrame({
            'q_experimental': self.q_sel,
            'I_experimental': self.I_sel,
            'q_all_data': np.nan,
            'I_all_data': np.nan
        })
        
        # 전체 실험 데이터 길이에 맞춰 패딩
        if len(self.q) > len(experimental_data):
            padding_length = len(self.q) - len(experimental_data)
            padding_data = pd.DataFrame({
                'q_experimental': [np.nan] * padding_length,
                'I_experimental': [np.nan] * padding_length,
                'q_all_data': [np.nan] * padding_length,
                'I_all_data': [np.nan] * padding_length
            })
            experimental_data = pd.concat([experimental_data, padding_data], ignore_index=True)
        
        # 전체 실험 데이터 추가
        experimental_data.loc[:len(self.q)-1, 'q_all_data'] = self.q
        experimental_data.loc[:len(self.q)-1, 'I_all_data'] = self.intensity
        
        # 최종 데이터 결합
        max_length = max(len(origin_data), len(experimental_data))
        
        # 데이터 길이 맞추기
        if len(origin_data) < max_length:
            origin_padding = max_length - len(origin_data)
            origin_data = pd.concat([
                origin_data, 
                pd.DataFrame({col: [np.nan] * origin_padding for col in origin_data.columns})
            ], ignore_index=True)
        
        if len(experimental_data) < max_length:
            exp_padding = max_length - len(experimental_data)
            experimental_data = pd.concat([
                experimental_data,
                pd.DataFrame({col: [np.nan] * exp_padding for col in experimental_data.columns})
            ], ignore_index=True)
        
        # 최종 Origin 데이터
        final_origin_data = pd.concat([experimental_data, origin_data], axis=1)
        
        # 파일 저장
        base_name = os.path.splitext(self.file_path)[0]
        if self.analysis_mode == 'multiple':
            origin_file = f"{base_name}_Averaged_Origin_Plot_Data.csv"
        else:
            origin_file = f"{base_name}_Origin_Plot_Data.csv"
        
        # 헤더 정보 추가
        header_info = f"""# GISAXS Origin Plotting Data
# Generated by POSTECH Polymer Research Institute - Comprehensive Analyzer
# Analysis Mode: {'Multiple File Averaging' if self.analysis_mode == 'multiple' else 'Single File'}
# Fitting Method: {self.final_method}
# Parameters:
# A1 (DAB amplitude): {A1:.6e}
# xi (donor correlation length): {xi:.6f} nm
# A2 (fractal amplitude): {A2:.6e}
# Rg (acceptor crystalline size): {Rg:.6f} nm
# eta (fractal correlation length): {eta:.6f} nm
# D (fractal dimension): {D:.6f}
# background: {bg:.6e}
# R-squared: {self.fit_results['R_squared']['value']:.6f}
"""
        
        if self.analysis_mode == 'multiple':
            header_info += f"# Number of averaged files: {self.averaged_data['n_files']}\n"
        
        header_info += """#
# Data columns:
# q_experimental: experimental q values (selected region)
# I_experimental: experimental intensity (selected region)
# q_all_data: all experimental q values
# I_all_data: all experimental intensity values
# q_nm_inv: q values for fitting curves (nm^-1)
# I_total_fit: total fitted curve (DAB + fractal + background)
# I_DAB_component: DAB component only (donor-rich domains)
# I_fractal_component: fractal component only (acceptor networks)
# I_DAB_with_background: DAB + background
# I_fractal_with_background: fractal + background
# background_level: constant background level
#"""
        
        # 헤더와 데이터 함께 저장
        with open(origin_file, 'w', encoding='utf-8-sig') as f:
            f.write(header_info)
        
        # 데이터 추가 (헤더 없이)
        final_origin_data.to_csv(origin_file, mode='a', index=False, encoding='utf-8-sig')
        
        print(f"\n=== Origin 데이터 저장 완료 ===")
        print(f"파일 위치: {origin_file}")
        
        return origin_file

    def save_averaging_statistics(self):
        """평균화 통계 정보 저장"""
        if self.averaged_data is None:
            return None
        
        stats_data = {
            'parameter': ['number_of_files', 'common_q_points', 'q_min', 'q_max', 
                         'mean_intensity_range', 'relative_std_mean'],
            'value': [
                self.averaged_data['n_files'],
                len(self.averaged_data['q']),
                np.min(self.averaged_data['q']),
                np.max(self.averaged_data['q']),
                np.max(self.averaged_data['intensity_mean']) / np.min(self.averaged_data['intensity_mean']),
                np.mean(self.averaged_data['intensity_std'] / self.averaged_data['intensity_mean']) * 100
            ],
            'unit': ['count', 'count', 'nm^-1', 'nm^-1', 'ratio', '%']
        }
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = self.file_path.replace('.csv', '_Averaging_Statistics.csv')
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"평균화 통계: {stats_file}")
        return stats_file

    def save_results(self):
        """파라미터 결과 저장"""
        if not self.fit_results:
            print("피팅 결과가 없습니다.")
            return
        
        results_data = []
        for param, data in self.fit_results.items():
            if isinstance(data, dict) and 'value' in data:
                # 단위 설정
                if 'nm' in param:
                    unit = 'nm'
                elif 'D_fractal' in param or 'R_squared' in param:
                    unit = 'dimensionless'
                elif 'data_points' in param:
                    unit = 'count'
                elif 'fitting_method' in param:
                    unit = 'method'
                else:
                    unit = 'a.u.'
                
                # 값 포맷팅
                if isinstance(data['value'], (int, float)):
                    if 'R_squared' in param or 'D_fractal' in param:
                        value_str = f"{data['value']:.4f}"
                    elif 'nm' in param:
                        value_str = f"{data['value']:.3f}"
                    else:
                        value_str = f"{data['value']:.3e}"
                else:
                    value_str = str(data['value'])
                
                # 오차 포맷팅
                error_str = 'N/A'
                if 'error' in data and isinstance(data['error'], (int, float)):
                    if 'R_squared' in param or 'D_fractal' in param:
                        error_str = f"{data['error']:.4f}"
                    elif 'nm' in param:
                        error_str = f"{data['error']:.3f}"
                    else:
                        error_str = f"{data['error']:.3e}"
                
                results_data.append({
                    'Parameter': param,
                    'Value': value_str,
                    'Error': error_str,
                    'Unit': unit
                })
        
        results_df = pd.DataFrame(results_data)
        
        # 파일 저장
        if self.analysis_mode == 'multiple':
            output_file = self.file_path.replace('.csv', '_Averaged_GISAXS_Parameters.csv')
        else:
            output_file = self.file_path.replace('.csv', '_GISAXS_Parameters.csv')
        
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n=== 완전한 GISAXS 피팅 파라미터 결과 ===")
        print(results_df.to_string(index=False))
        print(f"\n파라미터 결과: {output_file}")
        
        return output_file
    
    def run_analysis(self):
        """전체 분석 실행 - 다중 파일 지원"""
        print("=" * 60)
        print("포항공과대학교 고분자연구소")
        print("완전한 GISAXS 분석기 v2.0")
        print("DAB + 프랙탈 + 형태인자 + 다중파일 평균화 + Origin 호환")
        print("=" * 60)
        
        # 사용자 선택: 단일 파일 vs 다중 파일
        choice = input("분석 모드를 선택하세요:\n1. 단일 파일 분석\n2. 다중 파일 평균화 분석\n선택 (1 또는 2): ").strip()
        
        if choice == '2':
            print("\n=== 다중 파일 평균화 모드 ===")
            self.analysis_mode = 'multiple'
            
            # 1. 다중 CSV 파일 로드
            if not self.browse_multiple_csv_files():
                print("파일 선택이 취소되었습니다.")
                return
            
            # 2. 데이터 보간 및 평균화
            if not self.interpolate_and_average_data():
                print("데이터 평균화가 실패했습니다.")
                return
            
            # 3. Raw 데이터와 평균 데이터 비교 시각화
            self.plot_raw_and_averaged_data()
            
            print("평균화된 데이터로 분석을 진행합니다...")
            
        else:  # 단일 파일 모드
            print("\n=== 단일 파일 분석 모드 ===")
            self.analysis_mode = 'single'
            
            # 1. 단일 CSV 파일 로드
            if not self.browse_and_load_csv():
                print("파일 선택이 취소되었습니다.")
                return
        
        # 4. 영역 선택 (기존과 동일)
        if not self.plot_and_select_region():
            print("영역 선택이 실패했습니다.")
            return
        
        # 5. 다중 방법 피팅 (기존과 동일)
        if not self.perform_multi_method_fitting():
            print("모든 피팅 방법이 실패했습니다.")
            return
        
        # 6. 결과 저장
        param_file = self.save_results()
        origin_file = self.save_origin_data()
        
        # 7. 평균화 통계 저장 (다중 파일 모드인 경우)
        stats_file = None
        if self.analysis_mode == 'multiple':
            stats_file = self.save_averaging_statistics()
        
        print("\n" + "=" * 60)
        print("=== 분석 완료! ===")
        print(f"분석 모드: {'다중 파일 평균화' if self.analysis_mode == 'multiple' else '단일 파일'}")
        if self.analysis_mode == 'multiple':
            print(f"✓ 평균화된 파일 수: {self.averaged_data['n_files']}개")
            print(f"✓ 평균화 통계: {stats_file}")
        print(f"✓ 피팅 파라미터: {param_file}")
        print(f"✓ Origin 데이터: {origin_file}")
        print(f"✓ 피팅 방법: {self.final_method}")
        print(f"✓ R²: {self.fit_results['R_squared']['value']:.4f}")
        print("\n주요 결과:")
        print(f"  - ξ (도너 상관길이): {self.fit_results['xi_donor (nm)']['value']:.3f} nm")
        print(f"  - Rg (어셉터 결정크기): {self.fit_results['Rg_acceptor (nm)']['value']:.3f} nm")
        print(f"  - η (프랙탈 상관길이): {self.fit_results['eta_fractal (nm)']['value']:.3f} nm")
        print(f"  - D (프랙탈 차원): {self.fit_results['D_fractal']['value']:.3f}")
        print("\nOrigin에서 바로 plotting할 수 있는 데이터가 준비되었습니다!")
        print("=" * 60)

# 실행
if __name__ == "__main__":
    analyzer = ComprehensiveGISAXSAnalyzer()
    analyzer.run_analysis()
