import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.special import spherical_jn
import warnings
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple
warnings.filterwarnings('ignore')

class InteractiveSAXSFitter:
    def __init__(self):
        """
        대화형 SAXS 피팅 클래스
        파일 브라우징과 피팅 범위 선택 기능 포함
        """
        self.filename = None
        self.q = None
        self.I = None
        self.error = None
        
        # 피팅 범위 관련
        self.q_min = None
        self.q_max = None
        self.fit_mask = None
        
        # 피팅 결과
        self.cs_params = None
        self.cs_fit = None
        self.cs_chi2 = None
        self.rg_params = None
        self.rg_fit = None
        self.rg_chi2 = None
        self.best_model = None
        
        # GUI 관련
        self.fig = None
        self.ax = None
        self.span_selector = None
        self.selected_range = None
        
    def browse_files(self) -> List[str]:
        """
        GUI 파일 브라우저로 .dat 파일 선택
        """
        root = tk.Tk()
        root.withdraw()  # 메인 윈도우 숨기기
        
        print("📂 파일 선택 창이 열립니다...")
        
        filetypes = [
            ('DAT files', '*.dat'),
            ('All files', '*.*')
        ]
        
        filenames = filedialog.askopenfilenames(
            title="SAXS 데이터 파일 선택 (.dat)",
            filetypes=filetypes,
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        if filenames:
            print(f"✅ {len(filenames)}개 파일이 선택되었습니다:")
            for i, f in enumerate(filenames, 1):
                print(f"  {i}. {os.path.basename(f)}")
        else:
            print("❌ 파일이 선택되지 않았습니다.")
            
        return list(filenames)
    
    def load_data(self, filename: str) -> bool:
        """
        데이터 로딩 (구분자 자동 감지)
        """
        try:
            self.filename = filename
            
            # 다양한 구분자로 시도
            separators = [',', '\t', ' ', ';']
            data = None
            
            for sep in separators:
                try:
                    temp_data = np.loadtxt(filename, delimiter=sep)
                    if temp_data.ndim == 2 and temp_data.shape[1] >= 2:
                        data = temp_data
                        break
                except:
                    continue
            
            if data is None:
                raise ValueError("데이터 형식을 인식할 수 없습니다")
            
            self.q = data[:, 0]
            self.I = data[:, 1]
            self.error = data[:, 2] if data.shape[1] > 2 else np.ones_like(self.I)
            
            # 유효한 데이터만 선택 (q > 0, I > 0)
            valid_mask = (self.q > 0) & (self.I > 0) & np.isfinite(self.I)
            self.q = self.q[valid_mask]
            self.I = self.I[valid_mask]
            self.error = self.error[valid_mask]
            
            print(f"✅ 데이터 로딩 완료: {len(self.q)}개 포인트")
            print(f"   q 범위: {self.q.min():.6f} ~ {self.q.max():.6f}")
            print(f"   I 범위: {self.I.min():.2e} ~ {self.I.max():.2e}")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            return False
    
    def select_fitting_range_interactive(self) -> bool:
        """
        마우스 드래그로 피팅 범위 선택하는 대화형 인터페이스
        """
        if self.q is None or self.I is None:
            print("❌ 데이터가 로딩되지 않았습니다.")
            return False
        
        print(f"\n🎯 피팅 범위 선택")
        print("=" * 60)
        print("📋 사용법:")
        print("  1. 그래프에서 마우스로 드래그하여 피팅할 q 범위를 선택하세요")
        print("  2. 선택이 완료되면 그래프를 닫으세요")
        print("  3. 선택된 범위로 피팅을 시작합니다")
        print("-" * 60)
        
        # 그래프 생성
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 원본 데이터 플롯
        self.ax.loglog(self.q, self.I, 'o-', label='원본 데이터', 
                      markersize=4, alpha=0.7, color='blue')
        
        self.ax.set_xlabel('q (Å⁻¹)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'피팅 범위 선택: {os.path.basename(self.filename)}', fontsize=14)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # SpanSelector 추가 (범위 선택 도구) - matplotlib 버전 호환성 개선
        def onselect(q_min, q_max):
            self.q_min = q_min
            self.q_max = q_max
            self.selected_range = (q_min, q_max)
            
            # 선택된 범위 표시
            self.ax.axvspan(q_min, q_max, alpha=0.3, color='red', 
                           label=f'선택 범위: {q_min:.4f} ~ {q_max:.4f}')
            
            # 범위 내 데이터 하이라이트
            mask = (self.q >= q_min) & (self.q <= q_max)
            if np.any(mask):
                self.ax.loglog(self.q[mask], self.I[mask], 'ro', 
                              markersize=6, alpha=0.8, label='피팅 데이터')
            
            self.ax.legend()
            self.fig.canvas.draw()
            
            print(f"📍 선택된 범위: q = {q_min:.6f} ~ {q_max:.6f}")
            print(f"   포함된 데이터 포인트: {np.sum(mask)}개")
        
        # matplotlib 버전 호환성을 위한 SpanSelector 생성
        try:
            # 최신 버전 시도
            self.span_selector = SpanSelector(
                self.ax, onselect, direction='horizontal',
                useblit=True, interactive=True,
                props=dict(alpha=0.3, facecolor='red')
            )
        except TypeError:
            try:
                # 중간 버전 시도
                self.span_selector = SpanSelector(
                    self.ax, onselect, direction='horizontal',
                    useblit=True, interactive=True
                )
            except TypeError:
                # 구버전 시도
                self.span_selector = SpanSelector(
                    self.ax, onselect, direction='horizontal',
                    useblit=True
                )
        
        plt.tight_layout()
        plt.show()
        
        # 범위 선택 확인
        if self.selected_range is None:
            print("❌ 피팅 범위가 선택되지 않았습니다. 전체 범위를 사용합니다.")
            self.q_min = self.q.min()
            self.q_max = self.q.max()
            self.selected_range = (self.q_min, self.q_max)
        
        # 피팅 마스크 생성
        self.fit_mask = (self.q >= self.q_min) & (self.q <= self.q_max)
        
        if np.sum(self.fit_mask) < 10:
            print("⚠️  선택된 범위의 데이터 포인트가 너무 적습니다 (10개 미만).")
            print("   전체 범위를 사용합니다.")
            self.fit_mask = np.ones_like(self.q, dtype=bool)
            self.q_min = self.q.min()
            self.q_max = self.q.max()
        
        print(f"✅ 피팅 범위 설정 완료!")
        print(f"   q 범위: {self.q_min:.6f} ~ {self.q_max:.6f}")
        print(f"   피팅 포인트: {np.sum(self.fit_mask)}개")
        
        return True
    
    def get_fitting_data(self):
        """
        선택된 범위의 피팅 데이터 반환
        """
        if self.fit_mask is None:
            return self.q, self.I, self.error
        
        return self.q[self.fit_mask], self.I[self.fit_mask], self.error[self.fit_mask]
    
    def core_shell_model(self, q, R_core, t_shell, rho_core, rho_shell, rho_solvent, scale, background):
        """
        Monodisperse core-shell structure 모델 (PEDOT:PSS용)
        """
        R_total = R_core + t_shell
        
        # Core form factor
        x_core = q * R_core
        x_core = np.where(x_core == 0, 1e-10, x_core)
        F_core = 3 * (np.sin(x_core) - x_core * np.cos(x_core)) / x_core**3
        
        # Total sphere form factor
        x_total = q * R_total
        x_total = np.where(x_total == 0, 1e-10, x_total)
        F_total = 3 * (np.sin(x_total) - x_total * np.cos(x_total)) / x_total**3
        
        # Scattering amplitude
        V_core = (4/3) * np.pi * R_core**3
        V_shell = (4/3) * np.pi * (R_total**3 - R_core**3)
        
        F = (rho_core - rho_shell) * V_core * F_core + (rho_shell - rho_solvent) * V_shell * F_total
        P_q = F**2
        
        return scale * P_q + background
    
    def rod_gaussian_model(self, q, L_rod, R_rod, rho_rod, rho_solvent, Rg, scale_rod, scale_gauss, background):
        """
        Rod + Gaussian coil 모델 (aged data용)
        """
        # Rod contribution (rigid cylinder)
        qL = q * L_rod
        qR = q * R_rod
        
        qL_safe = np.where(qL == 0, 1e-10, qL)
        qR_safe = np.where(qR == 0, 1e-10, qR)
        
        # Form factor for cylinder - 수정된 계산
        try:
            F_rod = 2 * spherical_jn(1, qR_safe) / qR_safe * np.sin(qL_safe/2) / (qL_safe/2)
        except:
            # spherical_jn이 없는 경우 대체 계산
            F_rod = 2 * np.sin(qR_safe) / qR_safe * np.sin(qL_safe/2) / (qL_safe/2)
        
        P_rod = F_rod**2
        
        # Gaussian coil contribution (Debye function)
        x = (q * Rg)**2
        x_safe = np.where(x == 0, 1e-10, x)
        P_gauss = 2 * (np.exp(-x_safe) - 1 + x_safe) / x_safe**2
        P_gauss = np.where(x < 1e-4, 1.0 - x/3.0 + x**2/12.0, P_gauss)
        
        return scale_rod * P_rod + scale_gauss * P_gauss + background
    
    def fit_model(self, model_type='both', max_iterations=10000):
        """
        선택된 범위에서 모델 피팅
        """
        q_fit, I_fit, error_fit = self.get_fitting_data()
        
        print(f"\n🔬 모델 피팅 시작")
        print(f"피팅 데이터: {len(q_fit)}개 포인트 (q: {q_fit.min():.6f} ~ {q_fit.max():.6f})")
        print("=" * 60)
        
        results = {}
        
        # Core-Shell 모델 피팅
        if model_type in ['core_shell', 'both']:
            print("\n🔄 Core-Shell 모델 피팅...")
            
            initial_guess = [10.0, 5.0, 2.0, 1.0, 0.0, 1.0, 0.1]
            bounds = (
                [1.0, 1.0, 0.1, 0.1, -1.0, 1e-6, 0.0],
                [50.0, 20.0, 10.0, 10.0, 1.0, 1e6, 100.0]
            )
            
            best_chi2 = np.inf
            best_params = None
            
            print_interval = max(1, max_iterations // 10)
            
            for iteration in range(max_iterations):
                try:
                    if iteration > 0:
                        noise = np.random.normal(0, 0.1, len(initial_guess))
                        current_guess = np.array(initial_guess) * (1 + noise)
                        current_guess = np.clip(current_guess, bounds[0], bounds[1])
                    else:
                        current_guess = initial_guess
                    
                    popt, pcov = curve_fit(
                        self.core_shell_model, q_fit, I_fit,
                        p0=current_guess, bounds=bounds,
                        maxfev=5000, method='trf'
                    )
                    
                    y_fit = self.core_shell_model(q_fit, *popt)
                    chi2 = np.sum((I_fit - y_fit)**2 / error_fit**2) / (len(q_fit) - len(popt))
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = popt
                    
                    if iteration % print_interval == 0:
                        progress = (iteration / max_iterations) * 100
                        print(f"  진행률: {progress:5.1f}% | 현재 최적 χ²: {best_chi2:.6f}")
                        
                except:
                    continue
            
            if best_params is not None:
                self.cs_params = best_params
                self.cs_fit = self.core_shell_model(self.q, *best_params)
                self.cs_chi2 = best_chi2
                results['core_shell'] = best_chi2
                
                print(f"✅ Core-Shell 피팅 완료!")
                print(f"   R_core = {best_params[0]:.2f} nm")
                print(f"   t_shell = {best_params[1]:.2f} nm")
                print(f"   χ² = {best_chi2:.6f}")
        
        # Rod + Gaussian 모델 피팅
        if model_type in ['rod_gaussian', 'both']:
            print(f"\n🔄 Rod + Gaussian 모델 피팅...")
            
            initial_guess = [50.0, 2.0, 2.0, 0.0, 10.0, 1.0, 1.0, 0.1]
            bounds = (
                [10.0, 0.5, 0.1, -1.0, 1.0, 1e-6, 1e-6, 0.0],
                [200.0, 10.0, 10.0, 1.0, 50.0, 1e6, 1e6, 100.0]
            )
            
            best_chi2 = np.inf
            best_params = None
            
            print_interval = max(1, max_iterations // 10)
            
            for iteration in range(max_iterations):
                try:
                    if iteration > 0:
                        noise = np.random.normal(0, 0.2, len(initial_guess))
                        current_guess = np.array(initial_guess) * (1 + noise)
                        current_guess = np.clip(current_guess, bounds[0], bounds[1])
                    else:
                        current_guess = initial_guess
                    
                    popt, pcov = curve_fit(
                        self.rod_gaussian_model, q_fit, I_fit,
                        p0=current_guess, bounds=bounds,
                        maxfev=5000, method='trf'
                    )
                    
                    y_fit = self.rod_gaussian_model(q_fit, *popt)
                    chi2 = np.sum((I_fit - y_fit)**2 / error_fit**2) / (len(q_fit) - len(popt))
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = popt
                    
                    if iteration % print_interval == 0:
                        progress = (iteration / max_iterations) * 100
                        print(f"  진행률: {progress:5.1f}% | 현재 최적 χ²: {best_chi2:.6f}")
                        
                except:
                    continue
            
            if best_params is not None:
                self.rg_params = best_params
                self.rg_fit = self.rod_gaussian_model(self.q, *best_params)
                self.rg_chi2 = best_chi2
                results['rod_gaussian'] = best_chi2
                
                print(f"✅ Rod + Gaussian 피팅 완료!")
                print(f"   L_rod = {best_params[0]:.2f} nm")
                print(f"   R_rod = {best_params[1]:.2f} nm")
                print(f"   Rg = {best_params[4]:.2f} nm")
                print(f"   χ² = {best_chi2:.6f}")
        
        # 최적 모델 선택
        if len(results) > 1:
            best_model_name = min(results.items(), key=lambda x: x[1])[0]
            self.best_model = best_model_name
            self.best_chi2 = results[best_model_name]
            
            print(f"\n🏆 최적 모델: {best_model_name} (χ² = {self.best_chi2:.6f})")
        elif len(results) == 1:
            self.best_model = list(results.keys())[0]
            self.best_chi2 = list(results.values())[0]
            print(f"\n🏆 피팅 모델: {self.best_model} (χ² = {self.best_chi2:.6f})")
        else:
            print("❌ 모든 피팅이 실패했습니다.")
            self.best_model = None
        
        return len(results) > 0
    
    def plot_results(self):
        """
        피팅 결과 시각화
        """
        if self.q is None or self.I is None:
            print("❌ 데이터가 없습니다.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 원본 데이터
        ax.loglog(self.q, self.I, 'o', label='실험 데이터', 
                 markersize=4, alpha=0.7, color='black')
        
        # 피팅 범위 표시
        if self.fit_mask is not None:
            ax.loglog(self.q[self.fit_mask], self.I[self.fit_mask], 'o', 
                     label='피팅 데이터', markersize=5, alpha=0.8, color='blue')
            ax.axvspan(self.q_min, self.q_max, alpha=0.2, color='blue', 
                      label=f'피팅 범위: {self.q_min:.4f} ~ {self.q_max:.4f}')
        
        # 피팅 결과
        colors = ['red', 'green']
        linestyles = ['-', '--']
        
        if self.cs_fit is not None:
            ax.loglog(self.q, self.cs_fit, linestyles[0], linewidth=2, color=colors[0],
                     label=f'Core-Shell (χ² = {self.cs_chi2:.3f})')
        
        if self.rg_fit is not None:
            ax.loglog(self.q, self.rg_fit, linestyles[1], linewidth=2, color=colors[1],
                     label=f'Rod + Gaussian (χ² = {self.rg_chi2:.3f})')
        
        ax.set_xlabel('q (Å⁻¹)', fontsize=14)
        ax.set_ylabel('Intensity', fontsize=14)
        ax.set_title(f'SAXS 피팅 결과: {os.path.basename(self.filename)}', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, output_file=None):
        """
        결과 저장
        """
        if output_file is None and self.filename:
            base_name = os.path.splitext(self.filename)[0]
            output_file = f"{base_name}_interactive_fit_results.txt"
        
        if not output_file:
            print("❌ 출력 파일명을 지정할 수 없습니다.")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"대화형 SAXS 피팅 결과 보고서\n")
            f.write("=" * 60 + "\n")
            f.write(f"파일명: {self.filename}\n")
            f.write(f"전체 데이터 포인트: {len(self.q)}개\n")
            f.write(f"전체 q 범위: {self.q.min():.6f} ~ {self.q.max():.6f}\n\n")
            
            if self.fit_mask is not None:
                f.write(f"피팅 범위: {self.q_min:.6f} ~ {self.q_max:.6f}\n")
                f.write(f"피팅 포인트: {np.sum(self.fit_mask)}개\n\n")
            
            if self.cs_params is not None:
                f.write("Core-Shell 모델 결과:\n")
                f.write("-" * 40 + "\n")
                f.write(f"R_core = {self.cs_params[0]:.2f} nm\n")
                f.write(f"t_shell = {self.cs_params[1]:.2f} nm\n")
                f.write(f"χ² = {self.cs_chi2:.6f}\n\n")
            
            if self.rg_params is not None:
                f.write("Rod + Gaussian 모델 결과:\n")
                f.write("-" * 40 + "\n")
                f.write(f"L_rod = {self.rg_params[0]:.2f} nm\n")
                f.write(f"R_rod = {self.rg_params[1]:.2f} nm\n")
                f.write(f"Rg = {self.rg_params[4]:.2f} nm\n")
                f.write(f"χ² = {self.rg_chi2:.6f}\n\n")
            
            if self.best_model:
                f.write(f"최적 모델: {self.best_model}\n")
                f.write(f"최적 χ² = {self.best_chi2:.6f}\n")
        
        print(f"📄 결과가 저장되었습니다: {output_file}")

# 메인 실행 함수들
def run_interactive_saxs_analysis():
    """
    대화형 SAXS 분석 실행
    """
    print("🔬 대화형 SAXS 데이터 분석 프로그램")
    print("=" * 60)
    print("Adv. Funct. Mater. Figure 2 기반 Core-Shell → Rod-like 구조 전이 분석")
    print("파일 브라우징 + 마우스 드래그 피팅 범위 선택 기능")
    
    fitter = InteractiveSAXSFitter()
    
    # 1. 파일 선택
    print(f"\n📂 STEP 1: 파일 선택")
    files = fitter.browse_files()
    
    if not files:
        print("❌ 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 각 파일에 대해 분석 수행
    for i, filename in enumerate(files, 1):
        print(f"\n{'='*80}")
        print(f"📊 [{i}/{len(files)}] 파일 분석: {os.path.basename(filename)}")
        print(f"{'='*80}")
        
        # 데이터 로딩
        if not fitter.load_data(filename):
            print(f"❌ {filename} 로딩 실패. 다음 파일로...")
            continue
        
        # 피팅 범위 선택
        print(f"\n🎯 STEP 2: 피팅 범위 선택")
        if not fitter.select_fitting_range_interactive():
            print("❌ 피팅 범위 선택 실패. 다음 파일로...")
            continue
        
        # 피팅 모델 선택
        print(f"\n🔬 STEP 3: 피팅 모델 선택")
        print("1. Core-Shell 모델만")
        print("2. Rod + Gaussian 모델만") 
        print("3. 두 모델 모두 비교")
        
        while True:
            choice = input("선택 (1-3): ").strip()
            if choice == '1':
                model_type = 'core_shell'
                break
            elif choice == '2':
                model_type = 'rod_gaussian'
                break
            elif choice == '3':
                model_type = 'both'
                break
            else:
                print("❌ 1, 2, 3 중에서 선택해주세요.")
        
        # 반복 횟수 설정
        while True:
            try:
                iterations = input("최대 반복 횟수 (기본값: 10000): ").strip()
                if iterations == "":
                    iterations = 10000
                else:
                    iterations = int(iterations)
                if iterations < 1000:
                    print("⚠️  최소 1000회 이상을 권장합니다.")
                break
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
        
        # 피팅 실행
        print(f"\n🚀 STEP 4: 피팅 실행")
        if fitter.fit_model(model_type, iterations):
            # 결과 시각화
            fitter.plot_results()
            
            # 결과 저장
            fitter.save_results()
            
            print(f"✅ {os.path.basename(filename)} 분석 완료!")
        else:
            print(f"❌ {os.path.basename(filename)} 피팅 실패!")
        
        # 다음 파일 진행 확인 (마지막 파일이 아닌 경우)
        if i < len(files):
            continue_analysis = input("\n다음 파일을 분석하시겠습니까? (y/n): ").lower()
            if continue_analysis != 'y':
                break
    
    print(f"\n🎉 분석 완료!")
    print("=" * 60)
    print("모든 결과 파일이 각 데이터 파일과 같은 폴더에 저장되었습니다.")

def analyze_single_file():
    """
    단일 파일 대화형 분석
    """
    fitter = InteractiveSAXSFitter()
    
    print("🔬 단일 파일 SAXS 분석")
    print("=" * 40)
    
    # 파일 선택
    files = fitter.browse_files()
    if not files:
        print("❌ 파일이 선택되지 않았습니다.")
        return None
    
    filename = files[0]  # 첫 번째 파일만 사용
    print(f"선택된 파일: {os.path.basename(filename)}")
    
    # 데이터 로딩
    if not fitter.load_data(filename):
        print("❌ 데이터 로딩 실패")
        return None
    
    # 피팅 범위 선택
    if not fitter.select_fitting_range_interactive():
        print("❌ 피팅 범위 선택 실패")
        return None
    
    # 두 모델 모두 피팅
    if fitter.fit_model('both', 10000):
        fitter.plot_results()
        fitter.save_results()
        print("✅ 분석 완료!")
        return fitter
    else:
        print("❌ 피팅 실패")
        return None

if __name__ == "__main__":
    print("🎯 실행 모드 선택:")
    print("1. 여러 파일 일괄 분석")
    print("2. 단일 파일 분석")
    
    while True:
        choice = input("선택 (1-2): ").strip()
        if choice == '1':
            run_interactive_saxs_analysis()
            break
        elif choice == '2':
            analyze_single_file()
            break
        else:
            print("❌ 1 또는 2를 선택해주세요.")
