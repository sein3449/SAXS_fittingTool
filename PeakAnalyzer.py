import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from tkinter import filedialog, Tk
from matplotlib.widgets import SpanSelector
from datetime import datetime
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

from matplotlib import font_manager, rc
font_path = "c:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
else:
    print("맑은 고딕 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")


class GIWAXSPeakAnalyzer:
    """
    GIWAXS 데이터의 피크를 대화형으로 선택하고,
    수치-모델 하이브리드 방식으로 피크를 분석하며 결과를 저장하는 클래스.
    """
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        self.file_path = file_path
        self.base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}.")

            print(f"파일 로드 성공: {file_path}")
            print(f"데이터 형태: {self.data.shape}")
        except Exception as e:
            raise Exception(f"파일 읽기 오류: {e}")

        self.x_axis = self.data.iloc[:, 0].values
        self.y_data = self.data.iloc[:, 1:22].values
        self.analysis_results_for_file = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.base_name}_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"분석 결과는 '{self.output_dir}' 폴더에 저장됩니다.")

    def set_saving_preferences(self, save_summary, save_detailed, save_fit_data):
        self.save_summary_plots = save_summary
        self.save_detailed_plots = save_detailed
        self.save_fit_data = save_fit_data

        if self.save_summary_plots: os.makedirs(os.path.join(self.output_dir, "summary_plots"), exist_ok=True)
        if self.save_detailed_plots: os.makedirs(os.path.join(self.output_dir, "detailed_fitting_plots"), exist_ok=True)
        if self.save_fit_data: os.makedirs(os.path.join(self.output_dir, "fitting_data"), exist_ok=True)

    def select_rois_interactively(self):
        selected_rois = []
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(self.x_axis, self.y_data[:, 0], label=f'기준 데이터: {self.base_name}')
        ax.set_title(f"[{self.base_name}] 분석할 피크 영역(ROI) 선택\n(선택 후 'Enter' 키, 다음 파일로 넘어가기)", fontsize=14)
        ax.set_xlabel("x축 값", fontsize=12); ax.set_ylabel("강도", fontsize=12)
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

        def onselect(xmin, xmax):
            print(f"ROI 임시 선택: [{xmin:.4f}, {xmax:.4f}]")
            if (xmin, xmax) not in selected_rois:
                selected_rois.append((xmin, xmax))
                ax.axvspan(xmin, xmax, color='red', alpha=0.3)
                fig.canvas.draw()

        def on_key_press(event):
            if event.key == 'enter':
                if not selected_rois: print("선택된 ROI가 없습니다. 먼저 영역을 선택해주세요."); return
                selected_rois.sort(key=lambda item: item[0])
                print(f"\n파일 '{self.base_name}'에 대한 ROI {len(selected_rois)}개 선택 확정.")
                plt.close(fig)
            elif event.key == 'escape':
                print("\nROI 선택이 취소되었습니다."); selected_rois.clear()
                plt.close(fig)

        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, props=dict(facecolor='yellow', alpha=0.5), interactive=True, drag_from_anywhere=True)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.manager.set_window_title(f"ROI 선택: {self.base_name}"); plt.show()
        return selected_rois
    
    def pseudo_voigt(self, x, amplitude, center, sigma, eta):
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        lorentzian = 1 / (1 + ((x - center) / sigma) ** 2)
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian)

    def calculate_fwhm_pseudo_voigt(self, sigma, eta):
        fwhm_g = 2 * sigma * np.sqrt(2 * np.log(2)); fwhm_l = 2 * sigma
        return (fwhm_g**5 + 2.69269*fwhm_g**4*fwhm_l + 2.42843*fwhm_g**3*fwhm_l**2 +
                4.47163*fwhm_g**2*fwhm_l**3 + 0.07842*fwhm_g*fwhm_l**4 + fwhm_l**5)**(1/5)

    def _save_individual_fit_plot(self, roi_index, col_idx, roi_x, y_values, popt):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(roi_x, y_values, 'o', label='원본 데이터', markersize=4, alpha=0.6)
        
        x_fit_curve = np.linspace(roi_x[0], roi_x[-1], 300)
        y_fit_curve = self.pseudo_voigt(x_fit_curve, *popt)
        ax.plot(x_fit_curve, y_fit_curve, 'r-', label='Pseudo-Voigt 피팅')

        peak_center = popt[1]
        ax.axvline(peak_center, color='green', linestyle=':', label=f'Peak Center: {peak_center:.4f}')
        
        ax.set_title(f'ROI #{roi_index+1} / Column #{col_idx+1} 피팅 결과')
        ax.set_xlabel("x축 값"); ax.set_ylabel("강도")
        ax.legend(); ax.grid(True, alpha=0.4)
        
        plot_dir = os.path.join(self.output_dir, "detailed_fitting_plots")
        file_path = os.path.join(plot_dir, f"ROI_{roi_index+1}_Col_{col_idx+1}_fit.png")
        fig.savefig(file_path, dpi=150)
        plt.close(fig)

    def _save_fitting_data(self, roi_index, col_idx, roi_x, y_values, popt):
        x_fit_curve = np.linspace(roi_x[0], roi_x[-1], 500)
        y_fit_curve = self.pseudo_voigt(x_fit_curve, *popt)
        raw_df = pd.DataFrame({'raw_x': roi_x, 'raw_y': y_values})
        fit_df = pd.DataFrame({'fit_x': x_fit_curve, 'fit_y': y_fit_curve})
        
        combined_df = pd.concat([raw_df, fit_df], axis=1)
        data_dir = os.path.join(self.output_dir, "fitting_data")
        file_path = os.path.join(data_dir, f"ROI_{roi_index+1}_Col_{col_idx+1}_data.csv")
        combined_df.to_csv(file_path, index=False, float_format='%.6f')

    def analyze_peaks_in_roi(self, roi_index, x_min, x_max):
        roi_mask = (self.x_axis >= x_min) & (self.x_axis <= x_max)
        roi_x = self.x_axis[roi_mask]
        
        if len(roi_x) < 5: print(f"ROI {roi_index+1}: 데이터 포인트가 부족하여 분석을 건너뜁니다."); return

        roi_y_data = self.y_data[roi_mask, :]
        print(f"\n--- 파일: {self.base_name} | ROI #{roi_index+1} ({x_min:.3f} ~ {x_max:.3f}) 분석 중 ---")
        
        peak_positions, fwhm_values = [], []

        for col_idx in range(self.y_data.shape[1]):
            y_values = roi_y_data[:, col_idx]
            non_zero_mask = y_values > 0
            active_roi_x, active_y_values = roi_x[non_zero_mask], y_values[non_zero_mask]

            if len(active_y_values) < 5:
                peak_positions.append(np.nan); fwhm_values.append(np.nan); continue

            try:
                # 1단계: 수치적 방법으로 안정적인 초기값 추정
                window_size = 5 if len(active_y_values) > 10 else 3
                smoothed_y = np.convolve(active_y_values, np.ones(window_size)/window_size, mode='valid')
                peak_idx = np.argmax(smoothed_y) + (window_size - 1) // 2
                
                peak_x_init, peak_y_init = active_roi_x[peak_idx], active_y_values[peak_idx]
                half_max_y = peak_y_init / 2.0
                
                fwhm_x_left = np.interp(half_max_y, np.maximum.accumulate(active_y_values[:peak_idx+1]), active_roi_x[:peak_idx+1])
                fwhm_x_right = np.interp(half_max_y, active_y_values[peak_idx:][::-1], active_roi_x[peak_idx:][::-1])
                fwhm_init = fwhm_x_right - fwhm_x_left
                sigma_init = fwhm_init / 2.355 # Gaussian 근사

                if fwhm_init <= 0 or sigma_init <= 0: raise ValueError("초기 FWHM 추정 실패")

                # 2단계: 추정된 초기값을 사용하여 모델 피팅
                popt, _ = curve_fit(self.pseudo_voigt, active_roi_x, active_y_values,
                                    p0=[peak_y_init, peak_x_init, sigma_init, 0.5],
                                    bounds=([0, roi_x[0], 1e-5, 0], [peak_y_init*2, roi_x[-1], (roi_x[-1]-roi_x[0]), 1]),
                                    maxfev=8000)

                # 3단계: 피팅 결과로부터 최종 파라미터 추출
                peak_positions.append(popt[1])
                fwhm_values.append(self.calculate_fwhm_pseudo_voigt(popt[2], popt[3]))
                
                if self.save_detailed_plots: self._save_individual_fit_plot(roi_index, col_idx, roi_x, y_values, popt)
                if self.save_fit_data: self._save_fitting_data(roi_index, col_idx, roi_x, y_values, popt)

            except Exception as e:
                peak_positions.append(np.nan); fwhm_values.append(np.nan)
        
        self.analysis_results_for_file.extend([{'FileName': self.base_name, 'ROI_Index': roi_index + 1, 'ROI_Start_X': x_min, 'ROI_End_X': x_max,
                                                'Column_Number': col_idx + 1, 'Peak_Position': peak_positions[col_idx],
                                                'FWHM': fwhm_values[col_idx]} for col_idx in range(self.y_data.shape[1])])
        
        self.plot_roi_results(roi_index, x_min, x_max, peak_positions, fwhm_values)

    def plot_roi_results(self, roi_index, x_min, x_max, peak_positions, fwhm_values):
        valid_indices = ~np.isnan(peak_positions)
        columns = np.arange(1, self.y_data.shape[1] + 1)[valid_indices]
        valid_peaks, valid_fwhms = np.array(peak_positions)[valid_indices], np.array(fwhm_values)[valid_indices]
        
        if len(columns) == 0: print(f"ROI #{roi_index+1}: 유효한 분석 결과가 없어 그래프를 생성하지 않습니다."); return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'파일: {self.base_name} - ROI #{roi_index+1} 분석 결과 ({x_min:.3f} ~ {x_max:.3f})', fontsize=18, fontweight='bold')

        ax1.plot(columns, valid_peaks, 'bo-', lw=2, ms=8, label='Peak Position')
        ax1.set_title('피크 위치 (Fitted)', fontsize=14); ax1.set_ylabel('피크 위치 (x축 값)', fontsize=12); ax1.grid(True, alpha=0.5)
        for i, txt in enumerate(valid_peaks): ax1.annotate(f'{txt:.3f}', (columns[i], valid_peaks[i]), textcoords="offset points", xytext=(0,5), ha='center')
        
        ax2.plot(columns, valid_fwhms, 'ro-', lw=2, ms=8, label='FWHM')
        ax2.set_title('FWHM (Fitted)', fontsize=14); ax2.set_xlabel('컬럼 번호', fontsize=12); ax2.set_ylabel('FWHM', fontsize=12); ax2.grid(True, alpha=0.5)
        ax2.set_xticks(range(1, self.y_data.shape[1] + 1))
        for i, txt in enumerate(valid_fwhms): ax2.annotate(f'{txt:.3f}', (columns[i], valid_fwhms[i]), textcoords="offset points", xytext=(0,5), ha='center')
        
        if self.save_summary_plots:
            plot_dir = os.path.join(self.output_dir, "summary_plots")
            file_path = os.path.join(plot_dir, f"ROI_{roi_index+1}_summary.png")
            fig.savefig(file_path, dpi=200, bbox_inches='tight')
            print(f"ROI #{roi_index+1} 요약 그래프 저장 완료: {file_path}")

        fig.canvas.manager.set_window_title(f'결과: {self.base_name} - ROI #{roi_index+1}'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


def select_multiple_data_files():
    root = Tk(); root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="분석할 데이터 파일들을 모두 선택하세요 (Excel, CSV)",
        filetypes=[("Data files", "*.xlsx *.xls *.csv"), ("All files", "*.*")],
        initialdir=os.getcwd())
    root.destroy()
    return list(file_paths)

def save_final_summary_table(all_results):
    if not all_results: print("저장할 분석 결과가 없습니다."); return
    results_df = pd.DataFrame(all_results)
    
    root = Tk(); root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="통합 결과 요약 테이블 저장", defaultextension=".csv",
        filetypes=[("CSV file", "*.csv"), ("Excel file", "*.xlsx")],
        initialdir=os.getcwd(),
        initialfile="Total_Analysis_Summary.csv")
    root.destroy()
    
    if file_path:
        try:
            if file_path.endswith('.csv'): results_df.to_csv(file_path, index=False, float_format='%.5f')
            elif file_path.endswith('.xlsx'): results_df.to_excel(file_path, index=False, float_format='%.5f', engine='openpyxl')
            print(f"\n통합 분석 요약 테이블을 성공적으로 저장했습니다: {file_path}")
        except Exception as e: print(f"\n파일 저장 중 오류가 발생했습니다: {e}")

def main():
    print("=" * 50); print("  GIWAXS 피크 하이브리드 분석 프로그램 (Batch Mode)"); print("=" * 50)
    
    file_paths = select_multiple_data_files()
    if not file_paths: print("\n파일이 선택되지 않았습니다. 프로그램을 종료합니다."); return
        
    file_roi_map = {}
    for file_path in file_paths:
        try:
            print(f"\n--- ROI 지정 시작: {os.path.basename(file_path)} ---")
            # Analyzer 인스턴스는 ROI 선택 시에만 임시로 사용
            analyzer_for_roi = GIWAXSPeakAnalyzer(file_path)
            selected_rois = analyzer_for_roi.select_rois_interactively()
            if selected_rois:
                file_roi_map[file_path] = selected_rois
        except Exception as e:
            print(f"'{os.path.basename(file_path)}' 파일 처리 중 오류 발생: {e}")
            continue

    if not file_roi_map: print("\n유효한 ROI가 선택되지 않았습니다. 분석을 종료합니다."); return
    
    print("\n\n" + "=" * 50); print("      모든 파일 ROI 지정 완료, 통합 분석 시작"); print("=" * 50)
    
    all_files_results = []
    
    save_summary = input("각 ROI의 요약 그래프를 이미지로 저장하시겠습니까? (y/n, 기본값 y): ").lower() != 'n'
    save_detailed = input("각 컬럼의 상세 피팅 그래프를 이미지로 저장하시겠습니까? (y/n, 기본값 y): ").lower() != 'n'
    save_fit_data = input("피팅된 곡선 데이터를 파일로 저장하시겠습니까? (y/n, 기본값 n): ").lower() == 'y'

    for file_path, rois in file_roi_map.items():
        try:
            analyzer = GIWAXSPeakAnalyzer(file_path)
            analyzer.set_saving_preferences(save_summary, save_detailed, save_fit_data)
            
            for i, (xmin, xmax) in enumerate(rois):
                analyzer.analyze_peaks_in_roi(i, xmin, xmax)
            
            all_files_results.extend(analyzer.analysis_results_for_file)
        
        except Exception as e:
            print(f"'{os.path.basename(file_path)}' 파일 분석 중 오류 발생: {e}")
            continue

    if all_files_results:
        print("\n" + "=" * 50); print("          모든 분석 완료 - 최종 결과 요약"); print("=" * 50)
        results_df = pd.DataFrame(all_files_results)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(results_df)
        
        save_prompt = input("\n이 통합 분석 결과 요약 테이블을 파일로 저장하시겠습니까? (y/n): ").lower()
        if save_prompt == 'y':
            save_final_summary_table(all_files_results)
    else:
        print("\n유효한 분석 결과가 없습니다.")

if __name__ == "__main__":
    main()
