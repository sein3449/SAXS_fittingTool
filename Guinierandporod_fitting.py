import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

def load_data(filename):
    q, I = [], []
    with open(filename, 'r') as f:
        for line in f:
            if ',' in line:
                q_val, I_val = map(float, line.strip().split(','))
                if I_val > 0 and q_val > 0:
                    q.append(q_val)
                    I.append(I_val)
    return np.array(q), np.array(I)

# 파일 경로 설정
file_path = os.path.join(os.path.expanduser(r"C:\Users\owner\OneDrive - postech.ac.kr\SOMEBODY HELP\김민주 박사\GI-SAXS\POSTECH_SeinChung_s1h_026_ca_008_30s_0001"), 
                        "POSTECH_SeinChung_s1h_026_ca_008_30s_0001_180degree_Q.csv")
q, I = load_data(file_path)

#--------------------------------------------------
# 사용자 조정 가능 파라미터
#--------------------------------------------------
initial_guess = {
    'G': np.max(I),     # 초기 G 값
    'Rg': 10.0,         # 초기 Rg 추정값 (Å 단위)
    'q_min': 0.003,     # Guinier 핏팅 최소 q 값
    'q_max': 0.1        # Guinier 핏팅 최대 q 값
}
#--------------------------------------------------

def guinier_model(q, G, Rg):
    return G * np.exp(-(q**2 * Rg**2)/3)

def porod_model(q, D, p):
    return D * q**-p

def fit_guinier(q, I, initial_params):
    fit_mask = (q >= initial_params['q_min']) & (q <= initial_params['q_max'])
    q_fit = q[fit_mask]
    I_fit = I[fit_mask]
    
    p0 = [initial_params['G'], initial_params['Rg']]
    
    try:
        popt, pcov = curve_fit(guinier_model, q_fit, I_fit, p0=p0, 
                               bounds=([0, 1], [np.inf, 100]))
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Guinier fitting failed: {e}")
        return None, None
    
    return popt, perr

def fit_porod(q, I, q_crossover):
    porod_range = q >= q_crossover
    try:
        popt, pcov = curve_fit(porod_model, q[porod_range], I[porod_range],
                               p0=[1, 4], bounds=([0, 1], [np.inf, 5]))
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Porod fitting failed: {e}")
        return None, None
    
    return popt, perr

# Guinier 핏팅 실행
popt_guinier, perr_guinier = fit_guinier(q, I, initial_guess)

if popt_guinier is not None:
    G, Rg = popt_guinier
    G_err, Rg_err = perr_guinier
    
    print("\n=== Guinier Fit Results ===")
    print(f"Initial G: {initial_guess['G']:.2e}")
    print(f"Fitted G: {G:.2e} ± {G_err:.2e}")
    print(f"Initial Rg: {initial_guess['Rg']:.2f} Å")
    print(f"Fitted Rg: {Rg:.2f} ± {Rg_err:.2f} Å")
    
    # Porod 핏팅을 위한 경계점 계산
    q_crossover = 1.3 / Rg
    
    # Porod 핏팅 실행
    popt_porod, perr_porod = fit_porod(q, I, q_crossover)
    
    if popt_porod is not None:
        D, p = popt_porod
        D_err, p_err = perr_porod
        
        print("\n=== Porod Fit Results ===")
        print(f"D: {D:.2e} ± {D_err:.2e}")
        print(f"Porod exponent p: {p:.2f} ± {p_err:.2f}")
        
        # 결과 시각화
        plt.figure(figsize=(12, 6))
        plt.loglog(q, I, 'ko', markersize=3, alpha=0.3, label='Raw Data')
        
        # Guinier 영역
        guinier_range = q <= q_crossover
        plt.loglog(q[guinier_range], guinier_model(q[guinier_range], G, Rg), 
                   'r-', lw=2, label='Guinier Fit')
        
        # Porod 영역
        porod_range = q >= q_crossover
        plt.loglog(q[porod_range], porod_model(q[porod_range], D, p),
                   'b--', lw=2, label='Porod Fit')
        
        plt.axvline(q_crossover, color='g', linestyle=':', 
                    label=f'q*Rg = 1.3 ({q_crossover:.3f} Å⁻¹)')
        plt.xlabel('q (Å⁻¹)', fontsize=12)
        plt.ylabel('I(q) (a.u.)', fontsize=12)
        plt.title('Segmented Guinier-Porod Fit Analysis', fontsize=14)
        plt.legend()
        plt.grid(True, which='both', alpha=0.4)
        plt.show()
    else:
        print("Porod fitting failed.")
else:
    print("Guinier fitting failed. Try adjusting initial parameters.")

# 초기값 튜닝 가이드
print("\n** Initial Parameter Tuning Guide **")
print("1. G 초기값: 데이터의 최대 강도 값 부근에서 시작")
print("2. Rg 초기값: 예상되는 나노구조 크기의 1/√3 근처 값 사용")
print("3. q_min/q_max: Guinier 직선 구간이 관측되는 영역 선택")
print("4. 핏팅 실패 시 Rg 초기값을 50%~200% 범위에서 변경해가며 시도")
