import numpy as np
import matplotlib.pyplot as plt

def q_to_2theta(q, energy_keV):
    """
    q (Å⁻¹) 값을 받아 주어진 에너지(keV)에 해당하는 파장을 
    이용해 2θ (도) 값을 계산하는 함수.
    """
    # 에너지에 따른 파장 (Å) 계산
    wavelength = 12.398 / energy_keV
    # 변환식 내의 인수 계산: (q * λ) / (4π)
    argument = (q * wavelength) / (4 * np.pi)
    # 수치적 안정성을 위해 argument를 -1 ~ 1 사이로 제한
    argument = np.clip(argument, -1, 1)
    # θ를 라디안 단위로 계산
    theta_rad = np.arcsin(argument)
    # 2θ 값 (라디안)
    two_theta_rad = 2 * theta_rad
    # 라디안을 도(degree)로 변환
    two_theta_deg = np.degrees(two_theta_rad)
    return two_theta_deg

# q 값 범위 생성 (0에서 6 Å⁻¹)
q_values = np.linspace(0, 6, 300)

# 에너지 조건 설정 (keV)
energy_a = 8.95
energy_b = 11.25
energy_c = 11.08
energy_d = 15.51
energy_e = 19.99

# 각 에너지에 따른 2θ 값 계산
two_theta_a = q_to_2theta(q_values, energy_a)
two_theta_b = q_to_2theta(q_values, energy_b)
two_theta_c = q_to_2theta(q_values, energy_c)
two_theta_d = q_to_2theta(q_values, energy_d)
two_theta_e = q_to_2theta(q_values, energy_e)

# 그래프 그리기
plt.figure(figsize=(8,6))
plt.plot(q_values, two_theta_a, label=f'{energy_a} keV')
plt.plot(q_values, two_theta_b, label=f'{energy_b} keV')
plt.plot(q_values, two_theta_c, label=f'{energy_c} keV')
plt.plot(q_values, two_theta_d, label=f'{energy_d} keV')
plt.plot(q_values, two_theta_e, label=f'{energy_e} keV')
plt.xlabel('q (Å⁻¹)')
plt.ylabel('2θ (degrees)')
plt.title('q to 2θ convert')
plt.legend()
plt.grid(True)
plt.show()
