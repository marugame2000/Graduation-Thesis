import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

# 既知の波動関数
def known_wave_function_1(x):
    return np.sin(x)  # 基底状態（例）

def known_wave_function_2(x):
    return np.cos(x)  # 第一励起状態（例）

# 試行波動関数
def trial_wave_function(x, params):
    a, b, c = params
    return a * np.sin(b * x) + c * np.cos(b * x)

# ハミルトニアン関数
def hamiltonian(x, psi):
    kinetic_energy = -0.5 * np.gradient(np.gradient(psi, x), x)
    potential_energy = x**2 * psi
    return kinetic_energy + potential_energy

# 直交性を確保する関数
def orthogonality_condition(params, *args):
    func1 = args[0]
    func2 = args[1]
    result1, _ = quad(lambda x: trial_wave_function(x, params) * func1(x), -np.pi, np.pi)
    result2, _ = quad(lambda x: trial_wave_function(x, params) * func2(x), -np.pi, np.pi)
    return result1**2 + result2**2

# エネルギー期待値の計算
def energy_expectation(params, *args):
    H_func = args[0]
    x_values = np.linspace(-np.pi, np.pi, 1000)
    psi_values = trial_wave_function(x_values, params)
    total_energy = H_func(x_values, psi_values)
    # 積分計算で配列を処理する
    integral, _ = quad(lambda x: trial_wave_function(x, params) * np.interp(x, x_values, total_energy), -np.pi, np.pi)
    norm, _ = quad(lambda x: trial_wave_function(x, params)**2, -np.pi, np.pi)
    return integral / norm

# 初期パラメータ
initial_params = [1, 1, 1]

# 最適化
cons = ({'type': 'eq', 'fun': orthogonality_condition, 'args': (known_wave_function_1, known_wave_function_2)})
result = minimize(energy_expectation, initial_params, args=(hamiltonian,), constraints=cons)

# 結果の表示
if result.success:
    optimized_params = result.x
    print("Optimized Parameters:", optimized_params)
else:
    print("Optimization failed:", result.message)
