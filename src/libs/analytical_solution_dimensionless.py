import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def solution(V0,k):
    
    d = 1

    # 定数の値を設定
    m = 1
    #V0 = 100 * 1.6e-19
    #d = 1 * 1e-9
    hbar = 1
    R = np.sqrt(m * V0 * d**2 / (2 * hbar**2))
    #print(R)

    # αの範囲を設定
    alpha_values = np.linspace(0.01, 100, 2000000)  # 0を避けるために0.1から開始

    # 各方程式に対応するβの値を計算
    beta1_pos = alpha_values * np.tan(alpha_values)  # β = α tan(α)
    beta1_neg = -alpha_values / np.tan(alpha_values)  # β = -α cot(α)
    beta2 = np.sqrt(R**2 - alpha_values**2)

    # β = α tan(α)との交点を計算
    threshold = 0.5
    differences_pos = np.abs(beta1_pos - beta2)
    approx_intersections_indices_pos = np.where(differences_pos < threshold)[0]

    # β = -α cot(α)との交点を計算
    differences_neg = np.abs(beta1_neg - beta2)
    approx_intersections_indices_neg = np.where(differences_neg < threshold)[0]

    # 連続する近似交点をグルーピング
    groups_pos = []
    current_group = [approx_intersections_indices_pos[0]]
    for i in range(1, len(approx_intersections_indices_pos)):
        if approx_intersections_indices_pos[i] - approx_intersections_indices_pos[i - 1] == 1:
            current_group.append(approx_intersections_indices_pos[i])
        else:
            groups_pos.append(current_group)
            current_group = [approx_intersections_indices_pos[i]]
    if current_group:
        groups_pos.append(current_group)

    groups_neg = []
    current_group = [approx_intersections_indices_neg[0]]
    for i in range(1, len(approx_intersections_indices_neg)):
        if approx_intersections_indices_neg[i] - approx_intersections_indices_neg[i - 1] == 1:
            current_group.append(approx_intersections_indices_neg[i])
        else:
            groups_neg.append(current_group)
            current_group = [approx_intersections_indices_neg[i]]
    if current_group:
        groups_neg.append(current_group)

    # 各グループの中央の交点を取得
    final_intersections_pos = [(alpha_values[int(np.median(group))], beta1_pos[int(np.median(group))]) for group in groups_pos]
    final_intersections_neg = [(alpha_values[int(np.median(group))], beta1_neg[int(np.median(group))]) for group in groups_neg]


    # 交点の座標を出力
    #for intersection in final_intersections_pos:
        #print(f"Intersection with tan: α = {intersection[0]:.2f}, β = {intersection[1]:.2f}")

    #for intersection in final_intersections_neg:
        #print(f"Intersection with cot: α = {intersection[0]:.2f}, β = {intersection[1]:.2f}")

    # エネルギーEの計算
    def compute_energy(alpha):
        return (2 * hbar**2 * alpha**2) / (m * d**2)

    # tanとの交点でのエネルギー
    energies_pos = [compute_energy(alpha) for alpha, _ in final_intersections_pos]

    # cotとの交点でのエネルギー
    energies_neg = [compute_energy(alpha) for alpha, _ in final_intersections_neg]

    # エネルギーの出力
    #print("Energies from intersections with tan:")
    I=1
    for E in energies_pos:
        #print(f"E{I} = {E:.5f} (eV)")
        I=I+2

    #print("\nEnergies from intersections with cot:")
    N=2
    for E in energies_neg:
        #print(f"E{N} = {E:.5f} (eV)")
        N=N+2

    P=max(I,N)
    P=P-2 
    #print(P)

    if k % 2==1 :
        # 定義する変数
        k=(k-1)//2
        #print(k)
        A, B, x = sp.symbols('A B x', real=True)
        K = sp.sqrt((2 * m * energies_pos[k])/(hbar ** 2))
        Q = sp.sqrt((2 * m * ( V0 - energies_pos[k]))/(hbar ** 2))

        # 波動関数
        psi_1 = A * sp.cos(K*x)
        psi_2 = B * sp.exp(Q*x)
        psi_3 = B * sp.exp(-Q*x)

        # 各領域での規格化積分
        integral_1 = sp.integrate(psi_1**2, (x, -d/2, d/2))
        integral_2 = sp.integrate(psi_2**2, (x, -sp.oo, -d/2))
        integral_3 = sp.integrate(psi_3**2, (x, d/2, sp.oo))

        # 合計が1になるようにする
        eq = sp.Eq(integral_1 + integral_2 + integral_3, 1)
        eq2 = sp.Eq(psi_1.subs(x, -d/2), psi_2.subs(x, -d/2))
        eq3 = sp.Eq(psi_1.subs(x, d/2), psi_3.subs(x, d/2))

        # AとBの方程式を解く
        solution = sp.solve([eq, eq2, eq3])

        #print(solution)
        
    elif k % 2==0 :
        
        # 定義する変数
        k=k//2-1
        A, B, x = sp.symbols('A B x', real=True)
        K = sp.sqrt((2 * m * energies_neg[k])/(hbar ** 2))
        Q = sp.sqrt((2 * m * ( V0 - energies_neg[k]))/(hbar ** 2))

        # 波動関数
        psi_1 = A * sp.sin(K*x)
        psi_2 = B * sp.exp(Q*x)
        psi_3 = -B * sp.exp(-Q*x)

        # 各領域での規格化積分
        integral_1 = sp.integrate(psi_1**2, (x, -d/2, d/2))
        integral_2 = sp.integrate(psi_2**2, (x, -sp.oo, -d/2))
        integral_3 = sp.integrate(psi_3**2, (x, d/2, sp.oo))

        # 合計が1になるようにする
        eq = sp.Eq(integral_1 + integral_2 + integral_3, 1)
        eq2 = sp.Eq(psi_1.subs(x, -d/2), psi_2.subs(x, -d/2))
        eq3 = sp.Eq(psi_1.subs(x, d/2), psi_3.subs(x, d/2))

        # AとBの方程式を解く
        solution = sp.solve([eq, eq2, eq3])

        #print(solution)
        
    psi_1_subs = psi_1.subs(A, solution[0][A])
    psi_2_subs = psi_2.subs(B, solution[0][B])
    psi_3_subs = psi_3.subs(B, solution[0][B])
    #print(psi_1_subs.free_symbols)

    f_psi_1 = sp.lambdify(x, psi_1_subs, "numpy")
    f_psi_2 = sp.lambdify(x, psi_2_subs, "numpy")
    f_psi_3 = sp.lambdify(x, psi_3_subs, "numpy")

    x_vals1 = np.linspace(-d/2, d/2, 300)
    y_vals1 = -f_psi_1(x_vals1)

    x_vals2 = np.linspace(-5, -d/2, 1350)
    y_vals2 = -f_psi_2(x_vals2)

    x_vals3 = np.linspace(d/2, 5, 1350)
    y_vals3 = -f_psi_3(x_vals3)
    
    x_vals = np.linspace(-5, 5, 3000)
    y_vals = np.concatenate([y_vals2, y_vals1, y_vals3])

    return y_vals

