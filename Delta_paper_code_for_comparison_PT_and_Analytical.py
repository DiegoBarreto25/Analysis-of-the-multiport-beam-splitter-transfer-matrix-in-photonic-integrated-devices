

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:57:17 2025

@author: Diego Guimarães Barreto
"""


import numpy as np
import math



def omega(u, p, N):
    """
    omega_{u,p} = cos(u*pi/(N+1)) - cos(p*pi/(N+1))
    """
    
    return np.cos(u*np.pi/(N+1)) - np.cos(p*np.pi/(N+1))

def gamma(u, p, N):
    """
    gamma_{u,p} = sin(u*pi/(N+1)) * sin(p*pi/(N+1))
                  + sin(u*N*pi/(N+1)) * sin(p*N*pi/(N+1))
    """
    
    return (np.sin(u*np.pi/(N+1)) * np.sin(p*np.pi/(N+1))
            + np.sin(u*N*np.pi/(N+1)) * np.sin(p*N*np.pi/(N+1)))



def X_pk(i,p, k, N, kappa, delta):
    """
    X_{p,k} = A_{p,k} - B_{p,k} + termos de segunda ordem,
      onde A_{p,k} = sqrt(2/(N+1)) * sin(p*k*pi/(N+1))
      e   B_{p,k} = sum_{u != p} [ C_up(u,p) * sin(u*k*pi/(N+1)) ]
      
    Agora incluímos também os termos de segunda ordem em delta.
    """
    # Termo A_{p,k}
    A_pk = np.sqrt(2.0 / (N + 1)) * np.sin(p * k * np.pi / (N + 1))
    
    # Soma das correções (B_{p,k} de primeira ordem)
    somatorio = 0.0
    for u in range(1, N + 1):
        if u != p:
            num = (2.0/(N+1))**(3/2)
            denom = 2*kappa * omega(u, p, N)
            somatorio +=  num * (delta / denom) * gamma(u, p, N) * np.sin(u * k * np.pi / (N + 1))
    
    if i==2:
        B_pk = somatorio
    else:
        B_pk = 0

    return A_pk - B_pk #+ second_order




def omega_matrix(N):
    matriz = np.zeros((N, N))
    for u in range(1, N+1):
        for p in range(1, N+1):
            matriz[u-1, p-1] = omega(u, p, N)
    return matriz



def lambdas_ps(p, N, kappa, delta):
    """
    Calcula lambda_p considerando as contribuições de 0ª, 1ª, 2ª e agora 3ª ordem em delta.
    """
    matriz = omega_matrix(N)
    # 0ª ordem
    lam0 = -delta/2 + 2.0 * kappa * np.cos(p * np.pi / (N + 1))
    
    # 1ª ordem
    gamma_pp = (np.sin(p * np.pi / (N + 1))**2 + np.sin(p * N * np.pi / (N + 1))**2)
    lam1 = (2 * delta / (N + 1)) * gamma_pp
    # 2ª ordem
    lam2 = 0.0
    for m in range(1, N + 1):
        if m != p:
            
            gmp = gamma(m, p, N)
            om_mp = omega(m, p, N)
            val = ((2.0 / (N + 1))**2) * ((delta)**2)
            lam2 += val * ((gmp**2) / (2 * kappa * om_mp))
    lamb = [lam0,lam0 + lam1,lam0 + lam1 - lam2]

    return lamb

def build_X_matrix(i,N, kappa, delta):
    """
    Constrói a matriz X NxN (cada elemento X_{p,k}).
    """
    X = np.zeros((N,N), dtype=float)
    for p in range(1, N+1):
        for k in range(1, N+1):
            X[p-1, k-1] = X_pk(i,p, k, N, kappa, delta)

    return X

def build_lambda_vector(i,N, kappa, delta):
    """
    Retorna array [lambda_1, lambda_2, ..., lambda_N].
    """
    lambdas = []
    for p in range(1, N+1):
        lambd = lambdas_ps(p, N, kappa, delta)
        lambdas.append(lambd[i])
    return np.array(lambdas, dtype=complex)

def build_expLambda_matrix(lambdas, z):
    """
    Constroi diag( e^{i * lambda_1 * z}, e^{i * lambda_2 * z}, ..., e^{i * lambda_N * z} ).
    """
    N = len(lambdas)
    Lam = np.zeros((N,N), dtype=complex)
    for l in range(N):
        Lam[l,l] = np.exp(1j * lambdas[l] * z)
    return Lam

# Função de fidelidade entre duas matrizes
def unitary_fidelity(U, V):
    numerator = np.abs(np.trace(U.conj().T @ V)) ** 2
    denominator = np.trace(U.conj().T @ U) * np.trace(V.conj().T @ V)
    fidelity = np.abs(numerator / denominator)
    return fidelity

def generate_matrix(i,N, kappa, delta, z):
    X = build_X_matrix(i,N, kappa, delta)
    lambdas = build_lambda_vector(i,N, kappa, delta)
    expLam = build_expLambda_matrix(lambdas, z)
    X_inv = np.linalg.inv(X)
    BS = X.T @ expLam @ X

    return BS



def calculate_nbs_numerically(n_val, delta_val, kappa_val, L_val):
    import sympy
    """
    Calculates the NBS evolution matrix for a system of 'n_val' levels numerically.
    """
    # --- PASSO 1: SETUP SIMBÓLICO ---
    x, delta, kappa = sympy.symbols('x delta kappa')
    I = sympy.I
    
    # Constrói o polinômio característico simbolicamente
    poly_in_x = (kappa**2 * sympy.chebyshevu(n_val, x)
                 + (delta**2 - 4 * kappa * delta * x) * sympy.chebyshevu(n_val - 2, x)
                 + 2 * kappa * delta * sympy.chebyshevu(n_val - 3, x))

    # --- PASSO 2: SUBSTITUIÇÃO E RESOLUÇÃO NUMÉRICA ---
    poly_numerical = poly_in_x.subs({delta: delta_val, kappa: kappa_val})
    
    try:
        solutions_for_x = [sympy.re(sol.evalf()) for sol in sympy.solve(poly_numerical, x)]
        print(f'N={n_val}')
        print(solutions_for_x)
        print('')
    except NotImplementedError:
        print(f"SymPy could not solve the polynomial for N={n_val}.")
        return None, None

    # --- PASSO 3: CÁLCULO DE AUTOVETORES E AUTOVALORES ---
    def U(n_u, x_val):
        return sympy.chebyshevu(n_u, x_val) if n_u >= 0 else 0

    def CalcEigenvector_MapleLogic(N, x_k, d_val, k_val):
        v = sympy.zeros(N, 1)
        factor = -d_val / k_val + 2 * x_k
        for i in range(1, N + 1):
            if i == 1:
                #Para 3 dimensões isso pode acontecer
                v[i-1] = 1 if sympy.simplify(factor).is_zero else factor
            elif i < N:
                term = U(i - 1, x_k) - (d_val / k_val) * U(i - 2, x_k)
                v[i-1] = factor * term
            else:
                previous_element = v[i-2]
                #Para 3 dimensões isso pode acontecer
                if sympy.simplify(previous_element).is_zero: v[i-1] = -v[0]
                else:
                    term = U(N - 2, x_k) - (d_val / k_val) * U(N - 3, x_k)
                    v[i-1] = term
        return v
    
    def NormalizeEigenvector(v):
        norm_v = v.norm()
        return v / norm_v if norm_v != 0 else v
    
    eigenvectors, diagonal_elements = [], []
    for x_k in solutions_for_x:
        if not x_k.is_real: continue
        lambda_k = -delta_val / 2 + 2 * kappa_val * x_k
        v_k = CalcEigenvector_MapleLogic(n_val, x_k, delta_val, kappa_val)
        v_k_normalized = NormalizeEigenvector(v_k)
        eigenvectors.append(v_k_normalized)
        diagonal_elements.append(sympy.exp(I * L_val * lambda_k))
        
    # --- PASSO 4: MONTAGEM FINAL E RETORNO ---
    if eigenvectors and len(eigenvectors) == n_val:
        MatrixEigenvectors = sympy.Matrix.hstack(*eigenvectors)
        M_kk = sympy.diag(*diagonal_elements)
        NBS_sympy = MatrixEigenvectors @ M_kk @ MatrixEigenvectors.H
        NBS_numpy = np.array(NBS_sympy.evalf(), dtype=np.complex128)
        prod = NBS_numpy @ NBS_numpy.T
        return NBS_numpy, prod
    else:
        print(f"Calculation failed for N={n_val}: Expected {n_val} real solutions, but found {len(eigenvectors)}.")
        return None, None


import matplotlib.pyplot as plt

# Estilo do gráfico
plt.style.use('seaborn-v0_8-talk')

# Define o tamanho da figura ANTES de plotar
plt.figure(figsize=(40, 35))

# --- NOVA FUNÇÃO DE PLOTAGEM QUE DESENHA EM UM EIXO (SUBPLOT) ESPECÍFICO ---
def plotar_em_subplot(ax, N_list, k, delta, l):
    """
    Função modificada para desenhar em um 'ax' (eixo/subplot) fornecido.
    """
    correction_orders = [0, 1, 2]
    min_fidelity = 1.0
    
    for N in N_list:
        fidelidadeBS = []
        # A função numérica retorna apenas um valor, não uma tupla
        BS_numerical,_ = calculate_nbs_numerically(N, delta, k, l)
        
        if BS_numerical is None:
            print(f"\nPulando N={N} para δ={delta} pois a solução numérica falhou.")
            continue
            
        print(f"Calculando para N={N} com δ={delta}...")
        for i in correction_orders:
            M_perturbative = generate_matrix(i, N, k, delta, l)
            fidelidade_elm = unitary_fidelity(M_perturbative, BS_numerical)
            fidelidadeBS.append(fidelidade_elm)
            if fidelidade_elm < min_fidelity:
                min_fidelity = fidelidade_elm
        
        # Usa 'ax.plot' em vez de 'plt.plot'
        ax.plot(correction_orders, fidelidadeBS, marker='o', label=f"N = {N}",linewidth = 6)

    # Configurações específicas do subplot (ax)
    ax.set_xticks(correction_orders, labels=["0", "1", "2"])
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax.set_title(f'δ = {delta}'+r'$µm^{-1}$',fontsize=45) # Título de cada subplot
    ax.grid(True)
    
    ax.set_ylim(bottom=min_fidelity*0.9999995, top=1)# Ajuste dinâmico
    


# Parâmetros
k = 0.04
delta = [0.0001,0.001,0.01,0.03]
N_list = [5,6,7,8,9,10,11,12]
l = 30

# 1. Criar a figura e a grade de subplots (2 linhas, 2 colunas)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 35))
plt.style.use('seaborn-v0_8-talk')

# 2. Iterar sobre os deltas e os eixos, preenchendo cada subplot
# 'axes.flatten()' transforma a grade 2x2 em uma lista simples de 4 eixos
for ax, u_delta in zip(axes.flatten(), delta):
    print("\n" + "="*40)
    print(f"=== PREENCHENDO SUBPLOT PARA δ = {u_delta} ===")
    plotar_em_subplot(ax, N_list, k, u_delta, l)


# 3. Adicionar um título geral à figura
fig.suptitle(f'Process Fidelity vs Perturbation Order for κ = {k}'+r'$\mathbf{µm^{-1}}$', fontsize=50,weight = 'bold')

# 4. Ajustar o layout para evitar sobreposição de títulos e eixos
fig.tight_layout(rect=[0, 0, 1, 0.96]) # Deixa espaço no topo para o suptitle

# --- NOVA SEÇÃO PARA A LEGENDA ÚNICA ---
# Pegar os 'handles' (as linhas) e 'labels' (os textos) de um dos gráficos (ex: o primeiro)
handles, labels = axes.flatten()[0].get_legend_handles_labels()
# Ajuste fino para posição dos labels
fig.text(0.5, 0.08, "Perturbation Order", ha='center', fontsize=45)   # Eixo X
fig.text(0, 0.5, "Process Fidelity", va='center', rotation='vertical', fontsize=45)  # Eixo Y

# Criar a legenda na figura (fig), posicionada na parte de baixo, com 8 colunas
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(N_list), fontsize=45,fancybox= True)

# Ajustar o layout para criar espaço para o título e a legenda de baixo
# (substitui o fig.tight_layout)
plt.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.92, hspace=0.2, wspace=0.2)
plt.savefig('perturbationtheory.png', dpi=200)
# Mostrar a figura final com todos os subplots e a legenda única
plt.show()


