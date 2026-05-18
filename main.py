import numpy as np
import matplotlib.pyplot as plt

# исходная функция f(x) на [0, 4]
def f(x):
    x = np.asarray(x)
    return np.where(x < 2, x / 2, 1)


# общий тригонометрический ряд Фурье

def a_general(n):
    if n == 0:
        return 3 / 2
    return ((-1)**n - 1) / (np.pi**2 * n**2)

def b_general(n):
    return -1 / (np.pi * n)

def S_general(x, N):
    s = a_general(0) / 2
    for n in range(1, N + 1):
        s += a_general(n) * np.cos(n * np.pi * x / 2)
        s += b_general(n) * np.sin(n * np.pi * x / 2)
    return s

def exact_S_general(x):
    """Сумма общего ряда"""
    x_mod = x % 4
    y = np.where(x_mod < 2, x_mod / 2, 1.0)
    y[np.isclose(x_mod, 0)] = 0.5
    return y



# ряд Фурье по косинусам

def a_cos(n):
    if n == 0:
        return 3 / 2
    return 4 * (np.cos(n * np.pi / 2) - 1) / (np.pi**2 * n**2)

def S_cos(x, N):
    s = a_cos(0) / 2
    for n in range(1, N + 1):
        s += a_cos(n) * np.cos(n * np.pi * x / 4)
    return s

def exact_S_cos(x):
    """Сумма ряда по косинусам"""
    x_mod = np.abs((x + 4) % 8 - 4)
    return np.where(x_mod < 2, x_mod / 2, 1.0)




# ряд Фурье по синусам

def b_sin(n):
    return 4 * np.sin(n * np.pi / 2) / (np.pi**2 * n**2) - 2 * (-1)**n / (np.pi * n)

def S_sin(x, N):
    s = np.zeros_like(x)
    for n in range(1, N + 1):
        s += b_sin(n) * np.sin(n * np.pi * x / 4)
    return s

def exact_S_sin(x):
    """Сумма ряда по синусам"""
    x_mod = (x + 4) % 8 - 4
    y = np.sign(x_mod) * np.where(np.abs(x_mod) < 2, np.abs(x_mod) / 2, 1.0)
    y[np.isclose(np.abs(x_mod), 4)] = 0.0
    y[np.isclose(x_mod, 0)] = 0.0
    return y


VALUES = [3, 10, 30]

# построение графиков для общего ряда
x_gen = np.linspace(-4, 8, 4000)

for N in VALUES:
    plt.figure(figsize=(8, 4))
    
    plt.plot(x_gen, exact_S_general(x_gen), 'k--', linewidth=1.5, alpha=0.8, label='Сумма ряда')
    plt.plot(x_gen, S_general(x_gen, N), 'r-', linewidth=1.5, label=f'S_{N}(x)')
    
    plt.title(f'Общий ряд Фурье, N = {N}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    filename = f'general_N{N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()



# построение графиков для ряда по косинусам
x_cos = np.linspace(-8, 8, 4000)

for N in VALUES:
    plt.figure(figsize=(8, 4))
    
    plt.plot(x_cos, exact_S_cos(x_cos), 'k--', linewidth=1.5, alpha=0.8, label='Сумма ряда')
    plt.plot(x_cos, S_cos(x_cos, N), 'b-', linewidth=1.5, label=f'S_{N}(x)')
    
    plt.title(f'Ряд Фурье по косинусам, N = {N}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    filename = f'cos_N{N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# построение графиков для ряда по синусам
x_sin = np.linspace(-8, 8, 4000)

for N in VALUES:
    plt.figure(figsize=(8, 4))
    
    plt.plot(x_sin, exact_S_sin(x_sin), 'k--', linewidth=1.5, alpha=0.8, label='Сумма ряда')
    plt.plot(x_sin, S_sin(x_sin, N), 'g-', linewidth=1.5, label=f'S_{N}(x)')
    
    plt.title(f'Ряд Фурье по синусам, N = {N}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    filename = f'sin_N{N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

