import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def get_kernel_from_c(c, basis, support=None):
    """
    Construct a kernel function from coefficients and basis functions.

    Parameters:
        c (np.ndarray): Coefficient vector
        basis (list): List of basis functions (SymPy or NumPy compatible)
        support (tuple, optional): Function domain (default [0,10])

    Returns:
        tuple: (phi (numpy function), phi_cheb (approximation), phi_kernel_sympy (sympy expression))
    """
    x = sp.symbols('x')
    phi = 0
    for i in range(len(basis)):
        phi += c[i] * basis[i]
    phi_kernel_sympy = phi

    # Convert symbolic expression to numerical function
    phi = sp.lambdify(x, phi, 'numpy')

    # Currently Chebyshev approximation is just a placeholder
    phi_cheb = phi
    return phi, phi_cheb, phi_kernel_sympy


def learning_settings(basisType, dyn_sys=None, opts=None):


    learning_setup = {}
    x=sp.Symbol('x')
    if isinstance(basisType, (int, float)):
        print(f"Parametric inference: kernel-type = {basisType}")
    else:
        print(f"Nonparametric inference: kernel-type = {basisType}")


    if basisType == 1:  
        p, q, cut = 8, 2, 0.5
        learning_setup['dict'] = [
            sp.Piecewise((x ** (-p - 1), sp.Abs(x) > cut), (0, True)),
            sp.Piecewise((x ** (-q - 1), sp.Abs(x) > cut), (0, True)),
            sp.Piecewise((1, sp.Abs(x) <= cut), (0, True)),
            1,
            x ** 2
        ]
        learning_setup['c'] = np.array([-0.3333, 1.3333, -160, 3, -2])

    elif basisType == 2:  
        learning_setup['dict'] = [
            lambda x: x ** 2,
            lambda x: np.abs(x),
            lambda x: np.cos(x)
        ]
        learning_setup['c'] = np.array([0.1, 0.1, 3]).reshape(-1, 1)

    elif basisType == 3:  
        n = 10
        learning_setup['dict'] = [lambda x, i=i: np.sin(x * i + i) * 2 ** (i - 2) for i in range(1, n + 1)]
        learning_setup['c'] = np.ones(n)

    elif basisType == 4:  # 字典 1 (另一种定义)
        p, q, cut = 2, 0.2, 0.5
        dict1 = lambda x: x ** (-p - 1) * (np.abs(x) >= cut)
        dict2 = lambda x: -x ** (-q - 1) * (np.abs(x) >= cut)
        dict3 = lambda x: -1 * (np.abs(x) <= cut)
        c1, c2 = -1, 1
        c3 = (c1 * dict1(cut) + c2 * dict2(cut)) / dict3(cut)
        learning_setup['dict'] = [dict1, dict2, dict3]
        learning_setup['c'] = np.array([c1, c2, c3])

    elif basisType == 5:  # 核函数设置 5
        learning_setup['dict'] = [
            lambda x: x ** 2,
            lambda x: x ** 4,
            lambda x: np.zeros_like(x) + 1
        ]
        learning_setup['c'] = np.array([-10, -10, -0.1])

    elif basisType == 6:  
        p, q, cut = 8, 2, 0.5
        learning_setup['dict'] = [
            x ** (-p - 1) * sp.Piecewise((1,np.abs(x) > cut),(0,True)),
            x ** (-q - 1) * sp.Piecewise((1,np.abs(x) > cut),(0,True)),
            sp.Piecewise((1,np.abs(x) <= cut),(0,True)),
        ]
        learning_setup['c'] = np.array([-0.3333, 1.3333, -160])

    elif basisType=='Polynomial':
        learning_setup['dict'] = [
            1,
            x,
            x*x,
            x**3,
            x**4,
            x**5,
                    ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([1, 1, 1, 1, 1,1])

    elif basisType=='ictal':
        learning_setup['dict']=[
            1,
            x,
            sp.tanh(x),
            sp.tanh(2*x),
            sp.exp(-x)*sp.sign(x),
            sp.exp(-2*x)*sp.sign(x),
            -x*sp.exp(-x**2)
        ]
        learning_setup['dict_sympy'] = []
        for i in range(7):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([1,1,1,1,1,1,1])

    elif basisType=='Polynomial+activation function':
        learning_setup['dict'] = [
            1,
            x,
            # x*x,
            x**3,
            # sp.Abs(x),
            # x*x*sp.sign(x),
            sp.exp(-x*x),
            sp.tanh(x),

        ]
        learning_setup['dict_sympy'] = []
        for i in range(5):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([1,1,1,1,1])

    elif basisType=='Local':
        learning_setup['dict'] = [
            1,
            x,
            sp.tanh(x),
            sp.tanh(2 * x),
            sp.exp(-x) * sp.sign(x),
            sp.exp(-2 * x) * sp.sign(x),
            -x * sp.exp(-x ** 2),
            0,
            0,
            0,

        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100, 100, 100, 100, 100, 100, 100, 100,100,100])



    elif basisType=='HHmodel':
        learning_setup['dict']=[1/(1+sp.exp(-x)),0,0]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy=sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100,100,100])




    elif basisType=='Test':
        learning_setup['dict'] = [
            1,
            x,
            sp.tanh(x),
            sp.tanh(2 * x),
            sp.exp(-x) * sp.sign(x),
            sp.exp(-2 * x) * sp.sign(x),
            -x * sp.exp(-x ** 2),
            0,
            0,
            0,
        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100, 100, 100, 100, 100, 100, 100, 200, 200,200])



    elif basisType=='huifuqi':
        learning_setup['dict'] = [
            sp.Piecewise((1/(1+sp.exp(-x+10)),x<0),(0,True)),
            sp.sign(x)/(1+sp.Abs(x)),
            sp.exp(-x)/sp.sin(500*x),
            1,
            x,
            sp.exp(-x),
            0,
            0
        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([1, 1, 1, 1, 1, 1,1,1])

    elif basisType=='typical_example_Lenard_Jones':
        learning_setup['dict'] = []
        learning_setup['dict_sympy']=[]
        for k in range(3):
            learning_setup['dict'].append(
                (x ** (-9)) * (sp.Piecewise((1, sp.Abs(x) > 0.5+0.25*k), (0, True)))
            )

        for k in range(3,6):
            learning_setup['dict'].append(
                (x ** (-3)) * (sp.Piecewise((1, sp.Abs(x) > 0.5+0.25*(k-3)), (0, True)))
            )

        for k in range(6,10):
            learning_setup['dict'].append(
                sp.Piecewise((1.0, sp.Abs(x) <= (0.5+0.25*(k-6))), (0, True))
            )
        for i in range(10):
            dict_sympy=sp.lambdify(x,learning_setup['dict'][i],modules='numpy' )
            learning_setup['dict_sympy'].append(dict_sympy)
            
        learning_setup['c']=np.zeros(10)
        learning_setup['c'][0] = -1 / 3  
        learning_setup['c'][3] = 4 / 3 
        learning_setup['c'][6] = -160  

        learning_setup['basis_case'] = 'typical_example_Lenard_Jones'

    elif basisType=='build data':
        learning_setup['dict'] = [
            sp.sin(x),
            0
        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100,200])

    elif basisType=='sanjiao':
        learning_setup['dict'] = [
            sp.sin(x),
            sp.cos(x),
            sp.sin(2*x),
            sp.cos(2*x),
            sp.sin(3*x),
            sp.cos(3*x),
            sp.sin(4*x),
            sp.cos(4*x),
        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100, 100,100,100,100,100,100,100])

    elif basisType=='learn from data':
        learning_setup['dict'] = [
            sp.cos(x),
            sp.sin(2*x),
            0*x,
            0*x,
        ]
        learning_setup['dict_sympy'] = []
        for i in range(len(learning_setup['dict'])):
            dict_sympy = sp.lambdify(x, learning_setup['dict'][i], modules='numpy')
            learning_setup['dict_sympy'].append(dict_sympy)
        learning_setup['c'] = np.array([100,100,100, 100])


    learning_setup['kernelSupp'] = [0, 10]
    learning_setup['n'] = len(learning_setup['dict'])
    learning_setup['phi_kernel'],_,learning_setup['phi_kernel_sympy'] = get_kernel_from_c(learning_setup['c'], learning_setup['dict'], learning_setup['kernelSupp'])


    return learning_setup


def plot_orange_heatmap(matrix, title="Heatmap", save=False):
    """
    Plot an orange gradient heatmap without axes or colorbar.

    Parameters:
        matrix (np.ndarray): 2D matrix with values (expected range [0,1])
        title (str): Plot title
        save (bool): Save figure to file if True

    Returns:
        fig, ax: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrix, cmap='YlGn', vmin=-0.1, vmax=1, aspect='auto')
    ax.axis('off')
    plt.tight_layout()

    if save:
        fig.savefig(f"{title}_heatmap.png", bbox_inches='tight')

    plt.show()
    return fig, ax


def set_graph(graph_type):
    """
    Construct adjacency matrix for graph.

    Parameters:
        graph_type (str): 'random' for random adjacency, 'ones' for uniform adjacency

    Returns:
        np.ndarray: NxN adjacency matrix
    """
    N = 16
    sparsity = 1.0

    if graph_type == 'random':
        np.random.seed(42)
        A = np.zeros((N, N))
        for i in range(N):
            r = np.random.rand(N - 1)

            # Set a number of entries to zero according to sparsity
            num = max(min(N - 1 - int(np.floor(sparsity * (N - 1))), N - 2), 0)
            zero_indices = np.random.choice(N - 1, size=num, replace=False)
            r[zero_indices] = 0

            # Insert into adjacency matrix
            A[:, i] = np.insert(r, i, 0)

        # Normalize columns
        column_norm = np.sqrt(np.sum(A ** 2, axis=0))
        A = A / column_norm
        return A

    elif graph_type == 'ones':
        x = 1 / np.sqrt(N - 1)
        A = np.full((N, N), x)
        np.fill_diagonal(A, 0)
        return A



# Kernel type
kernel_type = 'sanjiao'
learning_setup_ref = learning_settings(kernel_type)

# Plot random initialization
A_random = set_graph('random')
matrix_0 = np.load('figure_3_data\ZZ_sanjiao_0_(42)_matrix19.npy')
vector_0 = np.load('figure_3_data\ZZ_sanjiao_0_(42)_vector19.npy')

plot_orange_heatmap(A_random, title="Random Graph")
plot_orange_heatmap(matrix_0, title="Random Matrix")

f0, _, _ = get_kernel_from_c(vector_0, learning_setup_ref['dict'])
fig= plt.subplots(figsize=(8, 2))
x = np.linspace(0, 20, 1000)
plt.plot(x, f0(x), '-', color='darkgreen', lw=6)
plt.axis('off')
plt.show()

# Plot all-ones initialization
A_ones = set_graph('ones')
matrix_1 = np.load('figure_3_data\ZZ_sanjiao_0_(0)_matrix19.npy')
vector_1 = np.load('figure_3_data\ZZ_sanjiao_0_(0)_vector19.npy')

plot_orange_heatmap(A_ones, title="Ones Graph")
plot_orange_heatmap(matrix_1, title="Ones Matrix")

fig= plt.subplots(figsize=(8, 2))
f1, _, _ = get_kernel_from_c(vector_1, learning_setup_ref['dict'])
plt.plot(x, f1(x), '-', color='darkgreen', lw=6)
plt.axis('off')
plt.show()
