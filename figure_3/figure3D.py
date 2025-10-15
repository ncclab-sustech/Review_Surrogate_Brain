import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style

matrix_1 = np.load(f'figure_3_data\ZZ_{0}_({42})_matrix{19}.npy')
vector_1 = np.load(f'figure_3_data\ZZ_{0}_({42})_vector{19}.npy')
matrix_2 = np.load(f'figure_3_data\ZZ_sanjiao_0_({42})_matrix{19}.npy')
vector_2 = np.load(f'figure_3_data\ZZ_sanjiao_0_({42})_vector{19}.npy')

style.use('default')

U, sing_vals_1, Vh = np.linalg.svd(matrix_1)
U2, sing_vals_2, Vh2 = np.linalg.svd(matrix_2)

# Create visualization figure
plt.figure(figsize=(10, 4))  # Wider canvas for better readability

# Plot singular values for the first matrix
plt.plot(
    range(1, len(sing_vals_1)+1),  # 1-based index
    sing_vals_1,
    marker='o',         # Circular markers
    markersize=6,
    linestyle='-',      # Solid line
    linewidth=2,
    color='cornflowerblue',
    label='Matrix Polynomial'  # Data label
)

# Plot singular values for the second matrix
plt.plot(
    range(1, len(sing_vals_2)+1),
    sing_vals_2,
    marker='s',         # Square markers
    markersize=6,
    linestyle='-',
    linewidth=2,
    color='salmon',
    label='Matrix Trigonometric'  # Data label
)

# Configure plot aesthetics
plt.xlabel('Index', fontsize=14, labelpad=10)
plt.ylabel('Singular Value', fontsize=14, labelpad=10)
plt.xticks(range(1, len(sing_vals_1)+1), fontsize=12)  # Explicit index labels
plt.yticks(fontsize=12)
plt.title('Singular Value Spectrum Comparison', fontsize=16, pad=15, weight='bold')

# Add gridlines (subdued style)
plt.grid(linestyle=':', alpha=0.3, color='gray')

# Add legend with semi-transparent background
plt.legend(
    fontsize=12,
    loc='upper right',
    frameon=True,
    framealpha=0.9
)

# Customize plot borders
ax = plt.gca()
ax.spines['top'].set_visible(False)      # Remove top border
ax.spines['right'].set_visible(False)     # Remove right border
ax.spines['left'].set_alpha(0.7)          # Subdue left border
ax.spines['bottom'].set_alpha(0.7)        # Subdue bottom border

# Ensure proper element spacing
plt.tight_layout()
plt.show()


# ------------------- 1. Solve Chinese font display issue -------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Set font that supports Chinese
plt.rcParams['axes.unicode_minus'] = False             # Fix minus sign shown as square

# Load reference vectors and matrices
Vector_ref = np.load(f'figure_3_data\ZZ_{0}_({42})_vector{19}.npy')
Matrix_ref = np.load(f'figure_3_data\ZZ_{0}_({42})_matrix{19}.npy')
Vector_ref_sanjiao = np.load(f'figure_3_data\ZZ_sanjiao_{0}_({42})_vector{19}.npy')
Matrix_ref_sanjiao = np.load(f'figure_3_data\ZZ_sanjiao_{0}_({42})_matrix{19}.npy')

data_matrix, data_vector = [], []
data_matrix_sanjiao, data_vector_sanjiao = [], []
t = 15  # selected column index

# ------------------- 2. Collect data samples -------------------
for i in range(1, 11):
    sample_matrix, sample_vector = [], []
    sanjiao_sample_matrix, sanjiao_sample_vector = [], []
    for j in range(31, 40):
        # Load normal kernel data
        Vector = np.load(f'figure_3_data\ZZ_{i}_({j})_vector{9}.npy')
        Matrix = np.load(f'figure_3_data\ZZ_{i}_({j})_matrix{9}.npy')
        matrix_change = np.linalg.norm(Matrix[:, t] - Matrix_ref[:, t]) / np.linalg.norm(Matrix_ref[:, t])
        vector_change = np.linalg.norm(Vector - Vector_ref) / np.linalg.norm(Vector_ref)
        sample_matrix.append(matrix_change)
        sample_vector.append(vector_change)

        # Load triangular kernel data
        Vector_sanjiao = np.load(f'figure_3_data\ZZ_sanjiao_{i}_({j})_vector{9}.npy')
        Matrix_sanjiao = np.load(f'figure_3_data\ZZ_sanjiao_{i}_({j})_matrix{9}.npy')
        sanjiao_matrix_change = np.linalg.norm(Matrix_sanjiao[:, t] - Matrix_ref_sanjiao[:, t]) / np.linalg.norm(Matrix_ref_sanjiao[:, t])
        sanjiao_vector_change = np.linalg.norm(Vector_sanjiao - Vector_ref_sanjiao) / np.linalg.norm(Vector_ref_sanjiao)
        sanjiao_sample_matrix.append(sanjiao_matrix_change)
        sanjiao_sample_vector.append(sanjiao_vector_change)

    data_matrix.append(sample_matrix)
    data_vector.append(sample_vector)
    data_matrix_sanjiao.append(sanjiao_sample_matrix)
    data_vector_sanjiao.append(sanjiao_sample_vector)

# ------------------- 3. Data generation for plotting -------------------
# Perturbation intensity: 10^-2 → 10^-1 (x-axis increments)
perturbations = np.linspace(0.01, 0.1, 10)
num_samples = 9  # number of data points per group

# ------------------- 4. Plotting -------------------
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

# Boxplot for standard kernel
boxplot1 = ax.boxplot(
    data_vector,
    positions=perturbations,
    notch=False,
    patch_artist=True,
    widths=0.002,
)

# Boxplot for triangular kernel
boxplot2 = ax.boxplot(
    data_vector_sanjiao,
    positions=perturbations,
    notch=False,
    patch_artist=True,
    widths=0.002,
)

# Median lines
medians1 = [np.median(group) for group in data_vector]
medians2 = [np.median(group) for group in data_vector_sanjiao]

ax.plot(
    perturbations,
    medians1,
    color='cornflowerblue',
    linewidth=2,
    markersize=8,
    alpha=0.7,
    zorder=2
)
ax.plot(
    perturbations,
    medians2,
    color='salmon',
    linewidth=2,
    markersize=8,
    alpha=0.7,
    zorder=2
)

# ------------------- 5. Axis configuration -------------------
ax.set_xscale('linear')
ax.set_yscale('linear')

ax.set_xticks(perturbations)
ax.set_xticklabels([f'{p:.2f}' for p in perturbations], rotation=45)

# Y-axis range (slightly larger than min/max values)
all_data = np.concatenate(data_vector + data_vector_sanjiao)
min_val = all_data.min() * 0.8
max_val = all_data.max() * 1.2
ax.set_ylim(min_val, max_val)
ax.set_xlim([0.01 - 0.005, 0.1 + 0.005])

# ------------------- 6. Style customization -------------------
for box in boxplot1['boxes']:
    box.set_facecolor('cornflowerblue')
    box.set_edgecolor('cornflowerblue')
    box.set_linewidth(1.2)

for box in boxplot2['boxes']:
    box.set_facecolor('salmon')
    box.set_edgecolor('salmon')
    box.set_linewidth(1.2)

# Outlier style
for flier in boxplot1['fliers']:
    flier.set_marker('.')
    flier.set_markersize(4)
    flier.set_markeredgewidth(0.5)
    flier.set_markeredgecolor('cornflowerblue')

for flier in boxplot2['fliers']:
    flier.set_marker('.')
    flier.set_markersize(4)
    flier.set_markeredgewidth(0.5)
    flier.set_markeredgecolor('salmon')

ax.set_xlabel('Disturbance intensity', fontsize=14)
ax.set_ylabel('Relative variation', fontsize=14)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()
