import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.io import loadmat


plt.ion()


def create_cols(x_range):
    x_range = list(map(lambda x: str(x), x_range))
    x_range = zip(x_range, x_range[1:])
    return list(map(lambda x: x[0] + '-' + x[1], x_range)) + ['40-']


def create_data_frame(matrix, group, x_range):
    cols = create_cols(x_range)
    df = pandas.DataFrame(matrix, columns=cols)
    df['group'] = group
    return df


def plot_dfs(ax, data_frame, interval=['2-4', '4-6', '6-8']):
    new_df = data_frame.loc[data_frame.variable.isin(interval)]
    sns.boxplot(x='variable', y='value', hue='group', data=new_df, ax=ax)


def plot_curves(ax, patients, color, var, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    for index, patient in enumerate(patients):
        matrix[index, :] = patient.__getattribute__(var)

    x_values = range(matrix.shape[1])
    avg = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_u = avg + std
    std_l = avg - std
    line = ax.plot(avg, color=color, marker='o')
    ax.fill_between(x_values, y1=avg, y2=std_u, color=color, alpha=0.2)
    ax.fill_between(x_values, y1=avg, y2=std_l, color=color, alpha=0.2)
    return line, matrix


def plot_curves_rel_first(ax, patients, color, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    for index, patient in enumerate(patients):
        first_val = patient.df[0]
        matrix[index, :] = patient.df / first_val

    x_values = range(matrix.shape[1])
    avg = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_u = avg + std
    std_l = avg - std
    line = ax.plot(avg, color=color, marker='o')
    ax.fill_between(x_values, y1=avg, y2=std_u, color=color, alpha=0.2)
    ax.fill_between(x_values, y1=avg, y2=std_l, color=color, alpha=0.2)
    return line, matrix


def plot_curves_rel_abs(ax, patients, color, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    for index, patient in enumerate(patients):
        first_val = patient.abs[0]
        matrix[index, :] = patient.df / first_val

    x_values = range(matrix.shape[1])
    avg = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std_u = avg + std
    std_l = avg - std
    line = ax.plot(avg, color=color, marker='o')
    ax.fill_between(x_values, y1=avg, y2=std_u, color=color, alpha=0.2)
    ax.fill_between(x_values, y1=avg, y2=std_l, color=color, alpha=0.2)
    return line, matrix


def plot_curves_interval(ax, patients, color, group):
    for pat in patients:
        vals = [pat.abs[0] - pat.abs[3], pat.abs[3] - pat.abs[6]]
        line = ax.plot(
                (0, 1),
                vals, color=color, marker='o',
                label=group)

    return line


def create_xaxis(ax, x_range, df=False):
    ax.set_xticks(range(0, 20))
    if df:
        x_range = list(map(lambda x: str(x), x_range))
        x_range = zip(x_range, x_range[1:])
        x_range = list(map(lambda x: x[0] + '-' + x[1], x_range)) + ['40-']

    plt.xticks(rotation=70)
    ax.set_xticklabels(x_range)


fig, ax = plt.subplots(1, 1)


ins = loadmat(
        '../Results/Analise_2_2_40/ins_2_2_40.mat',
        struct_as_record=False,
        squeeze_me=True
)

exp = loadmat(
        '../Results/Analise_2_2_40/exp_2_2_40.mat',
        struct_as_record=False,
        squeeze_me=True
)


var = 'df'
# line1, matrix1 = plot_curves(ax, ins['res_alpha_ins_2'], 'k', var, 'alpha')
# line2, matrix2 = plot_curves(ax, ins['res_copd_ins_2'], 'b', var, 'copd')
# line3, matrix3 = plot_curves(ax, ins['res_control_ins_2'], 'g', var, 'control')

line1, matrix1 = plot_curves_rel_first(ax, exp['res_alpha_exp_2'], 'k', 'alpha')
line2, matrix2 = plot_curves_rel_first(ax, exp['res_copd_exp_2'], 'b', 'copd')
line3, matrix3 = plot_curves_rel_first(ax, exp['res_control_exp_2'], 'g', 'control')

# line1, matrix1 = plot_curves_rel_abs(ax, ins['res_alpha_ins_2'], 'k', 'alpha')
# line2, matrix2 = plot_curves_rel_abs(ax, ins['res_copd_ins_2'], 'b', 'copd')
# line3, matrix3 = plot_curves_rel_abs(ax, ins['res_control_ins_2'], 'g', 'control')

rng = list(range(2, 42, 2))
cols = create_cols(rng)
create_xaxis(ax, rng, df=True)

df_alpha = create_data_frame(matrix1, 'alpha1', rng)
df_copd = create_data_frame(matrix2, 'copd', rng)
df_control = create_data_frame(matrix3, 'control', rng)

dfs = pandas.concat([df_alpha, df_copd, df_control])
names = list(dfs.columns)
names.pop(-1)
dfs_melted = pandas.melt(dfs, id_vars=['group'], value_vars=names)

ax.set_xlabel('Length Scale (mm)')
ax.set_ylabel('Coefficient of Variation')
plt.title(var)
plt.legend([line1[0], line2[0], line3[0]], ['alpha', 'copd', 'control'])


fig, ax = plt.subplots(2, 3)

plot_dfs(ax[0][0], dfs_melted, interval=cols[:3])
plot_dfs(ax[0][1], dfs_melted, interval=cols[3:6])
plot_dfs(ax[0][2], dfs_melted, interval=cols[6:9])

plot_dfs(ax[1][0], dfs_melted, interval=cols[9:12])
plot_dfs(ax[1][1], dfs_melted, interval=cols[12:15])
plot_dfs(ax[1][2], dfs_melted, interval=cols[15:18])
