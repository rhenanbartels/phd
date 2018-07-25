import matplotlib.pyplot as plt

from scipy.io import loadmat


plt.ion()


def plot_curves(ax, patients, color, var, group):
    for patient in patients:
        line = ax.plot(
                range(1, 5),
                patient.__getattribute__(var), color=color, marker='o',
                label=group)

    return line


def plot_curves_rel_first(ax, patients, color, group):
    for patient in patients:
        first_val = patient.df[0]
        line = ax.plot(
                range(1, 5),
                patient.df / first_val * 100, color=color, marker='o',
                label=group)

    return line


def plot_curves_rel_abs(ax, patients, color, group):
    for patient in patients:
        first_val = patient.abs[0]
        line = ax.plot(
                range(1, 5),
                patient.df / first_val, color=color, marker='o',
                label=group)

    return line


def plot_curves_interval(ax, patients, color, group):
    for pat in patients:
        vals = [pat.abs[0] - pat.abs[3], pat.abs[3] - pat.abs[6]]
        line = ax.plot(
                (0, 1),
                vals, color=color, marker='o',
                label=group)

    return line


def create_xaxis(ax, x_range, df=False):
    ax.set_xticks(range(1, 5))
    if df:
        ax.set_xticklabels(['2-4', '4-6', '6-8', '8-'])
    else:
        ax.set_xticklabels(x_range)


fig, ax = plt.subplots(1, 1)


ins = loadmat(
        '../Results/Analise_2_2_8/ins_2_2_8.mat',
        struct_as_record=False,
        squeeze_me=True
)

exp = loadmat(
        '../Results/Analise_2_2_8/exp_2_2_8.mat',
        struct_as_record=False,
        squeeze_me=True
)


var = 'df'
# line1 = plot_curves(ax, exp['res_alpha_exp_2'], 'k', var, 'alpha')
# line2 = plot_curves(ax, exp['res_copd_exp_2'], 'b', var, 'copd')
# line3 = plot_curves(ax, exp['res_control_exp_2'], 'g', var, 'control')

# line1 = plot_curves_rel_first(ax, ins['res_alpha_ins_2'], 'k', 'alpha')
# line2 = plot_curves_rel_first(ax, ins['res_copd_ins_2'], 'b', 'copd')
# line3 = plot_curves_rel_first(ax, ins['res_control_ins_2'], 'g', 'control')

line1 = plot_curves_rel_abs(ax, exp['res_alpha_exp_2'], 'k', 'alpha')
line2 = plot_curves_rel_abs(ax, exp['res_copd_exp_2'], 'b', 'copd')
line3 = plot_curves_rel_abs(ax, exp['res_control_exp_2'], 'g', 'control')

create_xaxis(ax, range(2, 10, 2), df=True)

ax.set_xlabel('Length Scale (mm)')
ax.set_ylabel('Coefficient of Variation')
plt.title(var)
plt.legend([line1[0], line2[0], line3[0]], ['alpha', 'copd', 'control'])
