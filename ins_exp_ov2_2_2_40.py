import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.io import loadmat

plt.ion()

X_RANGE = list(range(2, 42, 2))
TITLE = 'DF / DF[0]'


def create_cols(x_range):
    x_range = list(map(lambda x: str(x), x_range))
    x_range = zip(x_range, x_range[1:])
    return list(map(lambda x: x[0] + '-' + x[1], x_range)) + ['40-']


def create_data_frame(matrix, patient_names, group, x_range):
    cols = create_cols(x_range)
    df = pandas.DataFrame(matrix, columns=cols)
    df['group'] = group
    df['name'] = patient_names
    return df


def create_matrix(patients, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    patient_names = []
    for index, patient in enumerate(patients):
        matrix[index, :] = patient.df
        patient_names.append(patient.name)

    return matrix, patient_names


def create_matrix_rel_first(patients, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    patient_names = []
    for index, patient in enumerate(patients):
        first_val = patient.df[0]
        matrix[index, :] = patient.df / first_val
        patient_names.append(patient.name)

    return matrix, patient_names


def create_matrix_rel_first_abs(patients, group):
    n_patients = len(patients)
    n_cols = 20
    matrix = np.zeros((n_patients, n_cols))
    patient_names = []
    for index, patient in enumerate(patients):
        first_val = patient.abs[0]
        matrix[index, :] = patient.df / first_val
        patient_names.append(patient.name)

    return matrix, patient_names


def create_all_frames(alpha, copd, control, func):
    alpha_df, alpha_names = func(alpha, 'alpha')
    copd_df, copd_names = func(copd, 'copd')
    control_df, control_names = func(control, 'control')

    alpha_df = create_data_frame(alpha_df, alpha_names, 'alpha', X_RANGE)
    copd_df = create_data_frame(copd_df, copd_names, 'copd', X_RANGE)
    control_df = create_data_frame(control_df, control_names, 'control',
                                   X_RANGE)

    return alpha_df, copd_df, control_df


def match_ins_exp(alpha_ins, copd_ins, control_ins, alpha_exp, copd_exp,
                  control_exp):
    new_alpha = alpha_ins[alpha_ins.name.isin(alpha_exp.name)]
    new_copd = copd_ins[copd_ins.name.isin(copd_exp.name)]
    new_control = control_ins[control_ins.name.isin(control_exp.name)]

    return new_alpha, new_copd, new_control


def show_diff(all_ins, all_exp, interval):
    avg_ins = all_ins[['group'] + interval].groupby('group').mean()
    avg_exp = all_exp[['group'] + interval].groupby('group').mean()

    avg_diff = (np.abs(avg_ins - avg_exp) / avg_exp * 100).reset_index()
    df_diff_melted = pandas.melt(avg_diff, id_vars=['group'])
    plot = sns.pointplot(x='variable', y='value', hue='group',
                         data=df_diff_melted)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=70)
    plt.xlabel('Length Scale (mm)')
    plt.ylabel('Percentage Diff (%) Ins - Exp')
    plt.title(TITLE)


def group_cv(all_ins, all_exp, interval):
    cv = lambda x: x.std() / x.mean()
    cv_ins = all_ins[['group'] + interval].groupby('group').agg(cv).reset_index()
    cv_exp = all_exp[['group'] + interval].groupby('group').agg(cv).reset_index()

    cv_ins = pandas.melt(cv_ins, id_vars=['group']).reset_index()
    cv_exp = pandas.melt(cv_exp, id_vars=['group']).reset_index()

    plt.figure()
    plot = sns.pointplot(x='variable', y='value', hue='group',
                         data=cv_ins)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=70)
    plt.xlabel('Length Scale (mm)')
    plt.ylabel('Percentage Diff (%) Ins - Exp')
    plt.title(TITLE)


if __name__ == "__main__":
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

    alpha_ins, copd_ins, control_ins = create_all_frames(
            ins['res_alpha_ins_2'],
            ins['res_copd_ins_2'],
            ins['res_control_ins_2'],
            create_matrix_rel_first
    )
    alpha_exp, copd_exp, control_exp = create_all_frames(
            exp['res_alpha_exp_2'],
            exp['res_copd_exp_2'],
            exp['res_control_exp_2'],
            create_matrix_rel_first
    )

    # Jorcelene de COPD não está no condição INS
    copd_exp.drop(3, axis=0, inplace=True)

    alpha_ins, copd_ins, control_ins = match_ins_exp(
            alpha_ins,
            copd_ins,
            control_ins,
            alpha_exp,
            copd_exp,
            control_exp
    )

    all_ins = pandas.concat([alpha_ins, copd_ins, control_ins]).reset_index()
    all_exp = pandas.concat([alpha_exp, copd_exp, control_exp]).reset_index()

    show_diff(all_ins, all_exp, create_cols(X_RANGE))
    group_cv(all_ins, all_exp, create_cols(X_RANGE))
