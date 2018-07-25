import pandas
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat

plt.ion()

X_RANGE = list(range(2, 42, 2))


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


def create_all_frames(alpha, copd, control):
    alpha_df, alpha_names = create_matrix(alpha, 'alpha')
    copd_df, copd_names = create_matrix(copd, 'copd')
    control_df, control_names = create_matrix(control, 'control')

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


def diff_ins_exp(ins, exp):
    cols = list(ins.columns)[:-2]
    diff = ins[cols] - exp[cols]
    diff[['group', 'name']] = ins[['group', 'name']]
    return diff


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
            ins['res_control_ins_2']
    )
    alpha_exp, copd_exp, control_exp = create_all_frames(
            exp['res_alpha_exp_2'],
            exp['res_copd_exp_2'],
            exp['res_control_exp_2']
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

    diff = diff_ins_exp(all_ins, all_exp)
