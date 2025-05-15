# train_anfis.py
# Реализует алгоритмы обучения для ANFIS контроллера.

import numpy as np
import config as cfg
import random
import os

def train_anfis_full_backprop(anfis_model, training_data, epochs=cfg.NUM_EPOCHS,
                              learning_rate_premise=cfg.LEARNING_RATE_PREMISE,
                              learning_rate_consequence=cfg.LEARNING_RATE_CONSEQUENCE,
                              model_save_path="data/anfis_model_parameters.npz"):
    """
    Обучает ANFIS, используя полный алгоритм обратного распространения ошибки.
    Настраивает параметры функций принадлежности (предпосылки) и параметры заключений правил.
    Сохраняет параметры модели, показавшей наилучший (минимальный) MSE.
    """
    if not training_data:
        print("Нет данных для обучения.")
        return [], anfis_model

    if cfg.MF_TYPE != 'gauss':
        raise ValueError("Полное обучение (full_backprop) требует MF_TYPE='gauss' в config.py")

    print(f"\nНачало ПОЛНОГО обучения ANFIS на {epochs} эпох...")
    print(f"  Скорость обучения (предпосылки ФП): {learning_rate_premise}")
    print(f"  Скорость обучения (заключения правил): {learning_rate_consequence}")

    mse_history = []
    best_mse = float('inf')
    best_model_params = None

    def dMF_dMu(x, mu, sigma, mf_val):
        if sigma == 0: return 0.0
        return mf_val * (x - mu) / (sigma ** 2)

    def dMF_dSigma(x, mu, sigma, mf_val):
        if abs(sigma) < 1e-9: return 0.0
        return mf_val * ((x - mu) ** 2) / (sigma ** 3)

    for epoch in range(epochs):
        total_squared_error = 0
        random.shuffle(training_data)

        for data_point_idx, data_point in enumerate(training_data):
            inputs = data_point['inputs']
            y_true = data_point['target_pwm']
            input_val1, input_val2 = inputs[0], inputs[1]

            # --- ЭТАП 1: Прямой проход (вычисление выхода ANFIS y_pred) ---

            # Слой 1: Фаззификация
            mf_outputs_input1 = [anfis_model._get_mf_value(input_val1, 0, i) for i in range(cfg.NUM_MF_E)]
            mf_outputs_input2 = [anfis_model._get_mf_value(input_val2, 1, i) for i in range(cfg.NUM_MF_DE)]

            # Слой 2: Активация правил
            rule_strengths = np.zeros(anfis_model.num_rules)
            for i in range(anfis_model.num_rules):
                mf_idx1, mf_idx2 = anfis_model.rule_antecedents[i]
                rule_strengths[i] = mf_outputs_input1[mf_idx1] * mf_outputs_input2[mf_idx2]

            # Слой 3: Нормализация сил правил
            sum_rule_strengths = np.sum(rule_strengths)
            normalized_strengths = np.zeros_like(rule_strengths)
            if sum_rule_strengths > 1e-12:  # Защита от деления на ноль
                normalized_strengths = rule_strengths / sum_rule_strengths

            # Слой 4: Выходы отдельных правил
            rule_f_values = np.zeros(anfis_model.num_rules)
            for i in range(anfis_model.num_rules):
                rule_f_values[i] = anfis_model.consequent_params[i, 0] * input_val1 + \
                                   anfis_model.consequent_params[i, 1] * input_val2 + \
                                   anfis_model.consequent_params[i, 2]

            # Слой 5: Общий выход ANFIS
            y_pred = np.sum(normalized_strengths * rule_f_values)

            error = y_true - y_pred
            total_squared_error += error ** 2

            # --- ЭТАП 2: Обратное распространение ошибки и обновление параметров ---

            delta_output = -error  # Производная функции потерь

            delta_consequent = np.zeros_like(anfis_model.consequent_params)
            for i in range(anfis_model.num_rules):
                w_bar_i = normalized_strengths[i]
                delta_consequent[i, 0] = delta_output * w_bar_i * input_val1
                delta_consequent[i, 1] = delta_output * w_bar_i * input_val2
                delta_consequent[i, 2] = delta_output * w_bar_i * 1.0
            anfis_model.consequent_params -= learning_rate_consequence * delta_consequent

            # -- Обновление параметров предпосылок --
            # Вычисление "дельт" (сигналов ошибки), распространяемых к предыдущим слоям
            delta_normalized_strengths = delta_output * rule_f_values

            delta_rule_strengths = np.zeros_like(rule_strengths)
            if sum_rule_strengths > 1e-12:
                for j_idx in range(anfis_model.num_rules):
                    sum_term_for_dw_j = 0
                    for i_idx in range(anfis_model.num_rules):
                        delta_ij = 1.0 if i_idx == j_idx else 0.0
                        dw_bar_i_dw_j = (delta_ij - normalized_strengths[i_idx]) / sum_rule_strengths
                        sum_term_for_dw_j += delta_normalized_strengths[i_idx] * dw_bar_i_dw_j
                    delta_rule_strengths[j_idx] = sum_term_for_dw_j

            # Распространение ошибки к выходам функций принадлежности
            delta_mf_outputs1 = np.zeros(cfg.NUM_MF_E)
            delta_mf_outputs2 = np.zeros(cfg.NUM_MF_DE)

            for k_mf1_idx in range(cfg.NUM_MF_E):
                sum_term_for_dMF1 = 0
                for j_rule_idx in range(anfis_model.num_rules):
                    mf_idx_e_in_rule_j, mf_idx_de_in_rule_j = anfis_model.rule_antecedents[j_rule_idx]
                    if mf_idx_e_in_rule_j == k_mf1_idx:
                        dw_j_dMF1_k = mf_outputs_input2[mf_idx_de_in_rule_j]
                        sum_term_for_dMF1 += delta_rule_strengths[j_rule_idx] * dw_j_dMF1_k
                delta_mf_outputs1[k_mf1_idx] = sum_term_for_dMF1

            for l_mf2_idx in range(cfg.NUM_MF_DE):
                sum_term_for_dMF2 = 0
                for j_rule_idx in range(anfis_model.num_rules):
                    mf_idx_e_in_rule_j, mf_idx_de_in_rule_j = anfis_model.rule_antecedents[j_rule_idx]
                    if mf_idx_de_in_rule_j == l_mf2_idx:
                        dw_j_dMF2_l = mf_outputs_input1[mf_idx_e_in_rule_j]
                        sum_term_for_dMF2 += delta_rule_strengths[j_rule_idx] * dw_j_dMF2_l
                delta_mf_outputs2[l_mf2_idx] = sum_term_for_dMF2

            delta_premise_params1 = [[0.0, 0.0] for _ in range(cfg.NUM_MF_E)]
            delta_premise_params2 = [[0.0, 0.0] for _ in range(cfg.NUM_MF_DE)]

            for k_mf1_idx in range(cfg.NUM_MF_E):
                mf_val = mf_outputs_input1[k_mf1_idx]
                mu, sigma = anfis_model.mf_params_input1[k_mf1_idx]
                dEdMF = delta_mf_outputs1[k_mf1_idx]

                delta_premise_params1[k_mf1_idx][0] = dEdMF * dMF_dMu(input_val1, mu, sigma, mf_val)
                delta_premise_params1[k_mf1_idx][1] = dEdMF * dMF_dSigma(input_val1, mu, sigma, mf_val)

            for l_mf2_idx in range(cfg.NUM_MF_DE):
                mf_val = mf_outputs_input2[l_mf2_idx]
                mu, sigma = anfis_model.mf_params_input2[l_mf2_idx]
                dEdMF = delta_mf_outputs2[l_mf2_idx]
                delta_premise_params2[l_mf2_idx][0] = dEdMF * dMF_dMu(input_val2, mu, sigma, mf_val)
                delta_premise_params2[l_mf2_idx][1] = dEdMF * dMF_dSigma(input_val2, mu, sigma, mf_val)

            # --- Обновление параметров предпосылок (ФП) методом градиентного спуска ---
            for k in range(cfg.NUM_MF_E):
                anfis_model.mf_params_input1[k][0] -= learning_rate_premise * delta_premise_params1[k][
                    0]
                anfis_model.mf_params_input1[k][1] -= learning_rate_premise * delta_premise_params1[k][
                    1]
                MIN_SIGMA = 0.05
                if anfis_model.mf_params_input1[k][1] < MIN_SIGMA:
                    anfis_model.mf_params_input1[k][1] = MIN_SIGMA

            for l in range(cfg.NUM_MF_DE):
                anfis_model.mf_params_input2[l][0] -= learning_rate_premise * delta_premise_params2[l][
                    0]
                anfis_model.mf_params_input2[l][1] -= learning_rate_premise * delta_premise_params2[l][
                    1]
                if anfis_model.mf_params_input2[l][1] < MIN_SIGMA:
                    anfis_model.mf_params_input2[l][1] = MIN_SIGMA

        mse = total_squared_error / len(training_data) if len(training_data) > 0 else 0
        mse_history.append(mse)

        # Сохранение параметров модели, если достигнут лучший (минимальный) MSE на данный момент
        if mse < best_mse:
            best_mse = mse
            best_model_params = {
                'mf_params_input1': [p_list.copy() for p_list in anfis_model.mf_params_input1],
                'mf_params_input2': [p_list.copy() for p_list in anfis_model.mf_params_input2],
                'consequent_params': anfis_model.consequent_params.copy()
            }

        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs - 1:
            print(f"  Эпоха {epoch + 1}/{epochs}, MSE: {mse:.6f}")

    print(f"Полное обучение завершено. Лучший MSE: {best_mse:.6f}")

    # Восстановление параметров модели, показавшей лучший MSE за все эпохи
    if best_model_params:
        print("Восстановление модели с лучшим MSE.")
        anfis_model.mf_params_input1 = best_model_params['mf_params_input1']
        anfis_model.mf_params_input2 = best_model_params['mf_params_input2']
        anfis_model.consequent_params = best_model_params['consequent_params']

        try:
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mf1_np = np.array(best_model_params['mf_params_input1'], dtype=object)
            mf2_np = np.array(best_model_params['mf_params_input2'], dtype=object)

            np.savez(model_save_path,
                     mf_params_input1=mf1_np,
                     mf_params_input2=mf2_np,
                     consequent_params=best_model_params['consequent_params'])
            print(f"Параметры лучшей модели сохранены в: {model_save_path}")
        except Exception as e:
            print(f"Ошибка при сохранении параметров модели: {e}")
    else:
        print(
            "Не удалось улучшить модель во время обучения")

    return mse_history, anfis_model


def train_anfis_simple(anfis_model, training_data, epochs=cfg.NUM_EPOCHS,
                       learning_rate_consequence=cfg.LEARNING_RATE_CONSEQUENCE):  # Используем LR из config
    if not training_data:
        print("Нет данных для упрощенного обучения.")
        return [], anfis_model

    print(f"\nНачало обучения ANFIS на {epochs} эпох (только заключения)...")
    print(f"  Скорость обучения (заключения правил): {learning_rate_consequence}")

    mse_history = []
    best_mse = float('inf')
    best_consequent_params = None

    for epoch in range(epochs):
        total_squared_error = 0
        random.shuffle(training_data)

        for data_point in training_data:
            inputs = data_point['inputs']
            y_true = data_point['target_pwm']
            input_val1, input_val2 = inputs[0], inputs[1]

            mf_outputs_input1 = [anfis_model._get_mf_value(input_val1, 0, i) for i in range(cfg.NUM_MF_E)]
            mf_outputs_input2 = [anfis_model._get_mf_value(input_val2, 1, i) for i in range(cfg.NUM_MF_DE)]
            rule_strengths = np.zeros(anfis_model.num_rules)
            for i in range(anfis_model.num_rules):
                mf_idx1, mf_idx2 = anfis_model.rule_antecedents[i]
                rule_strengths[i] = mf_outputs_input1[mf_idx1] * mf_outputs_input2[mf_idx2]

            sum_rule_strengths = np.sum(rule_strengths)
            current_normalized_strengths = np.zeros_like(rule_strengths)
            if sum_rule_strengths > 1e-12:
                current_normalized_strengths = rule_strengths / sum_rule_strengths

            rule_f_values = np.zeros(anfis_model.num_rules)
            for i in range(anfis_model.num_rules):
                rule_f_values[i] = anfis_model.consequent_params[i, 0] * input_val1 + \
                                   anfis_model.consequent_params[i, 1] * input_val2 + \
                                   anfis_model.consequent_params[i, 2]
            y_pred = np.sum(current_normalized_strengths * rule_f_values)

            error = y_true - y_pred
            total_squared_error += error ** 2

            for i in range(anfis_model.num_rules):
                w_bar_i = current_normalized_strengths[i]
                anfis_model.consequent_params[
                    i, 0] += learning_rate_consequence * error * w_bar_i * input_val1
                anfis_model.consequent_params[
                    i, 1] += learning_rate_consequence * error * w_bar_i * input_val2
                anfis_model.consequent_params[
                    i, 2] += learning_rate_consequence * error * w_bar_i * 1.0

        mse = total_squared_error / len(training_data) if len(training_data) > 0 else 0
        mse_history.append(mse)

        if mse < best_mse:
            best_mse = mse
            best_consequent_params = anfis_model.consequent_params.copy()

        if (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0 or epoch == epochs - 1:
            print(f"  Эпоха {epoch + 1}/{epochs}, MSE: {mse:.6f}")

    print(f"Упрощенное обучение завершено. Лучший MSE: {best_mse:.6f}")
    if best_consequent_params is not None:
        anfis_model.consequent_params = best_consequent_params
        print("Восстановлены лучшие параметры заключений (из упрощенного обучения).")

    return mse_history, anfis_model


if __name__ == '__main__':
    print("Модуль train_anfis.py запущен напрямую (обычно используется для импорта).")
    pass
