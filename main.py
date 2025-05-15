# main.py
# Главный скрипт для обучения и тестирования ANFIS контроллера системы "Шарик в трубке".

import config as cfg
import anfis_structure
import train_anfis
import ball_tube_simulation as bts

import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
from data_generator import SimplePDController

script_dir = os.path.dirname(os.path.abspath(__file__))
data_filename = "anfis_COMBINED_data_PD250_FINAL.json"
data_filepath = os.path.join(script_dir, "data", data_filename)
model_params_path = os.path.join(script_dir, "data", "anfis_model_parameters.npz")

def load_training_data(filepath):
    """
    Загружает обучающие данные для ANFIS из указанного JSON-файла.
    Проверяет существование файла и базовый формат данных.
    Возвращает: список обучающих данных или пустой список в случае ошибки.
    """
    training_dataset = []
    print(f"Пытаюсь загрузить данные из: {filepath}")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                training_dataset = json.load(f)
            print(f"Загружено {len(training_dataset)} точек данных из файла: {filepath}")
            if not isinstance(training_dataset, list) or \
                    (len(training_dataset) > 0 and not all(
                        isinstance(d, dict) and 'inputs' in d and 'target_pwm' in d and
                        isinstance(d['inputs'], list) and len(d['inputs']) == 2
                        for d in training_dataset
                    )):
                print("Ошибка: Формат данных в файле некорректен.")
                return []
        except Exception as e:
            print(f"Ошибка при загрузке данных из {filepath}: {e}")
            return []
    else:
        print(f"Файл с данными не найден: {filepath}")
    return training_dataset

def load_anfis_model_parameters(anfis_model, filepath):
    """
    Загружает сохраненные параметры (ФП и заключений) в существующий экземпляр anfis_model.
    Параметры загружаются из файла .npz.
    Возвращает: True, если параметры успешно загружены, False в противном случае.
    """
    if os.path.exists(filepath):
        try:
            data = np.load(filepath, allow_pickle=True)
            anfis_model.mf_params_input1 = [list(p) for p in data['mf_params_input1']]
            anfis_model.mf_params_input2 = [list(p) for p in data['mf_params_input2']]
            anfis_model.consequent_params = data['consequent_params']
            print(f"Параметры ANFIS модели успешно загружены из: {filepath}")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке параметров ANFIS модели: {e}")
            return False
    else:
        print(f"Файл с параметрами ANFIS модели не найден: {filepath}. Будет проведено обучение.")
        return False

def test_trained_anfis(anfis_model, num_test_simulations=5, sim_duration_test=10.0):
    """
    Тестирует обученный anfis_model
    """
    print(
        f"\n--- Тестирование обученного ANFIS и ПД-регулятора на {num_test_simulations} симуляциях (до {sim_duration_test}с каждая) ---")

    pd_kp_for_comparison = 250
    pd_kd_for_comparison = 20
    pd_base_pwm_for_comparison = 95.0

    pd_controller_for_test = SimplePDController(
        kp=pd_kp_for_comparison,
        kd=pd_kd_for_comparison,
        setpoint=cfg.Z_DESIRED,
        base_pwm=pd_base_pwm_for_comparison
    )
    print(
        f"ПД-регулятор для теста: Kp={pd_kp_for_comparison}, Kd={pd_kd_for_comparison}, BasePWM={pd_controller_for_test.base_pwm}")

    successful_anfis_tests = 0  # Счетчик успешных тестов для ANFIS
    successful_pd_tests = 0  # Счетчик успешных тестов для ПД

    fixed_difficult_states = [
        [0.088, -0.051, 101.2, 0.0],
        [0.070, 0.0, 50.0, 0.0],
        [0.158, 0.097, 916.9, 0.0]
    ]
    num_fixed_tests = len(fixed_difficult_states)

    actual_num_test_simulations = max(num_test_simulations, num_fixed_tests)
    if num_test_simulations < num_fixed_tests and num_test_simulations > 0:
        print(
            f"Предупреждение: num_test_simulations ({num_test_simulations}) меньше числа фиксированных тестов ({num_fixed_tests}). Будет проведено {num_fixed_tests} тестов.")
    elif num_test_simulations == 0 and num_fixed_tests == 0:
        print("Количество тестовых симуляций и фиксированных тестов равно 0. Тестирование не будет проводиться.")
        denominator = 1
        print(f"\n--- Тестирование ANFIS завершено. Успешных тестов: {successful_anfis_tests}/{denominator} ---")
        print(f"--- Тестирование ПД завершено. Успешных тестов: {successful_pd_tests}/{denominator} ---")
        return

    # Минимальная безопасная начальная высота z0 для всех тестов
    Z0_MIN_SAFE_FOR_TESTS = 0.15
    Z0_MAX_SAFE_FOR_TESTS = cfg.H_TUBE - cfg.BALL_RADIUS * 1.1

    for i in range(actual_num_test_simulations):
        if i < num_fixed_tests:
            temp_initial_state = list(fixed_difficult_states[i])
            z0_orig, z0_dot_orig, eta0_orig, eta0_dot_orig = temp_initial_state
            z0, z0_dot, eta0, eta0_dot = z0_orig, z0_dot_orig, eta0_orig, eta0_dot_orig

            test_type_log_msg = f"\nТест {i + 1} (ФИКСИРОВАННЫЙ СТАРТ): "
            if z0 < Z0_MIN_SAFE_FOR_TESTS:
                print(f"    Исходный z0={z0:.3f} в фикс. тесте {i + 1} скорректирован до {Z0_MIN_SAFE_FOR_TESTS:.3f}")
                z0 = Z0_MIN_SAFE_FOR_TESTS
        else:
            temp_z0 = random.uniform(cfg.BALL_RADIUS * 1.1, Z0_MAX_SAFE_FOR_TESTS)
            if temp_z0 < Z0_MIN_SAFE_FOR_TESTS:
                z0 = Z0_MIN_SAFE_FOR_TESTS
            else:
                z0 = temp_z0

            z0_dot = random.uniform(-0.1, 0.1)
            eta0 = random.uniform(0, 1000.0)
            eta0_dot = 0.0
            test_type_log_msg = f"\nТест {i + 1} (СЛУЧАЙНЫЙ СТАРТ С ОГРАНИЧЕНИЕМ z0): "

        initial_state_common = [z0, z0_dot, eta0, eta0_dot]
        print(f"{test_type_log_msg}Начальные z={z0:.3f}, z_dot={z0_dot:.3f}, eta={eta0:.1f}, eta_dot={eta0_dot:.1f}")

        # --- Тестирование ANFIS контроллера ---
        current_state_anfis = list(initial_state_common)
        time_points_anfis = [0.0];
        z_trajectory_anfis = [z0];
        zd_trajectory_anfis = [z0_dot];
        eta_trajectory_anfis = [eta0];
        pwm_values_anfis = []
        current_time_anfis = 0.0;
        simulation_ok_anfis = True
        reason_anfis = "OK"
        control_step = 0.02

        # Цикл пошаговой симуляции для ANFIS
        while current_time_anfis < sim_duration_test:
            current_z_anfis, current_z_dot_anfis = current_state_anfis[0], current_state_anfis[1]
            error_e_anfis = cfg.Z_DESIRED - current_z_anfis
            error_de_anfis = -current_z_dot_anfis
            predicted_pwm_anfis = anfis_model.predict([error_e_anfis, error_de_anfis])
            predicted_pwm_anfis = np.clip(predicted_pwm_anfis, cfg.PWM_MIN, cfg.PWM_MAX)
            pwm_values_anfis.append(predicted_pwm_anfis)

            step_duration = control_step
            time_remaining_anfis = sim_duration_test - current_time_anfis
            if step_duration > time_remaining_anfis: step_duration = time_remaining_anfis
            if step_duration <= 1e-6: break

            step_success_anfis = False
            reason_anfis_step = "Неизвестная причина шага ANFIS"
            try:
                sol_step_anfis, step_success_anfis, reason_anfis_step_ret = bts.run_simulation_v3(
                    current_state_anfis[0], current_state_anfis[1], current_state_anfis[2], current_state_anfis[3],
                    predicted_pwm_anfis, duration=step_duration)
                if not step_success_anfis: reason_anfis = reason_anfis_step_ret
            except Exception as sim_err:
                print(f"  ANFIS: !!! Ошибка во время симуляции: {sim_err}");
                simulation_ok_anfis = False;
                reason_anfis = str(sim_err);
                break

            if not step_success_anfis:
                simulation_ok_anfis = False;
                if reason_anfis == "OK": reason_anfis = reason_anfis_step
                break

            current_state_anfis = sol_step_anfis.y[:, -1]
            actual_time_step_anfis = sol_step_anfis.t[-1]
            if len(sol_step_anfis.t) > 1:
                time_points_anfis.extend(current_time_anfis + sol_step_anfis.t[1:])
                z_trajectory_anfis.extend(sol_step_anfis.y[0, 1:])
                zd_trajectory_anfis.extend(sol_step_anfis.y[1, 1:])
                eta_trajectory_anfis.extend(sol_step_anfis.y[2, 1:])
            else:
                time_points_anfis.append(current_time_anfis + actual_time_step_anfis)
                z_trajectory_anfis.append(current_state_anfis[0])
                zd_trajectory_anfis.append(current_state_anfis[1])
                eta_trajectory_anfis.append(current_state_anfis[2])
            current_time_anfis += actual_time_step_anfis

        # Определение результата теста для ANFIS и вывод информации
        if simulation_ok_anfis and current_time_anfis >= sim_duration_test - 1e-5:
            print(f"  ANFIS: Тест {i + 1} успешно пройден ({current_time_anfis:.2f} c).");
            successful_anfis_tests += 1
        elif simulation_ok_anfis:
            print(f"  ANFIS: Тест {i + 1} завершился ({current_time_anfis:.2f} c).");
            successful_anfis_tests += 1
        else:
            if reason_anfis == "OK": reason_anfis = "Неизвестная причина прерывания ANFIS"
            print(f"  ANFIS: Тест {i + 1} не пройден. Причина: {reason_anfis} на t={current_time_anfis:.2f} c.")

        # --- Тестирование ПД-регулятора ---
        current_state_pd = list(initial_state_common)
        time_points_pd = [0.0];
        z_trajectory_pd = [z0];
        zd_trajectory_pd = [z0_dot];
        eta_trajectory_pd = [eta0];
        pwm_values_pd = []
        current_time_pd = 0.0;
        simulation_ok_pd = True
        reason_pd = "OK"
        reason_pd_step = "OK"

        while current_time_pd < sim_duration_test:
            current_z_pd, current_z_dot_pd = current_state_pd[0], current_state_pd[1]
            predicted_pwm_pd = pd_controller_for_test.calculate(current_z_pd, current_z_dot_pd)
            pwm_values_pd.append(predicted_pwm_pd)
            step_duration = control_step
            time_remaining_pd = sim_duration_test - current_time_pd
            if step_duration > time_remaining_pd: step_duration = time_remaining_pd
            if step_duration <= 1e-6: break

            step_success_pd = False
            reason_pd_step_ret = "Неизвестная причина шага ПД"
            try:
                sol_step_pd, step_success_pd, reason_pd_step_ret = bts.run_simulation_v3(
                    current_state_pd[0], current_state_pd[1], current_state_pd[2], current_state_pd[3],
                    predicted_pwm_pd, duration=step_duration)
                if not step_success_pd: reason_pd = reason_pd_step_ret
            except Exception as sim_err:
                print(f"  ПД:    !!! Ошибка во время симуляции: {sim_err}");
                simulation_ok_pd = False;
                reason_pd = str(sim_err);
                break

            if not step_success_pd:
                simulation_ok_pd = False;
                if reason_pd == "OK": reason_pd = reason_pd_step_ret
                break

            current_state_pd = sol_step_pd.y[:, -1]
            actual_time_step_pd = sol_step_pd.t[-1]
            if len(sol_step_pd.t) > 1:
                time_points_pd.extend(current_time_pd + sol_step_pd.t[1:])
                z_trajectory_pd.extend(sol_step_pd.y[0, 1:])
                zd_trajectory_pd.extend(sol_step_pd.y[1, 1:])
                eta_trajectory_pd.extend(sol_step_pd.y[2, 1:])
            else:
                time_points_pd.append(current_time_pd + actual_time_step_pd)
                z_trajectory_pd.append(current_state_pd[0])
                zd_trajectory_pd.append(current_state_pd[1])
                eta_trajectory_pd.append(current_state_pd[2])
            current_time_pd += actual_time_step_pd

        if simulation_ok_pd and current_time_pd >= sim_duration_test - 1e-5:
            print(f"  ПД:    Тест {i + 1} успешно пройден ({current_time_pd:.2f} c).");
            successful_pd_tests += 1
        elif simulation_ok_pd:
            print(f"  ПД:    Тест {i + 1} завершился ({current_time_pd:.2f} c).");
            successful_pd_tests += 1
        else:
            if reason_pd == "OK": reason_pd = "Неизвестная причина прерывания ПД"
            print(f"  ПД:    Тест {i + 1} не пройден. Причина: {reason_pd} на t={current_time_pd:.2f} c.")

        # --- Визуализация результатов текущего теста ---
        plt.figure(figsize=(12, 9))
        # График 1: Положение шарика z(t)
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time_points_anfis, z_trajectory_anfis, label=f'ANFIS Z (Успех: {simulation_ok_anfis})', color='blue')
        ax1.plot(time_points_pd, z_trajectory_pd, label=f'ПД Z (Успех: {simulation_ok_pd})', color='green',
                 linestyle='--')
        ax1.axhline(0, color='k', linestyle=':', label='Низ/Верх трубки');
        ax1.axhline(cfg.H_TUBE, color='k', linestyle=':')
        ax1.axhline(cfg.Z_DESIRED, color='r', linestyle='-.', label=f'Желаемое Z ({cfg.Z_DESIRED:.2f}м)')
        ax1.set_ylabel('z (м)');
        ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.7));
        ax1.grid(True)
        ax1.set_title(f'Тест {i + 1} (z0={z0:.2f}, zd0={z0_dot:.2f}, eta0={eta0:.1f})')

        # Дополнительная ось для скорости шарика z_dot(t)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_points_anfis, zd_trajectory_anfis, label='ANFIS z_dot', linestyle=':', color='cyan',
                      alpha=0.7)
        ax1_twin.plot(time_points_pd, zd_trajectory_pd, label='ПД z_dot', linestyle=':', color='lightgreen', alpha=0.7)
        ax1_twin.set_ylabel('z_dot (м/с)');
        ax1_twin.legend(loc='center left', bbox_to_anchor=(1.05, 0.3))

        # График 2: Скорость вентилятора eta(t)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(time_points_anfis, eta_trajectory_anfis, label='ANFIS Eta', color='blue')
        ax2.plot(time_points_pd, eta_trajectory_pd, label='ПД Eta', color='green', linestyle='--')
        ax2.axhline(cfg.ETA_MAX, color='orange', linestyle=':', label=f'Max Eta ({cfg.ETA_MAX:.0f})')
        ax2.set_ylabel('eta (ед.)');
        ax2.legend(loc='upper right');
        ax2.grid(True)

        # График 3: Управляющий сигнал PWM(t)
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        if pwm_values_anfis:
            pwm_time_plot_anfis = []
            for k_pwm_idx in range(len(pwm_values_anfis)):
                t_start = k_pwm_idx * control_step
                t_end = (k_pwm_idx + 1) * control_step
                if t_end > current_time_anfis: t_end = current_time_anfis
                if t_start < current_time_anfis: pwm_time_plot_anfis.extend([t_start, t_end])
            plot_pwm_val_anfis = [val for val in pwm_values_anfis for _ in (0, 1)][:len(pwm_time_plot_anfis)]
            if plot_pwm_val_anfis: ax3.plot(pwm_time_plot_anfis, plot_pwm_val_anfis, label='ANFIS PWM', color='blue',
                                            alpha=0.8)
        if pwm_values_pd:
            pwm_time_plot_pd = []
            for k_pwm_idx in range(len(pwm_values_pd)):
                t_start = k_pwm_idx * control_step
                t_end = (k_pwm_idx + 1) * control_step
                if t_end > current_time_pd: t_end = current_time_pd
                if t_start < current_time_pd: pwm_time_plot_pd.extend([t_start, t_end])
            plot_pwm_val_pd = [val for val in pwm_values_pd for _ in (0, 1)][:len(pwm_time_plot_pd)]
            if plot_pwm_val_pd: ax3.plot(pwm_time_plot_pd, plot_pwm_val_pd, label='ПД PWM', color='green',
                                         linestyle='--', alpha=0.8)

        ax3.set_xlabel('Время (с)');
        ax3.set_ylabel('PWM');
        ax3.set_ylim(cfg.PWM_MIN - 10, cfg.PWM_MAX + 10)
        ax3.legend(loc='upper right');
        ax3.grid(True)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Оставляем место для легенд справа
        plt.show()  # Показать график для текущего теста

    # Вывод итоговой статистики по всем тестам
    denominator = actual_num_test_simulations if actual_num_test_simulations > 0 else 1
    print(f"\n--- Тестирование ANFIS завершено. Успешных тестов: {successful_anfis_tests}/{denominator} ---")
    print(f"--- Тестирование ПД завершено. Успешных тестов: {successful_pd_tests}/{denominator} ---")


if __name__ == "__main__":
    print("--- Курсовая работа: ANFIS контроллер для системы 'Шарик в трубке' ---")

    # 1. Создание экземпляра ANFIS контроллера (структура берется из config.py)
    anfis_controller = anfis_structure.ANFIS(num_inputs=2)

    # 2. Попытка загрузить сохраненные параметры обученной модели ANFIS
    model_loaded = load_anfis_model_parameters(anfis_controller, model_params_path)

    # 3. Если модель не была загружена (например, нет файла .npz или ошибка), то запускаем обучение
    if not model_loaded:
        print("\nОбученные параметры не найдены или не удалось загрузить. Запуск обучения...")
        training_dataset = load_training_data(data_filepath)
        if not training_dataset:
            print("Нет данных для обучения. Программа завершается.")
            exit()
        print(f"Подготовлено {len(training_dataset)} точек данных для обучения.")

        print("\nНачальные параметры функций принадлежности (первые 2 для каждого входа):")
        for i in range(min(2, cfg.NUM_MF_E)): print(f"  ФП_e {i}: {anfis_controller.mf_params_input1[i]}")
        for i in range(min(2, cfg.NUM_MF_DE)): print(f"  ФП_de {i}: {anfis_controller.mf_params_input2[i]}")

        mse_history = []
        if cfg.MF_TYPE == 'gauss':
            mse_history, anfis_controller = train_anfis.train_anfis_full_backprop(
                anfis_controller, training_dataset, epochs=cfg.NUM_EPOCHS,
                learning_rate_premise=cfg.LEARNING_RATE_PREMISE,
                learning_rate_consequence=cfg.LEARNING_RATE_CONSEQUENCE,
                model_save_path=model_params_path  # Путь для сохранения лучшей модели
            )
        else:
            print("ПРЕДУПРЕЖДЕНИЕ: Запускается упрощенное обучение (только заключения), т.к. MF_TYPE не 'gauss'.")
            mse_history, anfis_controller = train_anfis.train_anfis_simple(
                anfis_controller, training_dataset, epochs=cfg.NUM_EPOCHS,
                learning_rate_consequence=cfg.LEARNING_RATE_CONSEQUENCE  # Используем LR из config
            )

        if mse_history:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(mse_history) + 1), mse_history)
            plt.title('История MSE во время обучения ANFIS')
            plt.xlabel('Эпоха');
            plt.ylabel('MSE');
            plt.grid(True);
            plt.show()
    else:
        print("ANFIS модель успешно загружена с сохраненными параметрами.")

    print("\nПараметры функций принадлежности ПОСЛЕ обучения/загрузки (первые 2):")
    for i in range(min(2, cfg.NUM_MF_E)): print(f"  ФП_e {i}: {anfis_controller.mf_params_input1[i]}")
    for i in range(min(2, cfg.NUM_MF_DE)): print(f"  ФП_de {i}: {anfis_controller.mf_params_input2[i]}")

    # 4. Запуск тестирования обученной (или загруженной) модели ANFIS
    test_trained_anfis(anfis_controller, num_test_simulations=5, sim_duration_test=10.0)

    print("\n--- Работа программы завершена ---")
