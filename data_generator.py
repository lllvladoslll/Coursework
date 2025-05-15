# data_generator.py

import numpy as np
import ball_tube_simulation as bts  # Убедитесь, что bts - это ваш ball_tube_simulation
import config as cfg
import random
import os
import json
import time
import matplotlib.pyplot as plt  # Если нужна будет отладочная визуализация здесь


class SimplePDController:
    def __init__(self, kp, kd, pwm_min=cfg.PWM_MIN, pwm_max=cfg.PWM_MAX, setpoint=cfg.Z_DESIRED, base_pwm=95.0):
        self.kp = kp
        self.kd = kd
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.setpoint = setpoint
        self.base_pwm = base_pwm

    def calculate(self, current_z, current_z_dot):
        error = self.setpoint - current_z
        error_derivative = -current_z_dot
        output = self.base_pwm + self.kp * error + self.kd * error_derivative
        output_clipped = np.clip(output, self.pwm_min, self.pwm_max)
        return output_clipped


def generate_data_with_pd(
        pd_controller,
        initial_conditions_sampler,  # Функция, которая возвращает [z0, zd0, eta0, eta0_dot]
        num_simulations=100,
        sim_duration_per_run=5.0,  # Длительность одной симуляции для сбора данных
        sample_time=0.05,  # Как часто собирать точки внутри успешной симуляции
        min_successful_duration=1.0,  # Минимальная длительность успешного пробега для сбора данных
        verbose=True
):
    """
    Общая функция для генерации данных с заданным ПД-контроллером и генератором начальных условий.
    Собирает данные только из УСПЕШНЫХ и достаточно длительных симуляций.
    """
    dataset = []
    total_data_points_collected = 0
    successful_runs_for_data = 0

    print(
        f"Начало генерации данных: {num_simulations} симуляций, ПД: Kp={pd_controller.kp}, Kd={pd_controller.kd}, BasePWM={pd_controller.base_pwm}")

    for i_sim in range(num_simulations):
        initial_state = initial_conditions_sampler()
        current_state_full = list(initial_state)  # Создаем копию

        current_time = 0.0
        last_sample_time = -sample_time
        points_from_this_run = []
        run_was_successful_so_far = True

        if verbose and (i_sim + 1) % (num_simulations // 10 if num_simulations >= 10 else 1) == 0:
            print(
                f"  Симуляция {i_sim + 1}/{num_simulations}... (z0={initial_state[0]:.2f}, eta0={initial_state[2]:.1f})")

        while current_time < sim_duration_per_run:
            current_z, current_z_dot = current_state_full[0], current_state_full[1]
            u_pwm = pd_controller.calculate(current_z, current_z_dot)

            step_duration = cfg.SIM_TIMESTEP  # Шаг симуляции из config
            time_remaining = sim_duration_per_run - current_time
            if step_duration > time_remaining:
                step_duration = time_remaining
            if step_duration <= 1e-6:
                break

            sol_step, step_success, reason = bts.run_simulation_v3(
                current_state_full[0], current_state_full[1],
                current_state_full[2], current_state_full[3],
                u_pwm, duration=step_duration
            )

            if step_success:
                if current_time - last_sample_time >= sample_time:
                    error_e = cfg.Z_DESIRED - current_z
                    error_de = -current_z_dot
                    points_from_this_run.append({'inputs': [error_e, error_de], 'target_pwm': u_pwm})
                    last_sample_time = current_time

                current_state_full = sol_step.y[:, -1]
                current_time += sol_step.t[-1]
            else:
                run_was_successful_so_far = False
                if verbose and (i_sim + 1) % (
                num_simulations // 20 if num_simulations >= 20 else 1) == 0:  # Печатаем реже о провалах
                    print(
                        f"    Симуляция {i_sim + 1} прервана ({reason}) на t={current_time + sol_step.t[-1]:.2f}с. Точки не добавлены.")
                break  # Прерываем текущую симуляцию, если шаг не удался

        # Добавляем точки, только если вся симуляция была успешной и достаточно долгой
        if run_was_successful_so_far and current_time >= min_successful_duration:
            dataset.extend(points_from_this_run)
            total_data_points_collected += len(points_from_this_run)
            successful_runs_for_data += 1
            if verbose and successful_runs_for_data % (num_simulations // 20 if num_simulations >= 20 else 1) == 0:
                print(
                    f"    Успешная симуляция {i_sim + 1} добавила {len(points_from_this_run)} точек. Всего собрано: {total_data_points_collected}.")

    print(
        f"Генерация завершена. Собрано {total_data_points_collected} точек из {successful_runs_for_data} успешных и достаточно длительных симуляций (из {num_simulations} попыток).")
    return dataset


# --- Функции для генерации различных начальных условий ---

def sampler_good_conditions():
    """Генерирует начальные условия для "хороших" данных."""
    # Старты вокруг Z_DESIRED или в широком безопасном диапазоне
    z0 = random.uniform(0.2 * cfg.H_TUBE, 0.8 * cfg.H_TUBE)
    z0_dot = random.uniform(-0.2, 0.2)  # Более широкий диапазон скоростей
    eta0 = random.uniform(cfg.ETA_MAX * 0.1, cfg.ETA_MAX * 0.8)  # Начать с работающим вентилятором
    eta0_dot = random.uniform(-100, 100)
    return [np.clip(z0, cfg.BALL_RADIUS * 1.1, cfg.H_TUBE - cfg.BALL_RADIUS * 1.1), z0_dot,
            np.clip(eta0, 0, cfg.ETA_MAX), eta0_dot]


def sampler_challenging_conditions():
    """Генерирует начальные условия для "сложных, но проходимых ПД(250,20,95)" данных."""
    # Акцент на z0 >= 0.15м, но с низкой начальной eta
    if random.random() < 0.8:  # 80% шанс на старт в "сложной, но проходимой" зоне
        z0 = random.uniform(0.15, 0.35 * cfg.H_TUBE)  # От 0.15м до ~0.175м
    else:  # 20% шанс на другие "средне-сложные" старты
        z0 = random.uniform(0.35 * cfg.H_TUBE, 0.6 * cfg.H_TUBE)

    z0_dot = random.uniform(-0.1, 0.1)  # Небольшие начальные скорости

    if random.random() < 0.7:  # 70% шанс на очень низкую eta
        eta0 = random.uniform(0, cfg.ETA_MAX * 0.15)  # 0 - 600
    else:  # 30% шанс на eta чуть выше
        eta0 = random.uniform(cfg.ETA_MAX * 0.15, cfg.ETA_MAX * 0.4)  # 600 - 1600

    eta0_dot = random.uniform(-50, 50)
    return [np.clip(z0, cfg.BALL_RADIUS * 1.1, cfg.H_TUBE - cfg.BALL_RADIUS * 1.1), z0_dot,
            np.clip(eta0, 0, cfg.ETA_MAX), eta0_dot]


if __name__ == '__main__':
    start_time_total_main = time.time()

    # --- Определяем "лучший" ПД-контроллер для генерации ВСЕХ данных ---
    best_pd_kp = 250
    best_pd_kd = 20
    best_pd_base_pwm = 95.0  # Убедитесь, что SimplePDController использует это значение

    pd_ideal_teacher = SimplePDController(
        kp=best_pd_kp,
        kd=best_pd_kd,
        base_pwm=best_pd_base_pwm,
        setpoint=cfg.Z_DESIRED
    )

    # --- 1. Генерация "хороших" данных ---
    print("\n--- Генерация 'хороших' данных с ПД(250,20,95) ---")
    good_data = generate_data_with_pd(
        pd_controller=pd_ideal_teacher,
        initial_conditions_sampler=sampler_good_conditions,
        num_simulations=1000,  # Количество попыток для хороших данных
        sim_duration_per_run=4.0,  # Длительность одной симуляции
        sample_time=0.04,  # Частота сбора точек
        min_successful_duration=2.0  # Успешная симуляция должна быть не короче
    )

    data_dir_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir_main): os.makedirs(data_dir_main)

    good_data_filename = "anfis_data_pd250_good_FINAL.json"
    good_data_filepath = os.path.join(data_dir_main, good_data_filename)
    if good_data:
        with open(good_data_filepath, 'w') as f:
            json.dump(good_data, f, indent=4)
        print(f"'Хорошие' данные сохранены в: {good_data_filepath}")
    else:
        print(f"Не удалось сгенерировать 'хорошие' данные.")

    # --- 2. Генерация "сложных, но проходимых" данных ---
    print("\n--- Генерация 'сложных, но проходимых ПД(250,20,95)' данных ---")
    challenging_data = generate_data_with_pd(
        pd_controller=pd_ideal_teacher,  # Тот же самый ПД-учитель
        initial_conditions_sampler=sampler_challenging_conditions,
        num_simulations=2000,  # Больше попыток для сложных данных
        sim_duration_per_run=3.0,  # Могут быть короче, важен сам факт прохождения
        sample_time=0.02,
        min_successful_duration=1.0
    )

    difficult_data_filename = "anfis_data_pd250_difficult_FINAL.json"
    difficult_data_filepath = os.path.join(data_dir_main, difficult_data_filename)
    if challenging_data:
        with open(difficult_data_filepath, 'w') as f:
            json.dump(challenging_data, f, indent=4)
        print(f"'Сложные, но проходимые' данные сохранены в: {difficult_data_filepath}")
    else:
        print(f"Не удалось сгенерировать 'сложные, но проходимые' данные.")

    end_time_total_main = time.time()
    print(f"\nВсё обучение заняло: {end_time_total_main - start_time_total_main:.2f} секунд.")