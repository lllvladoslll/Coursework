# ball_tube_simulation.py
# Описывает физическую модель системы "Шарик в трубке" и выполняет ее симуляцию.

import numpy as np
from scipy.integrate import solve_ivp
import config as cfg


def system_dynamics_v3(t, state, u_pwm):
    """
    Вычисляет производные состояний системы для решателя ОДУ.
    Возвращает: список производных [dz/dt, dz_dot/dt, deta/dt, deta_dot/dt].
    """
    z, z_dot, eta, eta_dot = state

    u_pwm_clipped = np.clip(u_pwm, cfg.PWM_MIN, cfg.PWM_MAX)

    tm_squared = cfg.T_M_MOTOR ** 2
    if tm_squared == 0:
        deta_ddot = 0
    else:
        deta_ddot = (cfg.K_M_MOTOR * u_pwm_clipped - eta - 2 * cfg.D_M_MOTOR * cfg.T_M_MOTOR * eta_dot) / tm_squared

    d_eta_dt_actual = eta_dot  # Производная скорости
    d_eta_dot_actual = deta_ddot  # Производная ускорения

    current_eta_next_step = eta + d_eta_dt_actual * cfg.SIM_TIMESTEP
    if current_eta_next_step >= cfg.ETA_MAX and d_eta_dt_actual > 0:
        d_eta_dt_actual = max(0, (cfg.ETA_MAX - eta) / cfg.SIM_TIMESTEP * 0.1)
        d_eta_dot_actual = -eta_dot / cfg.SIM_TIMESTEP * 0.1
    elif current_eta_next_step <= 0 and d_eta_dt_actual < 0:
        d_eta_dt_actual = min(0, (0 - eta) / cfg.SIM_TIMESTEP * 0.1)
        d_eta_dot_actual = -eta_dot / cfg.SIM_TIMESTEP * 0.1

    v_air_free_stream = cfg.K_ETA_TO_VAIR * eta  # Скорость воздуха в свободной трубке
    if v_air_free_stream < 0: v_air_free_stream = 0  # Скорость воздуха не может быть отрицательной

    v_air_effective = v_air_free_stream
    if cfg.USE_GAP_VELOCITY_ENHANCEMENT:
        if cfg.A_GAP > 1e-7:
            gap_factor = cfg.A_TUBE_CROSS_SECTION / cfg.A_GAP
            effective_gap_factor = min(gap_factor, cfg.MAX_GAP_VELOCITY_FACTOR)
            if effective_gap_factor < 1.0: effective_gap_factor = 1.0
            v_air_effective = v_air_free_stream * effective_gap_factor

    force_gravity = -cfg.M_BALL * cfg.G  # Сила тяжести

    v_relative_to_ball = v_air_effective - z_dot  # Относительная скорость воздуха относительно шарика

    force_aerodynamic = 0.5 * cfg.RHO_AIR * cfg.C_DRAG_BALL * cfg.A_BALL_CROSS_SECTION * (
                v_relative_to_ball ** 2) * np.sign(v_relative_to_ball)

    force_friction_ball = -cfg.K_FRICTION_BALL * z_dot  # Сила вязкого трения шарика (против скорости)

    total_force_on_ball = force_gravity + force_aerodynamic + force_friction_ball
    z_ddot = total_force_on_ball / cfg.M_BALL


    dz_dt = z_dot  # Производная положения z
    dz_dot_dt = z_ddot  # Производная скорости z_dot
    return [dz_dt, dz_dot_dt, d_eta_dt_actual, d_eta_dot_actual]


def run_simulation_v3(initial_ball_pos, initial_ball_vel, initial_eta, initial_eta_dot, u_pwm_value,
                      duration=cfg.SIM_DURATION):

    initial_state_full = [initial_ball_pos, initial_ball_vel, initial_eta, initial_eta_dot]

    def event_hit_top(t, state, u_pwm):
        return state[0] - cfg.H_TUBE

    event_hit_top.terminal = True
    event_hit_top.direction = 1

    def event_hit_bottom(t, state, u_pwm):
        return state[0] - 0.0001

    event_hit_bottom.terminal = True
    event_hit_bottom.direction = -1

    def event_eta_runaway(t, state, u_pwm):
        return abs(state[2]) - (cfg.ETA_MAX * 1.5)

    event_eta_runaway.terminal = True

    # Численное решение системы ОДУ
    sol = solve_ivp(
        fun=system_dynamics_v3,
        t_span=[0, duration],
        y0=initial_state_full,
        args=(u_pwm_value,),
        dense_output=True,
        events=[event_hit_top, event_hit_bottom, event_eta_runaway],
        max_step=cfg.SIM_TIMESTEP,
    )

    # Анализ результата завершения симуляции
    successful_run = True
    reason = "OK"

    if sol.status == -1:
        successful_run = False
        reason = "Integration failed"
    elif sol.t_events[0].size > 0:
        successful_run = False
        reason = "Hit top"
    elif sol.t_events[1].size > 0:
        successful_run = False
        reason = "Hit bottom"
    elif sol.t_events[2].size > 0:
        successful_run = False
        reason = f"Eta runaway: {sol.y[2, -1]:.1f}"


    final_z = sol.y[0, -1]

    is_within_bounds = (0 < final_z < cfg.H_TUBE)
    if successful_run and not is_within_bounds:
        successful_run = False
        reason = "Final Z out of bounds"  # Общее сообщение

    return sol, successful_run, reason


if __name__ == '__main__':

    print("Тестирование модуля симуляции v3 (с PT2-двигателем)...")

    # Примерные начальные условия и параметры для одного тестового прогона
    z0_test = cfg.H_TUBE * 0.1
    z_dot0_test = 0.0
    eta0_test = 500.0
    eta_dot0_test = 0.0
    pwm_test_value = 138
    sim_duration_test = 20.0

    print(
        f"Запуск симуляции v3: z0={z0_test:.2f}, z_dot0={z_dot0_test:.2f}, eta0={eta0_test:.1f}, eta_dot0={eta_dot0_test:.1f}, pwm={pwm_test_value}")

    solution, success, reason_fail = run_simulation_v3(z0_test, z_dot0_test, eta0_test, eta_dot0_test,
                                                       pwm_test_value, duration=sim_duration_test)

    print(f"\nТест симуляция завершена. Успех: {success}")
    if not success:
        print(f"  Причина: {reason_fail}")
    print(f"Время окончания: {solution.t[-1]:.2f} c / {sim_duration_test} c")
    final_state_sim = solution.y[:, -1]
    print(
        f"Конечное состояние: z={final_state_sim[0]:.3f} м, z_dot={final_state_sim[1]:.3f} м/с, eta={final_state_sim[2]:.1f}, eta_dot={final_state_sim[3]:.1f}")


    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)  # 4 графика: z, z_dot, eta, eta_dot

    # График Z
    axs[0].plot(solution.t, solution.y[0], label='Положение шарика (z)')
    axs[0].axhline(0, color='r', linestyle=':', label='Низ трубки')
    axs[0].axhline(cfg.H_TUBE, color='r', linestyle=':', label='Верх трубки')
    axs[0].axhline(cfg.Z_DESIRED, color='g', linestyle='-.', label=f'Желаемое Z ({cfg.Z_DESIRED:.2f}м)')
    axs[0].set_ylabel('z (м)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title(f'Тест симуляции v3 (PWM={pwm_test_value}, Успех: {success})')

    # График Z_dot
    axs[1].plot(solution.t, solution.y[1], label='Скорость шарика (z_dot)')
    axs[1].set_ylabel('z_dot (м/с)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # График Eta
    axs[2].plot(solution.t, solution.y[2], label='Скорость вентилятора (eta)')
    axs[2].axhline(cfg.ETA_MAX, color='orange', linestyle=':', label=f'Max Eta ({cfg.ETA_MAX:.0f})')
    axs[2].set_ylabel('eta (ед.)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    # График Eta_dot
    axs[3].plot(solution.t, solution.y[3], label='Ускорение eta (eta_dot)')
    axs[3].set_ylabel('eta_dot (ед./с)')
    axs[3].set_xlabel('Время (с)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
