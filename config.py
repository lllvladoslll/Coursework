# config.py
# Файл конфигурации: содержит все основные настройки и константы для проекта.

import numpy as np

H_TUBE = 0.5            # Высота трубки [метры]
BALL_RADIUS = 0.012     # Радиус шарика [метры]
TUBE_RADIUS = 0.020     # Внутренний радиус трубки [метры]
M_BALL = 0.0015         # Масса шарика [кг] (1.5 грамма)

G = 9.81                # Ускорение свободного падения [м/с^2]
RHO_AIR = 1.2           # Плотность воздуха [кг/м^3]
C_DRAG_BALL = 0.47      # Коэффициент аэродинамического сопротивления для сферы (безразмерный)

A_BALL_CROSS_SECTION = np.pi * BALL_RADIUS**2   # Площадь поперечного сечения шарика [м^2]
A_TUBE_CROSS_SECTION = np.pi * TUBE_RADIUS**2   # Площадь поперечного сечения трубки [м^2]
A_GAP = A_TUBE_CROSS_SECTION - A_BALL_CROSS_SECTION # Площадь зазора [м^2]

# Защита от слишком малого зазора
if A_GAP <= 1e-7:
    print(f"ПРЕДУПРЕЖДЕНИЕ в config: Расчетный зазор A_GAP ({A_GAP:.2e}) очень мал. Установлен в 10% от площади трубки.")
    A_GAP = A_TUBE_CROSS_SECTION * 0.1

PWM_MIN = 0             # Минимальное значение управляющего сигнала
PWM_MAX = 255           # Максимальное значение сигнала
ETA_MAX = 4000.0        # Максимальная скорость вращения вентилятора

K_M_MOTOR = 15.69       # Коэффициент усиления двигателя
T_M_MOTOR = 0.07        # инерционность [с]
D_M_MOTOR = 0.85        # характер отклика

K_ETA_TO_VAIR = 0.0045
K_FRICTION_BALL = 0.002



SIM_DURATION = 2.0      # Длительность одной симуляции для сбора обучающих данных [с]
SIM_TIMESTEP = 0.01     # Шаг времени для численного решения ОДУ

NUM_SIMULATIONS_FOR_DATA = 100 # Количество попыток симуляций для создания одного набора данных

Z_DESIRED = H_TUBE / 2  # Целевая высота шарика в трубке [м]

E_MIN = Z_DESIRED - H_TUBE
E_MAX = Z_DESIRED
DE_MIN = -2.0
DE_MAX = 2.0

NUM_MF_E = 5
NUM_MF_DE = 5
MF_TYPE = 'gauss'

LEARNING_RATE_CONSEQUENCE = 0.001
LEARNING_RATE_PREMISE = 0.00005
NUM_EPOCHS = 150

USE_GAP_VELOCITY_ENHANCEMENT = True
MAX_GAP_VELOCITY_FACTOR = 2.5