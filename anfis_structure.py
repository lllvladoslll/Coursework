# anfis_structure.py
import numpy as np
import itertools
import config as cfg


# --- Функции принадлежности ---
def triangular_mf(x, params):
    a, b, c = params
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b - a != 0 else (1.0 if x == b else 0.0)
    elif b < x < c:
        return (c - x) / (c - b) if c - b != 0 else (1.0 if x == b else 0.0)
    return 0.0  # На случай, если что-то пойдет не так

def gaussian_mf(x, params):
    mu, sigma = params
    if sigma == 0:  # Предотвращение деления на ноль
        return 1.0 if x == mu else 0.0
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


# --- Класс для ANFIS контроллера ---
class ANFIS:
    def __init__(self, num_inputs=2):

        self.num_inputs = num_inputs
        self.mf_params_input1 = []
        self.mf_params_input2 = []
        self._initialize_mf_params()
        self.num_rules = cfg.NUM_MF_E * cfg.NUM_MF_DE
        self.consequent_params = np.random.randn(self.num_rules,
                                                 self.num_inputs + 1) * 0.1
        self.rule_antecedents = list(itertools.product(range(cfg.NUM_MF_E), range(cfg.NUM_MF_DE)))

    def _initialize_mf_params(self):

        centers_e = np.linspace(cfg.E_MIN, cfg.E_MAX, cfg.NUM_MF_E)
        if cfg.NUM_MF_E > 1:
            sigma_e = (cfg.E_MAX - cfg.E_MIN) / (2 * (cfg.NUM_MF_E - 1)) if cfg.NUM_MF_E > 1 else (cfg.E_MAX - cfg.E_MIN) / 2
            if sigma_e == 0: sigma_e = 0.1
        else:
            sigma_e = (cfg.E_MAX - cfg.E_MIN) / 2
            if sigma_e == 0: sigma_e = 0.1

        for i in range(cfg.NUM_MF_E):
            mu = centers_e[i]
            if cfg.MF_TYPE == 'gauss':
                self.mf_params_input1.append([mu, sigma_e])
            elif cfg.MF_TYPE == 'triangular':
                a = mu - sigma_e
                c = mu + sigma_e
                self.mf_params_input1.append([a, mu, c])

        centers_de = np.linspace(cfg.DE_MIN, cfg.DE_MAX, cfg.NUM_MF_DE)
        if cfg.NUM_MF_DE > 1:
            sigma_de = (cfg.DE_MAX - cfg.DE_MIN) / (2 * (cfg.NUM_MF_DE - 1)) if cfg.NUM_MF_DE > 1 else (cfg.DE_MAX - cfg.DE_MIN) / 2
            if sigma_de == 0: sigma_de = 0.1
        else:
            sigma_de = (cfg.DE_MAX - cfg.DE_MIN) / 2
            if sigma_de == 0: sigma_de = 0.1

        for i in range(cfg.NUM_MF_DE):
            mu = centers_de[i]
            if cfg.MF_TYPE == 'gauss':
                self.mf_params_input2.append([mu, sigma_de])
            elif cfg.MF_TYPE == 'triangular':
                a = mu - sigma_de
                c = mu + sigma_de
                self.mf_params_input2.append([a, mu, c])

    def _get_mf_value(self, x, input_index, mf_index):

        if input_index == 0:
            params = self.mf_params_input1[mf_index]
        elif input_index == 1:
            params = self.mf_params_input2[mf_index]
        else:
            raise ValueError("Неверный индекс входной переменной")

        if cfg.MF_TYPE == 'gauss':
            return gaussian_mf(x, params)
        elif cfg.MF_TYPE == 'triangular':
            return triangular_mf(x, params)
        else:
            raise ValueError(f"Неизвестный тип функции принадлежности: {cfg.MF_TYPE}")

    def predict(self, inputs):

        if len(inputs) != self.num_inputs:
            raise ValueError(f"Ожидалось {self.num_inputs} входов, получено {len(inputs)}")

        input_val1 = inputs[0]
        input_val2 = inputs[1]

        # --- Слой 1: Фаззификация ---
        # Вычисляем значения всех функций принадлежности для каждого входа.
        mf_outputs_input1 = [self._get_mf_value(input_val1, 0, i) for i in range(cfg.NUM_MF_E)]
        mf_outputs_input2 = [self._get_mf_value(input_val2, 1, i) for i in range(cfg.NUM_MF_DE)]

        # --- Слой 2: Активация правил (степень выполнения предпосылок) ---
        rule_strengths = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            mf_idx_input1, mf_idx_input2 = self.rule_antecedents[i]
            # Сила правила w_i = MF1(x1) * MF2(x2)
            rule_strengths[i] = mf_outputs_input1[mf_idx_input1] * mf_outputs_input2[mf_idx_input2]

        # --- Слой 3: Нормализация сил активации правил ---
        sum_rule_strengths = np.sum(rule_strengths)
        if sum_rule_strengths == 0:

            normalized_strengths = np.zeros_like(rule_strengths)
        else:
            normalized_strengths = rule_strengths / sum_rule_strengths

        # --- Слой 4: Вычисление выхода каждого правила (дефаззификация предпосылок) ---
        rule_outputs = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            p_i, q_i, r_i = self.consequent_params[i]
            f_i = self.consequent_params[i, 0] * input_val1 + \
                  self.consequent_params[i, 1] * input_val2 + \
                  self.consequent_params[i, 2]
            rule_outputs[i] = normalized_strengths[i] * f_i

        # --- Слой 5: Конечное выходное значение ---
        final_output = np.sum(rule_outputs)

        return final_output


if __name__ == '__main__':
    print("Тестирование модуля ANFIS структуры...")
    anfis_test = ANFIS(num_inputs=2)

    print(f"Количество правил: {anfis_test.num_rules}")
    print("Параметры функций принадлежности для входа 1 (e):")
    for i, params in enumerate(anfis_test.mf_params_input1):
        print(f"  ФП {i}: {params}")
    print("Параметры функций принадлежности для входа 2 (de):")
    for i, params in enumerate(anfis_test.mf_params_input2):
        print(f"  ФП {i}: {params}")

    print(f"\nСтруктура правил (индексы ФП для входа1, входа2): {anfis_test.rule_antecedents}")
    print(f"\nНачальные параметры заключений (p_i, q_i, r_i для каждого правила):")
    for i in range(anfis_test.num_rules):
        print(f"  Правило {i}: {anfis_test.consequent_params[i, :]}")

    test_e = 0.1
    test_de = -0.05
    test_inputs = [test_e, test_de]

    predicted_pwm = anfis_test.predict(test_inputs)
    print(f"\nТестовый прямой проход ANFIS для входов e={test_e}, de={test_de}:")
    print(f"Предсказанный выход (u_pwm): {predicted_pwm}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    x_e_plot = np.linspace(cfg.E_MIN - 0.1, cfg.E_MAX + 0.1, 200)
    for i, params in enumerate(anfis_test.mf_params_input1):
        y_mf = [anfis_test._get_mf_value(x_val, 0, i) for x_val in x_e_plot]
        plt.plot(x_e_plot, y_mf, label=f'ФП_e {i}')
    plt.title('Функции принадлежности для входа "e" (ошибка)')
    plt.xlabel('Значение ошибки e')
    plt.ylabel('Степень принадлежности')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x_de_plot = np.linspace(cfg.DE_MIN - 0.1, cfg.DE_MAX + 0.1, 200)
    for i, params in enumerate(anfis_test.mf_params_input2):
        y_mf = [anfis_test._get_mf_value(x_val, 1, i) for x_val in x_de_plot]
        plt.plot(x_de_plot, y_mf, label=f'ФП_de {i}')
    plt.title('Функции принадлежности для входа "de" (производная ошибки)')
    plt.xlabel('Значение de')
    plt.ylabel('Степень принадлежности')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()