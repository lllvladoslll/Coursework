# merge_datasets.py
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "data")

file1_path = os.path.join(data_folder, "anfis_data_pd250_good_FINAL.json")
file2_path = os.path.join(data_folder, "anfis_data_pd250_difficult_FINAL.json")
output_file_path = os.path.join(data_folder, "anfis_COMBINED_data_PD250_FINAL.json")

dataset1 = []
dataset2 = []

# Загрузка первого файла
if os.path.exists(file1_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            dataset1 = json.load(f)
        print(f"Загружено {len(dataset1)} точек из {file1_path}")
    except Exception as e:
        print(f"Ошибка при загрузке {file1_path}: {e}")
else:
    print(f"Файл не найден: {file1_path}")

# Загрузка второго файла
if os.path.exists(file2_path):
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            dataset2 = json.load(f)
        print(f"Загружено {len(dataset2)} точек из {file2_path}")
    except Exception as e:
        print(f"Ошибка при загрузке {file2_path}: {e}")
else:
    print(f"Файл не найден: {file2_path}")

# Объединение датасетов
combined_dataset = dataset1 + dataset2
print(f"\nВсего точек в объединенном датасете: {len(combined_dataset)}")

# Сохранение объединенного датасета
if combined_dataset:
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(combined_dataset, f, indent=4)
        print(f"Объединенные данные сохранены в: {output_file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении объединенных данных: {e}")
else:
    print("Нет данных для объединения.")

