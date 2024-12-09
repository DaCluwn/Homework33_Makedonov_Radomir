import os
import json


import dill
import pandas as pd
import pickle
from datetime import datetime
#path = '/opt/airflow/airflow_hw'
#model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
path = os.environ.get('PROJECT_PATH', '.')

def predict():
    # Путь к последней сохраненной модели
    model_directory_path = path+'/data/models/'
    models_files_list = os.listdir(model_directory_path)
    paths = [os.path.join(model_directory_path, basename) for basename in models_files_list]
    latest_model = max(paths, key=os.path.getctime)
    print(latest_model)
    model_path = latest_model

    # Путь к папке с тестовыми данными и папке для сохранения предсказаний
    test_data_path = path+'/data/test'
    predictions_path = path+'/data/predictions/predictions'+datetime.now().strftime("%Y%m%d%H%M")+'.csv'

    # Проверяем, что папка для сохранения предсказаний существует
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    # Загружаем модель
    with open(model_path, 'rb') as file:
        model = dill.load(file)

    # Инициализируем пустой DataFrame для хранения предсказаний
    predictions_df = pd.DataFrame()

    # Читаем каждый файл в тестовой папке и делаем предсказание
    for filename in os.listdir(test_data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(test_data_path, filename)

            with open(file_path, 'r') as f:
                json_data = json.load(f)

            data = pd.json_normalize(json_data)

            pred = model.predict(data).tolist()

            # Сохраняем предсказания в DataFrame
            pred_df = pd.DataFrame([pred], columns=['prediction'])
            pred_df['filename'] = filename
            predictions_df = pd.concat([predictions_df, pred_df], ignore_index=True)

    # Сохраняем предсказания в CSV
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Predictions saved to {predictions_path}')


if __name__ == '__main__':
    predict()
