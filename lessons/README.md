# mle-uplift

Материалы спринта 5 курса Яндекс Практикум: uplift-моделирование (метрики, деревья, S/T/X/R-learner).

Рабочая папка: корень репозитория. CSV читаются относительно неё (`pd.read_csv("ab_results.csv")`).

## Окружение

- Python **3.10.11**
- Виртуальное окружение: `.venv`

Создание и установка:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install scikit-uplift duecredit
```

`scikit-uplift` (`sklift`) в `requirements.txt` нет — пакет нужен для `uplift_auc_score` и `qini_auc_score`.  
`duecredit` убирает предупреждение при импорте `causalml`.

Основные библиотеки: `causalml==0.15.5`, `catboost`, `xgboost`, `lightgbm`, `scikit-learn==1.7.0`, `shap`, `pandas`, `jupyterlab`.

### Jupyter Lab

Запуск из корня проекта, из `.venv`:

```powershell
jupyter lab --ip=0.0.0.0 --no-browser
```

Правки ноутбука на диске Jupyter часто не подхватывает. Нужно **перезагрузить файл с диска** и не сохранять старую вкладку — иначе автосейв затрёт изменения.

### Graphviz

Нужен для отрисовки uplift-деревьев (`UpliftTreeClassifier`). Бинарники, например:

`C:\Users\usachevae\AppData\Local\Programs\Graphviz\Graphviz-15.1.1-win64\bin`

Путь к `dot.exe` должен быть в `PATH`.

## Датасеты

Файлы лежат в корне. Скачать заново:

```powershell
curl.exe -L --fail -o ИМЯ.csv "URL"
```

| Файл | Откуда | Что внутри |
|---|---|---|
| `uplift_data_less3.csv` | [скачать](https://code.s3.yandex.net/landings-v2-machine-learning/Files/uplift_data_less3.csv?etag=c7503abfa46bf7d3ce822f36b3608f1f) | Тема 3, урок 3: признаки, `treatment`, целевая |
| `discount_uplift_metrics.csv` | [скачать](https://code.s3.yandex.net/landings-v2-machine-learning/Files/discount_uplift_metrics.csv?etag=27ad1d1f431b3606ae682740e785349a) | Тема 3, урок 3: готовые метрики скидки |
| `ab_results.csv` | [скачать](https://code.s3.yandex.net/landings-v2-machine-learning/Files/ab_results.csv?etag=f0c2f4829489fc94b0ffa598dd4bf3b0) | Такси Comfort A/B, ~400k строк, `treatment` 0/1, `target`. Тема 3 урок 4, тема 4 уроки 2, 4, 5, 6 |
| `predictions_less4.csv` | [скачать](https://code.s3.yandex.net/landings-v2-machine-learning/Files/predictions_less4.csv?etag=581ca1885ca5718a1fca575c2f6f9a55) | Тема 3, урок 4: готовые предсказания |
| `yandex_plus.csv` | [скачать](https://code.s3.yandex.net/landings-v2-machine-learning/Files/yandex_plus.csv?etag=dcb987c51f0c38179af580594174dd94) | Яндекс Плюс, `treatment`: строки `control` / `treatment1`, `conversion`. Тема 3 урок 5, тема 4 уроки 2, 4, 5, 6 |
| `top_users.csv` | локальный результат | Тема 3, урок 5: пользователи с `uplift_score > 0.2`. Не скачивается |

Для `yandex_plus.csv` метки воздействия перед `causalml` маппятся: `control → 0`, `treatment1 → 1`.

## Ноутбуки

### Тема 3 — метрики и uplift-деревья

| Файл | Содержание |
|---|---|
| `sprint5_theme3_lesson3.ipynb` | Группировки, Qini / uplift-кривые |
| `sprint5_theme3_lesson4.ipynb` | Uplift Random Forest, топ-5%, кривые по `predictions_less4.csv` |
| `sprint5_theme3_lesson5.ipynb` | Яндекс Плюс: конверсии, grid search Uplift@10%, `UpliftRandomForestClassifier`, `top_users.csv` |

### Тема 4 — meta-learners

| Файл | Содержание |
|---|---|
| `sprint5_theme4_lesson2.ipynb` | S-learner: вручную и `BaseSClassifier` (такси и Яндекс Плюс) |
| `sprint5_theme4_lesson3.ipynb` | Пустой (теория без кода) |
| `sprint5_theme4_lesson4.ipynb` | T-learner: две модели вручную и `BaseTClassifier` |
| `sprint5_theme4_lesson5.ipynb` | X-learner: вручную и `BaseXClassifier`, propensity |
| `sprint5_theme4_lesson6.ipynb` | R-learner: вручную (OOF + R-loss) и `BaseRClassifier` |

## Ориентиры по метрикам (такси `ab_results.csv`)

На одном и том же A/B такси causalml даёт примерно:

| Модель | Uplift AUC | Qini AUC |
|---|---|---|
| S-learner | 0.16 | 0.19–0.20 |
| T-learner (RF) | 0.16 | 0.20 |
| X-learner | 0.16–0.17 | 0.20 |
| R-learner (causalml) | 0.16 | 0.20 |
| R-learner вручную из урока | 0.07 | 0.09 |

На **Яндекс Плюс** R-learner с эталонными параметрами выше: Uplift AUC ≈ **0.43**, Qini AUC ≈ **0.36–0.38**.  
Эталон задания 9: `n_estimators=100`, `learning_rate=0.005`, `max_depth=6`, **без** `p=` в `fit` (propensity считает сам `BaseRClassifier`).

S/T-learner на Плюсе: Uplift ≈ 0.25.

## Особенности causalml

- `UpliftRandomForestClassifier(control_name='control')` ждёт **строковые** метки treatment. Числа `0`/`1` дают `TypeError: can only concatenate str ...`.
- `BaseSClassifier` / `BaseTClassifier` / `BaseXClassifier` / `BaseRClassifier` с `control_name=0` работают с числовыми метками.
- В `fit`/`predict` колонку `treatment` в матрицу признаков **не** подавать — она идёт отдельным аргументом.
- `predict` часто возвращает `(n, 1)` — для `sklift` нужен `.squeeze()`.
- У `BaseSClassifier.predict` с CatBoost иногда `ValueError: assignment destination is read-only` — подавать копию массива (`np.array(...values.copy())`).
- `yandex_plus`: `treatment` сначала маппить в `0`/`1`.
