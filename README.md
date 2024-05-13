# Распознавание рукописных символов EMNIST

## 1. Описание решения
Задача состоит в построении модели для распознавания рукописных символов.
Исходные данные: массив данных EMNIST, содержащих наборы пикселей.
У меня были проблемы с установкой пакета emnist для pyhton, поэтому я скачал данные с официального сайта проекта EMNIST.
Данные уже содержат тренировочную и тестовую выборки.
Для работы со стандартными классификаторами изображения переводятся в одномерные массивы.
В качестве классификатора я использую LogisticRegression с опцией multi_class = 'auto' из библиотеки scikit-learn.
Для оптимизации гиперпараметров был проведен поиск с помощью GridSearchCV (проводится подбор регуляризующего коэффициента), для трех фолдов.
Лучший классификатор показывает метрику accuracy 0.688 - по условию задания требуется не менее 0.68.


## 2. Установка и запуск сервиса

_Опишите в этом разделе, как запустить ваше решение, где должен запуститься сервис, как им пользоваться. Если вы хотите сообщить пользователям и проверяющим дополнительную информацию, сделайте это здесь._

```bash
git clone <link/to/your/repo>
cd <your_repo_name>
docker built <parameters>
docker run <parameters>
```