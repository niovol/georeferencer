# Руководство администратора

## Зависимости

- Docker
- Docker Compose

## Установка

1. Клонируйте репозиторий с субмодулями:

    ```bash
    git clone https://github.com/niovol/georeferencer --recurse-submodules
    cd <repository_directory>
    ```

2. Настройте файл `docker-compose.yml`, указав путь к директории с подложками на вашей локальной машине. Измените строку `- /layouts:/layouts`, заменив `/layouts` справа от двоеточия на путь к директории с подложками. Пример:

    ```yaml
    services:
      fastapi-app:
        container_name: nikolove18_fastapi
        image: nikolove18
        build:
          context: ./
          dockerfile: Dockerfile
        command: uvicorn src.api.server:app --reload --host 0.0.0.0
        volumes:
          - .:/app
          - /layouts:/path/to/local/layouts
        shm_size: '2gb'
        ports:
          - 8000:8000
    ```

3. Соберите и запустите сервис с использованием Docker Compose:

    ```bash
    docker-compose up --build
    ```

Сервис будет доступен по адресу `http://localhost:8000`.

## В случае проблем со стороны GitHub

Проект имеет внешнюю зависимость https://github.com/magicleap/SuperGluePretrainedNetwork/ в виде субмодуля
По какой-то причине этот файлы с этого проекта не всегда получается клонировать.
Если по какой-то причине не получается клонировать репозиторий, то есть следующие варианты решения:

1. Можете скачать с `Прототипа решения` с Яндекс Диска: https://disk.yandex.ru/d/SeeYQ8Zcr6mELQ

2. Можете скачать с GitHub проект https://github.com/niovol/georeferencer в виде .zip, и отдельно скачать субмодуль https://github.com/magicleap/SuperGluePretrainedNetwork
    Этот субмодуль необходимо разместить в папке `src/superglue`
