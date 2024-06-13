# Руководство администратора

## Установка

### Зависимости

- Docker
- Docker Compose

### Шаги установки

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/niovol/georeferencer
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
