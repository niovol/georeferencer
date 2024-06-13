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

2. Соберите и запустите сервис с использованием Docker Compose:

    ```bash
    docker-compose up --build
    ```

Сервис будет доступен по адресу `http://localhost:8000`.
