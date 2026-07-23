import json
import os
import uuid
from datetime import datetime

# Папка для хранения файлов диалогов в формате JSON
SESSIONS_DIR = "sessions"

if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)


def get_session_file(session_id: str) -> str:
    """Формирование пути к JSON-файлу сессии."""
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def get_sessions() -> list[dict]:
    """Получение списка всех сохраненных сессий с сортировкой по времени обновления."""
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            file_path = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    title = data.get("title", "Новый чат")
                    updated_at = data.get("updated_at", "")
                    sessions.append({
                        "id": session_id,
                        "title": title,
                        "updated_at": updated_at
                    })
            except Exception:
                pass
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def create_session(title: str = "Новый чат") -> str:
    """Создание нового JSON-файла сессии диалога."""
    session_id = str(uuid.uuid4())
    data = {
        "id": session_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    with open(get_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return session_id


def load_session(session_id: str) -> dict:
    """Чтение сообщений из файла сессии."""
    file_path = get_session_file(session_id)
    if not os.path.exists(file_path):
        return {"messages": []}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки сессии {session_id}: {e}")
        return {"messages": []}


def add_message(session_id: str, role: str, content: str, sources: list = None):
    """Добавление сообщения пользователя или ассистента в историю сессии."""
    if not session_id:
        return
    file_path = get_session_file(session_id)
    if not os.path.exists(file_path):
        data = {
            "id": session_id,
            "title": "Новый чат",
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

    msg = {"role": role, "content": content}
    if sources:
        msg["sources"] = sources

    data["messages"].append(msg)

    # Автоматическое переименование заголовка по первому вопросу пользователя
    if data.get("title") == "Новый чат" and role == "user":
        data["title"] = content[:30] + ("..." if len(content) > 30 else "")

    data["updated_at"] = datetime.now().isoformat()

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения сообщения в сессию {session_id}: {e}")


def delete_session(session_id: str):
    """Удаление файла сессии."""
    file_path = get_session_file(session_id)
    if os.path.exists(file_path):
        os.remove(file_path)
