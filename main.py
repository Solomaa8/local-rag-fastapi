import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager

# Настройка системных переменных для работы Ollama и локального кэша моделей
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["HF_HOME"] = "D:\\hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

# Настройка логирования сервера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from sentence_transformers import CrossEncoder

from history_manager import get_sessions, create_session, load_session, add_message, delete_session
from config import settings

load_dotenv()

# Глобальные объекты для ретриверов и модели переранжирования
vectorstore = None
bm25_retriever = None
reranker = None


def load_reranker():
    """Загрузка Cross-Encoder модели для переранжирования найденных фрагментов."""
    global reranker
    if reranker is None:
        logger.info("Загрузка модели реранкера BAAI/bge-reranker-base...")
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        logger.info("Реранкер успешно загружен.")
    return reranker


def get_loaders(docs_dir: str):
    """Инициализация лоадеров для чтения файлов разных форматов из директории."""
    return [
        DirectoryLoader(
            docs_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
        ),
        DirectoryLoader(
            docs_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=False,
            loader_kwargs={"encoding": "utf-8"},
        ),
        DirectoryLoader(
            docs_dir,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=False,
        ),
    ]


def build_contextual_chunks(docs):
    """
    Двухуровневая разбивка документов (Contextual Chunking):
    1. Делим текст на крупные родительские фрагменты (parent chunks).
    2. Каждый родительский фрагмент разбиваем на мелкие дочерние блоки (child chunks).
    3. Привязываем текст родительского контекста к метаданным каждого дочернего блока.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )

    parent_chunks = parent_splitter.split_documents(docs)
    child_chunks = []

    for parent in parent_chunks:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_content"] = parent.page_content
            child_chunks.append(child)

    logger.info(
        f"Контекстное разбиение завершено: {len(parent_chunks)} родительских -> {len(child_chunks)} дочерних блоков."
    )
    return child_chunks


def init_vector_db():
    """Инициализация гибридного поиска: векторного хранилища Chroma и полнотекстового поиска BM25."""
    global vectorstore, bm25_retriever

    docs_dir = settings.docs_dir
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    all_docs = []
    for loader in get_loaders(docs_dir):
        try:
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            logger.error(f"Ошибка загрузки документов: {e}")

    if not all_docs:
        logger.warning("В директории docs/ файлы не найдены.")
        vectorstore = None
        bm25_retriever = None
        return

    logger.info(f"Загружено {len(all_docs)} исходных документов.")
    child_chunks = build_contextual_chunks(all_docs)

    # Индексация в векторную базу Chroma с обработкой повторных попыток
    batch_size = 5
    vs = None
    for embed_attempt in range(10):
        try:
            vs = Chroma.from_documents(documents=child_chunks[:batch_size], embedding=embeddings)
            break
        except Exception as e:
            if embed_attempt < 9:
                logger.warning(
                    f"Ошибка создания эмбеддингов (попытка {embed_attempt + 1}/10): {type(e).__name__}. Повтор через 5 сек..."
                )
                time.sleep(5)
            else:
                logger.error("Не удалось создать эмбеддинги после 10 попыток.")
                vectorstore = None
                bm25_retriever = None
                return

    for i in range(batch_size, len(child_chunks), batch_size):
        batch = child_chunks[i: i + batch_size]
        vs.add_documents(documents=batch)
    vectorstore = vs
    logger.info("Векторное хранилище Chroma инициализировано.")

    # Создание индекса BM25 для поиска по точным ключевым словам
    bm25_retriever = BM25Retriever.from_documents(child_chunks)
    bm25_retriever.k = settings.retrieval_k
    logger.info("Индекс BM25 успешно создан.")

    load_reranker()
    logger.info("RAG система готова к работе.")


def wait_for_ollama(host: str = "127.0.0.1", port: int = 11434, max_tries: int = 30, delay: float = 2.0):
    """Проверка доступности порта Ollama перед стартом сервиса."""
    import socket
    logger.info(f"Проверка подключения к Ollama на {host}:{port}...")
    for i in range(max_tries):
        try:
            with socket.create_connection((host, port), timeout=3):
                logger.info("Порт Ollama открыт, ожидаем готовности модели...")
                time.sleep(3)
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass
        logger.info(f"  Ollama не отвечает, повтор {i + 1}/{max_tries}...")
        time.sleep(delay)
    logger.warning("Не удалось подключиться к Ollama, продолжаем запуск.")
    return False


def rerank_docs(query: str, docs, top_n: int = None):
    """Переранжирование найденных фрагментов с помощью Cross-Encoder реранкера."""
    if top_n is None:
        top_n = settings.reranker_top_n

    model = load_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    result = []
    for score, doc in scored[:top_n]:
        doc.metadata["reranker_score"] = float(score)
        result.append(doc)
    return result


def build_prompt_and_chain(mode: str, context: str = ""):
    """Сборка цепочки обращений к LLM для обычного диалога или режим ответов по документам."""
    if mode == "chat":
        system_prompt = "Ты — продвинутый ИИ-помощник. Отвечай подробно и на русском языке."
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
    else:
        system_prompt = (
            "Ты — эксперт по анализу документов. "
            "Используй ТОЛЬКО следующий контекст для ответа на вопрос. "
            "Если ответа нет в контексте, так и скажи. "
            "Отвечай на русском языке.\n\n"
            f"Контекст:\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
    return prompt_template | llm


# Инициализация моделей Ollama
embeddings = OllamaEmbeddings(model=settings.embeddings_model)
llm = ChatOllama(
    model=settings.llm_model,
    temperature=0.3,
    base_url=settings.ollama_host,
    timeout=settings.llm_timeout,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Асинхронная инициализация базы знаний при старте FastAPI."""
    logger.info("Запуск бэкенда RAG системы...")
    wait_for_ollama()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_vector_db)
    logger.info("Инициализация завершена.")
    yield
    logger.info("Остановка сервера.")


app = FastAPI(title="Local LLM API — Advanced RAG", lifespan=lifespan)

# Настройка CORS для обращения со стороны фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    mode: str = "chat"
    session_id: str


def prepare_chat_history(session_id: str):
    """Формирование истории диалога в формате сообщений LangChain."""
    session_data = load_session(session_id)
    history = session_data.get("messages", [])

    chat_history = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    return chat_history[-settings.max_history:]


def get_rag_context(query: str):
    """Поиск контекста по документам с использованием гибридного ретривера и реранкера."""
    if vectorstore is None or bm25_retriever is None:
        return None, []

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    candidates = ensemble.invoke(query)
    top_docs = rerank_docs(query, candidates)

    context_parts = []
    for doc in top_docs:
        parent = doc.metadata.get("parent_content")
        context_parts.append(parent if parent else doc.page_content)
    context = "\n\n---\n\n".join(context_parts)

    sources = []
    seen = set()
    for doc in top_docs:
        source_path = doc.metadata.get("source", "Unknown")
        source_name = os.path.basename(source_path)
        page = doc.metadata.get("page", 0)
        score = doc.metadata.get("reranker_score", 0.0)
        key = (source_name, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source_name,
                "page": page,
                "score": round(score, 4),
            })

    return context, sources


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Синхронный эндпоинт отправки сообщений."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    chat_history = prepare_chat_history(req.session_id)
    sources = []

    if req.mode == "chat":
        chain = build_prompt_and_chain("chat")
        response = chain.invoke({"input": req.message, "chat_history": chat_history})
        answer = response.content

    elif req.mode == "pdf":
        if vectorstore is None or bm25_retriever is None:
            answer = (
                "База документов пуста. "
                "Пожалуйста, добавьте файлы (.pdf, .txt, .docx) в папку docs/ и нажмите «Перезагрузить документы»."
            )
        else:
            context, sources = get_rag_context(req.message)
            chain = build_prompt_and_chain("pdf", context)
            response = chain.invoke({"input": req.message, "chat_history": chat_history})
            answer = response.content
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'chat' or 'pdf'.")

    add_message(req.session_id, "user", req.message)
    add_message(req.session_id, "assistant", answer, sources=sources if req.mode == "pdf" else None)

    return {"reply": answer, "sources": sources}


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Стриминговый эндпоинт генерации ответа через Server-Sent Events (SSE)."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    if not req.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    chat_history = prepare_chat_history(req.session_id)

    if req.mode == "pdf" and (vectorstore is None or bm25_retriever is None):
        async def no_docs_stream():
            msg = "База документов пуста. Добавьте файлы в папку docs/ и нажмите «Перезагрузить документы»."
            yield f"data: {msg}\n\n"
            yield "data: [DONE]\n\n"
            add_message(req.session_id, "user", req.message)
            add_message(req.session_id, "assistant", msg)
        return StreamingResponse(no_docs_stream(), media_type="text/event-stream")

    context = ""
    sources = []
    if req.mode == "pdf":
        context, sources = get_rag_context(req.message)

    chain = build_prompt_and_chain(req.mode, context)

    async def generate():
        import json
        full_answer = ""
        async for chunk in chain.astream({"input": req.message, "chat_history": chat_history}):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                full_answer += content
                yield f"data: {content}\n\n"

        if req.mode == "pdf" and sources:
            yield f"data: [SOURCES] {json.dumps(sources, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        add_message(req.session_id, "user", req.message)
        add_message(
            req.session_id,
            "assistant",
            full_answer,
            sources=sources if req.mode == "pdf" else None,
        )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/sessions")
async def api_get_sessions():
    """Получить список существующих сессий."""
    return {"sessions": get_sessions()}


@app.post("/api/sessions")
async def api_create_session():
    """Создать новую сессию чата."""
    session_id = create_session()
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    """Загрузить историю сообщений конкретной сессии."""
    return load_session(session_id)


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """Удалить сессию чата."""
    delete_session(session_id)
    return {"status": "ok"}


@app.get("/api/docs")
async def list_docs():
    """Список проиндексированных документов в папке docs/."""
    docs_dir = settings.docs_dir
    if not os.path.exists(docs_dir):
        return {"docs": []}

    files = []
    supported = {".pdf", ".txt", ".docx", ".doc"}
    for fname in os.listdir(docs_dir):
        fp = os.path.join(docs_dir, fname)
        if os.path.isfile(fp) and os.path.splitext(fname)[1].lower() in supported:
            stat = os.stat(fp)
            files.append({
                "name": fname,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": stat.st_mtime,
            })

    files.sort(key=lambda x: x["name"])
    return {"docs": files}


@app.get("/api/docs/{filename}")
async def download_doc(filename: str):
    """Скачивание исходного файла из папки docs/."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(settings.docs_dir, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл '{safe_name}' не найден.")

    return FileResponse(
        path=file_path,
        filename=safe_name,
        media_type="application/octet-stream",
    )


@app.post("/api/reload-docs")
async def reload_docs():
    """Динамическая переиндексация документов из папки docs/."""
    logger.info("Переиндексация документов...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_vector_db)
    status = "ok" if vectorstore is not None else "empty"
    logger.info(f"Переиндексация завершена. Статус: {status}")
    return {"status": status, "message": "Документы успешно переиндексированы."}