import os
import re
import json
import asyncio
import threading
import torch
import httpx
import aiosqlite
import logging
from typing import List, Union, AsyncGenerator, Generator
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from bs4 import BeautifulSoup
from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_GEN_KEYWORDS = [r"\bнарисуй\b", r"\bсгенерируй\b", r"\bdraw\b", r"\bкартинк\b"]
SEARCH_KEYWORDS    = [r"\bнайди\b", r"\bчто такое\b", r"\bновости\b", r"\bпоиск\b", r"\bгугл\b"]
CODE_KEYWORDS      = [r"\bкод\b", r"\bскрипт\b", r"\bpython\b", r"\bsql\b", r"\bпрограмм\b"]
RESEARCH_KEYWORDS  = [r"deep research", r"исследуй", r"отчёт", r"подробный анализ"]

class Pipeline:
    class Valves(BaseModel):
        MWS_API_KEY:        str   = os.getenv("MWS_API_KEY", "")
        RETRIEVER_MODEL:    str   = "BAAI/bge-m3"
        RERANKER_MODEL:     str   = "BAAI/bge-reranker-v2-m3"
        RERANK_THRESHOLD:   float = 0.3
        TOP_K:              int   = 10
        MEMORY_FACTS_LIMIT: int   = 8
        HISTORY_LIMIT:      int   = 10
        DB_PATH:            str   = "/app/pipelines/memory.db"
        API_BASE:           str   = "https://api.gpt.mws.ru/v1"
        DEFAULT_MODEL:      str   = "mws-gpt-alpha"
        CODER_MODEL:        str   = "qwen3-coder-480b-a35b"
        VLM_MODEL:          str   = "qwen2.5-vl-72b"
        ASSISTANT_NAME:     str   = "GPTHub"

    def __init__(self):
        self.valves    = self.Valves()
        self.device    = "cpu"
        self.retriever = None
        self.reranker  = None
        self._model_lock = threading.Lock()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def on_startup(self):
        logger.info(f"[STARTUP] {self.valves.ASSISTANT_NAME} инициализация...")
        async with aiosqlite.connect(self.valves.DB_PATH) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    TEXT NOT NULL,
                    fact       TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, fact)
                )
            """)
            await db.commit()

    def _load_models(self):
        if self.retriever is not None: return
        with self._model_lock:
            if self.retriever is None:
                logger.info("[MODELS] Загрузка SentenceTransformer...")
                self.retriever = SentenceTransformer(self.valves.RETRIEVER_MODEL, device=self.device)
                self.reranker  = CrossEncoder(self.valves.RERANKER_MODEL, device=self.device)

    async def get_user_memory(self, user_id: str) -> str:
        try:
            async with aiosqlite.connect(self.valves.DB_PATH) as db:
                async with db.execute(
                    "SELECT fact FROM user_memories WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                    (user_id, self.valves.MEMORY_FACTS_LIMIT)
                ) as cur:
                    rows = await cur.fetchall()
            return "\n- ".join([r[0] for r in rows]) if rows else ""
        except: return ""

    async def save_fact(self, user_id: str, text: str):
        if len(text) < 20 or len(text) > 600 or "{" in text: return
        try:
            prompt = (
                f"Извлеки ОДИН краткий важный факт о пользователе из текста: «{text}». "
                f"Ответь только фактом. Если ничего нового, ответь 'НЕТ'."
            )
            fact = await self._llm_call(prompt, [], model=self.valves.DEFAULT_MODEL, stream=False)
            if not fact or "НЕТ" in fact.upper(): return
            fact = fact.strip().strip('"-.*')
            async with aiosqlite.connect(self.valves.DB_PATH) as db:
                await db.execute("INSERT OR IGNORE INTO user_memories (user_id, fact) VALUES (?, ?)", (user_id, fact))
                await db.commit()
        except: pass

    async def search_web(self, query: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=12.0, headers=self.headers) as client:
                r = await client.get(f"https://duckduckgo.com/html/?q={quote(query)}")
                soup = BeautifulSoup(r.text, "html.parser")
                links = [a["href"] for a in soup.find_all("a", class_="result__a", limit=1)]
                if links: return await self.fetch_url(links[0])
        except: pass
        return "Поиск не удался."

    async def fetch_url(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=18.0, headers=self.headers) as client:
                r = await client.get(url)
                soup = BeautifulSoup(r.text, "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header"]): s.decompose()
                return " ".join(soup.get_text().split())[:3500]
        except: return "Ошибка парсинга."

    async def get_rag_context(self, query: str, body: dict) -> str:
        files = body.get("files", [])
        chunks = [c["text"] for f in files for c in f.get("data", []) if "text" in c]
        if not chunks: return ""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_models)
        def _process():
            q_emb = self.retriever.encode(query, convert_to_tensor=True)
            c_embs = self.retriever.encode(chunks[:300], convert_to_tensor=True)
            scores = util.cos_sim(q_emb, c_embs)[0]
            top_idx = torch.topk(scores, k=min(self.valves.TOP_K, len(chunks))).indices
            return "\n---\n".join([chunks[i] for i in top_idx])[:3500]
        return await loop.run_in_executor(None, _process)

    def _format_content(self, content: Union[str, list], as_vlm: bool = False) -> Union[str, list]:
        """Приводит контент к нужному формату для выбранной модели."""
        if not isinstance(content, list):
            return [{"type": "text", "text": content}] if as_vlm else content
        
        if not as_vlm:
            return " ".join([part["text"] for part in content if part.get("type") == "text"])
        
        return content

    async def _llm_call(self, prompt: Union[str, list], history: List[dict], model: str = None, stream: bool = False) -> Union[str, AsyncGenerator]:
        model = model or self.valves.DEFAULT_MODEL
        is_vlm = (model == self.valves.VLM_MODEL)
        
        formatted_prompt = self._format_content(prompt, as_vlm=is_vlm)
        formatted_history = []
        for msg in history[-self.valves.HISTORY_LIMIT:]:
            formatted_history.append({
                "role": msg["role"],
                "content": self._format_content(msg["content"], as_vlm=is_vlm)
            })

        headers = {"Authorization": f"Bearer {self.valves.MWS_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": model, 
            "messages": formatted_history + [{"role": "user", "content": formatted_prompt}], 
            "stream": stream, 
            "temperature": 0.45
        }
        
        if stream:
            return self._stream_generator(payload, headers)
        
        async with httpx.AsyncClient(timeout=70.0) as client:
            r = await client.post(f"{self.valves.API_BASE}/chat/completions", headers=headers, json=payload)
            if r.status_code != 200:
                logger.error(f"[API ERROR] {r.status_code}: {r.text}")
                return f"Ошибка API: {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]

    async def _stream_generator(self, payload, headers) -> AsyncGenerator:
        async with httpx.AsyncClient(timeout=100.0) as client:
            try:
                async with client.stream("POST", f"{self.valves.API_BASE}/chat/completions", headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        yield f"Ошибка API ({r.status_code})"
                        return
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            if "[DONE]" in line: break
                            try:
                                chunk = json.loads(line[6:])["choices"][0]["delta"].get("content")
                                if chunk: yield chunk
                            except: continue
            except Exception as e:
                yield f"\n[Ошибка стрима: {e}]"

    async def run_async_pipeline(self, body: dict, **kwargs) -> Union[str, AsyncGenerator]:
        messages = body.get("messages", [])
        if not messages: return "Нет сообщений."
        
        user_id = body.get("user", {}).get("id", "guest")
        last_msg_content = messages[-1]["content"]
        text_for_intent = self._format_content(last_msg_content, as_vlm=False)
        msg_l = text_for_intent.lower()

        has_images = isinstance(last_msg_content, list) and any(p.get("type") == "image_url" for p in last_msg_content)
        history_has_images = any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in messages[-3:])
        has_files = bool(body.get("files"))
        urls = re.findall(r'https?://\S+', text_for_intent)

        context_tasks = []
        if has_files: 
            context_tasks.append(self.get_rag_context(text_for_intent, body))
        if urls: 
            context_tasks.append(self.fetch_url(urls[0]))
        if any(re.search(kw, msg_l) for kw in SEARCH_KEYWORDS):
            context_tasks.append(self.search_web(text_for_intent))

        gathered_contexts = await asyncio.gather(*context_tasks) if context_tasks else []
        context = "\n\n".join(filter(None, gathered_contexts))

        model = self.valves.DEFAULT_MODEL
        if has_images or history_has_images:
            model = self.valves.VLM_MODEL
        elif any(re.search(kw, msg_l) for kw in CODE_KEYWORDS):
            model = self.valves.CODER_MODEL
        
        if any(re.search(kw, msg_l) for kw in RESEARCH_KEYWORDS):
            return await self.deep_research(text_for_intent)
        
        if any(re.search(kw, msg_l) for kw in IMAGE_GEN_KEYWORDS):
            return f"Генерация: ![Image](https://image.pollinations.ai/prompt/{quote(text_for_intent)}?nologo=true)"

        logger.info(f"[PIPE] Модель: {model} | Контекст: {'есть' if context else 'нет'}")


        memory_str = await self.get_user_memory(user_id)

        if not has_images:
            asyncio.create_task(self.save_fact(user_id, text_for_intent))


        system = f"Ты {self.valves.ASSISTANT_NAME}. Отвечай профессионально."
        if memory_str: system += f"\n\nТЕБЕ ИЗВЕСТНО О ПОЛЬЗОВАТЕЛЕ:\n- {memory_str}"
        if context: system += f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ (Файлы/Веб):\n{context}"
        
        return await self._llm_call(
            prompt=last_msg_content,
            history=[{"role": "system", "content": system}] + messages[:-1], 
            model=model, 
            stream=body.get("stream", False)
        )

    def pipe(self, body: dict, **kwargs) -> Union[str, Generator]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.run_async_pipeline(body, **kwargs))
            if hasattr(result, '__aiter__'):
                def sync_gen():
                    try:
                        while True:
                            yield loop.run_until_complete(result.__anext__())
                    except StopAsyncIteration: pass
                    finally: loop.close()
                return sync_gen()
            else:
                loop.close()
                return result
        except Exception as e:
            if not loop.is_closed(): loop.close()
            return f"Ошибка выполнения: {e}"
