"""
m15_langchain_agent.py — Kapstone modul: ReAct hujjat yordamchisi agenti.

Shartnoma:
    class DocumentAssistantAgent:
        def run(user_message: str) -> str      # to'liq ReAct agent sikli

d16_p15_langchain_agent.ipynb dan olingan. Agent tool funksiyalarini QAYTA
implementatsiya qilmaydi — u vositalar reyestrini (registry) qabul qiladi va
cheklangan ReAct sikli (run_react) orqali ularni chaqiradi.

Standart holatda kichik demo vositalar (sentiment_classify, retrieve_docs,
summarize) ishlaydi, shuning uchun modul mustaqil ishga tushadi. Keyinchalik
DEFAULT_TOOLS o'rniga oldingi kapstone modullarining metodlarini berish mumkin:

    tools = {
        "sentiment_classify": {"function": FineTunedClassifier().predict, ...},
        "retrieve_docs":       {"function": lambda q: RAGEngine().answer(q)["answer"], ...},
        "summarize_text":      {"function": TransformerSummarizer().summarize, ...},
        ...
    }
    agent = DocumentAssistantAgent(tools=tools)
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Demo vositalar — modul mustaqil ishlashi uchun standart reyestr.
# Ishlab chiqarishda bular kapstone modul metodlari bilan almashtiriladi.
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """O'zbekcha apostroflarni saqlagan holda sodda tokenizatsiya."""
    return re.findall(r"[a-zA-Z0-9_ʻʼ’'-]+", text.lower())


_KNOWLEDGE_BASE = [
    {"title": "NLP", "text": "NLP kompyuterga inson tilidagi matnni tahlil qilishga yordam beradi."},
    {"title": "Transformer", "text": "Transformer attention yordamida tokenlar orasidagi bog'lanishni modellashtiradi."},
    {"title": "RAG", "text": "RAG avval hujjat topadi, keyin savol va kontekstni generatorga yuboradi."},
    {"title": "Agent", "text": "Agent maqsadga qarab vosita tanlaydi, natijani kuzatadi va keyingi qadamni belgilaydi."},
    {"title": "O'zbek NLP", "text": "O'zbek tili agglutinativ bo'lgani uchun qo'shimchalar matn tahlilida muhim."},
]


def _sentiment_classify(text: str) -> dict:
    """Kichik demo sentiment vositasi."""
    tokens = set(_tokenize(text))
    positive_words = {"yaxshi", "ajoyib", "zo'r", "foydali", "qulay"}
    negative_words = {"yomon", "qiyin", "sifatsiz", "muammo", "sekin"}
    positive_count = len(tokens & positive_words)
    negative_count = len(tokens & negative_words)
    if positive_count > negative_count:
        return {"label": "ijobiy", "confidence": 0.82}
    if negative_count > positive_count:
        return {"label": "salbiy", "confidence": 0.79}
    return {"label": "neytral", "confidence": 0.55}


def _retrieve_docs(query: str) -> dict:
    """So'rovga eng ko'p umumiy tokeni bor hujjatni topadi."""
    query_tokens = set(_tokenize(query))
    scored = []
    for document in _KNOWLEDGE_BASE:
        doc_tokens = set(_tokenize(document["text"] + " " + document["title"]))
        scored.append((len(query_tokens & doc_tokens), document))
    best_score, best_document = max(scored, key=lambda item: item[0])
    return {"title": best_document["title"], "text": best_document["text"], "score": best_score}


def _summarize(text: str) -> dict:
    """Birinchi ikki gapni qaytaruvchi ekstraktiv demo vosita."""
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    selected = sentences[:2]
    summary = ". ".join(selected) + ("." if selected else "")
    return {"summary": summary, "sentence_count": len(selected)}


DEFAULT_TOOLS: dict = {
    "sentiment_classify": {
        "function": _sentiment_classify,
        "description": "Matn hissiyotini aniqlaydi.",
        "keywords": ["hissiyot", "ijobiy", "salbiy", "yaxshi", "yomon", "baho"],
    },
    "retrieve_docs": {
        "function": _retrieve_docs,
        "description": "Bilimlar bazasidan mos hujjatni topadi.",
        "keywords": ["top", "qidir", "ma'lumot", "hujjat", "nima", "haqida"],
    },
    "summarize": {
        "function": _summarize,
        "description": "Berilgan matnni qisqartiradi.",
        "keywords": ["xulosa", "qisqa", "qisqartir", "umumlashtir"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ReAct yordamchilari — planner, guardrail, sikl va javob formatlash.
# ─────────────────────────────────────────────────────────────────────────────

def select_tool(query: str, history: list[dict], tools: dict) -> Optional[str]:
    """LLM o'rnidagi qoidaviy rejalashtiruvchi: keyword mosligi bo'yicha tool tanlaydi.

    Ishlatilgan toollar qayta tanlanmaydi. Hech bir moslik topilmasa None.
    """
    q = query.lower()
    used = {record["tool"] for record in history}
    best_tool, best_score = None, 0
    for name, info in tools.items():
        if name in used:
            continue
        score = sum(1 for kw in info["keywords"] if kw in q)
        if score > best_score:
            best_score, best_tool = score, name
    return best_tool if best_score > 0 else None


def execute_action(tool_name: str, query: str, tools: dict) -> dict:
    """Tanlangan vositani xavfsiz tekshiradi va bajaradi.

    Raises:
        ValueError: Agar tool_name reyestrda bo'lmasa (guardrail).
    """
    if tool_name not in tools:
        raise ValueError(
            f"Noma'lum vosita: {tool_name!r}. Mavjud vositalar: {list(tools)}"
        )
    return tools[tool_name]["function"](query)


def format_final_answer(history: list[dict]) -> str:
    """Kuzatuvlarni foydalanuvchiga ko'rinadigan qisqa javobga aylantiradi."""
    if not history:
        return "Mos vosita topilmadi. So'rovni aniqroq yozing."
    parts = []
    for record in history:
        observation = record["observation"]
        parts.append(f"{record['tool']}: {json.dumps(observation, ensure_ascii=False)}")
    return " | ".join(parts)


def run_react(
    query: str,
    tools: dict,
    max_steps: int = 3,
    planner: Callable = select_tool,
) -> list[dict]:
    """Harakat va kuzatuvlardan iborat cheklangan ReAct sikli.

    Har qadamda planner tool tanlaydi; None qaytsa sikl to'xtaydi. Har bir
    bajarilgan qadam {step, tool, input, observation} sifatida saqlanadi.
    """
    history: list[dict] = []
    for step in range(1, max_steps + 1):
        tool_name = planner(query, history, tools)
        if tool_name is None:
            break
        observation = execute_action(tool_name, query, tools)
        history.append({
            "step": step,
            "tool": tool_name,
            "input": query,
            "observation": observation,
        })
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Kapstone klass
# ─────────────────────────────────────────────────────────────────────────────

class DocumentAssistantAgent:
    """ReAct agent — barcha kapstone modullarini asbob sifatida birlashtiradi.

    Tools (ishlab chiqarishda):
        sentiment_classify  → FineTunedClassifier.predict()
        retrieve_docs       → RAGEngine.answer()
        summarize_text      → TransformerSummarizer.summarize()
        spell_correct       → SpellLSHRetriever.correct()
        extract_entities    → NERTagger.entities()

    Klass tool funksiyalarini qayta implementatsiya qilmaydi: u reyestrni qabul
    qiladi va run_react() orqali planner tanlagan vositalarni chaqiradi.
    """

    def __init__(
        self,
        tools: Optional[dict] = None,
        max_steps: int = 3,
        planner: Callable = select_tool,
    ):
        self.tools = tools or DEFAULT_TOOLS
        self.max_steps = max_steps
        self.planner = planner
        self.history: list[dict] = []

    def run(self, user_message: str) -> str:
        """Foydalanuvchi so'rovini ReAct siklidan o'tkazadi va javob qaytaradi.

        Args:
            user_message: Foydalanuvchining o'zbek tilidagi savoli.

        Returns:
            Vositalar kuzatuvlaridan shakllangan o'zbek tilidagi javob matni.
        """
        if not user_message:
            raise ValueError("user_message bo'sh bo'lmasligi kerak.")
        trace = run_react(user_message, self.tools, self.max_steps, self.planner)
        self.history = trace
        return format_final_answer(trace)

    def last_trace(self) -> list[dict]:
        """Oxirgi run() ning Thought→Action→Observation izini nusxa qilib qaytaradi."""
        return list(self.history)

    def reset(self) -> None:
        """Agent tarixini tozalaydi."""
        self.history = []

    def tool_names(self) -> list[str]:
        """Reyestrdagi mavjud vosita nomlarini qaytaradi."""
        return list(self.tools)
