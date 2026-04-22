#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List

import aiohttp
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from gigachat import GigaChat



# ================== ОБЩАЯ НАСТРОЙКА ==================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    PALETTE = "palette"
    MASCOT = "mascot"
    CRITIC = "critic"
    METRICS = "metrics"


@dataclass
class UserContext:
    user_id: int
    current_agent: Optional[AgentType] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    agent_specific_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMResponse:
    success: bool
    content: str
    agent_type: AgentType
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    def __init__(self, agent_type: AgentType, api_config: Dict[str, Any], timeout: int = 30):
        self.agent_type = agent_type
        self.api_config = api_config
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.api_key: Optional[str] = api_config.get('api_key')
        self.base_url: Optional[str] = api_config.get('base_url')
        self.model: Optional[str] = api_config.get('model')

    async def initialize(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        pass
    # async def _call_llm_api(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
    #     """Универсальный метод вызова LLM API"""
    #     if not self.api_key or not self.base_url:
    #         logger.warning(f"No API configuration for {self.agent_type.value}")
    #         return None
    #
    #     try:
    #         headers = {
    #             "Authorization": f"Bearer {self.api_key}",
    #             "Content-Type": "application/json"
    #         }
    #
    #         payload = {
    #             "model": self.model or "gpt-3.5-turbo",
    #             "messages": messages,
    #             "max_tokens": kwargs.get('max_tokens', 1000),
    #             "temperature": kwargs.get('temperature', 0.7),
    #         }
    #
    #         async with self.session.post(
    #                 f"{self.base_url}/chat/completions",ме
    #                 headers=headers,
    #                 json=payload,
    #                 timeout=self.timeout
    #         ) as response:
    #
    #             if response.status == 200:
    #                 data = await response.json()
    #                 return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    #             else:
    #                 error_text = await response.text()
    #                 logger.error(f"API error ({response.status}): {error_text}")
    #                 return None
    #
    #     except Exception as e:
    #         logger.error(f"Error calling API for {self.agent_type.value}: {e}")
    #         return None

    def _prepare_payload(self, user_input: str, context: UserContext) -> Dict[str, Any]:
        return {
            "input": user_input,
            "context": {
                "user_id": context.user_id,
                "conversation_history": context.conversation_history[-10:],
                "agent_specific_data": context.agent_specific_data
            }
        }



class ContextManager:
    def __init__(self, ttl: int = 3600):
        self.contexts: Dict[int, UserContext] = {}
        self.ttl = ttl
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_context(self, user_id: int) -> UserContext:
        if user_id not in self.contexts:
            self.contexts[user_id] = UserContext(user_id=user_id)
        ctx = self.contexts[user_id]
        ctx.last_activity = time.time()
        return ctx

    def update_context(self, user_id: int, agent_type: AgentType, user_message: str,
                       agent_response: str, metrics: Optional[Dict[str, Any]] = None):

        ctx = self.get_context(user_id)
        ctx.current_agent = agent_type
        ctx.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": agent_response},
        ])
        if len(ctx.conversation_history) > 20:
            ctx.conversation_history = ctx.conversation_history[-20:]
        if metrics:
            ctx.performance_metrics.append({
                "timestamp": time.time(),
                "agent_type": agent_type.value,
                "user_message": user_message,
                "metrics": metrics
            })
            if len(ctx.performance_metrics) > 50:
                ctx.performance_metrics = ctx.performance_metrics[-50:]

    def get_user_metrics_summary(self, user_id: int) -> Dict[str, Any]:
        if user_id not in self.contexts:
            return {"error": "User not found"}

        context = self.contexts[user_id]
        if not context.performance_metrics:
            return {"message": "No metrics available"}

        total_metrics = len(context.performance_metrics)
        avg_scores = {}

        if total_metrics > 0:
            scores_by_type = {}
            for metric in context.performance_metrics:
                agent_type = metric["agent_type"]
                if agent_type not in scores_by_type:
                    scores_by_type[agent_type] = []

                if "final_score" in metric["metrics"]:
                    scores_by_type[agent_type].append(metric["metrics"]["final_score"])

            for agent_type, scores in scores_by_type.items():
                if scores:
                    avg_scores[agent_type] = sum(scores) / len(scores)

        return {
            "total_interactions": total_metrics,
            "average_scores_by_agent": avg_scores,
            "recent_metrics": context.performance_metrics[-5:] if total_metrics > 0 else []
        }


    def cleanup_expired(self):
        current_time = time.time()
        expired_users = [
            user_id for user_id, context in self.contexts.items()
            if current_time - context.last_activity > self.ttl
        ]

        for user_id in expired_users:
            del self.contexts[user_id]

        if expired_users:
            logger.info(f"Cleaned up {len(expired_users)} expired contexts")

    async def start_periodic_cleanup(self, interval: int = 600):
        while True:
            await asyncio.sleep(interval)
            self.cleanup_expired()




class RequestRouter:
    """
    Простая маршрутизация по ключевым словам.
    """

    @staticmethod
    def route_request(text: str) -> AgentType:
        text_lower = text.lower()

        if any(word in text_lower for word in ['метрики', 'оценк', 'качество', 'score', 'metrics', 'анализ']):
            return AgentType.METRICS

        if any(word in text_lower for word in ['цвет', 'палитр', 'color', 'palette']):
            return AgentType.PALETTE
        elif any(word in text_lower for word in ['маскот', 'mascot', 'персонаж']):
            return AgentType.MASCOT
        elif any(word in text_lower for word in ['критик', 'review', 'critic']):
            return AgentType.CRITIC
        else:
            return AgentType.PALETTE


# ================== ОБРАБОТКА ФИДБЭКА ПОЛЬЗОВАТЕЛЯ ==================

def detect_feedback(text: str) -> Optional[str]:
    """
    Пытаемся понять, это фидбэк к существующей палитре/макету
    или обычный новый запрос.
    """
    t = text.lower()
    if any(w in t for w in ["нравится", "классно", "подходит", "окей", "ок"]):
        return "like"
    if any(w in t for w in ["не нравится", "не заходит", "фу", "ужасно", "плохо"]):
        return "dislike"
    if any(w in t for w in ["переделай", "измени", "переработай", "сделай по-другому"]):
        return "redo"
    if any(w in t for w in ["добавь", "добавить", "еще цвет", "ещё цвет", "расширь палитру"]):
        return "add"
    return None


class LLMAgentManager:
    def __init__(self, config):  # Убрать contextmanager из параметров
        self.agents = {}
        self.config = config
        self.context_manager = config.getcontextmanager()
        self.metrics_config = config.getmetricsconfig()
        self._enable_auto_metrics = config.getenableautometrics()
        self._cleanup_task = None


    def register_agent(self, agent_type: AgentType, agent: BaseAgent):
        self.agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type.value}")

    async def initialize(self):
        for agent in self.agents.values():
            await agent.initialize()

        self._cleanup_task = asyncio.create_task(
            self.context_manager.start_periodic_cleanup()

        )

        logger.info("All agents initialized")

    async def close(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for agent in self.agents.values():
            await agent.close()
        logger.info("All agents closed")

    async def _evaluate_response(self, question: str, answer: str,
                                 conversation_history: List[Dict], agent_type: AgentType) -> Optional[Dict[str, Any]]:
        if not self._enable_auto_metrics:
            return None

        from MetricColorBot import create_metrics_agent
        try:

            metrics_provider_type = self.metrics_config.get('provider_type', 'auto')
            metrics_provider_kwargs = self.metrics_config.get('provider_kwargs', {})

            metrics_agent = create_metrics_agent(
                provider_type=metrics_provider_type,
                provider_kwargs=metrics_provider_kwargs
            )

            metrics = metrics_agent.evaluate(
                question=question,
                answer=answer,
                conversation_history=conversation_history,
                run_async=False
            )

            metrics["evaluated_agent"] = agent_type.value

            logger.info(f"Auto-evaluated response from {agent_type.value}: {metrics.get('final_score', 0):.2%}")
            return metrics

        except ImportError:
            logger.warning("Metrics agent not available, skipping evaluation")
            return None
        except Exception as e:
            logger.error(f"Error evaluating metrics: {e}")
            return None

    async def process_message(self, user_id: int, text: str) -> LLMResponse:
        target_agent = RequestRouter.route_request(text)

        if target_agent == AgentType.METRICS:
            return await self._handle_metrics_request(user_id, text)

        if target_agent not in self.agents:
            return LLMResponse(
                success=False,
                content=f"Агент {target_agent.value} не зарегистрирован",
                agent_type=target_agent,
                error_message=f"Agent {target_agent.value} not registered"
            )

        context = self.context_manager.get_context(user_id)
        agent = self.agents[target_agent]

        try:
            pre_request_history = context.conversation_history.copy()

            response = await agent.process_request(text, context)

            if response.success:
                metrics = None
                if self._enable_auto_metrics:
                    metrics = await self._evaluate_response(
                        question=text,
                        answer=response.content,
                        conversation_history=pre_request_history,
                        agent_type=target_agent
                    )
                    response.metrics = metrics

                self.context_manager.update_context(
                    user_id, target_agent, text, response.content, metrics
                )

            return response

        except Exception as e:
            logger.error(f"Error processing message with agent {target_agent}: {e}")
            return LLMResponse(
                success=False,
                content="Произошла ошибка при обработке запроса",
                agent_type=target_agent,
                error_message=f"Internal server error: {str(e)}"
            )

    async def _handle_metrics_request(self, user_id: int, text: str) -> LLMResponse:
        try:
            context = self.context_manager.get_context(user_id)
            text_lower = text.lower()

            if 'сводк' in text_lower or 'summary' in text_lower:
                summary = self.context_manager.get_user_metrics_summary(user_id)

                if "error" in summary:
                    content = "❌ У вас пока нет истории для анализа метрик."
                else:
                    content = "📊 **Сводка ваших метрик:**\n\n"

                    content += f"📈 Всего взаимодействий: {summary['total_interactions']}\n\n"

                    if summary['average_scores_by_agent']:
                        content += "🎯 Средние оценки по агентам:\n"
                        for agent_type, score in summary['average_scores_by_agent'].items():
                            content += f"  • {agent_type.title()}: {score:.1%}\n"

                    if summary.get('recent_metrics'):
                        content += "\n📋 Последние оценки:\n"
                        for metric in summary['recent_metrics'][-3:]:
                            final_score = metric['metrics'].get('final_score', 0)
                            agent_type = metric['agent_type']
                            content += f"  • {agent_type}: {final_score:.1%}\n"

            elif 'оцен' in text_lower or 'eval' in text_lower:
                if len(context.conversation_history) < 2:
                    content = "❌ Недостаточно данных для оценки. Сначала пообщайтесь с ботом."
                else:
                    last_question = None
                    last_answer = None

                    for i in range(len(context.conversation_history) - 1, 0, -1):
                        if (context.conversation_history[i]['role'] == 'assistant' and
                                context.conversation_history[i-1]['role'] == 'user'):
                            last_answer = context.conversation_history[i]['content']
                            last_question = context.conversation_history[i-1]['content']
                            break

                    if last_question and last_answer:
                        try:
                            from metrics_code import create_metrics_agent

                            metrics_agent = create_metrics_agent(
                                provider_type=self.metrics_config.get('provider_type', 'auto'),
                                provider_kwargs=self.metrics_config.get('provider_kwargs', {})
                            )

                            metrics = metrics_agent.evaluate(
                                question=last_question,
                                answer=last_answer,
                                conversation_history=context.conversation_history[:-2]
                            )

                            content = f"📊 **Оценка последнего ответа:**\n\n"
                            content += f"❓ Вопрос: {last_question[:100]}...\n\n"
                            content += f"🎯 **Итоговая оценка: {metrics.get('final_score', 0):.1%}**\n\n"
                            content += f"📈 Детали:\n"

                            if 'relevancy' in metrics:
                                rel = metrics['relevancy']
                                status = "✅" if rel.get('passed', False) else "❌"
                                content += f"• Релевантность: {rel.get('score', 0):.1%} {status}\n"

                            if 'conciseness' in metrics:
                                conc = metrics['conciseness']
                                status = "✅" if conc.get('passed', False) else "❌"
                                content += f"• Лаконичность: {conc.get('score', 0):.1%} {status}\n"

                            if 'token_usage' in metrics:
                                tokens = metrics['token_usage']
                                content += f"• Использовано токенов: {tokens.get('total_tokens', 0)}\n"

                            content += f"\n📋 **Вердикт:** {metrics.get('assessment', 'Не оценено')}"
                        except ImportError:
                            content = "❌ Система метрик недоступна"
                        except Exception as e:
                            content = f"❌ Ошибка оценки: {str(e)}"
                    else:
                        content = "❌ Не удалось найти последний вопрос-ответ для оценки."

            else:
                content = "📊 **Система метрик качества**\n\n"
                content += "Доступные команды:\n"
                content += "• 'метрики сводка' - общая статистика ваших взаимодействий\n"
                content += "• 'оценить ответ' - оценка последнего ответа бота\n\n"
                content += "📈 Метрики включают:\n"
                content += "• Релевантность ответа\n• Лаконичность\n• Эффективность использования токенов"

            return LLMResponse(
                success=True,
                content=content,
                agent_type=AgentType.METRICS
            )

        except Exception as e:
            logger.error(f"Error handling metrics request: {e}")
            return LLMResponse(
                success=False,
                content="Произошла ошибка при обработке запроса метрик",
                agent_type=AgentType.METRICS,
                error_message=f"Error: {str(e)}"
            )



# ================== ЛОКАЛЬНЫЙ PALETTE AGENT (из PalletAgent.py, сокращённо) ==================


import json
import re
import csv

from typing import Tuple

HEX_RE = re.compile(r"^#([0-9A-Fa-f]{6})$")

DEFAULT_CATALOG_PATHS = [
    "./словарьЦветов - Catalog_Public (1).tsv",
    "./Catalog_Public.tsv",
    "./data/Catalog_Public.tsv",
]


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    m = HEX_RE.match(hex_str.strip())
    if not m:
        raise ValueError(f"Bad HEX: {hex_str}")
    v = m.group(1)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    r = min(255, max(0, int(round(r))))
    g = min(255, max(0, int(round(g))))
    b = min(255, max(0, int(round(b))))
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_hsl(r: int, g: int, b: int) -> Tuple[float, float, float]:
    rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
    cmax, cmin = max(rp, gp, bp), min(rp, gp, bp)
    d = cmax - cmin
    l = (cmax + cmin) / 2.0
    if d == 0:
        h = 0.0
        s = 0.0
    else:
        s = d / (1 - abs(2 * l - 1))
        if cmax == rp:
            h = 60 * (((gp - bp) / d) % 6)
        elif cmax == gp:
            h = 60 * (((bp - rp) / d) + 2)
        else:
            h = 60 * (((rp - gp) / d) + 4)
    return h, s, l


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = l - c / 2
    if 0 <= h < 60:
        rp, gp, bp = c, x, 0
    elif 60 <= h < 120:
        rp, gp, bp = x, c, 0
    elif 120 <= h < 180:
        rp, gp, bp = 0, c, x
    elif 180 <= h < 240:
        rp, gp, bp = 0, x, c
    elif 240 <= h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return (
        int(round((rp + m) * 255)),
        int(round((gp + m) * 255)),
        int(round((bp + m) * 255)),
    )


def relative_luminance(hex_str: str) -> float:
    def _lin(u):
        u = u / 255.0
        return u / 12.92 if u <= 0.04045 * 255 else ((u + 0.055) / 1.055) ** 2.4

    r, g, b = hex_to_rgb(hex_str)
    R, G, B = _lin(r), _lin(g), _lin(b)
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    L1 = relative_luminance(fg_hex)
    L2 = relative_luminance(bg_hex)
    Lmax, Lmin = (L1, L2) if L1 >= L2 else (L2, L1)
    return (Lmax + 0.05) / (Lmin + 0.05)


def tweak_lightness(hex_str: str, delta_l: float) -> str:
    r, g, b = hex_to_rgb(hex_str)
    h, s, l = rgb_to_hsl(r, g, b)
    l = min(1.0, max(0.0, l + delta_l))
    return rgb_to_hex(*hsl_to_rgb(h, s, l))


from dataclasses import dataclass as _dataclass2


@_dataclass2
class CatalogRow:
    hex: str
    name: str
    tags: str


def load_catalog(tsv_paths: List[str] = DEFAULT_CATALOG_PATHS) -> List[CatalogRow]:
    for p in tsv_paths:
        if os.path.isfile(p):
            path = p
            break
    else:
        raise FileNotFoundError(
            "Не найден TSV-словарь цветов. Проверьте путь в DEFAULT_CATALOG_PATHS."
        )
    rows: List[CatalogRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hex_val = (
                    row.get("hex")
                    or row.get("HEX")
                    or row.get("Hex")
                    or row.get("ColorHEX")
                    or row.get("#")
                    or ""
            ).strip()
            name_val = (
                    row.get("name")
                    or row.get("Name")
                    or row.get("ru_name")
                    or row.get("title")
                    or ""
            ).strip()
            if not HEX_RE.match(hex_val):
                digits = re.sub(r"[^0-9A-Fa-f]", "", hex_val)
                if len(digits) == 6:
                    hex_val = f"#{digits.upper()}"
            if not HEX_RE.match(hex_val):
                continue
            text_blobs = []
            for k, v in row.items():
                if v and isinstance(v, str):
                    if k.lower() in ("hex", "#"):
                        continue
                    text_blobs.append(v)
            tags = " ".join(text_blobs).lower()
            rows.append(
                CatalogRow(hex=hex_val.upper(), name=name_val or hex_val, tags=tags)
            )
    if not rows:
        raise ValueError("Каталог загружен, но пуст.")
    return rows


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9#\-]+")

PSYCHO_MAP = {
    "кофе": ["коричневый", "бежевый", "сливочный", "молочный", "чёрный"],
    "доверие": ["синий", "серый", "голубой"],
}


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def score_row_by_query(row: CatalogRow, tokens: List[str]) -> float:
    base = 0.0
    for t in tokens:
        if f" {t} " in (" " + row.tags + " "):
            base += 1.0
    psycho_targets = set()
    for t in tokens:
        if t in PSYCHO_MAP:
            psycho_targets.update(PSYCHO_MAP[t])
    for kw in psycho_targets:
        if kw in row.tags:
            base += 0.5
    return base


def hue_of(hex_str: str) -> float:
    r, g, b = hex_to_rgb(hex_str)
    h, s, l = rgb_to_hsl(r, g, b)
    return h


def nearest_by_hue(candidates: List[CatalogRow], target_h: float, k: int = 1) -> List[CatalogRow]:
    scored = []
    for r in candidates:
        dh = abs((hue_of(r.hex) - target_h + 180) % 360 - 180)
        scored.append((dh, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def pick_scheme(seed: CatalogRow, pool: List[CatalogRow], scheme: str, n: int) -> List[CatalogRow]:
    scheme = (scheme or "auto").lower()
    H = hue_of(seed.hex)
    if scheme in ("mono", "monochrome", "monochromatic"):
        targets = [H + i * 6 for i in range(6)]
    elif scheme in ("analog", "analogous"):
        targets = [H - 30, H - 15, H, H + 15, H + 30]
    elif scheme in ("comp", "complementary"):
        targets = [H, (H + 180) % 360, H + 15, (H + 195) % 360, (H - 15) % 360]
    elif scheme in ("triad", "triadic"):
        targets = [H, (H + 120) % 360, (H + 240) % 360]
    else:
        targets = [H, (H + 20) % 360, (H + 200) % 360]
    result: List[CatalogRow] = [seed]
    used = {seed.hex}
    for tgt in targets:
        if len(result) >= n:
            break
        picked = nearest_by_hue(pool, tgt, k=3)
        for cand in picked:
            if cand.hex not in used:
                result.append(cand)
                used.add(cand.hex)
                if len(result) >= n:
                    break
    if len(result) < n:
        extra = [r for r in pool if r.hex not in used]
        result.extend(extra[: (n - len(result))])
    return result[:n]


def ensure_accessibility(palette_hex: List[str]):
    notes = []
    roles_default = ["primary", "surface", "text", "background", "accent", "accent2"]
    role_map = {
        roles_default[i]: palette_hex[i]
        for i in range(min(len(palette_hex), len(roles_default)))
    }
    fg = role_map.get("text", palette_hex[0])
    bg = role_map.get("background", palette_hex[-1])
    cr = contrast_ratio(fg, bg)
    report = {
        "text_on_bg": {
            "fg": fg,
            "bg": bg,
            "contrast": round(cr, 2),
            "AA_4.5": cr >= 4.5,
            "AAA_7.0": cr >= 7.0,
        }
    }
    adjusted = palette_hex[:]
    if cr < 4.5:
        better = fg
        for delta in (-0.20, -0.12, -0.08, 0.08, 0.12, 0.20):
            cand = tweak_lightness(fg, delta)
            if contrast_ratio(cand, bg) >= 4.5:
                better = cand
                break
        if better != fg and fg in adjusted:
            idx = adjusted.index(fg)
            adjusted[idx] = better
            report["text_on_bg"]["adjusted_fg"] = better
            report["text_on_bg"]["contrast_after"] = round(
                contrast_ratio(better, bg), 2
            )
            notes.append("Скорректирована светлота текста до AA.")
    return report, adjusted, notes


def assign_roles(hex_list: List[str]) -> List[Dict[str, str]]:
    roles = ["primary", "surface", "text", "background", "accent", "accent2"]
    out = []
    for i, h in enumerate(hex_list):
        role = roles[i] if i < len(roles) else "extra"
        out.append({"hex": h, "name": h, "role": role})
    return out


def build_palette_local(query: str, n_colors: int = 4,
                        scheme: Optional[str] = None,
                        need_roles: bool = True) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"ok": False, "agent": "palette_local", "error": "Пустой запрос"}
    catalog = load_catalog()
    tokens = tokenize(query)
    scored = [(score_row_by_query(r, tokens), r) for r in catalog]
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        seed = catalog[0]
        pool = catalog[1:]
    else:
        seed = scored[0][1]
        pool = [r for _, r in scored[1:]]
    n = max(2, min(6, int(n_colors or 4)))
    picked = pick_scheme(seed, pool, scheme or "auto", n)
    base_hex = [r.hex for r in picked]
    report, adjusted_hex, notes = ensure_accessibility(base_hex)
    if need_roles:
        items = assign_roles(adjusted_hex)
        hex2name = {r.hex: r.name for r in picked}
        for it in items:
            it["name"] = hex2name.get(it["hex"], it["hex"])
    else:
        items = [{"hex": h, "name": h} for h in adjusted_hex]
    explain_short = f"Seed «{seed.name or seed.hex}», scheme={scheme or 'auto'}, WCAG check."
    feedback_hints = notes or []
    return {
        "ok": True,
        "agent": "palette_local",
        "palette": items,
        "contrast_report": report,
        "explain_short": explain_short,
        "feedback_hints": feedback_hints,
    }


# ================== GIGACHAT АГЕНТЫ ==================

def make_gigachat_client() -> GigaChat:
    cred = os.getenv("GIGACHAT_TOKEN")
    if not cred:
        raise ValueError("GIGACHAT_TOKEN не найден в переменных окружения")
    return GigaChat(
        credentials=cred,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,  # для тестов
    )


# ================== GIGACHAT АГЕНТЫ + TSV HYBRID PALETTE AGENT ==================

class HybridPaletteAgent(BaseAgent):
    """
    Гибридный агент палитры с нейросетью и TSV-словарем.
    Сочетает семантическое понимание GigaChat с точностью цветов из каталога.
    """

    def __init__(self, tsv_paths: List[str] = DEFAULT_CATALOG_PATHS):
        self.giga = make_gigachat_client()
        self.catalog = load_catalog(tsv_paths)
        super().__init__(AgentType.PALETTE, {"api_key":"", "base_url":"", "model":""})
        logger.info(f"HybridPaletteAgent инициализирован с {len(self.catalog)} цветами")

    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        try:
            # 1. Анализ запроса LLM
            analysis = await self._analyze_with_llm(user_input)

            # 2. Поиск цветов в TSV
            color_matches = self._find_colors_in_catalog(
                analysis.get("suggested_colors", []),
                analysis.get("emotions", [])
            )

            # 3. Построение палитры
            palette = self._build_palette(
                color_matches,
                analysis.get("color_scheme", "аналогичная"),
                6
            )

            # 4. Формирование JSON ответа
            result = {
                "success": True,
                "agent": "hybrid_palette",
                "query_analysis": analysis,
                "palette": palette,
                "metadata": {
                    "colors_from_tsv": len(color_matches) > 0,
                    "total_colors_found": len(color_matches)
                }
            }
            context.agent_specific_data["last_palette"] = result
            return LLMResponse(
                success=True,
                content=json.dumps(result, ensure_ascii=False, indent=2),
                agent_type=self.agent_type
            )

        except Exception as e:
            logger.error(f"HybridPaletteAgent error: {e}")
            return LLMResponse(
                success=False,
                content=json.dumps({
                    "error": str(e),
                    "fallback_suggestion": "Попробуйте более конкретный запрос"
                }),
                agent_type=self.agent_type,
                error_message=str(e)
            )

    async def _analyze_with_llm(self, user_input: str) -> dict:
        """Анализ запроса нейросетью"""
        prompt = f"""Проанализируй запрос о цветовой палитре: "{user_input}"

        Верни JSON с полями:
        - theme: тема/сфера (кофейня, IT, медицина и т.д.)
        - emotions: ключевые эмоции/настроения
        - style: стиль (современный, классический и т.д.)
        - suggested_colors: предполагаемые названия цветов на русском
        - color_scheme: подходящая схема (монохромная, аналогичная и т.д.)"""

        resp = await asyncio.to_thread(
            self.giga.chat,
            {"messages": [
                {"role": "system", "content": "Ты аналитик цветовых предпочтений"},
                {"role": "user", "content": prompt}
            ]}
        )

        import json, re
        text = resp.choices[0].message.content
        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:
            return json.loads(match.group())
        else:
            return {
                "theme": "общий",
                "emotions": ["нейтральный"],
                "style": "современный",
                "suggested_colors": ["синий", "серый", "белый"],
                "color_scheme": "аналогичная"
            }

    def _find_colors_in_catalog(self, color_names: list, emotions: list) -> list:
        """Поиск цветов в TSV по названиям и эмоциям"""
        matches = []

        for color_name in color_names:
            color_lower = color_name.lower()
            for row in self.catalog:
                if (color_lower in row.name.lower() or
                        color_lower in row.tags):
                    matches.append(row)

        # Поиск по психологическим ассоциациям
        for emotion in emotions:
            if emotion in PSYCHO_MAP:
                for color in PSYCHO_MAP[emotion]:
                    for row in self.catalog:
                        if color in row.tags:
                            matches.append(row)

        return matches[:10]  # Ограничиваем

    def _build_palette(self, colors: list, scheme: str, n_colors: int) -> list:
        """Построение гармоничной палитры"""
        if not colors:
            colors = self.catalog[:10]

        seed = colors[0]

        # Конвертация схемы
        scheme_map = {
            "монохромная": "mono",
            "аналогичная": "analogous",
            "комплементарная": "complementary",
            "триадная": "triad"
        }
        scheme_key = scheme_map.get(scheme.lower(), "auto")

        # Используем существующую функцию pick_scheme
        palette_rows = pick_scheme(seed, colors[1:], scheme_key, n_colors)

        # Проверяем доступность
        hex_list = [r.hex for r in palette_rows]
        report, adjusted_hex, notes = ensure_accessibility(hex_list)

        # Форматируем результат
        roles = ["primary", "surface", "text", "background", "accent", "accent2"]
        result = []

        for i, hex_val in enumerate(adjusted_hex):
            # Находим имя цвета
            color_name = next((r.name for r in palette_rows if r.hex == hex_val), hex_val)

            result.append({
                "hex": hex_val,
                "name": color_name,
                "role": roles[i] if i < len(roles) else "extra"
            })

        return result


class GigaChatCriticAgent(BaseAgent):
    def __init__(self):
        self.giga = make_gigachat_client()
        super().__init__(AgentType.CRITIC, {"api_key":"", "base_url":"", "model":""})

    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        try:
            system_prompt = (
                "Ты экспертный критик дизайна и UX. "
                "Анализируй палитру/макет/шрифты и давай конкретную конструктивную обратную связь: "
                "сильные стороны и что улучшить. Пиши по-русски, дружелюбно, но по делу. "
                "Если пользователь просит что-то изменить, в конце ответа выдели отдельной строкой "
                "краткое ТЗ для генератора палитры, начиная её с префикса 'TECH_SPEC: '."
            )
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ]
            }
            resp = await asyncio.to_thread(self.giga.chat, payload)
            content = resp.choices[0].message.content
            return LLMResponse(success=True, content=content, agent_type=self.agent_type)
        except Exception as e:
            logger.error(f"GigaChat Critic error: {e}")
            return LLMResponse(
                success=False,
                content="Не удалось проанализировать.",
                agent_type=self.agent_type,
                error_message=str(e),
            )


class GigaChatMascotAgent(BaseAgent):
    def __init__(self):
        self.giga = make_gigachat_client()
        super().__init__(AgentType.MASCOT, {"api_key":"", "base_url":"", "model":""})

    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        try:
            system_prompt = (
                "Ты креативный дизайнер логотипов и маскотов.\n"
                "На основе описания бизнеса создай подробное текстовое описание маскота (персонажа), "
                "который идеально подойдёт этому бизнесу. Укажи:\n"
                "1) Внешний вид, 2) Цветовую гамму с HEX, 3) Характер, 4) Позу, 5) Стиль анимации, "
                "6) Почему маскот подходит этому бизнесу. Ответь только описанием в Markdown."
            )
            messages = [{"role": "system", "content": system_prompt}]
            for msg in context.conversation_history[-10:]:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})
            payload = {"messages": messages}

            resp = await asyncio.to_thread(self.giga.chat, payload)
            content = resp.choices[0].message.content

            return LLMResponse(success=True, content=content, agent_type=self.agent_type)
        except Exception as e:
            logger.error(f"GigaChat Mascot error: {e}")
            return LLMResponse(
                success=False,
                content="Не удалось сгенерировать маскота.",
                agent_type=self.agent_type,
                error_message=str(e),
            )


# ================== КОНСОЛЬНЫЙ ДИСПЕТЧЕР ДЛЯ ТЕСТА ==================

async def main_console():
    if not os.getenv('GIGACHAT_TOKEN'):
        print("GIGACHAT_TOKEN не установлен в .env")
        return

    # ← ДОБАВИТЕ ЭТО (строки ~1230-1245):
    from dataclasses import dataclass, field

    @dataclass
    class Config:
        contextmanager: ContextManager = field(default_factory=lambda: ContextManager(ttl=3600))
        metricsconfig: dict = field(default_factory=lambda: {"provider_type": "auto"})
        enableautometrics: bool = False

        def getcontextmanager(self):
            return self.contextmanager

        def getmetricsconfig(self):
            return self.metricsconfig

        def getenableautometrics(self):
            return self.enableautometrics

    config = Config()
    manager = LLMAgentManager(config=config)
    try:
        from fixed_hybrid_agent import FixedHybridPaletteAgent
        manager.register_agent(AgentType.PALETTE, FixedHybridPaletteAgent())
    except ImportError as e:
        print(f"Не удалось загрузить FixedHybridPaletteAgent: {e}")
        print("Используем упрощённую версию")
        manager.register_agent(AgentType.PALETTE, HybridPaletteAgent())
    manager.register_agent(AgentType.CRITIC, GigaChatCriticAgent())
    manager.register_agent(AgentType.MASCOT, GigaChatMascotAgent())

    await manager.initialize()

    print("Консольный тестер. Ключевые слова для маршрутизации:")
    # print("- 'tsv', 'каталог' → локальная палитра по TSV")
    print("- 'палитра', 'цвет' → гибридная палитра (GigaChat + TSV-словарь)")
    print("- 'критик', 'оценка' → критик через GigaChat")
    print("- 'маскот', 'персонаж' → маскот через GigaChat")
    print("Пиши 'exit' для выхода.\n")

    try:
        while True:
            text = input("Ты: ").strip()
            if text.lower() in {"exit", "quit", "выход"}:
                break
            resp = await manager.process_message(user_id=1, text=text)
            if resp.success:
                print(f"\n[{resp.agent_type.value}] Ответ:\n{resp.content}\n")
            else:
                print(f"\nОшибка ({resp.agent_type.value}): {resp.error_message or resp.content}\n")
    finally:
        await manager.close()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_console())

# для экспорта:
__all__ = [
    'BaseAgent', 'AgentType', 'UserContext', 'LLMResponse',
    'ContextManager', 'RequestRouter', 'LLMAgentManager',
    'CatalogRow', 'load_catalog', 'pick_scheme', 'ensure_accessibility',
    'DEFAULT_CATALOG_PATHS', 'logger',
    'GigaChatCriticAgent', 'GigaChatMascotAgent'
]