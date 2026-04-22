# mascot_generator_agent.py
# Агент для генерации описания маскота. Интегрируется в test_agents_new.py

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

logger = logging.getLogger(__name__)


class AgentType(Enum):
    MASCOT = "mascot"


@dataclass
class LLMResponse:
    success: bool
    content: str
    agenttype: AgentType
    errormessage: Optional[str] = None


class GigaChatMascotAgent:
    """Агент генерации описания маскота для бизнеса"""

    def __init__(self, credentials: str, timeout: int = 30):
        self.credentials = credentials
        self.timeout = timeout
        self.giga = GigaChat(
            credentials=credentials,
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )

    async def process_request(self, user_input: str, context) -> LLMResponse:
        """
        Обрабатывает запрос пользователя и генерирует описание маскота

        Args:
            user_input: Описание бизнеса
            context: Контекст пользователя

        Returns:
            LLMResponse с описанием маскота
        """
        try:
            system_prompt = """Ты креативный дизайнер логотипов и маскотов. 

На основе описания бизнеса создай подробное текстовое описание маскота (персонажа), 
который идеально подойдет этому бизнесу. Описание должно включать:

1. Внешний вид персонажа (форма тела, стиль, пропорции, детали)
2. Цветовую гамму (основные цвета персонажа - укажи HEX коды)
3. Характер и эмоции (какой у персонажа характер, выражение лица)
4. Позу и жесты (как персонаж обычно позирует)
5. Стиль анимации (если бы был анимирован - 2D/3D, реалистичный/мультяшный)
6. Почему этот маскот подходит именно этому бизнесу (2-3 предложения)

Ответь ТОЛЬКО описанием маскота в формате Markdown. Без дополнительных комментариев."""

            payload = Chat(
                messages=Messages(
                    messages=[
                        {
                            "role": MessagesRole.SYSTEM,
                            "content": system_prompt
                        },
                        {
                            "role": MessagesRole.USER,
                            "content": user_input
                        }
                    ]
                )
            )

            response = await asyncio.to_thread(
                self.giga.chat, payload
            )

            content = response.choices[0].message.content

            return LLMResponse(
                success=True,
                content=content,
                agenttype=AgentType.MASCOT
            )

        except Exception as e:
            logger.error(f"GigaChat Mascot error: {e}")
            return LLMResponse(
                success=False,
                content="Не удалось сгенерировать описание маскота",
                agenttype=AgentType.MASCOT,
                errormessage=str(e)
            )



