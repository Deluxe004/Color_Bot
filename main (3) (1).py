import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List
import aiohttp
from abc import ABC, abstractmethod
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    PALETTE = "palette"
    FONT = "font" 
    MASCOT = "mascot"
    MOCKUP = "mockup"
    CRITIC = "critic"

@dataclass
class UserContext:
    """Контекст пользователя для кэширования"""
    user_id: int
    current_agent: Optional[AgentType] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    agent_specific_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Стандартизированный ответ от LLM агента"""
    success: bool
    content: str
    agent_type: AgentType
    error_message: Optional[str] = None

class BaseAgent(ABC):
    """Базовый класс для всех LLM агентов"""
    
    def __init__(self, agent_type: AgentType, base_url: str, timeout: int = 30):
        self.agent_type = agent_type
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Инициализация aiohttp сессии"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Закрытие сессии"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        """Обработка запроса пользователя"""
        pass
    
    def _prepare_payload(self, user_input: str, context: UserContext) -> Dict[str, Any]:
        """Подготовка payload для LLM API"""
        return {
            "input": user_input,
            "context": {
                "user_id": context.user_id,
                "conversation_history": context.conversation_history[-10:],  # Ограничиваем историю
                "agent_specific_data": context.agent_specific_data
            }
        }


class ContextManager:
    """Менеджер контекста пользователя"""
    
    def __init__(self, ttl: int = 3600):
        self.contexts: Dict[int, UserContext] = {}
        self.ttl = ttl  # Time-to-live в секундах
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def get_context(self, user_id: int) -> UserContext:
        """Получение или создание контекста пользователя"""
        if user_id not in self.contexts:
            self.contexts[user_id] = UserContext(user_id=user_id)
        
        context = self.contexts[user_id]
        context.last_activity = time.time()
        return context
    
    def update_context(self, user_id: int, agent_type: AgentType, user_message: str, agent_response: str):
        """Обновление контекста пользователя"""
        context = self.get_context(user_id)
        context.current_agent = agent_type
        
        # Ограничиваем размер истории
        context.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": agent_response}
        ])
        
        # Держим только последние 20 сообщений (10 пар вопрос-ответ)
        if len(context.conversation_history) > 20:
            context.conversation_history = context.conversation_history[-20:]
    
    def cleanup_expired(self):
        """Очистка устаревших контекстов"""
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
        """Запуск периодической очистки"""
        while True:
            await asyncio.sleep(interval)
            self.cleanup_expired()

class RequestRouter:
    """Маршрутизатор запросов к агентам"""
    
    @staticmethod
    def route_request(text: str) -> AgentType:
        """Простая логика маршрутизации на основе ключевых слов"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['цвет', 'палитр', 'color', 'palette']):
            return AgentType.PALETTE
        elif any(word in text_lower for word in ['шрифт', 'font', 'типографик']):
            return AgentType.FONT
        elif any(word in text_lower for word in ['маскот', 'mascot', 'персонаж']):
            return AgentType.MASCOT
        elif any(word in text_lower for word in ['мокап', 'mockup', 'макет']):
            return AgentType.MOCKUP
        elif any(word in text_lower for word in ['оцен', 'критик', 'review', 'critic']):
            return AgentType.CRITIC
        else:
            return AgentType.CRITIC  # Агент по умолчанию


class LLMAgentManager:
    """Менеджер LLM агентов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.config = config
        self.context_manager = config.get('context_manager', ContextManager())
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def register_agent(self, agent_type: AgentType, agent: BaseAgent):
        """Регистрация агента в менеджере"""
        self.agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type.value}")
    
    async def initialize(self):
        """Инициализация всех зарегистрированных агентов"""
        for agent in self.agents.values():
            await agent.initialize()
        
        # Запускаем периодическую очистку контекстов
        self._cleanup_task = asyncio.create_task(
            self.context_manager.start_periodic_cleanup()
        )
        
        logger.info("All agents initialized")
    
    async def close(self):
        """Закрытие всех агентов"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        for agent in self.agents.values():
            await agent.close()
        logger.info("All agents closed")
    
    async def process_message(self, user_id: int, text: str) -> LLMResponse:
        """Обработка сообщения через соответствующий агент"""
        target_agent = RequestRouter.route_request(text)
        
        if target_agent not in self.agents:
            return LLMResponse(
                success=False,
                content="",
                agent_type=target_agent,
                error_message=f"Agent {target_agent.value} not registered"
            )
        
        context = self.context_manager.get_context(user_id)
        agent = self.agents[target_agent]
        
        try:
            response = await agent.process_request(text, context)
            
            if response.success:
                self.context_manager.update_context(
                    user_id, target_agent, text, response.content
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message with agent {target_agent}: {e}")
            return LLMResponse(
                success=False,
                content="",
                agent_type=target_agent,
                error_message=f"Internal server error: {str(e)}"
            )


class TelegramBotDispatcher:
    """Основной диспетчер Telegram бота"""
    
    def __init__(self, telegram_token: str, llm_manager: LLMAgentManager):
        self.telegram_token = telegram_token
        self.llm_manager = llm_manager
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._polling_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Инициализация диспетчера"""
        self.session = aiohttp.ClientSession()
        await self.llm_manager.initialize()
        self.is_running = True
        logger.info("Telegram Bot Dispatcher initialized")
    
    async def close(self):
        """Завершение работы диспетчера"""
        self.is_running = False
        
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        
        await self.llm_manager.close()
        
        if self.session:
            await self.session.close()
        
        logger.info("Telegram Bot Dispatcher closed")
    
    async def send_telegram_message(self, chat_id: int, text: str):
        """Отправка сообщения в Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to send message: {error_text}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending telegram message: {e}")
            return False
    
    async def process_telegram_update(self, update: Dict[str, Any]):
        """Обработка обновления от Telegram"""
        try:
            message = update.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '').strip()
            user_id = message.get('from', {}).get('id')
            
            if not text or not chat_id or not user_id:
                return
            
            logger.info(f"Processing message from user {user_id}: {text}")
            
            response = await self.llm_manager.process_message(user_id, text)
            
            if response.success:
                reply_text = f"<b>{response.agent_type.value.title()} Agent:</b>\n{response.content}"
            else:
                reply_text = f"❌ Ошибка: {response.error_message}\nПопробуйте еще раз."
            
            await self.send_telegram_message(chat_id, reply_text)
            
        except Exception as e:
            logger.error(f"Error processing telegram update: {e}")
    
    async def start_polling(self):
        """Запуск polling для Telegram updates"""
        if not self.is_running:
            await self.initialize()
        
        offset = 0
        logger.info("Starting Telegram polling...")
        
        while self.is_running:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
                params = {
                    "offset": offset,
                    "timeout": 30,
                    "allowed_updates": ["message"]
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('ok'):
                            updates = data.get('result', [])
                            
                            for update in updates:
                                offset = update['update_id'] + 1

                                # Обрабатываем каждое обновление в отдельной задаче
                                asyncio.create_task(
                                    self.process_telegram_update(update)
                                )
                    else:
                        logger.warning(f"Telegram API returned status {response.status}")
                        await asyncio.sleep(5)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)

# Пример конкретной реализации агента для тестирования
class MockPaletteAgent(BaseAgent):
    """Mock агент для тестирования палитр"""
    
    def __init__(self):
        super().__init__(AgentType.PALETTE, "http://mock-palette-api.com")
    
    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        # Имитация обработки запроса
        await asyncio.sleep(0.1)  # Имитация задержки сети
        
        return LLMResponse(
            success=True,
            content=f"🎨 Подобранная палитра для запроса '{user_input}':\n- Основной: #FF6B6B\n- Акцентный: #4ECDC4\n- Фоновый: #F7FFF7\n- Текст: #292F36",
            agent_type=AgentType.PALETTE
        )

async def main():
    """Пример использования диспетчера"""
    
    # Создаем менеджер контекста
    context_manager = ContextManager()
    
    # Создаем менеджер агентов
    llm_config = {
        'context_manager': context_manager
    }
    agent_manager = LLMAgentManager(llm_config)
    
    # Регистрируем тестового агента
    agent_manager.register_agent(AgentType.PALETTE, MockPaletteAgent())
    
    # Создаем и запускаем диспетчер
    dispatcher = TelegramBotDispatcher(
        telegram_token='AAHg_QpTn6K2_WDkKT5diJqWJpwune2P-MY', 
        llm_manager=agent_manager
    )
    
    try:
        await dispatcher.start_polling()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await dispatcher.close()

if __name__ == "__main__":
    # Для тестирования можно использовать asyncio.run(main())
    # Но в продакшене лучше явно управлять event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped by user")
