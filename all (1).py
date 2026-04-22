# main.py
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Импортируем диспетчер
from dispatcher import (
    TelegramBotDispatcher, 
    LLMAgentManager, 
    ContextManager, 
    AgentType
)

# Импортируем ваших агентов
from palette_agent import PaletteAgent
from font_agent import FontAgent
from mascot_agent import MascotAgent
from critic_agent import CriticAgent

async def main():
    # Конфигурация
    context_manager = ContextManager()
    
    llm_config = {
        'context_manager': context_manager,
        'enable_auto_metrics': True,
        'metrics_config': {
            'provider_type': 'openai',
            'provider_kwargs': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': 'gpt-3.5-turbo'
            }
        }
    }
    
    # Создаем менеджер
    agent_manager = LLMAgentManager(llm_config)
    
    # Регистрируем ваших агентов
    agent_manager.register_agent(
        AgentType.PALETTE, 
        PaletteAgent({
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': 'gpt-4'
        })
    )
    
    agent_manager.register_agent(
        AgentType.FONT,
        FontAgent({
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': 'gpt-3.5-turbo'
        })
    )
    
    # ... остальные агенты
    
    # Создаем диспетчер
    dispatcher = TelegramBotDispatcher(
        telegram_token=os.getenv('TELEGRAM_TOKEN'),
        llm_manager=agent_manager
    )
    
    try:
        await dispatcher.start_polling()
    except KeyboardInterrupt:
        print("Bot stopped")
    finally:
        await dispatcher.close()

if __name__ == "__main__":
    asyncio.run(main())