#%% Импорт необходимых библиотек
import os
import json
import re
from typing import Dict, List, Any, Optional, Union, Callable
from pydantic import BaseModel, Field
import tiktoken
import asyncio
from abc import ABC, abstractmethod

#%% Базовый класс для LLM провайдеров (абстрактный интерфейс)
class BaseLLMProvider(ABC):
    """Абстрактный базовый класс для всех LLM провайдеров"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Синхронная генерация текста"""
        pass
    
    @abstractmethod
    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        pass
    
    def get_model_name(self) -> str:
        """Получение имени модели"""
        return self.__class__.__name__

#%% OpenAI-совместимые провайдеры
class OpenAIProvider(BaseLLMProvider):
    """Провайдер для OpenAI API (и совместимых)"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 **kwargs):
        
        try:
            from openai import OpenAI, AsyncOpenAI
            
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL', "https://api.openai.com/v1")
            self.model = model
            self.kwargs = kwargs
            
            # Создаем клиентов
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Инициализируем токенизатор
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self._tokenizer = None
                
        except ImportError:
            raise ImportError("Для использования OpenAIProvider установите: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Синхронная генерация"""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка OpenAI API: {e}")
            return ""
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация"""
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка OpenAI API (async): {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов"""
        if self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        return len(text.split()) * 4 // 3  # Примерное приближение
    
    def get_model_name(self) -> str:
        return f"OpenAI ({self.model})"

#%% Mistral провайдер
class MistralProvider(BaseLLMProvider):
    """Провайдер для Mistral AI"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "mistral-small-latest",
                 **kwargs):
        
        try:
            from mistralai import Mistral, AsyncMistral
            
            self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
            self.model = model
            self.kwargs = kwargs
            
            self._client = Mistral(api_key=self.api_key)
            self._async_client = AsyncMistral(api_key=self.api_key)
            
            # Токенизатор
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self._tokenizer = None
                
        except ImportError:
            raise ImportError("Для MistralProvider установите: pip install mistralai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка Mistral API: {e}")
            return ""
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        try:
            response = await self._async_client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка Mistral API (async): {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        if self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        return len(text.split()) * 4 // 3
    
    def get_model_name(self) -> str:
        return f"Mistral ({self.model})"

#%% Anthropic (Claude) провайдер
class AnthropicProvider(BaseLLMProvider):
    """Провайдер для Anthropic Claude API"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-3-haiku-20240307",
                 **kwargs):
        
        try:
            import anthropic
            
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            self.model = model
            self.kwargs = kwargs
            
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self._tokenizer = None
                
        except ImportError:
            raise ImportError("Для AnthropicProvider установите: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.content[0].text
        except Exception as e:
            print(f"Ошибка Anthropic API: {e}")
            return ""
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        try:
            response = await self._async_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.content[0].text
        except Exception as e:
            print(f"Ошибка Anthropic API (async): {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        if self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        return len(text.split()) * 4 // 3
    
    def get_model_name(self) -> str:
        return f"Anthropic ({self.model})"

#%% Google Gemini провайдер
class GeminiProvider(BaseLLMProvider):
    """Провайдер для Google Gemini API"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gemini-pro",
                 **kwargs):
        
        try:
            import google.generativeai as genai
            
            self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
            self.model = model
            self.kwargs = kwargs
            
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(model)
            
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self._tokenizer = None
                
        except ImportError:
            raise ImportError("Для GeminiProvider установите: pip install google-generativeai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                **{**self.kwargs, **kwargs}
            )
            return response.text
        except Exception as e:
            print(f"Ошибка Gemini API: {e}")
            return ""
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        # Gemini не имеет официального async API, используем sync в thread pool
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        func = partial(self.generate, prompt, **kwargs)
        return await loop.run_in_executor(None, func)
    
    def count_tokens(self, text: str) -> int:
        if self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        return len(text.split()) * 4 // 3
    
    def get_model_name(self) -> str:
        return f"Google Gemini ({self.model})"

#%% Локальные модели через LiteLLM
class LiteLLMProvider(BaseLLMProvider):
    """Универсальный провайдер через LiteLLM (поддерживает множество API и локальных моделей)"""
    
    def __init__(self,
                 model: str,
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 **kwargs):
        
        try:
            import litellm
            
            self.model = model
            self.api_base = api_base
            self.api_key = api_key
            self.kwargs = kwargs
            
            # Настраиваем LiteLLM
            if api_base:
                litellm.api_base = api_base
            if api_key:
                litellm.api_key = api_key
            
            self._litellm = litellm
            
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self._tokenizer = None
                
        except ImportError:
            raise ImportError("Для LiteLLMProvider установите: pip install litellm")
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self._litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка LiteLLM API: {e}")
            return ""
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        try:
            response = await self._litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **{**self.kwargs, **kwargs}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Ошибка LiteLLM API (async): {e}")
            return ""
    
    def count_tokens(self, text: str) -> int:
        if self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        return len(text.split()) * 4 // 3
    
    def get_model_name(self) -> str:
        return f"LiteLLM ({self.model})"

#%% Тестовый провайдер (без API)
class MockProvider(BaseLLMProvider):
    """Мок-провайдер для тестирования без реальных API"""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {
            "relevancy": "0.85",
            "correctness": "0.90",
            "default": "0.75"
        }
        self._tokenizer = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Логика для возврата тестовых оценок
        prompt_lower = prompt.lower()
        
        if "релевантность" in prompt_lower or "relevancy" in prompt_lower:
            return self.responses.get("relevancy", "0.85")
        elif "сравните" in prompt_lower or "correctness" in prompt_lower:
            return self.responses.get("correctness", "0.90")
        else:
            return self.responses.get("default", "0.75")
    
    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        return len(text.split()) * 4 // 3
    
    def get_model_name(self) -> str:
        return "Mock Provider (для тестирования)"

#%% Фабрика провайдеров
class LLMProviderFactory:
    """Фабрика для создания провайдеров LLM"""
    
    @staticmethod
    def create_provider(provider_type: str = "auto", **kwargs) -> BaseLLMProvider:
        """
        Создание провайдера LLM
        
        Args:
            provider_type: Тип провайдера:
                - "openai": OpenAI API
                - "mistral": Mistral AI
                - "anthropic": Anthropic Claude
                - "gemini": Google Gemini
                - "litellm": LiteLLM (универсальный)
                - "mock": Мок-провайдер для тестов
                - "auto": Автоматическое определение по переменным окружения
            **kwargs: Параметры для конкретного провайдера
        
        Returns:
            Экземпляр BaseLLMProvider
        """
        
        if provider_type == "auto":
            # Автоматическое определение по переменным окружения
            if os.getenv('OPENAI_API_KEY'):
                return LLMProviderFactory.create_provider("openai", **kwargs)
            elif os.getenv('MISTRAL_API_KEY'):
                return LLMProviderFactory.create_provider("mistral", **kwargs)
            elif os.getenv('ANTHROPIC_API_KEY'):
                return LLMProviderFactory.create_provider("anthropic", **kwargs)
            elif os.getenv('GOOGLE_API_KEY'):
                return LLMProviderFactory.create_provider("gemini", **kwargs)
            else:
                print("⚠️  API ключи не найдены, использую MockProvider")
                return LLMProviderFactory.create_provider("mock", **kwargs)
        
        elif provider_type == "openai":
            return OpenAIProvider(**kwargs)
        
        elif provider_type == "mistral":
            return MistralProvider(**kwargs)
        
        elif provider_type == "anthropic":
            return AnthropicProvider(**kwargs)
        
        elif provider_type == "gemini":
            return GeminiProvider(**kwargs)
        
        elif provider_type == "litellm":
            return LiteLLMProvider(**kwargs)
        
        elif provider_type == "mock":
            return MockProvider(**kwargs)
        
        else:
            raise ValueError(f"Неизвестный тип провайдера: {provider_type}")

#%% Основной класс агента метрик (УНИВЕРСАЛЬНЫЙ)
class UniversalMetricsAgent(BaseModel):
    """
    Универсальный агент для оценки качества RAG-бота
    Поддерживает любые LLM API через провайдеры
    """
    
    class Config:
        arbitrary_types_allowed = True
    
    # Провайдер LLM
    provider: BaseLLMProvider = Field(default=None)
    
    # Параметры оценки
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "relevancy": 0.7,
        "efficiency": 0.5,
        "clarifications": 0.3
    })
    
    # Конфигурация промптов
    evaluation_prompts: Dict[str, str] = Field(default_factory=lambda: {
        "relevancy": """
        Оцените релевантность ответа на вопрос пользователя по шкале от 0.0 до 1.0.
        
        Вопрос: {question}
        Ответ: {answer}
        {context_line}
        
        Критерии:
        1.0 - Ответ идеально соответствует вопросу, полностью раскрывает тему
        0.8 - Ответ хороший, но можно было бы полнее
        0.6 - Ответ частично релевантен, есть упущения
        0.4 - Ответ слабо связан с вопросом
        0.2 - Ответ почти не релевантен
        0.0 - Ответ совершенно не по теме
        
        Верните ТОЛЬКО число с плавающей точкой, без пояснений.
        """,
        
        "correctness": """
        Сравните два ответа на вопрос и оцените их фактологическую схожесть от 0.0 до 1.0.
        
        Вопрос: {question}
        Ответ 1 (предоставленный): {answer}
        Ответ 2 (ожидаемый): {expected}
        
        Критерии:
        1.0 - Факты полностью совпадают
        0.8 - Основные факты совпадают, есть мелкие различия
        0.6 - Факты частично совпадают
        0.4 - Минимальное совпадение фактов
        0.2 - Факты практически не совпадают
        0.0 - Факты совершенно разные
        
        Верните ТОЛЬКО число с плавающей точкой.
        """,
        
        "conciseness": """
        Оцените лаконичность и информативность ответа от 0.0 до 1.0.
        
        Вопрос: {question}
        Ответ: {answer}
        Количество токенов в ответе: {tokens}
        
        Критерии:
        1.0 - Идеальный баланс краткости и информативности
        0.8 - Хорошая информативность, можно было бы короче
        0.6 - Приемлемо, но есть избыточность или недостаток информации
        0.4 - Слишком многословно или слишком кратко
        0.2 - Очень плохой баланс
        0.0 - Совершенно не соответствует
        
        Верните ТОЛЬКО число.
        """
    })
    
    def __init__(self, provider: Optional[BaseLLMProvider] = None, **data):
        super().__init__(**data)
        
        if not self.provider:
            if provider:
                self.provider = provider
            else:
                # Автоматическое создание провайдера
                self.provider = LLMProviderFactory.create_provider("auto")
        
        # Инициализируем токенизатор
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Инициализация токенизатора"""
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self._tokenizer = None
            print("⚠️  Токенизатор не найден, использую приблизительный подсчет токенов")
    
    def count_tokens(self, text: str) -> int:
        """Подсчет токенов с использованием провайдера или локального токенизатора"""
        if hasattr(self.provider, 'count_tokens'):
            return self.provider.count_tokens(text)
        elif self._tokenizer and text:
            return len(self._tokenizer.encode(text))
        else:
            # Приблизительный подсчет
            return len(text.split()) * 4 // 3
    
    def _extract_score(self, response: str) -> float:
        """Извлечение числовой оценки из ответа LLM"""
        if not response:
            return 0.5  # Значение по умолчанию
        
        # Ищем числа с плавающей точкой
        patterns = [
            r'(\d+\.?\d*)',  # Любое число
            r'(\d+)%',       # Проценты
            r'(\d+)/10',     # Из 10
            r'(\d+)/100'     # Из 100
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    score = float(matches[0])
                    
                    # Нормализация
                    if '%' in response:
                        score = score / 100.0
                    elif '/10' in response:
                        score = score / 10.0
                    elif '/100' in response:
                        score = score / 100.0
                    
                    # Ограничение диапазона
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        return 0.5  # Значение по умолчанию, если не удалось извлечь
    
    def _llm_evaluate(self, prompt_type: str, **format_kwargs) -> float:
        """Выполнение оценки через LLM"""
        if prompt_type not in self.evaluation_prompts:
            raise ValueError(f"Неизвестный тип промпта: {prompt_type}")
        
        prompt_template = self.evaluation_prompts[prompt_type]
        prompt = prompt_template.format(**format_kwargs)
        
        try:
            response = self.provider.generate(prompt)
            return self._extract_score(response)
        except Exception as e:
            print(f"⚠️  Ошибка при оценке {prompt_type}: {e}")
            return 0.5  # Значение по умолчанию при ошибке
    
    async def _allm_evaluate(self, prompt_type: str, **format_kwargs) -> float:
        """Асинхронная оценка через LLM"""
        if prompt_type not in self.evaluation_prompts:
            raise ValueError(f"Неизвестный тип промпта: {prompt_type}")
        
        prompt_template = self.evaluation_prompts[prompt_type]
        prompt = prompt_template.format(**format_kwargs)
        
        try:
            response = await self.provider.a_generate(prompt)
            return self._extract_score(response)
        except Exception as e:
            print(f"⚠️  Ошибка при асинхронной оценке {prompt_type}: {e}")
            return 0.5
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict]] = None,
        expected_answer: Optional[str] = None,
        run_async: bool = False
    ) -> Dict[str, Any]:
        """
        Основной метод оценки качества ответа
        
        Args:
            run_async: Если True, запускает асинхронные оценки (требуется asyncio.run)
        
        Returns:
            Словарь с метриками
        """
        
        if run_async:
            return asyncio.run(self._evaluate_async(
                question, answer, context, conversation_history, expected_answer
            ))
        
        metrics = {
            "question": question,
            "answer": answer,
            "provider": self.provider.get_model_name(),
            "timestamp": self._get_timestamp()
        }
        
        try:
            # 1. Оценка релевантности
            context_line = f"Контекст: {' '.join(context)}" if context else "Контекст не предоставлен."
            
            relevancy_score = self._llm_evaluate(
                "relevancy",
                question=question,
                answer=answer,
                context_line=context_line
            )
            
            metrics["relevancy"] = {
                "score": relevancy_score,
                "passed": relevancy_score >= self.thresholds["relevancy"]
            }
            
            # 2. Оценка использования токенов
            token_metrics = self._evaluate_token_usage(question, answer, conversation_history)
            metrics.update(token_metrics)
            
            # 3. Оценка лаконичности
            conciseness_score = self._llm_evaluate(
                "conciseness",
                question=question,
                answer=answer,
                tokens=self.count_tokens(answer)
            )
            
            metrics["conciseness"] = {
                "score": conciseness_score,
                "passed": conciseness_score >= 0.6  # Порог для лаконичности
            }
            
            # 4. Оценка уточнений
            clarification_metrics = self._evaluate_clarifications(conversation_history)
            metrics.update(clarification_metrics)
            
            # 5. Оценка корректности (если есть эталон)
            if expected_answer:
                correctness_score = self._llm_evaluate(
                    "correctness",
                    question=question,
                    answer=answer,
                    expected=expected_answer
                )
                
                metrics["correctness"] = {
                    "score": correctness_score,
                    "passed": correctness_score >= 0.7
                }
            
            # 6. Расчет итоговых показателей
            metrics["final_score"] = self._calculate_final_score(metrics)
            metrics["assessment"] = self._get_assessment(metrics["final_score"])
            
        except Exception as e:
            metrics["error"] = str(e)
            metrics["final_score"] = 0.0
            metrics["assessment"] = "Ошибка оценки"
        
        return metrics
    
    async def _evaluate_async(
        self,
        question: str,
        answer: str,
        context: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict]] = None,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """Асинхронная версия оценки"""
        
        metrics = {
            "question": question,
            "answer": answer,
            "provider": self.provider.get_model_name(),
            "timestamp": self._get_timestamp()
        }
        
        try:
            # Создаем задачи для параллельного выполнения
            tasks = []
            
            # Оценка релевантности
            context_line = f"Контекст: {' '.join(context)}" if context else "Контекст не предоставлен."
            relevancy_task = self._allm_evaluate(
                "relevancy",
                question=question,
                answer=answer,
                context_line=context_line
            )
            tasks.append(("relevancy", relevancy_task))
            
            # Оценка лаконичности
            conciseness_task = self._allm_evaluate(
                "conciseness",
                question=question,
                answer=answer,
                tokens=self.count_tokens(answer)
            )
            tasks.append(("conciseness", conciseness_task))
            
            # Оценка корректности (если есть)
            if expected_answer:
                correctness_task = self._allm_evaluate(
                    "correctness",
                    question=question,
                    answer=answer,
                    expected=expected_answer
                )
                tasks.append(("correctness", correctness_task))
            
            # Запускаем все оценки параллельно
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Обрабатываем результаты
            for (metric_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"Ошибка в оценке {metric_name}: {result}")
                    score = 0.5
                else:
                    score = result
                
                if metric_name == "relevancy":
                    metrics["relevancy"] = {
                        "score": score,
                        "passed": score >= self.thresholds["relevancy"]
                    }
                elif metric_name == "conciseness":
                    metrics["conciseness"] = {
                        "score": score,
                        "passed": score >= 0.6
                    }
                elif metric_name == "correctness":
                    metrics["correctness"] = {
                        "score": score,
                        "passed": score >= 0.7
                    }
            
            # Синхронные оценки (токены и уточнения)
            token_metrics = self._evaluate_token_usage(question, answer, conversation_history)
            metrics.update(token_metrics)
            
            clarification_metrics = self._evaluate_clarifications(conversation_history)
            metrics.update(clarification_metrics)
            
            # Итоговая оценка
            metrics["final_score"] = self._calculate_final_score(metrics)
            metrics["assessment"] = self._get_assessment(metrics["final_score"])
            
        except Exception as e:
            metrics["error"] = str(e)
            metrics["final_score"] = 0.0
            metrics["assessment"] = "Ошибка оценки"
        
        return metrics
    
    def _evaluate_token_usage(
        self,
        question: str,
        answer: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Оценка использования токенов"""
        
        result = {}
        
        # Подсчет токенов
        question_tokens = self.count_tokens(question)
        answer_tokens = self.count_tokens(answer)
        total_tokens = question_tokens + answer_tokens
        
        result["token_usage"] = {
            "question_tokens": question_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": total_tokens
        }
        
        # Подсчет токенов в истории
        if conversation_history:
            history_tokens = sum(
                self.count_tokens(msg.get("content", "")) 
                for msg in conversation_history
            )
            result["token_usage"]["history_tokens"] = history_tokens
            result["token_usage"]["total_with_history"] = total_tokens + history_tokens
        
        # Расчет эффективности
        efficiency_score = self._calculate_efficiency_score(question_tokens, answer_tokens, answer)
        result["token_usage"]["efficiency_score"] = efficiency_score
        result["token_usage"]["efficiency_passed"] = efficiency_score >= self.thresholds["efficiency"]
        
        return result
    
    def _calculate_efficiency_score(
        self,
        question_tokens: int,
        answer_tokens: int,
        answer: str
    ) -> float:
        """Расчет эффективности использования токенов"""
        
        if answer_tokens == 0:
            return 0.0
        
        # Оптимальное соотношение ответа к вопросу
        optimal_ratio = 2.0  # Ответ в 2 раза длиннее вопроса
        
        if question_tokens > 0:
            ratio = answer_tokens / question_tokens
            
            # Оценка соотношения (ближе к оптимальному = лучше)
            if ratio <= optimal_ratio:
                ratio_score = ratio / optimal_ratio
            else:
                ratio_score = optimal_ratio / ratio
        else:
            ratio_score = 0.5
        
        # Оценка информативности (количество предложений)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        num_sentences = len(sentences)
        
        if num_sentences > 0:
            # Оптимально 2-5 предложений для большинства ответов
            if 2 <= num_sentences <= 5:
                structure_score = 1.0
            elif num_sentences == 1:
                structure_score = 0.7
            elif 6 <= num_sentences <= 8:
                structure_score = 0.5
            else:
                structure_score = 0.3
        else:
            structure_score = 0.3
        
        # Итоговая оценка эффективности
        efficiency = (ratio_score * 0.6 + structure_score * 0.4)
        
        return max(0.0, min(1.0, efficiency))
    
    def _evaluate_clarifications(
        self,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Оценка количества уточнений"""
        
        result = {"clarifications": {"count": 0, "score": 1.0}}
        
        if not conversation_history:
            return result
        
        # Подсчет сообщений пользователя
        user_messages = [
            msg for msg in conversation_history 
            if msg.get("role") == "user"
        ]
        
        clarification_count = max(0, len(user_messages) - 1)
        
        # Расчет оценки
        if clarification_count == 0:
            score = 1.0
            assessment = "Идеально"
        elif clarification_count == 1:
            score = 0.8
            assessment = "Хорошо"
        elif clarification_count == 2:
            score = 0.6
            assessment = "Удовлетворительно"
        elif clarification_count == 3:
            score = 0.4
            assessment = "Плохо"
        else:
            score = 0.2
            assessment = "Очень плохо"
        
        result["clarifications"].update({
            "count": clarification_count,
            "score": score,
            "assessment": assessment,
            "passed": score >= self.thresholds["clarifications"]
        })
        
        return result
    
    def _calculate_final_score(self, metrics: Dict) -> float:
        """Расчет итоговой оценки"""
        
        weights = {
            "relevancy": 0.4,
            "efficiency": 0.3,
            "conciseness": 0.2,
            "clarifications": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        # Релевантность
        if "relevancy" in metrics:
            total_score += metrics["relevancy"]["score"] * weights["relevancy"]
            total_weight += weights["relevancy"]
        
        # Эффективность токенов
        if "token_usage" in metrics:
            efficiency_score = metrics["token_usage"].get("efficiency_score", 0.5)
            total_score += efficiency_score * weights["efficiency"]
            total_weight += weights["efficiency"]
        
        # Лаконичность
        if "conciseness" in metrics:
            total_score += metrics["conciseness"]["score"] * weights["conciseness"]
            total_weight += weights["conciseness"]
        
        # Уточнения
        if "clarifications" in metrics:
            clar_score = metrics["clarifications"].get("score", 1.0)
            total_score += clar_score * weights["clarifications"]
            total_weight += weights["clarifications"]
        
        # Нормализация
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _get_assessment(self, score: float) -> str:
        """Текстовая оценка"""
        
        if score >= 0.9:
            return "Отлично 🏆"
        elif score >= 0.8:
            return "Очень хорошо 👍"
        elif score >= 0.7:
            return "Хорошо ✅"
        elif score >= 0.6:
            return "Удовлетворительно ⚠️"
        elif score >= 0.5:
            return "Ниже среднего 🔄"
        else:
            return "Требует улучшения ❌"
    
    def _get_timestamp(self) -> str:
        """Текущая временная метка"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def print_report(self, metrics: Dict):
        """Вывод отчета"""
        
        print("=" * 70)
        print("📊 УНИВЕРСАЛЬНЫЙ ОТЧЕТ ПО ОЦЕНКЕ КАЧЕСТВА")
        print("=" * 70)
        
        # Базовая информация
        print(f"\n🤖 Провайдер: {metrics.get('provider', 'Неизвестно')}")
        print(f"🕒 Время оценки: {metrics.get('timestamp', 'Неизвестно')}")
        
        # Вопрос и ответ
        question = metrics.get('question', 'N/A')
        answer = metrics.get('answer', 'N/A')
        print(f"\n❓ ВОПРОС: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"💬 ОТВЕТ: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        
        # Итоговая оценка
        final_score = metrics.get('final_score', 0.0)
        assessment = metrics.get('assessment', 'Не оценено')
        print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА: {final_score:.2%}")
        print(f"📋 ВЕРДИКТ: {assessment}")
        
        # Детальные метрики
        print("\n📈 ДЕТАЛЬНЫЕ МЕТРИКИ:")
        
        # Релевантность
        if "relevancy" in metrics:
            rel = metrics["relevancy"]
            status = "✅" if rel.get("passed", False) else "❌"
            print(f"  🔍 Релевантность: {rel.get('score', 0):.2%} {status}")
        
        # Эффективность токенов
        if "token_usage" in metrics:
            tokens = metrics["token_usage"]
            status = "✅" if tokens.get("efficiency_passed", False) else "❌"
            print(f"  💾 Эффективность токенов: {tokens.get('efficiency_score', 0):.2%} {status}")
            print(f"    - Вопрос: {tokens.get('question_tokens', 0)} токенов")
            print(f"    - Ответ: {tokens.get('answer_tokens', 0)} токенов")
        
        # Лаконичность
        if "conciseness" in metrics:
            conc = metrics["conciseness"]
            status = "✅" if conc.get("passed", False) else "❌"
            print(f"  📏 Лаконичность: {conc.get('score', 0):.2%} {status}")
        
        # Уточнения
        if "clarifications" in metrics:
            clar = metrics["clarifications"]
            status = "✅" if clar.get("passed", False) else "❌"
            print(f"  ❓ Уточнений: {clar.get('count', 0)} ({clar.get('assessment', '')}) {status}")
        
        # Корректность
        if "correctness" in metrics:
            corr = metrics["correctness"]
            status = "✅" if corr.get("passed", False) else "❌"
            print(f"  🎓 Фактическая корректность: {corr.get('score', 0):.2%} {status}")
        
        print("\n" + "=" * 70)

#%% Фабрика для быстрого создания агентов
def create_metrics_agent(
    provider_type: str = "auto",
    provider_kwargs: Optional[Dict] = None,
    thresholds: Optional[Dict] = None
) -> UniversalMetricsAgent:
    """
    Фабрика для быстрого создания агента метрик
    
    Args:
        provider_type: Тип провайдера ("auto", "openai", "mistral", и т.д.)
        provider_kwargs: Аргументы для провайдера
        thresholds: Пороговые значения для метрик
    
    Returns:
        UniversalMetricsAgent
    """
    
    provider_kwargs = provider_kwargs or {}
    thresholds = thresholds or {}
    
    # Создаем провайдер
    provider = LLMProviderFactory.create_provider(provider_type, **provider_kwargs)
    
    # Создаем агента
    agent = UniversalMetricsAgent(
        provider=provider,
        thresholds={**{
            "relevancy": 0.7,
            "efficiency": 0.5,
            "clarifications": 0.3
        }, **thresholds}
    )
    
    return agent

#%% Примеры использования
if __name__ == "__main__":
    
    print("🧪 ТЕСТИРОВАНИЕ УНИВЕРСАЛЬНОГО АГЕНТА МЕТРИК")
    print("=" * 50)
    
    # Тест 1: Mock провайдер (без API ключей)
    print("\n1. Тест с Mock провайдером (без API):")
    mock_agent = create_metrics_agent("mock")
    
    test_result = mock_agent.evaluate(
        question="Что такое искусственный интеллект?",
        answer="Искусственный интеллект - это область компьютерных наук, занимающаяся созданием систем, способных выполнять задачи, требующие человеческого интеллекта.",
        context=["ИИ включает машинное обучение, обработку естественного языка и компьютерное зрение."],
        conversation_history=[
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Привет! Как я могу помочь?"}
        ]
    )
    
    mock_agent.print_report(test_result)
    
    # Тест 2: Auto-detection (проверяем переменные окружения)
    print("\n2. Тест с автоопределением провайдера:")
    
    # Сохраняем текущие переменные окружения
    original_env = {}
    for key in ['OPENAI_API_KEY', 'MISTRAL_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']:
        original_env[key] = os.getenv(key)
    
    # Проверяем наличие API ключей
    has_any_api_key = any(original_env.get(key) for key in original_env)
    
    if has_any_api_key:
        print("   Найдены API ключи, создаем агента с автоопределением...")
        auto_agent = create_metrics_agent("auto")
        
        # Быстрая оценка
        quick_result = auto_agent.evaluate(
            question="Как работает ChatGPT?",
            answer="ChatGPT использует архитектуру трансформер и обучается на больших объемах текстовых данных для генерации человеко-подобных ответов."
        )
        
        auto_agent.print_report(quick_result)
    else:
        print("   API ключи не найдены. Для теста реальных API установите:")
        print("   - OpenAI: export OPENAI_API_KEY='ваш_ключ'")
        print("   - Mistral: export MISTRAL_API_KEY='ваш_ключ'")
        print("   - Anthropic: export ANTHROPIC_API_KEY='ваш_ключ'")
        print("   - Google: export GOOGLE_API_KEY='ваш_ключ'")
    
    print("\n" + "=" * 50)
    print("📋 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
    
    print("""
# 1. Быстрое создание агента
agent = create_metrics_agent("openai", {"model": "gpt-4"})

# 2. Оценка ответа
metrics = agent.evaluate(
    question="Ваш вопрос",
    answer="Ответ бота",
    context=["документ1", "документ2"],
    conversation_history=history
)

# 3. Вывод отчета
agent.print_report(metrics)

# 4. Асинхронная оценка
import asyncio
metrics = agent.evaluate(question, answer, run_async=True)

# 5. Создание кастомного провайдера
from my_custom_provider import MyLLMProvider
custom_provider = MyLLMProvider()
agent = UniversalMetricsAgent(provider=custom_provider)
    """)
    
    print("\n✅ Тестирование завершено!")