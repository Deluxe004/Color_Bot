#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import asyncio
import time
from typing import Dict, List, Any
from dataclasses import dataclass


from unification_gigachat import (
    BaseAgent, AgentType, UserContext, LLMResponse, 
    CatalogRow, load_catalog, pick_scheme, ensure_accessibility,
    DEFAULT_CATALOG_PATHS, logger
)

# Импортируем GigaChat для агента
import os
from gigachat import GigaChat

def make_gigachat_client() -> GigaChat:
    cred = os.getenv("GIGACHAT_TOKEN")
    if not cred:
        raise ValueError("GIGACHAT_TOKEN не найден в переменных окружения")
    return GigaChat(
        credentials=cred,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    )

@dataclass
class ColorMatch:
    hex: str
    name: str
    confidence: float
    source: str

class FixedHybridPaletteAgent(BaseAgent):
    """
    Исправленный гибридный агент палитры.
    Использует LLM для анализа запроса и TSV-словарь для точного подбора цветов.
    """
    
    def __init__(self, tsv_paths: List[str] = DEFAULT_CATALOG_PATHS):
        super().__init__(AgentType.PALETTE, {"api_key":"", "base_url":"", "model":""})  # Изменить с PALETTE_LLM на PALETTE
        self.giga = make_gigachat_client()
        self.catalog = load_catalog(tsv_paths)
        
        # Строим индекс для быстрого поиска
        self._build_indices()
        
        logger.info(f"FixedHybridPaletteAgent инициализирован с {len(self.catalog)} цветами")
    
    def _build_indices(self):
        """Строит индексы для поиска по тегам"""
        self.index_by_tag = {}
        for row in self.catalog:
            tags = row.tags.lower().replace(',', ' ').split()
            for tag in tags:
                if len(tag) > 2:  # Игнорируем короткие теги
                    if tag not in self.index_by_tag:
                        self.index_by_tag[tag] = []
                    self.index_by_tag[tag].append(row)
    
    async def _analyze_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Анализ запроса нейросетью с улучшенным промптом"""
        prompt = f"""Проанализируй запрос о цветовой палитре: "{user_input}"

Верни JSON с полями:
- theme: тема/сфера (кофейня, IT, медицина и т.д.)
- emotions: ключевые эмоции/настроения на русском (максимум 3)
- style: стиль (современный, классический и т.д.)
- english_keywords: ключевые слова на АНГЛИЙСКОМ для поиска в базе (например: warm, cool, natural, energetic, calm, professional)
- color_scheme: подходящая схема (монохромная, аналогичная, комплементарная, триадная)

ВАЖНО: english_keywords должны быть на английском, так как база цветов содержит английские теги."""

        try:
            resp = await asyncio.to_thread(
                self.giga.chat,
                {"messages": [
                    {"role": "system", "content": "Ты эксперт по цветовым палитрам. Отвечай только JSON."},
                    {"role": "user", "content": prompt}
                ]}
            )
            
            text = resp.choices[0].message.content
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Ошибка анализа LLM: {e}")
        
        # Fallback
        return self._fallback_analysis(user_input)
    
    def _fallback_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback анализ"""
        input_lower = user_input.lower()
        
        english_keywords = []
        if any(word in input_lower for word in ["кофе", "кафе", "уют"]):
            english_keywords = ["warm", "brown", "cozy", "comfortable"]
        elif any(word in input_lower for word in ["спокой", "медитац", "релакс"]):
            english_keywords = ["cool", "calm", "peaceful", "relaxing"]
        elif any(word in input_lower for word in ["технолог", "it", "современ"]):
            english_keywords = ["professional", "technological", "modern", "reliable"]
        
        return {
            "theme": "общий",
            "emotions": ["нейтральный"],
            "style": "современный",
            "english_keywords": english_keywords or ["neutral", "balanced"],
            "color_scheme": "аналогичная"
        }
    
    def _find_colors_by_keywords(self, english_keywords: List[str]) -> List[CatalogRow]:
        """Поиск цветов по английским ключевым словам"""
        found_rows = []
        seen_hex = set()
        
        for keyword in english_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.index_by_tag:
                for row in self.index_by_tag[keyword_lower]:
                    if row.hex not in seen_hex:
                        found_rows.append(row)
                        seen_hex.add(row.hex)
        
        return found_rows[:10]
    
    def _build_palette(self, found_colors: List[CatalogRow], 
                      scheme: str, n_colors: int = 6) -> List[Dict[str, Any]]:
        """Построение гармоничной палитры"""
        if not found_colors:
            # Если цвета не найдены, используем разнообразную выборку
            import random
            seed = random.choice(self.catalog)
            pool = [r for r in self.catalog if r.hex != seed.hex]
        else:
            seed = found_colors[0]
            pool = found_colors[1:] if len(found_colors) > 1 else self.catalog
        
        # Преобразуем схему
        scheme_map = {
            "монохромная": "mono",
            "аналогичная": "analogous",
            "комплементарная": "complementary",
            "триадная": "triad"
        }
        scheme_key = scheme_map.get(scheme.lower(), "auto")
        
        # Строим палитру
        palette_rows = pick_scheme(seed, pool, scheme_key, n_colors)
        
        # Проверяем доступность
        hex_list = [r.hex for r in palette_rows]
        report, adjusted_hex, notes = ensure_accessibility(hex_list)
        
        # Форматируем результат
        roles = ["primary", "secondary", "accent", "surface", "background", "text"]
        result = []
        
        for i, hex_val in enumerate(adjusted_hex):
            original_row = next((r for r in palette_rows if r.hex == hex_val), None)
            name = original_row.name if original_row else f"Цвет {i+1}"
            
            result.append({
                "hex": hex_val,
                "name": name,
                "role": roles[i] if i < len(roles) else "extra",
                "description": original_row.tags[:50] if original_row else ""
            })
        
        return result

    async def process_request(self, user_input: str, context: UserContext) -> LLMResponse:
        """Основной метод обработки"""
        start_time = time.time()  # Начало замера общего времени
        try:
            # 1. Анализ запроса
            analysis_start = time.time()
            analysis = await self._analyze_with_llm(user_input)
            analysis_time = time.time() - analysis_start

            # 2. Поиск цветов
            search_start = time.time()
            english_keywords = analysis.get('english_keywords', [])
            if isinstance(english_keywords, str):
                english_keywords = english_keywords.split(',')

            color_matches = self._find_colors_by_keywords(english_keywords)
            search_time = time.time() - search_start

            # 3. Построение палитры
            build_start = time.time()
            palette = self._build_palette(color_matches, analysis.get('color_scheme', 'аналогичная'))
            build_time = time.time() - build_start

            # Шаг 4: Сбор метрик
            metrics = {
                # Временные метрики
                "total_processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "analysis_time_ms": round(analysis_time * 1000, 2),
                "search_time_ms": round(search_time * 1000, 2),
                "build_time_ms": round(build_time * 1000, 2),

                # Метрики поиска
                "colors_found": len(color_matches),
                "query_analysis": {
                    "theme": analysis.get('theme', 'unknown'),
                    "emotions_count": len(analysis.get('emotions', [])),
                    "suggested_colors_count": len(analysis.get('english_keywords', [])),
                    # Исправлено: было suggested_colors
                    "color_scheme": analysis.get('color_scheme', 'auto')
                },

                # Метрики результата
                "palette_size": len(palette),
                "tsv_colors_used": len(color_matches) > 0,
                "agent_version": "1.0.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Шаг 5: Формирование результата
            result = {
                "success": True,
                "agent_type": self.agent_type.value,
                "analysis": analysis,
                "palette": palette,
                "metadata": {
                    "metrics": metrics,
                    "processing_time": metrics["total_processing_time_ms"]
                }
            }

            # Шаг 6: Возврат с метриками
            return LLMResponse(
                success=True,
                content=json.dumps(result, ensure_ascii=False, indent=2),
                agent_type=self.agent_type,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Ошибка в FixedHybridPaletteAgent: {e}")

            error_metrics = {
                "error_occurred": True,
                "error_type": type(e).__name__,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_version": "1.0.0"
            }

            return LLMResponse(
                success=False,
                content=json.dumps({
                    "error": f"Ошибка генерации палитры: {str(e)}",
                    "fallback_suggestion": "Попробуйте уточнить запрос"
                }),
                agent_type=self.agent_type,
                error_message=str(e),
                metrics=error_metrics
            )

    
