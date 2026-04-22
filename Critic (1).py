from collections import deque

class Critic:
    """Критик для оценки качества палитр и обучения"""

    def __init__(self, learning_rate=0.1, discount_factor=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # Таблица Q-значений
        self.memory = deque(maxlen=1000)

    def evaluate_palette(self, palette, business_type, user_feedback):
        """Оценка палитры и вычисление награды"""
        reward = 0

        # Базовые правила оценки
        if len(palette) >= 3:
            reward += 10

        # Анализ пользовательского фидбека
        feedback_score = self._analyze_feedback(user_feedback)
        reward += feedback_score

        # Проверка сочетаемости цветов
        harmony_score = self._check_color_harmony(palette)
        reward += harmony_score

        # Соответствие бизнес-профилю
        business_score = self._check_business_match(palette, business_type)
        reward += business_score

        return reward

    def _analyze_feedback(self, feedback):
        """Анализ текстового фидбека пользователя"""
        positive_words = ["нравится", "хорошо", "отлично", "супер", "красиво", "да"]
        negative_words = ["не нравится", "плохо", "ужасно", "нет", "переделай"]

        if any(word in feedback.lower() for word in positive_words):
            return 20
        elif any(word in feedback.lower() for word in negative_words):
            return -10
        return 0

    def _check_color_harmony(self, palette):
        """Проверка гармонии цветов"""
        if len(palette) < 2:
            return 0

        # Простая проверка контраста
        brightness_variation = len(set([self._get_color_brightness(color) for color in palette]))
        return min(brightness_variation * 5, 15)

    def _get_color_brightness(self, color):
        """Вычисление яркости цвета (упрощенное)"""
        if color in ["белый", "желтый", "розовый"]:
            return "light"
        elif color in ["черный", "синий", "фиолетовый"]:
            return "dark"
        else:
            return "medium"

    def _check_business_match(self, palette, business_type):
        """Проверка соответствия палитры типу бизнеса"""
        business_rules = {
            "технологии": ["синий", "серый", "белый"],
            "медицина": ["синий", "белый", "зеленый"],
            "еда": ["красный", "оранжевый", "желтый"],
            "природа": ["зеленый", "коричневый", "голубой"]
        }

        expected_colors = business_rules.get(business_type, [])
        matches = sum(1 for color in palette if color in expected_colors)
        return matches * 5