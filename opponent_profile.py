
from collections import defaultdict
from copy import deepcopy
from random import random
from cards import Card
from game_state import GameState
from evaluation import categorize_effect, is_combo

class OpponentProfile:
    def __init__(self):
        self.style = 'neutral'  # Начальный стиль: нейтральный
        self.history = []  # История действий оппонента
        self.transitions = defaultdict(lambda: defaultdict(int))  # Переходы для предсказания
        self.card_frequency = defaultdict(int)  # Частота карт для анализа стиля

    def update(self, action: Card, state: GameState):
        """Обновление профиля оппонента на основе сыгранной карты и состояния."""
        self.history.append((action, deepcopy(state)))
        self.card_frequency[action.name] += 1
        if len(self.history) >= 3:
            prev_key = '->'.join([h[0].name for h in self.history[-3:]])
            self.transitions[prev_key][action.name] += 1
        
        # Анализ стиля игры (улучшено: учитываем частоту категорий и последовательности)
        damage_count = sum(1 for h in self.history if categorize_effect(h[0].effect) in ['damage', 'direct_damage'])
        build_count = sum(1 for h in self.history if categorize_effect(h[0].effect) in ['build_wall', 'build_tower', 'build_building'])
        resource_count = sum(1 for h in self.history if categorize_effect(h[0].effect) in ['resource_gain', 'resource_loss'])
        combo_count = sum(1 for i in range(len(self.history)-1) if is_combo(self.history[i][0], self.history[i+1][0], self.history[i][1]))

        total_actions = max(1, len(self.history))
        if damage_count / total_actions > 0.4:
            self.style = 'aggressive'
        elif build_count / total_actions > 0.4:
            self.style = 'defensive'
        elif resource_count / total_actions > 0.4:
            self.style = 'resource'
        elif combo_count > 2 and combo_count / total_actions > 0.2:
            self.style = 'combo'
        else:
            self.style = 'neutral'

    def get_transition_probs(self, prev_key: str) -> dict[str, float]:
        """Вероятности следующей карты на основе истории переходов."""
        trans = self.transitions.get(prev_key, {})
        total = sum(trans.values())
        return {k: v / total if total > 0 else 0 for k, v in trans.items()}

    def predict_next_card(self, state: GameState) -> Card:
        """Предсказание следующей карты оппонента с учетом стиля и истории."""
        from cards import CARDS  # Локальный импорт для избежания циклической зависимости
        if len(self.history) >= 2:
            prev_key = '->'.join([h[0].name for h in self.history[-3:]])
            probs = self.get_transition_probs(prev_key)
            if probs:
                return CARDS[max(probs, key=probs.get)]
        # Если нет переходов, предсказываем по стилю
        if self.style == 'aggressive':
            damage_cards = [c for c in CARDS.values() if 'урона' in c.effect.lower() or 'damage' in c.effect.lower()]
            return random.choice(damage_cards) if damage_cards else random.choice(list(CARDS.values()))
        elif self.style == 'defensive':
            build_cards = [c for c in CARDS.values() if 'стене' in c.effect.lower() or 'башне' in c.effect.lower()]
            return random.choice(build_cards) if build_cards else random.choice(list(CARDS.values()))
        elif self.style == 'resource':
            res_cards = [c for c in CARDS.values() if 'шахта' in c.effect.lower() or 'монастырь' in c.effect.lower() or 'казарма' in c.effect.lower()]
            return random.choice(res_cards) if res_cards else random.choice(list(CARDS.values()))
        elif self.style == 'combo':
            combo_cards = [c for c in CARDS.values() if 'play again' in c.effect.lower() or 'если' in c.effect.lower()]
            return random.choice(combo_cards) if combo_cards else random.choice(list(CARDS.values()))
        return random.choice(list(CARDS.values()))
