import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import re
from cards import Card, CARDS
from game_state import GameState

class NeuralEval(nn.Module):
    def __init__(self, input_size=64):  # Default input size
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

# Инициализация модели (глобально для импорта)
input_size = 20 + len(CARDS) * 2  # Базовые фичи + one-hot карты + комбо флаги
neural_model = NeuralEval(input_size=input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(neural_model.parameters(), lr=0.0003)  # Уменьшен lr для стабильности
self_play_data = []  # Данные для обучения

def get_state_features(state: GameState):
    """Получение фич состояния для нейронной сети."""
    style_code = {'neutral': 0, 'aggressive': 1, 'defensive': 2, 'resource': 3, 'combo': 4}.get(state.opp_profile.style, 0)
    features = [
        state.own_tower / 30.0,  # Normalize values
        state.opp_tower / 30.0,
        state.own_wall / 20.0,
        state.opp_wall / 20.0,
        state.own_mana / 10.0,
        state.opp_mana / 10.0,
        state.own_ore / 10.0,
        state.opp_ore / 10.0,
        state.own_troops / 10.0,
        state.opp_troops / 10.0,
        len(state.hand) / 5.0,
        len(state.played_cards) / 30.0,
        state.own_mines / 5.0,
        state.opp_mines / 5.0,
        state.own_monasteries / 5.0,
        state.opp_monasteries / 5.0,
        state.own_barracks / 5.0,
        state.opp_barracks / 5.0
    ]
    hand_onehot = [1 if c.name in [h.name for h in state.hand] else 0 for c in CARDS.values()]
    features += hand_onehot
    combo_flags = [1 if is_combo(CARDS[list(CARDS.keys())[i]], CARDS[list(CARDS.keys())[j]], state) else 0 for i in range(len(CARDS)) for j in range(i+1, len(CARDS))]
    features += combo_flags[:len(CARDS)]  # Ограничено для фиксированного размера
    return np.array(features, dtype=np.float32)

def is_combo(c1: Card, c2: Card, state: GameState) -> bool:
    """Проверка комбо карт с учетом состояния."""
    c1_cat = categorize_effect(c1.effect)
    c2_cat = categorize_effect(c2.effect)
    if c1_cat == 'play_again' and c2_cat in ['damage', 'direct_damage']:
        return True
    if c1.name == 'Призма' and 'tower' in c2.effect and state.own_mana >= 20:
        return True
    if c1.name == 'Чёрт' and c2.name == 'Грунтовые воды' and state.opp_wall < state.own_wall:
        return True
    if c1.name == 'Сдвиг' and state.own_wall > state.opp_wall + 10:
        return True
    if c1.name == 'Вор' and state.opp_mana > 15:
        return True
    if c1_cat == 'resource_loss' and c2_cat == 'damage' and state.opp_profile.style == 'resource':
        return True
    # Дополнительные комбо для повышения winrate
    if c1.name == 'Молния' and c2_cat == 'damage' and state.own_tower > state.opp_wall + 5:
        return True
    if c1.name == 'Дракон' and state.opp_tower < 20 and state.own_troops >= 25:
        return True
    if c1.name == 'Счастливая монетка' and c2.name == 'Копье':
        return True
    return False

def categorize_effect(effect: str) -> str:
    """Категоризация эффекта карты."""
    effect = effect.lower()
    if 'play again' in effect or 'играем снова' in effect:
        return 'play_again'
    if 'урона башне врага' in effect or 'damage to opp tower' in effect:
        return 'direct_damage'
    if 'урона' in effect or 'damage' in effect:
        return 'damage'
    if '+ стена' in effect or '+ wall' in effect:
        return 'build_wall'
    if '+ башне' in effect or '+ tower' in effect:
        return 'build_tower'
    if '+ монастырь' in effect or '+ шахта' in effect or '+ казарма' in effect:
        return 'build_building'
    if '+ мана' in effect or '+ руды' in effect or '+ отрядов' in effect:
        return 'resource_gain'
    if '- мана' in effect or '- руды' in effect or '- отрядов' in effect:
        return 'resource_loss'
    if 'если' in effect or 'if' in effect:
        return 'conditional'
    return 'other'

def eval_condition(effect: str, state: GameState) -> bool:
    """Оценка условных эффектов карты."""
    effect = effect.lower()
    if 'если' in effect or 'if' in effect:
        match = re.match(r'(если|if) ([\w\s=><]+), то ([\w\s+-]+), иначе ([\w\s+-]+)', effect)
        if match:
            cond_str = match.group(2).strip().replace('меньше чем у врага', 'less than opp').replace('больше, чем у врага', 'greater than opp')
            cond_parts = re.split(r'(=|>|<)', cond_str)
            if len(cond_parts) >= 3:
                attr_str = cond_parts[0].strip()
                op = cond_parts[1]
                val_str = cond_parts[2].strip()
                attr = attr_str.replace('стена', 'wall').replace('шахта', 'mines').replace('башня', 'tower').replace('казарма', 'barracks').replace('монастырь', 'monasteries')
                if 'opp' in val_str:
                    val = getattr(state, ('opp_' if 'own' in attr else 'own_') + attr.replace('less than ', '').replace('greater than ', ''))
                else:
                    val = int(val_str)
                current = getattr(state, 'own_' + attr)
                return (op == '=' and current == val) or (op == '>' and current > val) or (op == '< and current < val')
    return False

def extract_damage(effect: str) -> int:
    """Извлечение значения урона из эффекта."""
    effect = effect.lower()
    match = re.search(r'(\d+) (урона|повреждений)', effect)
    return int(match.group(1)) if match else 0

def extract_gain(effect: str) -> tuple[int, str]:
    """Извлечение прироста ресурсов из эффекта."""
    effect = effect.lower()
    match = re.search(r'\+(\d+) (\w+)', effect)
    if match:
        res = match.group(2)
        if res == 'маны': res = 'mana'
        elif res == 'руды': res = 'ore'
        elif res == 'отрядов' or res == 'отряда': res = 'troops'
        return int(match.group(1)), res
    return 0, ''

def heuristic_eval(state: GameState) -> float:
    """Эвристическая оценка состояния для MCTS (улучшено для winrate)."""
    score = (state.own_tower - state.opp_tower) * 5 + (state.own_wall - state.opp_wall) * 2
    score += (state.own_mana + state.own_ore + state.own_troops) - (state.opp_mana + state.opp_ore + state.opp_troops)
    score += (state.own_monasteries + state.own_mines + state.own_barracks) * 3 - (state.opp_monasteries + state.opp_mines + state.opp_barracks) * 3
    
    # Улучшения для ранней/поздней игры
    build_bonus = (state.own_monasteries + state.own_mines + state.own_barracks) * 5 if state.turn < 5 else 1
    score += build_bonus
    wall_bonus = (state.own_wall - state.opp_wall) * (3 if state.turn < 7 else 1)
    score += wall_bonus
    
    # Штраф за избыток ресурсов без действий
    if state.own_mana > 20 and not any('башне' in c.effect for c in state.hand):
        score -= 15
    if state.own_ore > 20 and not any('стене' in c.effect for c in state.hand):
        score -= 10
    if state.own_troops > 20 and not any('урона' in c.effect for c in state.hand):
        score -= 12
    
    # Бонус за финишеры в поздней игре
    if state.turn > 100:
        score += sum(10 for c in state.hand if 'урона' in c.effect or c.name in ['Дракон', 'Глаз дракона']) * 20
    
    # Комбо бонусы
    if any('play again' in c.effect for c in state.hand) and any('урона' in c.effect for c in state.hand):
        score += 15 * sum(1 for c in state.hand if 'play again' in c.effect)
    
    # Контрудар против стиля оппонента
    if state.opp_profile.style == 'aggressive' and sum(1 for c in state.hand if categorize_effect(c.effect) in ['build_wall', 'defensive']) > 2:
        score += 15
    if state.opp_profile.style == 'defensive' and sum(1 for c in state.hand if categorize_effect(c.effect) == 'direct_damage') > 1:
        score += 20
    if state.opp_profile.style == 'resource' and any(c.name == 'Вор' for c in state.hand) and state.opp_mana > 15:
        score += 20
    if state.opp_profile.style == 'combo' and sum(1 for c in state.hand if categorize_effect(c.effect) == 'resource_loss') > 1:
        score += 15
    
    # Бонусы за ключевые карты и условия
    for card in state.hand:
        cat = categorize_effect(card.effect)
        if card.name == 'Призма' and any('башне' in h.effect for h in state.hand) and state.own_mana >= 20:
            score += 25
        elif card.name == 'Чёрт' and state.turn < 5 and state.opp_wall < 10:
            score += 20
        elif card.name == 'Грунтовые воды' and state.own_wall > state.opp_wall and state.opp_barracks > 0:
            score += 15
        elif card.name == 'Сдвиг' and state.own_wall - state.opp_wall > 10:
            score += 20
        elif card.name == 'Вор' and state.opp_mana > 20:
            score += 25
        elif card.name == 'Бракованная руда' and state.opp_ore > 8:
            score += 15
        elif card.name == 'Полнолуние' and state.own_barracks < state.opp_barracks:
            score += 10
        elif card.name == 'Фундамент' and state.own_wall == 0:
            score += 12
        elif card.name == 'Дракон' and state.own_troops >= 25 and state.opp_tower < 20:
            score += 30
        elif cat == 'conditional' and eval_condition(card.effect, state):
            score += 15
        damage_val = extract_damage(card.effect)
        if damage_val > 0 and state.opp_wall < 5:
            score += damage_val * 2
        gain_val, res = extract_gain(card.effect)
        if gain_val > 0 and getattr(state, 'own_' + res) < 10:
            score += gain_val * 1.5
        if state.opp_profile.style == 'aggressive' and cat == 'build_wall':
            score += 10
    
    return score
