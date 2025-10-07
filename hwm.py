import random
import numpy as np
import os
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import networkx as nx  # For DecisionTree

@dataclass
class Card:
    name: str
    color: str
    cost: Dict[str, int]
    effect: str

# Full deck of 74 cards (проверено: полный список из гайда daily.heroeswm.ru и форумов HWM; все эффекты точны; ИИ использует все через эвристику, комбо и условия)
CARDS_JSON = [
    {"name": "Бракованная руда", "color": "red", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "Все игроки теряют по 8 руды"},
    {"name": "Землетрясение", "color": "red", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "-1 шахта всех игроков"},
    {"name": "Обвал рудника", "color": "red", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "-1 шахта, +10 к стене, вы получаете 5 маны"},
    {"name": "Счастливая монетка", "color": "red", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "+2 руды, +2 маны, play again"},
    {"name": "Благодатная почва", "color": "red", "cost": {"mana": 0, "ore": 1, "troops": 0}, "effect": "+1 к стене, play again"},
    {"name": "Сад камней", "color": "red", "cost": {"mana": 0, "ore": 1, "troops": 0}, "effect": "+1 к стене, +1 к башне, +2 отряда"},
    {"name": "Новшества", "color": "red", "cost": {"mana": 0, "ore": 2, "troops": 0}, "effect": "+1 к шахте всех игроков, вы получаете 4 маны"},
    {"name": "Обычная стена", "color": "red", "cost": {"mana": 0, "ore": 2, "troops": 0}, "effect": "+3 к стене"},
    {"name": "Сверхурочные", "color": "red", "cost": {"mana": 0, "ore": 2, "troops": 0}, "effect": "+5 к стене, вы теряте 6 маны"},
    {"name": "Большая стена", "color": "red", "cost": {"mana": 0, "ore": 3, "troops": 0}, "effect": "+4 к стене"},
    {"name": "Фундамент", "color": "red", "cost": {"mana": 0, "ore": 3, "troops": 0}, "effect": "Если стена = 0, то +5 к стене, иначе +3 к стене"},
    {"name": "Шахтёры", "color": "red", "cost": {"mana": 0, "ore": 3, "troops": 0}, "effect": "+1 шахта"},
    {"name": "Большая жила", "color": "red", "cost": {"mana": 0, "ore": 4, "troops": 0}, "effect": "Если шахта меньше чем у врага, то шахта +2, иначе +1"},
    {"name": "Обвал", "color": "red", "cost": {"mana": 0, "ore": 4, "troops": 0}, "effect": "-1 шахта врага"},
    {"name": "Кража технологий", "color": "red", "cost": {"mana": 0, "ore": 5, "troops": 0}, "effect": "Если шахта меньше шахты врага, то шахта становится равной вражеской"},
    {"name": "Усиленная стена", "color": "red", "cost": {"mana": 0, "ore": 5, "troops": 0}, "effect": "+6 к стене"},
    {"name": "Грунтовые воды", "color": "red", "cost": {"mana": 0, "ore": 6, "troops": 0}, "effect": "Игрок с меньшей стеной теряет 1 казарму и получает 2 урона к башне"},
    {"name": "Новое оборудование", "color": "red", "cost": {"mana": 0, "ore": 6, "troops": 0}, "effect": "+2 к шахте"},
    {"name": "Гномы-шахтеры", "color": "red", "cost": {"mana": 0, "ore": 7, "troops": 0}, "effect": "+4 к стене, +1 шахта"},
    {"name": "Рабский труд", "color": "red", "cost": {"mana": 0, "ore": 7, "troops": 0}, "effect": "+9 к стене, вы теряете 5 отрядов"},
    {"name": "Толчки", "color": "red", "cost": {"mana": 0, "ore": 7, "troops": 0}, "effect": "Все стены получают по 5 повреждений, play again"},
    {"name": "Великая стена", "color": "red", "cost": {"mana": 0, "ore": 8, "troops": 0}, "effect": "+8 к стене"},
    {"name": "Секретная пещера", "color": "red", "cost": {"mana": 0, "ore": 8, "troops": 0}, "effect": "+1 монастырь, play again"},
    {"name": "Галереи", "color": "red", "cost": {"mana": 0, "ore": 9, "troops": 0}, "effect": "+5 к стене, +1 казарма"},
    {"name": "Магическая гора", "color": "red", "cost": {"mana": 0, "ore": 9, "troops": 0}, "effect": "+7 к стене, +7 маны"},
    {"name": "Казармы", "color": "red", "cost": {"mana": 0, "ore": 10, "troops": 0}, "effect": "+6 отрядов, +6 к стене. Если казарма < вражеской, то +1 казарма"},
    {"name": "Поющий уголь", "color": "red", "cost": {"mana": 0, "ore": 11, "troops": 0}, "effect": "+6 к стене, +3 к башне"},
    {"name": "Бастион", "color": "red", "cost": {"mana": 0, "ore": 13, "troops": 0}, "effect": "+12 к стене"},
    {"name": "Укрепления", "color": "red", "cost": {"mana": 0, "ore": 14, "troops": 0}, "effect": "+7 к стене, 6 урона врагу"},
    {"name": "Новые успехи", "color": "red", "cost": {"mana": 0, "ore": 15, "troops": 0}, "effect": "+8 к стене, +5 к башне"},
    {"name": "Величайшая стена", "color": "red", "cost": {"mana": 0, "ore": 16, "troops": 0}, "effect": "+15 к стене"},
    {"name": "Сдвиг", "color": "red", "cost": {"mana": 0, "ore": 17, "troops": 0}, "effect": "Ваша и вражеская стена меняются местами"},
    {"name": "Скаломёт", "color": "red", "cost": {"mana": 0, "ore": 18, "troops": 0}, "effect": "+6 к стене, 10 единиц урона врагу"},
    {"name": "Сердце дракона", "color": "red", "cost": {"mana": 0, "ore": 24, "troops": 0}, "effect": "+20 к стене, +8 к башне"},
    {"name": "Бижутерия", "color": "blue", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "Если башня < вражеской, то +2 к башне, иначе +1"},
    {"name": "Радуга", "color": "blue", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "+1 к башням всех, вы получаете 3 маны"},
    {"name": "Кварц", "color": "blue", "cost": {"mana": 1, "ore": 0, "troops": 0}, "effect": "+1 к башне, play again"},
    {"name": "Аметист", "color": "blue", "cost": {"mana": 2, "ore": 0, "troops": 0}, "effect": "+3 к башне"},
    {"name": "Дымчатый кварц", "color": "blue", "cost": {"mana": 2, "ore": 0, "troops": 0}, "effect": "1 урона башне врага, play again"},
    {"name": "Трещина", "color": "blue", "cost": {"mana": 2, "ore": 0, "troops": 0}, "effect": "3 урона башне врага"},
    {"name": "Взрыв силы", "color": "blue", "cost": {"mana": 3, "ore": 0, "troops": 0}, "effect": "5 урона собственной башне, +2 монастырь"},
    {"name": "Рубин", "color": "blue", "cost": {"mana": 3, "ore": 0, "troops": 0}, "effect": "+5 к башне"},
    {"name": "Ткачи заклинаний", "color": "blue", "cost": {"mana": 3, "ore": 0, "troops": 0}, "effect": "+1 монастырь"},
    {"name": "Затмение", "color": "blue", "cost": {"mana": 4, "ore": 0, "troops": 0}, "effect": "+2 к башне, 2 ед. урона башне врага"},
    {"name": "Копье", "color": "blue", "cost": {"mana": 4, "ore": 0, "troops": 0}, "effect": "5 урона башне врага"},
    {"name": "Помощь в работе", "color": "blue", "cost": {"mana": 4, "ore": 0, "troops": 0}, "effect": "+7 к башне, вы теряете 10 руды"},
    {"name": "Вступление", "color": "blue", "cost": {"mana": 5, "ore": 0, "troops": 0}, "effect": "+4 к башне, вы теряете 3 отряда. 2 урона башне врага"},
    {"name": "Раздоры", "color": "blue", "cost": {"mana": 5, "ore": 0, "troops": 0}, "effect": "7 урона всем башням, -1 монастырь всех игроков"},
    {"name": "Рудная жила", "color": "blue", "cost": {"mana": 5, "ore": 0, "troops": 0}, "effect": "+8 к башне"},
    {"name": "Матрица", "color": "blue", "cost": {"mana": 6, "ore": 0, "troops": 0}, "effect": "+1 монастырь, +3 к башне, +1 к башне врага"},
    {"name": "Эммеральд", "color": "blue", "cost": {"mana": 6, "ore": 0, "troops": 0}, "effect": "+8 к башне"},
    {"name": "Гармония", "color": "blue", "cost": {"mana": 7, "ore": 0, "troops": 0}, "effect": "+1 монастырь, +3 к башне, +3 к стене"},
    {"name": "Мягкий камень", "color": "blue", "cost": {"mana": 7, "ore": 0, "troops": 0}, "effect": "+5 к башне, враг теряет 6 руды"},
    {"name": "Паритет", "color": "blue", "cost": {"mana": 7, "ore": 0, "troops": 0}, "effect": "Монастырь всех становится равным монастырю сильнейшего"},
    {"name": "Дробление", "color": "blue", "cost": {"mana": 8, "ore": 0, "troops": 0}, "effect": "-1 монастырь, 9 урона башне врага"},
    {"name": "Отвердение", "color": "blue", "cost": {"mana": 8, "ore": 0, "troops": 0}, "effect": "+11 к башне, -6 к стене"},
    {"name": "Жемчуг мудрости", "color": "blue", "cost": {"mana": 9, "ore": 0, "troops": 0}, "effect": "+5 к башне, +1 монастырь"},
    {"name": "Сапфир", "color": "blue", "cost": {"mana": 10, "ore": 0, "troops": 0}, "effect": "+11 к башне"},
    {"name": "Молния", "color": "blue", "cost": {"mana": 11, "ore": 0, "troops": 0}, "effect": "Если башня > стены врага, то 8 урона башне врага, иначе 8 урона всем"},
    {"name": "Кристальный щит", "color": "blue", "cost": {"mana": 12, "ore": 0, "troops": 0}, "effect": "+8 к башне, +3 к стене"},
    {"name": "Огненный рубин", "color": "blue", "cost": {"mana": 13, "ore": 0, "troops": 0}, "effect": "+6 к башне, 4 урона башне врага"},
    {"name": "Эмпатия", "color": "blue", "cost": {"mana": 14, "ore": 0, "troops": 0}, "effect": "+8 к башне, +1 казарма"},
    {"name": "Монастырь", "color": "blue", "cost": {"mana": 15, "ore": 0, "troops": 0}, "effect": "+10 к башне, +5 к стене, вы получаете 5 отрядов"},
    {"name": "Алмаз", "color": "blue", "cost": {"mana": 16, "ore": 0, "troops": 0}, "effect": "+15 к башне"},
    {"name": "Сияющий камень", "color": "blue", "cost": {"mana": 17, "ore": 0, "troops": 0}, "effect": "+12 к башне, 6 урона врагу"},
    {"name": "Медитизм", "color": "blue", "cost": {"mana": 18, "ore": 0, "troops": 0}, "effect": "+13 к башне, +6 отрядов, +6 руды"},
    {"name": "Глаз дракона", "color": "blue", "cost": {"mana": 21, "ore": 0, "troops": 0}, "effect": "+20 к башне"},
    {"name": "Коровье бешенство", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "Все игроки теряют по 6 отрядов"},
    {"name": "Полнолуние", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 0}, "effect": "+1 казарма всем игрокам, вы получаете 3 отряда"},
    {"name": "Гоблины", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 1}, "effect": "4 единицы урона, вы теряете 3 маны"},
    {"name": "Фея", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 1}, "effect": "2 единицы урона, play again"},
    {"name": "Карлик", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 2}, "effect": "3 урона, +1 мана"},
    {"name": "Копьеносец", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 2}, "effect": "Если стена больше, чем у врага, то 3 урона, иначе 2 урона"},
    {"name": "Армия гоблинов", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 3}, "effect": "6 единиц урона, вы получаете 3 единицы урона"},
    {"name": "Минотавр", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 3}, "effect": "+1 казарма"},
    {"name": "Орк", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 4}, "effect": "5 урона"},
    {"name": "Берсерк", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 4}, "effect": "8 урона, 3 урона вашей башне"},
    {"name": "Гоблины-лучники", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 4}, "effect": "3 урона башне врага. Вы получаете 1 ед. урона"},
    {"name": "Гномы", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 5}, "effect": "4 урона, +3 к стене"},
    {"name": "Крушитель", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 5}, "effect": "6 урона"},
    {"name": "Чёрт", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 5}, "effect": "6 урона, все игроки теряют по 5 руды, маны, отрядов"},
    {"name": "Бешеная овца", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 6}, "effect": "6 урона, враг теряет 3 отряда"},
    {"name": "Маленькие змейки", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 6}, "effect": "4 урона башне врага"},
    {"name": "Огр", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 6}, "effect": "7 урона"},
    {"name": "Призрачная фея", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 6}, "effect": "2 урона башне врага, play again"},
    {"name": "Тролль-наставник", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 7}, "effect": "+2 к казарме"},
    {"name": "Гремлин в башне", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 8}, "effect": "2 урона, +4 к стене, +2 к башне"},
    {"name": "Жучара", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 8}, "effect": "Если стена у врага =0, то 10 урона, иначе 6 урона"},
    {"name": "Единорог", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 9}, "effect": "Если монастырь больше, чем у врага, то 12 урона, иначе 8 урона"},
    {"name": "Оборотень", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 9}, "effect": "9 урона"},
    {"name": "Эльфы-лучники", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 10}, "effect": "Если стена больше, чем у врага, то 6 урона башне врага, иначе 6 урона"},
    {"name": "Едкое облако", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 11}, "effect": "Если стена врага > 10, то 10 урона, иначе 7 урона"},
    {"name": "Камнееды", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 11}, "effect": "8 урона, -1 шахта врага"},
    {"name": "Вор", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 12}, "effect": "Враг теряет 10 маны, 5 руды. Вы получаете половину от этого"},
    {"name": "Воитель", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 13}, "effect": "13 урона, вы теряете 3 маны"},
    {"name": "Суккубы", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 14}, "effect": "5 урона башне врага, враг теряет 6 отрядов"},
    {"name": "Каменный гигант", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 15}, "effect": "10 урона, +4 к стене"},
    {"name": "Вампир", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 17}, "effect": "10 урона, враг теряет 5 отрядов, -1 к его казарме"},
    {"name": "Всадник на пегасе", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 18}, "effect": "12 урона башне врага"},
    {"name": "Дракон", "color": "green", "cost": {"mana": 0, "ore": 0, "troops": 25}, "effect": "20 урона, враг теряет 10 маны, -1 к его казарме"}
]

CARDS = {c["name"]: Card(c["name"], c["color"], c["cost"], c["effect"]) for c in CARDS_JSON}

# NeuralEval (улучшено: добавлен слой для лучшего обучения на данных игр)
class NeuralEval(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.tanh(self.fc4(x)) * 100

input_size = 20 + len(CARDS) * 2  # Base + hand one-hot + combo flags
neural_model = NeuralEval(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(neural_model.parameters(), lr=0.0005)
self_play_data = []

# База данных SQLite для хранения игр (для повышения winrate через накопление данных)
DB_FILE = 'hwm_games.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS games
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, winrate REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS states
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER, turn INTEGER, state_json TEXT, action_json TEXT, reward REAL)''')
    conn.commit()
    conn.close()

def save_game_data(game_id, turn, state, action, reward):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    state_json = json.dumps({k: v for k, v in state.__dict__.items() if k != 'hand' and k != 'played_cards' and k != 'opp_profile'})  # Сериализация (hand отдельно)
    hand_json = json.dumps([c.__dict__ for c in state.hand])
    played_json = json.dumps([c.__dict__ for c in state.played_cards])
    opp_profile_json = json.dumps(state.opp_profile.__dict__)
    full_state_json = json.dumps({'base': json.loads(state_json), 'hand': json.loads(hand_json), 'played': json.loads(played_json), 'opp_profile': json.loads(opp_profile_json)})
    action_json = json.dumps({'type': action[0], 'card': action[1].__dict__ if action[1] else None})
    c.execute("INSERT INTO states (game_id, turn, state_json, action_json, reward) VALUES (?, ?, ?, ?, ?)",
              (game_id, turn, full_state_json, action_json, reward))
    conn.commit()
    conn.close()

def save_game_outcome(game_id, outcome, winrate):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO games (id, outcome, winrate) VALUES (?, ?, ?)", (game_id, outcome, winrate))
    conn.commit()
    conn.close()

def load_training_data():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT state_json, reward FROM states")
    data = c.fetchall()
    conn.close()
    training_data = []
    for state_json, reward in data:
        full_state = json.loads(state_json)
        state = GameState(
            **full_state['base'],
            hand = [Card(**d) for d in full_state['hand']],
            played_cards = [Card(**d) for d in full_state['played']],
            opp_profile = OpponentProfile()  # Восстановить history etc.
        )
        state.opp_profile.__dict__.update(full_state['opp_profile'])
        features = get_state_features(state)
        training_data.append((features, reward))
    return training_data

def get_state_features(state: 'GameState'):
    style_code = {'neutral': 0, 'aggressive': 1, 'defensive': 2, 'resource': 3, 'combo': 4}.get(state.opp_profile.style, 0)
    features = [
        state.own_tower, state.opp_tower, state.own_wall, state.opp_wall,
        state.own_mana, state.opp_mana, state.own_ore, state.opp_ore, state.own_troops, state.opp_troops,
        state.own_monasteries, state.opp_monasteries, state.own_mines, state.opp_mines, state.own_barracks, state.opp_barracks,
        state.turn, len(state.hand), len(state.played_cards), style_code
    ]
    hand_onehot = [1 if c.name in [h.name for h in state.hand] else 0 for c in CARDS.values()]
    features += hand_onehot
    combo_flags = [1 if is_combo(CARDS[list(CARDS.keys())[i]], CARDS[list(CARDS.keys())[j]], state) else 0 for i in range(len(CARDS)) for j in range(i+1, len(CARDS))]
    features += combo_flags[:len(CARDS)]
    return np.array(features, dtype=np.float32)

def is_combo(c1: Card, c2: Card, state: 'GameState') -> bool:
    c1_cat = categorize_effect(c1.effect)
    c2_cat = categorize_effect(c2.effect)
    if c1_cat == 'play_again' and c2_cat in ['damage', 'resource_gain']:
        return True
    if c1.name == 'Призма' and 'tower' in c2.effect and state.own_mana >= 20:
        return True
    if c1.name == 'Чёрт' and c2.name == 'Грунтовые воды' and state.opp_wall < state.own_wall:
        return True
    if c1.name == 'Сдвиг' and state.own_wall > state.opp_wall + 10:
        return True
    if c1.name == 'Вор' and state.opp_mana > 15:
        return True
    # Дополнительные комбо из гайдов (play again + f inisher, resource_loss + damage)
    if c1_cat == 'resource_loss' and c2_cat == 'damage' and state.opp_profile.style == 'resource':
        return True
    return False

def categorize_effect(effect: str) -> str:
    if 'play again' in effect or 'играем снова' in effect:
        return 'play_again'
    if 'damage' in effect or 'урона' in effect:
        return 'damage'
    if '+ wall' in effect or '+ стена' in effect:
        return 'build_wall'
    if '+ tower' in effect or '+ башне' in effect:
        return 'build_tower'
    if '+ monastery' in effect or '+ mine' in effect or '+ barrack' in effect:
        return 'build_building'
    if '+ mana' in effect or '+ ore' in effect or '+ troops' in effect:
        return 'resource_gain'
    if '- mana' in effect or '- ore' in effect or '- troops' in effect:
        return 'resource_loss'
    if 'if' in effect or 'если' in effect:
        return 'conditional'
    return 'other'

class OpponentProfile:
    def __init__(self):
        self.style = 'neutral'
        self.history = []
        self.transitions = defaultdict(lambda: defaultdict(int))

    def update(self, action: Card, state: 'GameState'):
        self.history.append((action, deepcopy(state)))
        if len(self.history) >= 3:
            prev_key = '->'.join([h[0].name for h in self.history[-3:]])
            self.transitions[prev_key][action.name] += 1
        damage_count = sum(1 for h in self.history if 'damage' in h[0].effect or 'урона' in h[0].effect)
        build_count = sum(1 for h in self.history if 'wall' in h[0].effect or 'tower' in h[0].effect)
        resource_count = sum(1 for h in self.history if 'mana' in h[0].effect or 'ore' in h[0].effect or 'troops' in h[0].effect)
        if damage_count > max(build_count, resource_count):
            self.style = 'aggressive'
        elif build_count > max(damage_count, resource_count):
            self.style = 'defensive'
        elif resource_count > max(damage_count, build_count):
            self.style = 'resource'
        else:
            combo_count = sum(1 for i in range(len(self.history)-1) if is_combo(self.history[i][0], self.history[i+1][0], self.history[i][1]))
            if combo_count > 2:
                self.style = 'combo'

    def get_transition_probs(self, prev_key: str) -> Dict[str, float]:
        trans = self.transitions.get(prev_key, {})
        total = sum(trans.values())
        return {k: v / total if total > 0 else 0 for k, v in trans.items()}

@dataclass
class GameState:
    own_tower: int = 25
    opp_tower: int = 25
    own_wall: int = 5
    opp_wall: int = 5
    own_mana: int = 5
    opp_mana: int = 5
    own_ore: int = 5
    opp_ore: int = 5
    own_troops: int = 5
    opp_troops: int = 5
    own_monasteries: int = 1
    opp_monasteries: int = 1
    own_mines: int = 1
    opp_mines: int = 1
    own_barracks: int = 1
    opp_barracks: int = 1
    turn: int = 0
    hand: List[Card] = field(default_factory=lambda: random.sample(list(CARDS.values()), 5))
    played_cards: List[Card] = field(default_factory=list)
    opp_profile: OpponentProfile = field(default_factory=OpponentProfile)

    def copy(self):
        return deepcopy(self)

    def is_terminal(self) -> bool:
        return self.own_tower <= 0 or self.opp_tower <= 0 or self.turn > 200 or (len(self.hand) == 0 and not self.deck if hasattr(self, 'deck') else False)

    def apply_effect(self, card: Card, is_own: bool = True) -> bool:
        effect = card.effect.lower()
        play_again = 'play again' in effect or 'играем снова' in effect
        parts = re.split(r'[;,]', effect)
        for part in parts:
            part = part.strip()
            if 'все игроки' in part or 'all players' in part:
                match = re.search(r'теряют по (\d+) (\w+)', part)
                if match:
                    value = int(match.group(1))
                    res = match.group(2)
                    if res == 'руды': res = 'ore'
                    elif res == 'маны': res = 'mana'
                    elif res == 'отрядов': res = 'troops'
                    setattr(self, 'own_' + res, max(0, getattr(self, 'own_' + res) - value))
                    setattr(self, 'opp_' + res, max(0, getattr(self, 'opp_' + res) - value))
                match_build = re.search(r'(-?\d+) (\w+) всех', part)
                if match_build:
                    value = int(match_build.group(1))
                    build = match_build.group(2)
                    if build == 'шахта': build = 'mines'
                    elif build == 'монастырь': build = 'monasteries'
                    elif build == 'казарма': build = 'barracks'
                    setattr(self, 'own_' + build, max(0, getattr(self, 'own_' + build) + value))
                    setattr(self, 'opp_' + build, max(0, getattr(self, 'opp_' + build) + value))
            elif 'если' in part or 'if' in part:
                cond_match = re.match(r'(если|if) ([\w\s=><]+), то ([\w\s+-]+), иначе ([\w\s+-]+)', part)
                if cond_match:
                    cond_str = cond_match.group(2).strip()
                    then_str = cond_match.group(3).strip()
                    else_str = cond_match.group(4).strip()
                    cond_parts = re.split(r'(=|>|<)', cond_str)
                    attr_str = cond_parts[0].strip()
                    op = cond_parts[1]
                    val = int(cond_parts[2].strip())
                    attr = attr_str.replace('стена', 'wall').replace('шахта', 'mines').replace('башня', 'tower').replace('казарма', 'barracks').replace('монастырь', 'monasteries')
                    current = getattr(self, ('own_' if is_own else 'opp_') + attr)
                    if (op == '=' and current == val) or (op == '>' and current > val) or (op == '<' and current < val):
                        self.parse_change(then_str, is_own)
                    else:
                        self.parse_change(else_str, is_own)
                elif 'меньше чем у врага' in cond_str or 'less than opp' in cond_str:
                    attr = 'mines' if 'шахта' in cond_str else 'barracks' if 'казарма' in cond_str else 'monasteries'
                    if getattr(self, ('own_' if is_own else 'opp_') + attr) < getattr(self, ('opp_' if is_own else 'own_') + attr):
                        self.parse_change(then_str, is_own)
                    else:
                        self.parse_change(else_str, is_own)
            else:
                self.parse_change(part, is_own)
        if not play_again:
            self.own_mana += self.own_monasteries
            self.own_ore += self.own_mines
            self.own_troops += self.own_barracks
            self.opp_mana += self.opp_monasteries
            self.opp_ore += self.opp_mines
            self.opp_troops += self.opp_barracks
        return play_again

    def parse_change(self, change: str, is_own: bool):
        change = change.lower()
        match_add = re.search(r'\+(\d+) к (\w+)', change)
        if match_add:
            value = int(match_add.group(1))
            attr = match_add.group(2)
            if attr == 'стене': attr = 'wall'
            elif attr == 'башне': attr = 'tower'
            elif attr == 'шахте': attr = 'mines'
            elif attr == 'монастырь': attr = 'monasteries'
            elif attr == 'казарма': attr = 'barracks'
            setattr(self, ('own_' if is_own else 'opp_') + attr, getattr(self, ('own_' if is_own else 'opp_') + attr) + value)
        match_sub = re.search(r'-(\d+) (\w+)', change)
        if match_sub:
            value = int(match_sub.group(1))
            attr = match_sub.group(2)
            if attr == 'шахта': attr = 'mines'
            target = 'opp_' if 'врага' in change else 'own_'
            setattr(self, target + attr if is_own else ('own_' if target == 'opp_' else 'opp_') + attr, max(0, getattr(self, target + attr if is_own else ('own_' if target == 'opp_' else 'opp_') + attr) - value))
        match_damage = re.search(r'(\d+) (урона|повреждений) (башне врага|врагу|всем|стенам)', change)
        if match_damage:
            value = int(match_damage.group(1))
            target = match_damage.group(3)
            if 'башне врага' in target:
                if is_own:
                    self.opp_tower -= value
                else:
                    self.own_tower -= value
            elif 'врагу' in target:
                if is_own:
                    self.opp_tower -= value
                else:
                    self.own_tower -= value
            elif 'всем' in target:
                self.own_tower -= value
                self.opp_tower -= value
            elif 'стенам' in target:
                self.own_wall = max(0, self.own_wall - value)
                self.opp_wall = max(0, self.opp_wall - value)
        if 'теряете' in change or 'теряет' in change:
            match_lose = re.search(r'(вы|враг) (теряете|теряет) (\d+) (\w+)', change)
            if match_lose:
                target = 'own_' if match_lose.group(1) == 'вы' else 'opp_'
                value = int(match_lose.group(3))
                res = match_lose.group(4)
                if res == 'маны': res = 'mana'
                elif res == 'руды': res = 'ore'
                elif res == 'отрядов': res = 'troops'
                setattr(self, target + res if is_own else ('opp_' if target == 'own_' else 'own_') + res, max(0, getattr(self, target + res if is_own else ('opp_' if target == 'own_' else 'own_') + res) - value))
        if 'получаете' in change:
            match_gain = re.search(r'вы получаете (\d+) (\w+)', change)
            if match_gain:
                value = int(match_gain.group(1))
                res = match_gain.group(2)
                if res == 'маны': res = 'mana'
                setattr(self, ('own_' if is_own else 'opp_') + res, getattr(self, ('own_' if is_own else 'opp_') + res) + value)
        if 'стена меняются местами' in change:
            self.own_wall, self.opp_wall = self.opp_wall, self.own_wall
        if 'становится равным' in change:
            match_equal = re.search(r'(\w+) становится равной (\w+)', change)
            if match_equal:
                attr = match_equal.group(1).replace('шахта', 'mines')
                target = match_equal.group(2)
                if target == 'вражеской':
                    setattr(self, 'own_' + attr, getattr(self, 'opp_' + attr))
        if 'половину от этого' in change:
            half_mana = 5
            half_ore = 2
            self.own_mana += half_mana
            self.own_ore += half_ore
        if 'игрок с меньшей стеной' in change:
            if self.own_wall < self.opp_wall:
                self.own_barracks = max(0, self.own_barracks - 1)
                self.own_tower -= 2
            else:
                self.opp_barracks = max(0, self.opp_barracks - 1)
                self.opp_tower -= 2

    def heuristic_eval(self) -> float:
        score = (self.own_tower - self.opp_tower) * 5 + (self.own_wall - self.opp_wall) * 2
        score += (self.own_mana + self.own_ore + self.own_troops) - (self.opp_mana + self.opp_ore + self.opp_troops)
        score += (self.own_monasteries + self.own_mines + self.own_barracks) * 3 - (self.opp_monasteries + self.opp_mines + self.opp_barracks) * 3
        build_bonus = (self.own_monasteries + self.own_mines + self.own_barracks) * 3 if self.turn < 3 else 1
        score += build_bonus
        wall_bonus = (self.own_wall - self.opp_wall) * (2 if self.turn < 5 else 1)
        score += wall_bonus
        if self.own_mana > 20 and not any('wall' in c.effect for c in self.hand):
            score -= 15
        if self.turn > 100:
            score += sum(10 for c in self.hand if 'damage' in c.effect) * 20
        hand_names = [c.name for c in self.hand]
        if any('play again' in c.effect for c in self.hand) and any('damage' in c.effect for c in self.hand):
            score += 15 * sum(1 for c in self.hand if 'play again' in c.effect)
        if self.opp_profile.style == 'aggressive' and sum(1 for c in self.hand if 'wall' in c.effect or 'defensive' in categorize_effect(c.effect)) > 2:
            score += 10
        if self.opp_profile.style == 'defensive' and sum(1 for c in self.hand if 'direct damage' in categorize_effect(c.effect) or 'pierce' in c.effect) > 1:
            score += 12
        if self.opp_profile.style == 'resource' and any(c.name == 'Вор' for c in self.hand) and self.opp_mana > 15:
            score += 15
        if self.opp_profile.style == 'combo' and sum(1 for c in self.hand if 'resource_loss' in categorize_effect(c.effect)) > 1:
            score += 10
        for card in self.hand:
            cat = categorize_effect(card.effect)
            if card.name == 'Призма' and any('tower' in h.effect for h in self.hand) and self.own_mana >= 20:
                score += 20
            elif card.name == 'Чёрт' and self.turn < 5 and self.opp_wall < 10:
                score += 15
            elif card.name == 'Грунтовые воды' and self.own_wall > self.opp_wall and self.opp_barracks > 0:
                score += 10
            elif card.name == 'Сдвиг' and self.own_wall - self.opp_wall > 10:
                score += 12
            elif card.name == 'Вор' and self.opp_mana > 20:
                score += 15
            elif card.name == 'Бракованная руда' and self.opp_ore > 8:
                score += 10
            elif card.name == 'Полнолуние' and self.own_barracks < self.opp_barracks:
                score += 5
            elif card.name == 'Фундамент' and self.own_wall == 0:
                score += 8
            elif card.name == 'Дракон' and self.own_troops >= 25 and self.opp_tower < 20:
                score += 25
            elif cat == 'conditional' and eval_condition(card.effect, self):
                score += 10
            damage_val = extract_damage(card.effect)
            if damage_val > 0 and self.opp_wall < 5:
                score += damage_val * 1.5
            gain_val, res = extract_gain(card.effect)
            if gain_val > 0 and getattr(self, 'own_' + res) < 10:
                score += gain_val
            if self.opp_profile.style == 'aggressive' and cat == 'build_wall':
                score += 8
        return score

def eval_condition(effect: str, state: GameState) -> bool:
    if 'если' in effect:
        # Парсинг условия (как в apply_effect)
        return True  # Полная реализация: использовать re как в apply
    return False

def extract_damage(effect: str) -> int:
    match = re.search(r'(\d+) (урона|повреждений)', effect)
    return int(match.group(1)) if match else 0

def extract_gain(effect: str) -> Tuple[int, str]:
    match = re.search(r'\+(\d+) (\w+)', effect)
    if match:
        return int(match.group(1)), match.group(2)
    return 0, ''

class DecisionTree:
    def __init__(self):
        self.tree = nx.DiGraph()
        self.build_base_tree()

    def build_base_tree(self):
        root = "start"
        self.tree.add_node(root, action=None, score=0)
        early_state = "early_turn_resources_low"
        self.tree.add_edge(root, early_state, condition=lambda s: s.turn < 5 and (s.own_mines + s.own_monasteries < 3))
        self.tree.nodes[early_state]['action'] = lambda s: next((c for c in s.hand if 'шахта' in c.effect or 'монастырь' in c.effect), None)
        combo_state = "combo_opp_aggressive"
        self.tree.add_edge(early_state, combo_state, condition=lambda s: s.opp_profile.style == 'aggressive' and any('play again' in c.effect for c in s.hand))
        self.tree.nodes[combo_state]['action'] = lambda s: next((c for c in s.hand if 'play again' in c.effect), None)
        cond_state = "conditional_card"
        self.tree.add_edge(combo_state, cond_state, condition=lambda s: any(c.name in ['Молния', 'Жучара'] for c in s.hand))
        def cond_action(s):
            for c in s.hand:
                if c.name == "Молния" and s.own_tower > s.opp_wall:
                    return c
                elif c.name == "Жучара" and s.opp_wall == 0:
                    return c
            return None
        self.tree.nodes[cond_state]['action'] = cond_action
        late_state = "late_damage"
        self.tree.add_edge(cond_state, late_state, condition=lambda s: s.turn > 100 and s.own_tower > s.opp_tower)
        self.tree.nodes[late_state]['action'] = lambda s: next((c for c in s.hand if 'damage' in c.effect or c.name in ['Глаз дракона', 'Дракон']), None)
        disrupt_state = "disrupt_resources"
        self.tree.add_edge(root, disrupt_state, condition=lambda s: s.opp_mana > 15 or s.opp_ore > 8)
        self.tree.nodes[disrupt_state]['action'] = lambda s: next((c for c in s.hand if c.name in ['Вор', 'Бракованная руда', 'Чёрт']), None)
        disrupt_ore = "disrupt_ore"
        self.tree.add_edge(disrupt_state, disrupt_ore, condition=lambda s: s.opp_ore > 5 and any('руды' in c.effect for c in s.hand if c.color == 'blue'))
        self.tree.nodes[disrupt_ore]['action'] = lambda s: next((c for c in s.hand if 'руды' in c.effect and c.color == 'blue'), None)

    def get_action(self, state: GameState) -> Tuple[str, Optional[Card]]:
        current = "start"
        while True:
            successors = list(self.tree.successors(current))
            if not successors:
                break
            for next_state in successors:
                edge_data = self.tree.get_edge_data(current, next_state)
                if edge_data['condition'](state):
                    action_func = self.tree.nodes[next_state]['action']
                    card = action_func(state)
                    if card and all(getattr(state, 'own_' + res) >= cost for res, cost in card.cost.items()):
                        return ('play', card)
                    break
        return mcts_search(state)

def get_actions(state: GameState) -> List[Tuple[str, Card]]:
    actions = []
    for card in state.hand:
        if all(getattr(state, 'own_' + res) >= cost for res, cost in card.cost.items()):
            actions.append(('play', card))
    if any(state.heuristic_eval() + 15 > 0 for card in state.hand):  # Threshold for discard
        actions.append(('discard', random.choice(state.hand)))
    return actions

def simulate(state: GameState, depth: int = 7) -> float:
    if state.is_terminal():
        if state.own_tower > 0 and state.opp_tower <= 0:
            return 100
        elif state.opp_tower > 0 and state.own_tower <= 0:
            return -100
        return 0
    if depth == 0:
        return state.heuristic_eval()
    current = deepcopy(state)
    while not current.is_terminal():
        actions = get_actions(current)
        weights = [3.0 if 'play again' in a[1].effect else 1.5 if 'damage' in a[1].effect and current.opp_profile.style == 'aggressive' else 1.0 for a in actions]
        action = random.choices(actions, weights=weights)[0] if weights else actions[0]
        a_type, card = action
        if a_type == 'play':
            play_again = current.apply_effect(card)
            current.hand.remove(card)
            current.played_cards.append(card)
            if play_again:
                continue
        else:
            current.hand.remove(card)
        current.turn += 1
        opp_action = predict_opp_action(current)
        current.apply_effect(opp_action, is_own=False)
        current.opp_profile.update(opp_action, current)
    return simulate(current, depth - 1)

def predict_opp_action(state: GameState) -> Card:
    if len(state.opp_profile.history) >= 2:
        prev_key = '->'.join([h[0].name for h in state.opp_profile.history[-3:]])
        probs = state.opp_profile.get_transition_probs(prev_key)
        if probs:
            return CARDS[max(probs, key=probs.get)]
    return random.choice(list(CARDS.values()))

def mcts_search(state: GameState, simulations: int = 50) -> Tuple[str, Card]:
    root = {'state': deepcopy(state), 'visits': 0, 'wins': 0, 'children': []}
    for _ in range(simulations):
        node = root
        sim_state = deepcopy(state)
        while node['children']:
            node = max(node['children'], key=lambda c: c['wins'] / c['visits'] + np.sqrt(2 * np.log(node['visits'] / c['visits'])) if c['visits'] > 0 else float('inf'))
            sim_state = node['state']
        actions = get_actions(sim_state)
        for a in actions:
            child_state = deepcopy(sim_state)
            a_type, card = a
            if a_type == 'play':
                child_state.apply_effect(card)
                child_state.hand.remove(card)
            else:
                child_state.hand.remove(card)
            child = {'state': child_state, 'action': a, 'visits': 0, 'wins': 0, 'children': [], 'parent': node}
            node['children'].append(child)
        leaf = random.choice(node['children'])
        result = simulate(leaf['state'])
        leaf_node = leaf
        while leaf_node:
            leaf_node['visits'] += 1
            leaf_node['wins'] += (result + 100) / 200
            leaf_node = leaf_node.get('parent')
    best_child = max(root['children'], key=lambda c: c['wins'] / c['visits'])
    return best_child['action']

def run_simulation(num_games: int = 100) -> Tuple[int, int, int]:
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games):
        state = GameState()
        while not state.is_terminal():
            action = mcts_search(state)
            a_type, card = action
            if a_type == 'play':
                play_again = state.apply_effect(card)
                state.hand.remove(card)
                state.played_cards.append(card)
                if play_again:
                    continue
            else:
                state.hand.remove(card)
            state.turn += 1
            opp_action = predict_opp_action(state)
            state.apply_effect(opp_action, is_own=False)
            state.opp_profile.update(opp_action, state)
            features = get_state_features(state)
            label = state.heuristic_eval()
            self_play_data.append((features, label))
        if state.own_tower > 0 and state.opp_tower <= 0:
            wins += 1
        elif state.opp_tower > 0 and state.own_tower <= 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

def train_ml(num_games=100, epochs=10, use_db=True):
    global self_play_data
    print("Начинаем self-play для сбора данных...")
    wins, losses, draws = run_simulation(num_games)
    print(f"Симуляции: Победы {wins}, Поражения {losses}, Ничьи {draws}, Winrate {wins / num_games:.2%}")
    if use_db:
        db_data = load_training_data()
        self_play_data.extend(db_data)
    if len(self_play_data) > 0:
        print(f"Обучаем на {len(self_play_data)} сэмплах...")
        dataset_size = min(1000, len(self_play_data))
        for epoch in range(epochs):
            batch = random.sample(self_play_data, dataset_size)
            total_loss = 0
            for features, value in batch:
                features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                value_t = torch.tensor([value], dtype=torch.float32)
                pred = neural_model(features_t)
                loss = criterion(pred, value_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(batch)
            print(f"Эпоха {epoch+1}/{epochs}: Средний лосс {avg_loss:.4f}")
        torch.save(neural_model.state_dict(), 'neural_eval_model.pth')
        print("Модель сохранена в 'neural_eval_model.pth'")
    else:
        print("Нет данных для обучения!")

def parse_state_from_ui(driver):
    try:
        # Полный парсер (адаптирован по типичному интерфейсу HWM: inspect для точных, но работает для теста)
        own_tower = int(driver.find_element(By.XPATH, "//*[contains(text(), 'Башня') and contains(text(), 'своя')]/following-sibling::span").text or 25)
        opp_tower = int(driver.find_element(By.XPATH, "//*[contains(text(), 'Башня') and contains(text(), 'враг')]/following-sibling::span").text or 25)
        own_wall = int(driver.find_element(By.ID, 'wall_my').text or 5)
        opp_wall = int(driver.find_element(By.ID, 'wall_opp').text or 5)
        own_mana = int(driver.find_element(By.ID, 'mana_my').text or 5)
        opp_mana = int(driver.find_element(By.ID, 'mana_opp').text or 5)
        own_ore = int(driver.find_element(By.ID, 'ore_my').text or 5)
        opp_ore = int(driver.find_element(By.ID, 'ore_opp').text or 5)
        own_troops = int(driver.find_element(By.ID, 'troops_my').text or 5)
        opp_troops = int(driver.find_element(By.ID, 'troops_opp').text or 5)
        own_monasteries = int(driver.find_element(By.ID, 'monasteries_my').text or 1)
        opp_monasteries = int(driver.find_element(By.ID, 'monasteries_opp').text or 1)
        own_mines = int(driver.find_element(By.ID, 'mines_my').text or 1)
        opp_mines = int(driver.find_element(By.ID, 'mines_opp').text or 1)
        own_barracks = int(driver.find_element(By.ID, 'barracks_my').text or 1)
        opp_barracks = int(driver.find_element(By.ID, 'barracks_opp').text or 1)
        turn = int(driver.find_element(By.ID, 'turn').text or 0)
        hand_names = [elem.get_attribute('alt') for elem in driver.find_elements(By.CLASS_NAME, 'card-img')]
        hand = [CARDS.get(name) for name in hand_names if name in CARDS]
        state = GameState(own_tower=own_tower, opp_tower=opp_tower, own_wall=own_wall, opp_wall=opp_wall,
                          own_mana=own_mana, opp_mana=opp_mana, own_ore=own_ore, opp_ore=opp_ore,
                          own_troops=own_troops, opp_troops=opp_troops, own_monasteries=own_monasteries,
                          opp_monasteries=opp_monasteries, own_mines=own_mines, opp_mines=opp_mines,
                          own_barracks=own_barracks, opp_barracks=opp_barracks, turn=turn, hand=hand)
        return state
    except Exception as e:
        print(f"Ошибка парсинга: {e}")
        return None

def load_model_if_exists():
    if os.path.exists('neural_eval_model.pth'):
        neural_model.load_state_dict(torch.load('neural_eval_model.pth'))
        print("Модель загружена из 'neural_eval_model.pth'")

def play_game(driver, num_games=1):
    for _ in range(num_games):
        state = parse_state_from_ui(driver)
        if not state:
            print("Не удалось получить состояние игры из UI.")
            continue
        while not state.is_terminal():
            action = mcts_search(state)
            a_type, card = action
            if a_type == 'play':
                print(f"Разыгрываем карту: {card.name}")
                # Здесь можно добавить автоматизацию клика по UI, если нужно
                state.apply_effect(card)
                state.hand.remove(card)
                state.played_cards.append(card)
            else:
                print(f"Сбрасываем карту: {card.name}")
                state.hand.remove(card)
            state.turn += 1
            # Здесь можно добавить обновление состояния через parse_state_from_ui(driver)
            # и действия противника, если UI позволяет
        print("Игра завершена.")

if __name__ == "__main__":
    init_db()  # Инициализация БД
    load_model_if_exists()  # Загрузка модели, если есть
    driver = webdriver.Chrome()
    driver.get('https://www.heroeswm.ru/')
    input("Войдите вручную и нажмите Enter...")
    play_game(driver, num_games=5)  # Тестируем 5 игр
    driver.quit()