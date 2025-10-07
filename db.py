# db.py
import sqlite3
from typing import List, Dict, Any, Tuple
from game_state import GameState
from cards import Card, CARDS
from opponent_profile import OpponentProfile
import json

DB_FILE = 'hwm_games.db'

def init_db():
    """Инициализация базы данных."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS games
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  state TEXT,
                  result TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

def save_game_data(game_id: str, turn: int, state: GameState, action: Tuple[str, Card], reward: float):
    """Сохранение данных хода в базу."""
    state_dict = {
        'own_tower': state.own_tower, 'opp_tower': state.opp_tower,
        'own_wall': state.own_wall, 'opp_wall': state.opp_wall,
        'own_mana': state.own_mana, 'opp_mana': state.opp_mana,
        'own_ore': state.own_ore, 'opp_ore': state.opp_ore,
        'own_troops': state.own_troops, 'opp_troops': state.opp_troops,
        'own_monasteries': state.own_monasteries, 'opp_monasteries': state.opp_monasteries,
        'own_mines': state.own_mines, 'opp_mines': state.opp_mines,
        'own_barracks': state.own_barracks, 'opp_barracks': state.opp_barracks,
        'turn': state.turn,
        'hand': [c.name for c in state.hand],
        'played_cards': [c.name for c in state.played_cards],
        'opp_profile': state.opp_profile.__dict__
    }
    action_str = f"{action[0]}:{action[1].name}"
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO game_states (game_id, turn, state, action, reward)
            VALUES (?, ?, ?, ?, ?)
        ''', (game_id, turn, json.dumps(state_dict), action_str, reward))
        conn.commit()

def save_game_outcome(game_id: str, outcome: str, winrate: float):
    """Сохранение результата игры."""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO games (game_id, outcome, winrate)
            VALUES (?, ?, ?)
        ''', (game_id, outcome, winrate))
        conn.commit()

def load_training_data() -> list:
    """Загрузка данных для обучения."""
    data = []
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS game_states
                    (game_id TEXT, turn INTEGER, state TEXT, action TEXT, reward REAL)''')
        c.execute('SELECT state, action, reward FROM game_states')
        rows = c.fetchall()
        for row in rows:
            state_dict = json.loads(row[0])
            state = GameState()  # Create empty state first
            # Update attributes
            for key, value in state_dict.items():
                if key not in ['hand', 'played_cards', 'opp_profile']:
                    setattr(state, key, value)
            
            # Handle special cases
            state.hand = [CARDS[name] for name in state_dict['hand']]
            state.played_cards = [CARDS[name] for name in state_dict['played_cards']]
            
            action_parts = row[1].split(':')
            action = (action_parts[0], CARDS[action_parts[1]])
            data.append((state, action, row[2]))
    return data