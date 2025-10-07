# game_state.py
import random
from typing import List, Optional, Tuple, TYPE_CHECKING
from copy import deepcopy
import re
from cards import Card, CARDS

if TYPE_CHECKING:
    from opponent_profile import OpponentProfile

class GameState:
    def __init__(self):
        self.own_tower = 30
        self.opp_tower = 30
        self.hand = []
        self.played_cards = []
        self.turn = 0
        self.deck = []
        self.own_wall = 0
        self.opp_wall = 0
        self.own_mana = 0
        self.opp_mana = 0
        self.own_ore = 0
        self.opp_ore = 0
        self.own_troops = 0
        self.opp_troops = 0
        self.own_mines = 0
        self.opp_mines = 0
        self.own_monasteries = 0
        self.opp_monasteries = 0
        self.own_barracks = 0
        self.opp_barracks = 0
        self._opp_profile = None

    @property
    def opp_profile(self):
        if self._opp_profile is None:
            from opponent_profile import OpponentProfile
            self._opp_profile = OpponentProfile()
        return self._opp_profile

    def is_terminal(self):
        """Проверка завершения игры."""
        return self.own_tower <= 0 or self.opp_tower <= 0

    def copy(self):
        return deepcopy(self)

    def is_terminal(self) -> bool:
        return self.own_tower <= 0 or self.opp_tower <= 0 or self.turn > 200 or (len(self.hand) == 0 and not self.deck)

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
                    cond_str = cond_match.group(2).strip().replace('меньше чем у врага', 'less than opp').replace('больше, чем у врага', 'greater than opp')
                    then_str = cond_match.group(3).strip()
                    else_str = cond_match.group(4).strip()
                    cond_parts = re.split(r'(=|>|<)', cond_str)
                    if len(cond_parts) >= 3:
                        attr_str = cond_parts[0].strip()
                        op = cond_parts[1]
                        val_str = cond_parts[2].strip()
                        attr = attr_str.replace('стена', 'wall').replace('шахта', 'mines').replace('башня', 'tower').replace('казарма', 'barracks').replace('монастырь', 'monasteries')
                        if 'opp' in val_str:
                            val = getattr(self, ('opp_' if is_own else 'own_') + attr.replace('less than ', '').replace('greater than ', ''))
                        else:
                            val = int(val_str)
                        current = getattr(self, ('own_' if is_own else 'opp_') + attr)
                        cond_met = (op == '=' and current == val) or (op == '>' and current > val) or (op == '<' and current < val)
                        if cond_met:
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
            if len(self.hand) < 5 and self.deck:  # Добавлено: Дополнение руки из колоды для реализма
                new_card = random.choice(self.deck)
                self.hand.append(new_card)
                self.deck.remove(new_card)
            elif not self.deck:  # Перемешивание колоды, если закончилась
                self.deck = [c for c in CARDS.values() if c not in self.hand and c not in self.played_cards]
                random.shuffle(self.deck)
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
            elif attr == 'отряда' or attr == 'отрядов': attr = 'troops'
            elif attr == 'маны': attr = 'mana'
            elif attr == 'руды': attr = 'ore'
            setattr(self, ('own_' if is_own else 'opp_') + attr, getattr(self, ('own_' if is_own else 'opp_') + attr) + value)
        match_sub = re.search(r'-(\d+) (\w+)', change)
        if match_sub:
            value = int(match_sub.group(1))
            attr = match_sub.group(2)
            if attr == 'шахта': attr = 'mines'
            elif attr == 'монастырь': attr = 'monasteries'
            elif attr == 'казарма': attr = 'barracks'
            target = 'opp_' if 'врага' in change else 'own_'
            setattr(self, target + attr if is_own else ('own_' if target == 'opp_' else 'opp_') + attr, max(0, getattr(self, target + attr if is_own else ('own_' if target == 'opp_' else 'opp_') + attr) - value))
        match_damage = re.search(r'(\d+) (урона|повреждений) (башне врага|врагу|всем|стенам|башне|собственной башне)', change)
        if match_damage:
            value = int(match_damage.group(1))
            target = match_damage.group(3)
            if 'башне врага' in target or 'врагу' in target:
                if is_own:
                    damage_to_opp = value - self.opp_wall
                    self.opp_wall = max(0, self.opp_wall - value)
                    if damage_to_opp > 0:
                        self.opp_tower -= damage_to_opp  # Добавлено: Учет стены в уроне (урон сначала стене, остаток башне)
                else:
                    damage_to_own = value - self.own_wall
                    self.own_wall = max(0, self.own_wall - value)
                    if damage_to_own > 0:
                        self.own_tower -= damage_to_own
            elif 'собственной башне' in target:
                if is_own:
                    self.own_tower -= value
                else:
                    self.opp_tower -= value
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
                elif res == 'руды': res = 'ore'
                elif res == 'отрядов': res = 'troops'
                setattr(self, ('own_' if is_own else 'opp_') + res, getattr(self, ('own_' if is_own else 'opp_') + res) + value)
        if 'стена меняются местами' in change:
            self.own_wall, self.opp_wall = self.opp_wall, self.own_wall
        if 'становится равным' in change:
            match_equal = re.search(r'(\w+) становится равной (\w+)', change)
            if match_equal:
                attr = match_equal.group(1).replace('шахта', 'mines').replace('монастырь', 'monasteries').replace('казарма', 'barracks')
                target = match_equal.group(2)
                if target == 'вражеской':
                    setattr(self, 'own_' + attr, getattr(self, 'opp_' + attr))
        if 'половину от этого' in change:
            half_mana = 5
            half_ore = 2.5  # Уточнено: Половину от 10 маны/5 руды = 5/2.5, но int для реализма
            self.own_mana += half_mana
            self.own_ore += int(half_ore)
        if 'игрок с меньшей стеной' in change:
            if self.own_wall < self.opp_wall:
                self.own_barracks = max(0, self.own_barracks - 1)
                self.own_tower -= 2
            else:
                self.opp_barracks = max(0, self.opp_barracks - 1)
                self.opp_tower -= 2