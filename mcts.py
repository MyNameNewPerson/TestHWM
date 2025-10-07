
import random
import numpy as np
import torch
from typing import Tuple, Optional, List
from game_state import GameState
from cards import Card, CARDS
from evaluation import heuristic_eval, get_state_features, neural_model
from opponent_profile import OpponentProfile

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action: Optional[Tuple[str, Card]] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self) -> List[Tuple[str, Card]]:
        actions = []
        for card in self.state.hand:
            cost = card.cost
            if (self.state.own_mana >= cost['mana'] and
                self.state.own_ore >= cost['ore'] and
                self.state.own_troops >= cost['troops']):
                actions.append(('play', card))
            actions.append(('discard', card))
        return actions

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> 'MCTSNode':
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            if child.visits > 0 else float('inf') for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self) -> Tuple[str, Card]:
        legal_actions = self.get_legal_actions()
        return random.choice(legal_actions) if legal_actions else ('discard', random.choice(self.state.hand))

    def expand(self) -> 'MCTSNode':
        action = self.untried_actions.pop()
        next_state = self.state.copy()
        if action[0] == 'play':
            play_again = next_state.apply_effect(action[1])
            if not play_again:
                next_state.hand.remove(action[1])
                next_state.played_cards.append(action[1])
        else:  # discard
            next_state.hand.remove(action[1])
            if next_state.deck:
                new_card = random.choice(next_state.deck)
                next_state.hand.append(new_card)
                next_state.deck.remove(new_card)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

def simulate(state: GameState, depth: int = 7) -> float:
    """Симуляция игры до конца или максимальной глубины."""
    current_state = state.copy()
    current_depth = 0
    while not current_state.is_terminal() and current_depth < depth:
        legal_actions = [(a, c) for a, c in MCTSNode(current_state).get_legal_actions() if a == 'play']
        if not legal_actions:
            action = ('discard', random.choice(current_state.hand))
        else:
            # Улучшено: Приоритет комбо и контрудара по стилю оппонента
            combo_actions = [(a, c) for a, c in legal_actions if 'play again' in c.effect.lower() or 'если' in c.effect.lower()]
            if current_state.opp_profile.style == 'aggressive':
                defensive_actions = [(a, c) for a, c in legal_actions if 'стене' in c.effect.lower() or 'башне' in c.effect.lower()]
                action = random.choice(defensive_actions or combo_actions or legal_actions)
            elif current_state.opp_profile.style == 'defensive':
                direct_damage_actions = [(a, c) for a, c in legal_actions if 'урона башне врага' in c.effect.lower()]
                action = random.choice(direct_damage_actions or combo_actions or legal_actions)
            elif current_state.opp_profile.style == 'resource':
                resource_loss_actions = [(a, c) for a, c in legal_actions if 'теряет' in c.effect.lower()]
                action = random.choice(resource_loss_actions or combo_actions or legal_actions)
            else:
                action = random.choice(combo_actions or legal_actions)
        
        if action[0] == 'play':
            play_again = current_state.apply_effect(action[1])
            if not play_again:
                current_state.hand.remove(action[1])
                current_state.played_cards.append(action[1])
                # Симуляция хода оппонента
                opp_action = current_state.opp_profile.predict_next_card(current_state)
                current_state.apply_effect(opp_action, is_own=False)
        else:
            current_state.hand.remove(action[1])
            if current_state.deck:
                new_card = random.choice(current_state.deck)
                current_state.hand.append(new_card)
                current_state.deck.remove(new_card)
        current_depth += 1
    
    # Оценка: нейронная сеть или эвристика
    if torch.cuda.is_available():
        features = torch.tensor(get_state_features(current_state), dtype=torch.float32).cuda()
        neural_model.eval()
        with torch.no_grad():
            score = neural_model(features).item()
    else:
        score = heuristic_eval(current_state)
    
    # Усиление награды за победу
    if current_state.opp_tower <= 0:
        score += 100
    elif current_state.own_tower <= 0:
        score -= 100
    return score

def mcts_search(state: GameState, simulations: int = 1000) -> Tuple[str, Card]:
    """MCTS поиск лучшего хода."""
    root = MCTSNode(state)
    for _ in range(simulations):
        node = root
        # Выбор
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        
        # Расширение
        if not node.is_fully_expanded():
            node = node.expand()
        
        # Симуляция
        reward = simulate(node.state)
        
        # Обратное распространение
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    # Выбор лучшего действия
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action
