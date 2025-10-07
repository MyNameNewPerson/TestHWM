
import random
import torch
import uuid
from typing import Tuple
from game_state import GameState
from cards import CARDS
from mcts import mcts_search, simulate
from evaluation import NeuralEval, get_state_features, criterion, optimizer, self_play_data

# Initialize the neural network model
neural_model = NeuralEval()
from db import save_game_data, save_game_outcome, load_training_data

def run_simulation(num_games: int = 100) -> Tuple[int, int, int]:
    """Запуск симуляций для сбора данных обучения."""
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        game_id = str(uuid.uuid4())
        state = GameState()
        turn = 0
        while not state.is_terminal():
            action = mcts_search(state, simulations=1000)  # Увеличено для точности
            reward = simulate(state.copy(), depth=7)
            save_game_data(game_id, turn, state, action, reward)
            
            # Применение действия ИИ
            if action[0] == 'play':
                play_again = state.apply_effect(action[1])
                state.hand.remove(action[1])
                state.played_cards.append(action[1])
                if not play_again:
                    # Симуляция хода оппонента
                    opp_action = state.opp_profile.predict_next_card(state)
                    state.apply_effect(opp_action, is_own=False)
                    state.opp_profile.update(opp_action, state)
            else:  # discard
                state.hand.remove(action[1])
                if state.deck:
                    new_card = random.choice(state.deck)
                    state.hand.append(new_card)
                    state.deck.remove(new_card)
                # Оппонент играет
                opp_action = state.opp_profile.predict_next_card(state)
                state.apply_effect(opp_action, is_own=False)
                state.opp_profile.update(opp_action, state)
            turn += 1
        
        # Определение результата
        if state.opp_tower <= 0:
            wins += 1
            outcome = 'win'
            winrate = 1.0
        elif state.own_tower <= 0:
            losses += 1
            outcome = 'loss'
            winrate = 0.0
        else:
            draws += 1
            outcome = 'draw'
            winrate = 0.5
        save_game_outcome(game_id, outcome, winrate)
    
    return wins, losses, draws

def train_ml(num_games: int = 100, epochs: int = 10, use_db: bool = True):
    """Обучение нейронной сети на симуляциях или данных из БД."""
    if use_db:
        data = load_training_data()
    else:
        data = []
        wins, losses, draws = run_simulation(num_games)
        data.extend(load_training_data())  # Дополняем новыми симуляциями
        print(f"Симуляции: Побед {wins}, Поражений {losses}, Ничьих {draws}, Winrate: {wins / (wins + losses + draws):.2%}")
    
    for epoch in range(epochs):
        total_loss = 0
        for state, action, reward in data:
            features = torch.tensor(get_state_features(state), dtype=torch.float32)
            if torch.cuda.is_available():
                features = features.cuda()
                neural_model.cuda()
            
            optimizer.zero_grad()
            pred = neural_model(features)
            target = torch.tensor([reward], dtype=torch.float32)
            if torch.cuda.is_available():
                target = target.cuda()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Эпоха {epoch + 1}/{epochs}, Средний лосс: {total_loss / len(data):.4f}")
    
    # Сохранение модели
    torch.save(neural_model.state_dict(), 'neural_eval_model.pth')
    print("Модель сохранена в neural_eval_model.pth")
