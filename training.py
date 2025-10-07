import random
import torch
import uuid
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
from game_state import GameState
from cards import CARDS
from mcts import mcts_search, simulate
from evaluation import NeuralEval, get_state_features, criterion, optimizer, self_play_data, neural_model
from db import save_game_data, save_game_outcome, load_training_data

# Initialize the neural network model
neural_model = NeuralEval()

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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(neural_model.parameters())
    
    if use_db:
        training_data = load_training_data()
    else:
        # Use default training data if DB is not available
        training_data = []  # Add some default training examples
    
    for epoch in range(epochs):
        total_loss = 0
        for state, action, reward in training_data:
            features = get_state_features(state)
            prediction = neural_model(features)
            target = torch.tensor([reward], dtype=torch.float32)
            
            loss = criterion(prediction, target)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(training_data)}")
    
    # Save the trained model
    torch.save(neural_model.state_dict(), 'neural_eval_model.pth')
