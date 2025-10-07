from bs4 import BeautifulSoup
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime

class HWMAPI:
    def __init__(self, session: requests.Session):
        self.session = session
        self.base_url = 'https://www.heroeswm.ru'
    
    def get_csrf_token(self, html: str) -> Optional[str]:
        """Получение CSRF токена со страницы."""
        soup = BeautifulSoup(html, 'html.parser')
        csrf_input = soup.find('input', {'name': 'csrf_token'})
        return csrf_input.get('value') if csrf_input else None
    
    def parse_tavern_games(self, html: str) -> List[Dict]:
        """Парсинг списка игр в таверне."""
        games = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Находим таблицу с играми
        game_table = soup.find('table', {'class': 'table-list'})
        if game_table:
            rows = game_table.find_all('tr')[1:]  # Пропускаем заголовок
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    game = {
                        'id': cols[0].text.strip(),
                        'opponent': cols[1].text.strip(),
                        'status': cols[2].text.strip(),
                        'link': cols[0].find('a')['href'] if cols[0].find('a') else None
                    }
                    games.append(game)
        
        return games

    def parse_game_state(self, html: str) -> dict:
        """Парсинг состояния игры."""
        soup = BeautifulSoup(html, 'html.parser')
        state = {
            'own_tower': 30,
            'opp_tower': 30,
            'own_wall': 0,
            'opp_wall': 0,
            'own_mana': 1,
            'opp_mana': 1,
            'own_ore': 0,
            'opp_ore': 0,
            'hand': []
        }
        
        # Парсим значения из HTML
        try:
            # Башни
            towers = soup.find_all('div', class_='tower')
            if len(towers) >= 2:
                state['own_tower'] = int(towers[0].find('span', class_='health').text)
                state['opp_tower'] = int(towers[1].find('span', class_='health').text)
            
            # Карты в руке
            hand_cards = soup.find_all('div', class_='card')
            state['hand'] = [card.get('data-card') for card in hand_cards if card.get('data-card')]
            
        except Exception as e:
            print(f"Error parsing game state: {e}")
        
        return state