import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from game_state import GameState
from cards import CARDS
from mcts import mcts_search
from evaluation import get_state_features, neural_model

def init_driver():
    """Инициализация Selenium драйвера."""
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(options=options)
    return driver

def login(driver, login: str, password: str):
    """Авторизация на heroeswm.ru."""
    driver.get('https://www.heroeswm.ru/')
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'login')))
        driver.find_element(By.NAME, 'login').send_keys(login)
        driver.find_element(By.NAME, 'pass').send_keys(password)
        driver.find_element(By.NAME, 'plogin').click()
        WebDriverWait(driver, 10).until(EC.url_contains('home.php'))
        print("Авторизация успешна")
    except Exception as e:
        print(f"Ошибка авторизации: {e}")
        raise

def get_active_games(driver):
    """Получение списка активных игр в таверне."""
    driver.get('https://www.heroeswm.ru/tavern.php')
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'tavern-row')))
    games = []
    elements = driver.find_elements(By.CLASS_NAME, 'tavern-row')
    for el in elements:
        try:
            game_id = el.get_attribute('data-id') or f"game_{len(games)}"
            opponent = el.find_element(By.CLASS_NAME, 'opponent').text
            status = 'active' if 'playing' in el.text.lower() else 'waiting'
            games.append({'id': game_id, 'opponent': opponent, 'status': status})
        except:
            continue
    return games

def parse_state_from_ui(driver):
    """Парсинг текущего состояния игры из UI."""
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'game-state')))
    state = GameState()
    
    # Парсинг ресурсов и зданий (адаптировано под структуру сайта)
    state.own_tower = int(driver.find_element(By.ID, 'own_tower').text)
    state.opp_tower = int(driver.find_element(By.ID, 'opp_tower').text)
    state.own_wall = int(driver.find_element(By.ID, 'own_wall').text)
    state.opp_wall = int(driver.find_element(By.ID, 'opp_wall').text)
    state.own_mana = int(driver.find_element(By.ID, 'own_mana').text)
    state.opp_mana = int(driver.find_element(By.ID, 'opp_mana').text)
    state.own_ore = int(driver.find_element(By.ID, 'own_ore').text)
    state.opp_ore = int(driver.find_element(By.ID, 'opp_ore').text)
    state.own_troops = int(driver.find_element(By.ID, 'own_troops').text)
    state.opp_troops = int(driver.find_element(By.ID, 'opp_troops').text)
    state.own_monasteries = int(driver.find_element(By.ID, 'own_monasteries').text)
    state.opp_monasteries = int(driver.find_element(By.ID, 'opp_monasteries').text)
    state.own_mines = int(driver.find_element(By.ID, 'own_mines').text)
    state.opp_mines = int(driver.find_element(By.ID, 'opp_mines').text)
    state.own_barracks = int(driver.find_element(By.ID, 'own_barracks').text)
    state.opp_barracks = int(driver.find_element(By.ID, 'opp_barracks').text)
    
    # Парсинг руки
    hand_elements = driver.find_elements(By.CLASS_NAME, 'card')
    state.hand = [CARDS[el.get_attribute('data-name')] for el in hand_elements if el.get_attribute('data-name') in CARDS]
    
    # Парсинг сыгранных карт (история)
    played_cards = driver.find_elements(By.CLASS_NAME, 'played-card')
    state.played_cards = [CARDS[el.get_attribute('data-name')] for el in played_cards if el.get_attribute('data-name') in CARDS]
    
    return state

def select_game(driver, game_id: str):
    """Выбор игры в таверне."""
    try:
        driver.find_element(By.ID, f'game-{game_id}').click()
        WebDriverWait(driver, 10).until(EC.url_contains('tavern_game.php'))
        print(f"Игра {game_id} выбрана")
    except Exception as e:
        print(f"Ошибка выбора игры: {e}")
        raise

def play_game(driver, num_games: int = 1):
    """Автоматическая игра в таверне."""
    for _ in range(num_games):
        games = get_active_games(driver)
        if not games:
            print("Нет доступных игр")
            return
        
        # Выбор первой доступной игры
        game_id = next((g['id'] for g in games if g['status'] == 'waiting'), games[0]['id'])
        select_game(driver, game_id)
        
        while True:
            try:
                state = parse_state_from_ui(driver)
                if state.is_terminal():
                    print("Игра завершена")
                    break
                
                action = mcts_search(state, simulations=1000)  # Увеличено для точности
                if action[0] == 'play':
                    try:
                        # Клик по карте (адаптировано под структуру сайта)
                        card_element = driver.find_element(By.XPATH, f"//div[@data-name='{action[1].name}']")
                        card_element.click()
                        WebDriverWait(driver, 5).until(EC.staleness_of(card_element))
                        print(f"Сыграна карта: {action[1].name}")
                        state.apply_effect(action[1])
                        state.hand.remove(action[1])
                        state.played_cards.append(action[1])
                    except Exception as e:
                        print(f"Ошибка при игре карты {action[1].name}: {e}")
                        # Fallback: Вывод действия в консоль
                        print(f"Рекомендация: Сыграть {action[1].name}")
                else:
                    try:
                        # Сброс карты
                        card_element = driver.find_element(By.XPATH, f"//div[@data-name='{action[1].name}']")
                        driver.find_element(By.XPATH, f"//div[@data-name='{action[1].name}']/button[@class='discard']").click()
                        WebDriverWait(driver, 5).until(EC.staleness_of(card_element))
                        print(f"Сброшена карта: {action[1].name}")
                        state.hand.remove(action[1])
                        if state.deck:
                            new_card = random.choice(state.deck)
                            state.hand.append(new_card)
                            state.deck.remove(new_card)
                    except Exception as e:
                        print(f"Ошибка при сбросе карты {action[1].name}: {e}")
                        print(f"Рекомендация: Сбросить {action[1].name}")
                
                # Ожидание хода оппонента
                time.sleep(2)  # Адаптируй под тайминг сайта
                opp_action = state.opp_profile.predict_next_card(state)
                state.apply_effect(opp_action, is_own=False)
                state.opp_profile.update(opp_action, state)
            except Exception as e:
                print(f"Ошибка в игре: {e}")
                break
