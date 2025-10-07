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
import tkinter as tk
from tkinter import StringVar, Menu, messagebox
from tkinter import ttk
import json
import os

CONFIG_FILE = 'config.json'

def init_driver():
    """Инициализация Selenium драйвера."""
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-blink-features=AutomationControlled')  # Скрываем автоматизацию
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def login(driver, login: str, password: str):
    """Авторизация на heroeswm.ru."""
    try:
        driver.get('https://www.heroeswm.ru/')
        wait = WebDriverWait(driver, 10)
        
        # Ждем загрузки формы логина
        login_field = wait.until(EC.presence_of_element_located((By.NAME, "login")))
        pass_field = wait.until(EC.presence_of_element_located((By.NAME, "pass")))
        
        # Очищаем поля и вводим данные
        login_field.clear()
        login_field.send_keys(login)
        pass_field.clear()
        pass_field.send_keys(password)
        
        # Находим кнопку входа
        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']")))
        login_button.click()
        
        # Проверяем успешность входа
        try:
            # Ждем либо перехода на home.php, либо появления сообщения об ошибке
            wait.until(lambda driver: 
                'home.php' in driver.current_url or 
                len(driver.find_elements(By.CLASS_NAME, "error-message")) > 0
            )
            
            # Проверяем, вошли ли мы успешно
            if 'home.php' in driver.current_url:
                print("Авторизация успешна")
                return True
            else:
                print("Неверные учетные данные")
                return False
                
        except Exception as e:
            print(f"Ошибка при проверке входа: {e}")
            return False
            
    except Exception as e:
        print(f"Ошибка авторизации: {e}")
        return False

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
    """Парсинг состояния игры."""
    try:
        # Ждем загрузку игрового поля
        game_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "game-field"))
        )
        
        # Получаем состояние башен
        own_tower = int(driver.find_element(By.CSS_SELECTOR, ".my-tower .health").text)
        opp_tower = int(driver.find_element(By.CSS_SELECTOR, ".enemy-tower .health").text)
        
        # Получаем карты в руке
        hand_cards = []
        card_elements = driver.find_elements(By.CSS_SELECTOR, ".hand-cards .card")
        for card_el in card_elements:
            card_name = card_el.get_attribute("data-card")
            if card_name in CARDS:
                hand_cards.append(CARDS[card_name])
        
        # Создаем состояние игры
        state = GameState()
        state.own_tower = own_tower
        state.opp_tower = opp_tower
        state.hand = hand_cards
        
        return state
    except Exception as e:
        print(f"Ошибка парсинга состояния: {e}")
        return None

def select_game(driver, game_id: str):
    """Выбор игры в таверне."""
    try:
        driver.find_element(By.ID, f'game-{game_id}').click()
        WebDriverWait(driver, 10).until(EC.url_contains('tavern_game.php'))
        print(f"Игра {game_id} выбрана")
    except Exception as e:
        print(f"Ошибка выбора игры: {e}")
        raise

def play_card(driver, card):
    """Разыгрывание карты."""
    try:
        # Ждем возможности хода
        WebDriverWait(driver, 30).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f".card[data-card='{card.name}']"))
        )
        
        # Находим и кликаем по карте
        card_element = driver.find_element(By.CSS_SELECTOR, f".card[data-card='{card.name}']")
        driver.execute_script("arguments[0].click();", card_element)
        
        # Ждем анимации
        time.sleep(1.5)
        
        # Проверяем, что карта сыграна
        try:
            card_element = driver.find_element(By.CSS_SELECTOR, f".card[data-card='{card.name}']")
            return False  # Карта все еще в руке
        except:
            return True  # Карта успешно сыграна
            
    except Exception as e:
        print(f"Ошибка при игре картой {card.name}: {e}")
        return False

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

def save_credentials(login: str, password: str):
    """Сохранение учетных данных."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'login': login, 'password': password}, f)

def load_credentials() -> tuple:
    """Загрузка сохраненных учетных данных."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return data.get('login', ''), data.get('password', '')
    except:
        return '', ''

def create_context_menu(entry, root):
    """Создание контекстного меню для полей ввода."""
    menu = Menu(entry, tearoff=0)
    
    def paste():
        try:
            clipboard = root.clipboard_get()
            entry.delete(0, tk.END)
            entry.insert(0, clipboard)
        except Exception as e:
            print(f"Paste error: {e}")
            
    def copy():
        try:
            if entry.selection_present():
                root.clipboard_clear()
                root.clipboard_append(entry.selection_get())
        except Exception as e:
            print(f"Copy error: {e}")
            
    def cut():
        try:
            if entry.selection_present():
                root.clipboard_clear()
                root.clipboard_append(entry.selection_get())
                entry.delete("sel.first", "sel.last")
        except Exception as e:
            print(f"Cut error: {e}")

    menu.add_command(label="Вставить", command=paste)
    menu.add_command(label="Копировать", command=copy)
    menu.add_command(label="Вырезать", command=cut)
    
    # Биндим правую кнопку мыши
    entry.bind('<Button-3>', lambda e: menu.tk_popup(e.x_root, e.y_root))
    
    # Добавляем стандартные биндинги с явным вызовом функций
    entry.bind('<Control-v>', lambda e: paste())
    entry.bind('<Control-c>', lambda e: copy())
    entry.bind('<Control-x>', lambda e: cut())
    
    # Добавляем биндинги для стандартных событий
    entry.bind('<<Paste>>', lambda e: paste())
    entry.bind('<<Copy>>', lambda e: copy())
    entry.bind('<<Cut>>', lambda e: cut())

def start_gui():
    """Запуск GUI для авторизации и выбора игр."""
    root = tk.Tk()
    root.title("HWM Две Башни ИИ")
    root.geometry("600x400")
    
    # Стили для интерфейса
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)
    style.configure("TLabel", font=("Arial", 12))
    style.configure("TEntry", font=("Arial", 12))
    
    # Фрейм для авторизации
    login_frame = ttk.Frame(root, padding="10")
    login_frame.pack(fill="both", expand=True)
    
    # Переменные для хранения логина и пароля
    login_var = StringVar()
    pass_var = StringVar()
    
    # Загрузка сохраненных данных
    saved_login, saved_pass = load_credentials()
    login_var.set(saved_login)
    pass_var.set(saved_pass)
    
    ttk.Label(login_frame, text="Логин:").pack(pady=5)
    login_entry = ttk.Entry(login_frame, textvariable=login_var)
    login_entry.pack(pady=5)
    
    ttk.Label(login_frame, text="Пароль:").pack(pady=5)
    pass_entry = ttk.Entry(login_frame, show="*", textvariable=pass_var)
    pass_entry.pack(pady=5)
    
    # Включаем стандартные операции для полей ввода
    for entry in (login_entry, pass_entry):
        entry.bind('<Control-a>', lambda e: e.widget.select_range(0, tk.END))
        entry.bind('<Control-v>', lambda e: e.widget.event_generate('<<Paste>>'))
        entry.bind('<Control-c>', lambda e: e.widget.event_generate('<<Copy>>'))
        entry.bind('<Control-x>', lambda e: e.widget.event_generate('<<Cut>>'))
        create_context_menu(entry, root)
    
    # Чекбокс для сохранения данных
    save_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(login_frame, text="Сохранить данные", variable=save_var).pack(pady=5)
    
    driver = None
    
    def do_login():
        nonlocal driver
        try:
            driver = init_driver()
            login_result = login(driver, login_var.get(), pass_var.get())
            
            if login_result:
                if save_var.get():
                    save_credentials(login_var.get(), pass_var.get())
                
                # Скрываем окно логина
                login_frame.pack_forget()
                
                # Создаем новый фрейм для игр
                show_games(driver)
            else:
                messagebox.showerror("Ошибка", "Неверный логин или пароль")
                if driver:
                    driver.quit()
                    driver = None
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка авторизации: {e}")
            if driver:
                driver.quit()
                driver = None

    def show_games(driver):
        """Отображение списка активных игр."""
        games_frame = ttk.Frame(root, padding="10")
        games_frame.pack(fill="both", expand=True)
        
        ttk.Label(games_frame, text="Доступные игры в таверне").pack(pady=5)
        
        # Создаем список игр
        games_list = tk.Listbox(games_frame, height=10)
        games_list.pack(fill="both", expand=True, pady=5)
        
        def update_games():
            try:
                games = get_active_games(driver)
                games_list.delete(0, tk.END)
                for game in games:
                    games_list.insert(tk.END, f"{game['id']}: {game['opponent']} ({game['status']})")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось получить список игр: {e}")
        
        # Кнопки управления
        ttk.Button(games_frame, text="Обновить список", command=update_games).pack(pady=5)
        ttk.Button(games_frame, text="Выход", command=on_closing).pack(pady=5)
        
        # Первоначальное обновление списка
        update_games()
    
    ttk.Button(login_frame, text="Войти", command=do_login).pack(pady=10)
    
    def on_closing():
        if driver:
            driver.quit()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # ...existing code for show_games...
