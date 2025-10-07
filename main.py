import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import torch
import os
import requests
import json
from datetime import datetime
from db import init_db
from evaluation import neural_model
from api import HWMAPI
from PIL import Image, ImageTk
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import random
import time

class HWMClient:
    def __init__(self):
        self.session = requests.Session()
        self.base_url = 'https://www.heroeswm.ru'
        self.api = HWMAPI(self.session)
        
        # Генерируем случайный User-Agent
        self.chrome_version = f"{random.randint(90, 120)}.0.{random.randint(1000, 9999)}.{random.randint(10, 99)}"
        
        # Добавляем больше реалистичных заголовков
        self.session.headers.update({
            'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{self.chrome_version} Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'sec-ch-ua': f'"Google Chrome";v="{self.chrome_version}", "Not=A?Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive'
        })

    def login(self, username: str, password: str) -> bool:
        try:
            print("\n=== Начало процесса входа ===")
            
            # Очищаем куки
            self.session.cookies.clear()
            
            # Получаем начальные куки
            self.session.headers.update({
                'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{self.chrome_version} Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive',
                'Cache-Control': 'max-age=0'
            })
            
            # Получаем главную страницу для куки
            init_response = self.session.get(self.base_url)
            print(f"Инициализация: статус={init_response.status_code}")
            
            # Формируем данные для входа напрямую
            login_data = {
                'login': username,
                'pass': password,
                'server_id': '1',
                'pliv': str(int(time.time())),  # Временная метка
                'LOGIN_redirect': '1',
                'do': 'login'
            }
            
            # Добавляем специальные заголовки для POST запроса
            post_headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Origin': self.base_url,
                'Referer': f'{self.base_url}/login.php',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # Отправляем POST запрос
            print("Отправка данных авторизации...")
            login_response = self.session.post(
                f'{self.base_url}/login.php',
                data=login_data,
                headers=post_headers,
                allow_redirects=False  # Отключаем автоматические редиректы
            )
            
            print(f"Ответ сервера: {login_response.status_code}")
            print(f"Заголовки ответа: {dict(login_response.headers)}")
            
            # Если есть редирект, следуем ему
            if login_response.status_code in (301, 302, 303):
                redirect_url = login_response.headers.get('Location')
                if redirect_url:
                    if not redirect_url.startswith('http'):
                        redirect_url = urljoin(self.base_url, redirect_url)
                    print(f"Переход по редиректу: {redirect_url}")
                    final_response = self.session.get(redirect_url)
                    print(f"Финальный статус: {final_response.status_code}")
                    print(f"Финальный URL: {final_response.url}")
                    
                    # Сохраняем ответ для анализа
                    with open('login_final.html', 'w', encoding='utf-8') as f:
                        f.write(final_response.text)
                        
                    return 'home.php' in final_response.url
            
            return False
            
        except Exception as e:
            print(f"Ошибка входа: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_tavern_games(self) -> list:
        try:
            response = self.session.get(f'{self.base_url}/tavern.php')
            return self.api.parse_tavern_games(response.text)
        except Exception as e:
            print(f"Get games error: {e}")
            return []

def start_gui():
    root = tk.Tk()
    root.title("HWM Две Башни")
    root.geometry("800x600")

    hwm_client = HWMClient()
    
    # Основной фрейм
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)
    
    # Текстовое поле для логов
    log_text = scrolledtext.ScrolledText(main_frame, height=10)
    log_text.pack(fill="both", expand=True, pady=5)
    
    def log(message: str):
        log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        log_text.see(tk.END)
    
    # Функции для работы с буфером обмена
    def paste_login():
        try:
            clipboard_text = root.clipboard_get()
            login_var.set(clipboard_text)
            log("Логин вставлен из буфера обмена")
        except Exception as e:
            log(f"Ошибка вставки логина: {e}")

    def paste_password():
        try:
            clipboard_text = root.clipboard_get()
            pass_var.set(clipboard_text)
            log("Пароль вставлен из буфера обмена")
        except Exception as e:
            log(f"Ошибка вставки пароля: {e}")

    # Модифицируем фрейм авторизации
    login_frame = ttk.LabelFrame(main_frame, text="Авторизация", padding="5")
    login_frame.pack(fill="x", pady=5)
    
    # Создаем фрейм для логина с кнопкой вставки
    login_container = ttk.Frame(login_frame)
    login_container.pack(side="left", padx=5)
    
    # Логин с кнопкой вставки и предустановленным значением
    ttk.Label(login_container, text="Логин:").pack(side="left")
    login_var = tk.StringVar(value="Gruzdik")  # Предустановленное значение
    login_entry = ttk.Entry(login_container, textvariable=login_var)
    login_entry.pack(side="left", padx=5)
    
    # Пароль с кнопкой вставки и предустановленным значением
    ttk.Label(login_container, text="Пароль:").pack(side="left", padx=(10,0))
    pass_var = tk.StringVar(value="A19923003a$")  # Предустановленное значение
    pass_entry = ttk.Entry(login_container, textvariable=pass_var, show="*")
    pass_entry.pack(side="left", padx=5)
    
    # Биндинги для быстрой вставки через Ctrl+V
    login_entry.bind('<Control-v>', lambda e: paste_login())
    pass_entry.bind('<Control-v>', lambda e: paste_password())
    
    # Создаем биндинги для вставки
    def create_paste_bindings(entry, var, name):
        def paste(event=None):
            try:
                clipboard = root.clipboard_get()
                var.set(clipboard)
                log(f"{name} вставлен из буфера обмена")
            except Exception as e:
                log(f"Ошибка вставки {name}: {e}")
        
        # Bind both Ctrl+V and right-click menu
        entry.bind('<Control-v>', paste)
        
        # Create right-click menu
        menu = tk.Menu(root, tearoff=0)
        menu.add_command(label="Вставить", command=paste)
        
        def show_menu(event):
            menu.tk_popup(event.x_root, event.y_root)
            return "break"
        
        entry.bind('<Button-3>', show_menu)
        
        return menu

    # Применяем биндинги к полям логина и пароля
    login_menu = create_paste_bindings(login_entry, login_var, "Логин")
    pass_menu = create_paste_bindings(pass_entry, pass_var, "Пароль")

    # Модифицируем функцию входа для лучшего логирования
    def do_login():
        try:
            username = login_var.get().strip()
            password = pass_var.get().strip()
            
            if not username or not password:
                log("Ошибка: логин или пароль не может быть пустым")
                messagebox.showerror("Ошибка", "Введите логин и пароль")
                return
                
            log(f"Попытка входа для пользователя: {username}")
            
            # Получаем страницу логина для проверки доступности сервера
            try:
                response = hwm_client.session.get(f'{hwm_client.base_url}/login.php')
                if response.status_code != 200:
                    log(f"Ошибка доступа к серверу: {response.status_code}")
                    messagebox.showerror("Ошибка", "Сервер недоступен")
                    return
            except Exception as e:
                log(f"Ошибка подключения к серверу: {e}")
                messagebox.showerror("Ошибка", "Проблема с подключением к серверу")
                return
                
            if hwm_client.login(username, password):
                log("Успешный вход в систему")
                update_games()
            else:
                log("Ошибка входа: неверные учетные данные")
                messagebox.showerror("Ошибка", "Неверный логин или пароль")
        except Exception as e:
            log(f"Ошибка входа: {e}")
            messagebox.showerror("Ошибка", str(e))

    ttk.Button(login_frame, text="Войти", command=do_login).pack(side="left", padx=5)
    
    # Фрейм игр
    games_frame = ttk.LabelFrame(main_frame, text="Доступные игры", padding="5")
    games_frame.pack(fill="both", expand=True, pady=5)
    
    games_text = scrolledtext.ScrolledText(games_frame, height=10)
    games_text.pack(fill="both", expand=True)
    
    def update_games():
        try:
            games = hwm_client.get_tavern_games()
            games_text.delete('1.0', tk.END)
            if games:
                for game in games:
                    games_text.insert(tk.END, 
                        f"Игра #{game['id']}\n"
                        f"Противник: {game['opponent']}\n"
                        f"Статус: {game['status']}\n"
                        f"-------------------\n"
                    )
            else:
                games_text.insert(tk.END, "Нет доступных игр\n")
            log("Список игр обновлен")
        except Exception as e:
            log(f"Ошибка получения списка игр: {e}")
    
    ttk.Button(games_frame, text="Обновить", command=update_games).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        init_db()
        
        # Add site analysis before starting GUI
        from tools import analyze_site_structure
        print("Analyzing site structure...")
        analysis = analyze_site_structure()
        if analysis:
            print("Site analysis completed. Check site_analysis.json for details.")
        
        start_gui()
    except Exception as e:
        print(f"Ошибка запуска: {e}")


