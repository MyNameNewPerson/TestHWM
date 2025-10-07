
import tkinter as tk
from tkinter import messagebox, ttk
from db import init_db
from evaluation import neural_model
from training import train_ml
from ui import init_driver, login, get_active_games, select_game, play_game
import torch
import os

def load_model_if_exists():
    """Загрузка сохраненной модели, если существует."""
    model_path = 'neural_eval_model.pth'
    if os.path.exists(model_path):
        try:
            neural_model.load_state_dict(torch.load(model_path))
            print("Модель загружена из neural_eval_model.pth")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
    else:
        print("Модель не найдена, будет использоваться новая")

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
    
    ttk.Label(login_frame, text="Логин:").pack(pady=5)
    login_entry = ttk.Entry(login_frame)
    login_entry.pack(pady=5)
    
    ttk.Label(login_frame, text="Пароль:").pack(pady=5)
    pass_entry = ttk.Entry(login_frame, show="*")
    pass_entry.pack(pady=5)
    
    driver = None
    
    def do_login():
        nonlocal driver
        try:
            driver = init_driver()
            login(driver, login_entry.get(), pass_entry.get())
            messagebox.showinfo("Успех", "Авторизация выполнена!")
            login_frame.pack_forget()
            show_games(driver)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка авторизации: {e}")
            if driver:
                driver.quit()
    
    ttk.Button(login_frame, text="Войти", command=do_login).pack(pady=10)
    
    def show_games(driver):
        """Отображение списка активных игр."""
        games_frame = ttk.Frame(root, padding="10")
        games_frame.pack(fill="both", expand=True)
        
        ttk.Label(games_frame, text="Доступные игры в таверне").pack(pady=5)
        listbox = tk.Listbox(games_frame, height=10, font=("Arial", 12))
        listbox.pack(fill="both", expand=True, pady=5)
        
        games = []  # Список игр, доступен для select_and_play и update_games

        def update_games():
            nonlocal games
            try:
                games = get_active_games(driver)
                listbox.delete(0, tk.END)
                for g in games:
                    listbox.insert(tk.END, f"Игра {g['id']}: Оппонент {g['opponent']} ({g['status']})")
                root.after(5000, update_games)  # Обновление каждые 5 секунд
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка получения игр: {e}")
        
        def select_and_play():
            try:
                selected = listbox.curselection()
                if selected:
                    game_id = games[selected[0]]['id']
                    select_game(driver, game_id)
                    play_game(driver)
                    messagebox.showinfo("Игра", "Автоматическая игра завершена!")
                else:
                    messagebox.showwarning("Предупреждение", "Выберите игру!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при игре: {e}")
        
        def train_model():
            try:
                train_ml(num_games=100, epochs=10, use_db=False)
                messagebox.showinfo("Успех", "Обучение модели завершено!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка обучения: {e}")
        
        update_games()
        ttk.Button(games_frame, text="Выбрать и играть", command=select_and_play).pack(pady=5)
        ttk.Button(games_frame, text="Обучить модель", command=train_model).pack(pady=5)
        ttk.Button(games_frame, text="Выход", command=lambda: [driver.quit(), root.destroy()]).pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    init_db()  # Инициализация базы данных
    load_model_if_exists()  # Загрузка модели
    start_gui()  # Запуск интерфейса
