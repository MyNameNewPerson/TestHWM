# Конфигурация для работы с сайтом
BASE_URL = "https://www.heroeswm.ru"
TAVERN_URL = f"{BASE_URL}/tavern.php"
GAME_URL = f"{BASE_URL}/tavern_game.php"

# Задержки
ANIMATION_DELAY = 1.5
TURN_DELAY = 2.0
ACTION_DELAY = 0.5

# Селекторы
SELECTORS = {
    'login_field': 'input[name="login"]',
    'password_field': 'input[name="pass"]',
    'login_button': 'input[name="plogin"]',
    'game_field': '.game-field',
    'own_tower': '.my-tower .health',
    'opp_tower': '.enemy-tower .health',
    'hand_cards': '.hand-cards .card',
    'card_template': '.card[data-card="{}"]'
}