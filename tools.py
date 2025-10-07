import requests
from bs4 import BeautifulSoup
import json

def analyze_site_structure():
    """Анализ структуры сайта и форм."""
    try:
        session = requests.Session()
        base_url = 'https://www.heroeswm.ru'
        
        # Get main page
        main_response = session.get(base_url)
        main_soup = BeautifulSoup(main_response.text, 'html.parser')
        
        # Get login page
        login_response = session.get(f'{base_url}/login.php')
        login_soup = BeautifulSoup(login_response.text, 'html.parser')
        
        # Analyze forms
        forms = login_soup.find_all('form')
        
        analysis = {
            'main_page': {
                'status': main_response.status_code,
                'title': main_soup.title.text if main_soup.title else None,
                'forms': len(main_soup.find_all('form'))
            },
            'login_page': {
                'status': login_response.status_code,
                'title': login_soup.title.text if login_soup.title else None,
                'forms': [{
                    'name': form.get('name'),
                    'action': form.get('action'),
                    'method': form.get('method'),
                    'inputs': [{
                        'name': input_tag.get('name'),
                        'type': input_tag.get('type'),
                        'value': input_tag.get('value')
                    } for input_tag in form.find_all('input')]
                } for form in forms]
            }
        }
        
        # Save analysis
        with open('site_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
            
        return analysis
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return None