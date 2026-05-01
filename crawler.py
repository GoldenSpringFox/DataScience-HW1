import os
import requests
from bs4 import BeautifulSoup
import time

def fetch_with_retry(url, headers, max_retries=3):
    for attempt in range(1, max_retries + 1):
        response = requests.get(url, headers=headers)

        if response.status_code == 202:
            retry_after = response.headers.get('Retry-After')
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = 2 * attempt

            print(f"Received 202 Accepted. Waiting {wait} seconds before retry {attempt}/{max_retries}...")
            time.sleep(wait)
            continue

        return response

    return response

def load_html_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Failed to read local HTML file: {e}")
        return None


def extract_category_links(soup):
    categories = {}
    category_heading = soup.find('p', class_='subtitulofiltro', string=lambda t: t and t.strip().lower() == 'category')
    if not category_heading:
        return categories

    category_list = category_heading.find_next_sibling('ul')
    if not category_list:
        return categories

    for a in category_list.find_all('a', href=True):
        category_name = a.get_text(strip=True)
        category_href = a['href']
        if category_name:
            categories[category_name] = category_href

    return categories


def crawl_books(source):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        if source.startswith('file://'):
            source = source[7:]

        if os.path.isfile(source):
            print(f"Loading local HTML file: {source}")
            html = load_html_from_file(source)
            if html is None:
                return
            soup = BeautifulSoup(html, 'html.parser')
            categories = extract_category_links(soup)
            print(categories)
            return

        response = fetch_with_retry(source, headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            categories = extract_category_links(soup)
            print(categories)
        elif response.status_code == 202:
            print("Still got 202 Accepted after retries. The resource is not ready yet.")
        else:
            print(f"Failed to retrieve page. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Start source: either a URL or a local HTML file
# local_file = "Bookdelivery.com homepage.htm"
base_url = "https://www.bookdelivery.com/"
crawl_books(base_url)
