import os
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import math
import json

def fetch_with_selenium(url):
    """Fetch page using Selenium with anti-bot detection options"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run in headless mode
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        # Wait a bit for page to load
        time.sleep(2)
        html = driver.page_source
        return html
    except Exception as e:
        print(f"Error fetching page with Selenium: {e}")
        return None
    finally:
        driver.quit()

def parse_art_books(html_file_path):
    """Parse the art books HTML file and extract book information"""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    product_divs = soup.find_all('div', class_=lambda x: x and 'box-producto' in x)
    print(f"Found {len(product_divs)} product divs")
    
    books = []
    
    for product in product_divs:
        book = {}
        
        # Title
        title_elem = product.find('h3', class_='nombre')
        book['Title'] = title_elem.get_text(strip=True) if title_elem else 'N/A'
        
        # Category (where crawled from)
        book['Category'] = 'Art Books'
        
        # Categories (comma-separated) - not available in listing, use main category
        book['Categories'] = 'Arts'
        
        # Authors
        author_elem = product.find('div', class_='autor')
        if author_elem and not author_elem.get('class') == ['autor', 'color-dark-gray', 'metas']:
            book['Authors'] = author_elem.get_text(strip=True)
        else:
            book['Authors'] = 'N/A'
        
        # Price in NIS
        price_elem = product.find('strong')
        if price_elem:
            price_text = price_elem.get_text(strip=True).replace('₪', '').replace(',', '')
            try:
                nis_price = float(price_text)
                book['Price in NIS'] = math.ceil(nis_price * 100) / 100  # 2 decimal digits, rounded up
            except ValueError:
                book['Price in NIS'] = 0.00
        else:
            book['Price in NIS'] = 0.00
        
        # Price in USD (exchange rate 3.01)
        usd_price = book['Price in NIS'] / 3.01
        book['Price in USD'] = math.ceil(usd_price * 100) / 100
        
        # Year, Format, etc. from meta
        meta_elem = product.find('div', class_='autor color-dark-gray metas hide-on-hover')
        meta_text = meta_elem.get_text(strip=True) if meta_elem else ''
        
        # Parse meta text - format is like "Publisher, Year, Edition, Format, Condition"
        parts = [part.strip() for part in meta_text.split(',')]
        book['Year'] = 'N/A'
        book['Format'] = 'N/A'
        
        for part in parts:
            part_lower = part.lower()
            if part.isdigit() and len(part) == 4:  # Year
                book['Year'] = part
            elif 'hardcover' in part_lower:
                book['Format'] = 'Hardcover'
            elif 'paperback' in part_lower:
                book['Format'] = 'Paperback'
            elif 'spiral' in part_lower:
                book['Format'] = 'Spiral'
            elif 'sheet music' in part_lower:
                book['Format'] = 'Sheet Music'
        
        # Synopsis - not available in listing
        book['Synopsis'] = 'N/A'
        book['Synopsis length'] = 0
        
        # Star Rating
        stars_elem = product.find('span', class_=lambda x: x and 'stars' in x and 'stars-' in x)
        reviews_elem = product.find('span', class_='color-dark-gray font-weight-light margin-left-5 font-size-small')
        
        if reviews_elem and '(' in reviews_elem.get_text():
            num_reviews = reviews_elem.get_text().strip('()')
            try:
                book['NumberOfReviews'] = int(num_reviews)
            except ValueError:
                book['NumberOfReviews'] = 0
        else:
            book['NumberOfReviews'] = 0
        
        if book['NumberOfReviews'] == 0:
            book['StarRating'] = 'None'
        else:
            # Extract rating from class (stars-5 means 5 stars)
            if stars_elem:
                classes = stars_elem.get('class', [])
                rating_class = next((cls for cls in classes if cls.startswith('stars-')), None)
                if rating_class:
                    rating = int(rating_class.split('-')[1])
                    book['StarRating'] = math.ceil(rating * 100) / 100  # 2 decimal digits, rounded up
                else:
                    book['StarRating'] = 'None'
            else:
                book['StarRating'] = 'None'
        
        # Dimensions, Weight, ISBN - not available in listing
        book['Dimensions'] = 'N/A'
        book['Dimensions unit'] = 'N/A'
        book['Weight'] = 'N/A'
        book['Weight unit'] = 'N/A'
        
        # ISBN
        isbn = product.get('data-isbn')
        book['ISBN/ISBN13'] = isbn if isbn else 'N/A'
        
        books.append(book)
    
    return books

def get_next_page_links(html_file_path):
    """Extract links to next 4 pages"""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    # Find pagination links
    pagination = soup.find('div', id='pagn')
    if pagination:
        page_links = pagination.find_all('a', href=True)
        for link in page_links:
            href = link['href']
            if 'page=' in href and href not in links:
                links.append(href)
    
    return links[:4]  # Return first 4 next pages


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
    try:
        if source.startswith('file://'):
            source = source[7:]

        if os.path.isfile(source):
            print(f"Loading local HTML file: {source}")
            if 'Art Books' in source:
                # Parse art books
                books = parse_art_books(source)
                print(f"Found {len(books)} books")
                
                # Save to JSON
                with open('art_books.json', 'w', encoding='utf-8') as f:
                    json.dump(books, f, indent=2, ensure_ascii=False)
                
                # Get next page links
                next_links = get_next_page_links(source)
                print(f"Next page links: {next_links}")
                
                # Save links
                with open('next_pages.json', 'w', encoding='utf-8') as f:
                    json.dump(next_links, f, indent=2)
                
                return
            else:
                # Original functionality for other files
                html = load_html_from_file(source)
                if html is None:
                    return
                soup = BeautifulSoup(html, 'html.parser')
                categories = extract_category_links(soup)
                print(categories)
                return

        print(f"Fetching page: {source}")
        html = fetch_with_selenium(source)
        
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            categories = extract_category_links(soup)
            print(categories)
        else:
            print("Failed to retrieve page with Selenium")
    except Exception as e:
        print(f"An error occurred: {e}")


# Start source: either a URL or a local HTML file
# local_file = "Bookdelivery.com homepage.htm"
# base_url = "https://www.bookdelivery.com/il-en/"
art_page_local = "Bookdelivery.com Art Books.htm"
crawl_books(art_page_local)
