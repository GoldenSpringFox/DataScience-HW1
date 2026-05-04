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

def parse_books(html_content, category_name):
    """Parse the HTML content and extract book information"""
    soup = BeautifulSoup(html_content, 'html.parser')
    product_divs = soup.find_all('div', class_=lambda x: x and 'box-producto' in x)
    
    books = []
    
    for product in product_divs:
        book = {}
        
        # Title
        title_elem = product.find('h3', class_='nombre')
        book['Title'] = title_elem.get_text(strip=True) if title_elem else 'N/A'
        
        # Category (where crawled from)
        book['Category'] = category_name
        
        # Categories (comma-separated) - not available in listing, use main category
        book['Categories'] = category_name
        
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

def get_categories_from_homepage():
    """Fetch homepage and extract all category links"""
    base_url = "https://www.bookdelivery.com/il-en/"
    print("Fetching homepage to get categories...")
    html = fetch_with_selenium(base_url)
    if not html:
        print("Failed to fetch homepage")
        return {}
    
    soup = BeautifulSoup(html, 'html.parser')
    categories = {}
    
    # Find the category list
    category_heading = soup.find('p', class_='subtitulofiltro', string='Category')
    if category_heading:
        category_list = category_heading.find_next_sibling('ul')
        if category_list:
            for li in category_list.find_all('li', class_='category-li'):
                a_tag = li.find('a')
                if a_tag and a_tag.get('href'):
                    category_name = a_tag.find('span').get_text(strip=True) if a_tag.find('span') else a_tag.get_text(strip=True)
                    category_url = a_tag['href']
                    if category_name and category_url:
                        categories[category_name] = category_url
    
    print(f"Found {len(categories)} categories")
    return categories

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


def crawl_books(start_category_index=1, start_page_num=1):
    """Crawl all categories and their first 5 pages, starting from specified category and page"""
    all_books_dict = {}  # Use dict to handle duplicates by unique key
    
    # Load existing books if file exists
    json_file = 'all_books.json'
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_books = json.load(f)
            print(f"Loaded {len(existing_books)} existing books from {json_file}")
            # Convert to dict for easy lookup/update
            for book in existing_books:
                key = f"{book.get('Title', '')}_{book.get('Authors', '')}".lower().replace(' ', '_')
                all_books_dict[key] = book
        except:
            print("Could not load existing books, starting fresh")
            all_books_dict = {}
    
    # Get all categories from homepage
    categories = get_categories_from_homepage()
    if not categories:
        print("No categories found, aborting")
        return
    
    category_list = list(categories.items())
    
    # Start from specified category index (1-based)
    for cat_idx in range(start_category_index - 1, len(category_list)):
        category_name, category_url = category_list[cat_idx]
        print(f"\nCrawling category {cat_idx + 1}/{len(category_list)}: {category_name}")
        category_books = []
        
        # Determine starting page for this category
        page_start = start_page_num if cat_idx == start_category_index - 1 else 1
        
        # Crawl pages for this category
        for page_num in range(page_start, 6):  # 1 to 5
            if page_num == 1:
                page_url = category_url
            else:
                # Add page parameter
                if '?' in category_url:
                    page_url = f"{category_url}&page={page_num}"
                else:
                    page_url = f"{category_url}?page={page_num}"
            
            print(f"  Fetching page {page_num}: {page_url}")
            html = fetch_with_selenium(page_url)
            
            if html:
                # Parse books from this page
                page_books = parse_books(html, category_name)
                print(f"    Found {len(page_books)} books on page {page_num}")
                
                # Add/update books in the dictionary
                for book in page_books:
                    key = f"{book.get('Title', '')}_{book.get('Authors', '')}".lower().replace(' ', '_')
                    all_books_dict[key] = book  # This will overwrite if key exists
                
                category_books.extend(page_books)
                
                # Save progress after each page (convert dict to list)
                all_books_list = list(all_books_dict.values())
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(all_books_list, f, indent=2, ensure_ascii=False)
                print(f"    Saved progress: {len(all_books_list)} total books")
            else:
                print(f"    Failed to fetch page {page_num}")
            
            # Sleep for ~3 seconds between requests
            if page_num < 5:  # Don't sleep after last page
                print("    Sleeping for 3 seconds...")
                time.sleep(3)
        
        print(f"  Total books for {category_name}: {len(category_books)}")
        
        # Sleep between categories too (except for the last one)
        if cat_idx < len(category_list) - 1:
            print("  Sleeping for 3 seconds before next category...")
            time.sleep(3)
    
    # Final save (convert dict to list)
    all_books_list = list(all_books_dict.values())
    print(f"\nCrawling complete. Total books crawled: {len(all_books_list)}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_books_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved all books to {json_file}")


# Start crawling
if __name__ == "__main__":
    import sys
    start_category = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    crawl_books(start_category, start_page)
