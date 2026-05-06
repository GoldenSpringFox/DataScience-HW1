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
import re

hide_window = False  # Set to True to hide the browser window during crawling

def fetch_with_selenium(url):
    """Fetch page using Selenium with anti-bot detection options"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run in headless mode
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    if (hide_window):
        chrome_options.add_argument("--window-position=-2000,-2000")  # Position window off-screen
        chrome_options.add_argument("--window-size=800,600")  # Set a reasonable window size

    
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

def parse_books(html_content, category_name, existing_books_dict=None, page_num=None, total_pages=5, skip_only_if_missing=False):
    """Parse the HTML content and extract book information"""
    soup = BeautifulSoup(html_content, 'html.parser')
    product_divs = soup.find_all('div', class_=lambda x: x and 'box-producto' in x)
    
    books = []
    skipped_book_count = 0
    
    for idx, product in enumerate(product_divs, start=1):
        book = {}
        
        # Extract book URL first
        book_link = product.find('a', href=True)
        book_url = book_link['href'] if book_link else None
        # Convert relative URLs to absolute
        if book_url and book_url.startswith('/'):
            book_url = f"https://www.bookdelivery.com{book_url}"
        book['URL'] = book_url if book_url else 'N/A'
        
        # Title
        title_elem = product.find('h3', class_='nombre')
        book['Title'] = title_elem.get_text(strip=True) if title_elem else 'N/A'
        
        # Category (where crawled from)
        book['Category'] = category_name
        
        # Categories (comma-separated) - will be filled from individual page
        book['Categories'] = category_name
        
        # Authors
        author_elem = product.find('div', class_='autor')
        if author_elem and not author_elem.get('class') == ['autor', 'color-dark-gray', 'metas']:
            book['Authors'] = author_elem.get_text(strip=True)
        else:
            book['Authors'] = 'N/A'
        
        book_key = f"{book.get('Title', '')}_{book.get('Authors', '')}".lower().replace(' ', '_')
        if existing_books_dict and book_key in existing_books_dict:
            existing_book = existing_books_dict[book_key]
            if skip_only_if_missing:
                # print(f"  Skipping existing book: {book['Title']} by {book['Authors']}")
                books.append(existing_book)
                skipped_book_count += 1
                continue
            elif is_book_complete(existing_book):
                # print(f"  Skipping already complete book: {book['Title']} by {book['Authors']}")
                books.append(existing_book)
                skipped_book_count += 1
                continue
        
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
        
        # Year, Format, etc. from meta - will be updated from individual page
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
        
        # Synopsis - will be filled from individual page
        book['Synopsis'] = 'N/A'
        book['Synopsis length'] = 0
        
        # Star Rating - will be updated from individual page
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
        
        # Dimensions, Weight - will be filled from individual page
        book['Dimensions'] = 'N/A'
        book['Dimensions unit'] = 'N/A'
        book['Weight'] = 'N/A'
        book['Weight unit'] = 'N/A'
        
        # ISBN
        isbn = product.get('data-isbn')
        book['ISBN/ISBN13'] = isbn if isbn else 'N/A'
        
        # Now visit the individual book page to get detailed information
        if book_url:
            page_label = f"(page {page_num}/{total_pages})" if page_num is not None else ""
            print(f"  {category_name} {page_label} [book {idx}/{len(product_divs)}]: {book_url}")
            book_html = fetch_with_selenium(book_url)
            if book_html:
                book_soup = BeautifulSoup(book_html, 'html.parser')
                
                # Extract categories
                categories_div = book_soup.find('div', id='metadata-categorías')
                if categories_div:
                    category_links = categories_div.find_all('a')
                    categories_list = [link.get_text(strip=True) for link in category_links]
                    book['Categories'] = ', '.join(categories_list) if categories_list else category_name
                
                # Extract year (from metadata)
                year_div = book_soup.find('div', id='metadata-año')
                if year_div:
                    year_text = year_div.get_text(strip=True)
                    if year_text.isdigit() and len(year_text) == 4:
                        book['Year'] = year_text
                
                # Extract synopsis
                synopsis_span = book_soup.find('span', id='texto-descripcion')
                if synopsis_span:
                    synopsis_text = synopsis_span.get_text(strip=True)
                    book['Synopsis'] = synopsis_text
                    book['Synopsis length'] = len(synopsis_text)
                
                # Extract star rating from individual page using detailed review counts
                evaluation_ul = book_soup.find('ul', class_='evaluacion')
                if evaluation_ul:
                    total_votes = 0
                    total_score = 0
                    for rating_value in range(1, 6):
                        count = 0
                        li = evaluation_ul.find('li', class_=f'stars-{rating_value}-li')
                        if li:
                            count_text = li.get_text(strip=True)
                            match = re.search(r"\((\d+)\)", count_text)
                            if match:
                                count = int(match.group(1))
                        total_votes += count
                        total_score += rating_value * count

                    if total_votes > 0:
                        book['NumberOfReviews'] = total_votes
                        book['StarRating'] = math.ceil((total_score / total_votes) * 100) / 100
                    else:
                        book['StarRating'] = 'None'
                        book['NumberOfReviews'] = 0
                else:
                    rating_span = book_soup.find('span', class_=lambda x: x and 'stars' in x and 'stars-' in x)
                    if rating_span:
                        classes = rating_span.get('class', [])
                        rating_class = next((cls for cls in classes if cls.startswith('stars-')), None)
                        if rating_class:
                            rating = int(rating_class.split('-')[1])
                            book['StarRating'] = math.ceil(rating * 100) / 100

                # Extract dimensions
                dimensions_div = book_soup.find('div', id='metadata-dimensiones')
                if dimensions_div:
                    dimensions_text = dimensions_div.get_text(strip=True)
                    # Parse dimensions like "26.2 x 18.8 x 5.1 cm"
                    if 'x' in dimensions_text and 'cm' in dimensions_text:
                        book['Dimensions'] = dimensions_text.replace('cm', '').strip()
                        book['Dimensions unit'] = 'cm'
                
                # Extract weight
                weight_div = book_soup.find('div', id='metadata-peso')
                if weight_div:
                    weight_text = weight_div.get_text(strip=True)
                    # Parse weight like "1.84 kg."
                    if 'kg' in weight_text:
                        book['Weight'] = weight_text.replace('kg.', '').replace('kg', '').strip()
                        book['Weight unit'] = 'kg'
            
            # Small sleep between individual book page visits
            time.sleep(3)
        
        books.append(book)
    
    print(f"    Skipped {skipped_book_count} books that were already in JSON and complete.")

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


def is_book_complete(book):
    """Return True when no fields are left as placeholders or missing."""
    for value in book.values():
        if value == 'N/A' or value == '' or value is None:
            return False
    return True


def assign_book_ids(books):
    for idx, book in enumerate(books, start=1):
        book['ID'] = idx
    return books


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


def crawl_books(start_category_index=1, start_page_num=1, skip_only_if_missing=False):
    """Crawl all categories and their first 5 pages, starting from specified category and page.
    
    Args:
        skip_only_if_missing: If True, only skip books that are not in the JSON at all.
                              If False, skip books that are already complete (default behavior).
    """
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
                page_books = parse_books(html, category_name, existing_books_dict=all_books_dict, page_num=page_num, total_pages=5, skip_only_if_missing=skip_only_if_missing)
                # print(f"    Found {len(page_books)} books on page {page_num}")
                
                # Add/update books in the dictionary
                for book in page_books:
                    key = f"{book.get('Title', '')}_{book.get('Authors', '')}".lower().replace(' ', '_')
                    all_books_dict[key] = book  # This will overwrite if key exists
                
                category_books.extend(page_books)
                
                # Save progress after each page (convert dict to list)
                all_books_list = assign_book_ids(list(all_books_dict.values()))
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(all_books_list, f, indent=2, ensure_ascii=False)
                print(f"    Saved progress: {len(all_books_list)} total books")
            else:
                print(f"    Failed to fetch page {page_num}")
            
            
            if page_num < 5:  # Don't sleep after last page
                time.sleep(4)
        
        print(f"  Total books for {category_name}: {len(category_books)}")
        
        # Sleep between categories too (except for the last one)
        if cat_idx < len(category_list) - 1:
            time.sleep(5)
    
    # Final save (convert dict to list)
    all_books_list = assign_book_ids(list(all_books_dict.values()))
    print(f"\nCrawling complete. Total books crawled: {len(all_books_list)}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_books_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved all books to {json_file}")


# Start crawling
if __name__ == "__main__":
    import sys
    start_category = 10
    start_page = 1
    skip_only_if_missing = True
    hide_window = True
    
    crawl_books(start_category, start_page, skip_only_if_missing)
