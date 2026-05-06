import os
import pandas as pd
if __name__ == "__main__":
    #save the raw data in pandas data frame
    df_books = pd.read_json('all_books.json')

    #cast all numeric fields
    df_books['Price in NIS'] = pd.to_numeric(df_books['Price in NIS'], errors='coerce')
    df_books['Price in USD'] = pd.to_numeric(df_books['Price in USD'], errors='coerce')
    df_books['Year'] = pd.to_numeric(df_books['Year'], errors='coerce')
    df_books['NumberOfReviews'] = pd.to_numeric(df_books['NumberOfReviews'], errors='coerce')
    df_books['StarRating'] = pd.to_numeric(df_books['StarRating'], errors='coerce')

    #create an output directory and save
    #os.makedirs('output', exist_ok=True)
    #df_books.to_csv('output/books_raw.csv', index=False, encoding='utf-8')
    #df_books.to_json('output/books_raw.json', orient='records', force_ascii=False, indent=2)

    #print first 10 pages before sort by tytle
    pd.set_option('display.max_columns', None)
    print(df_books.head(10))

    #sort by title and print
    df_books_sorted = df_books.sort_values(by='Title', ascending=True)
    print(df_books_sorted.head(10))

    #add a binary feature Isexpensive and NumberOfAuthors
    median_price = df_books['Price in NIS'].median()
    df_books['IsExpensive'] = (df_books['Price in NIS'] > median_price).astype(int)
    df_books['NumberOfAuthors'] = df_books['Authors'].apply(
        lambda x: 0 if x == 'N/A' else len([a for a in x.split(',') if a.strip()]))

    #save
    #df_books.to_csv('output/books_raw.csv', index=False, encoding='utf-8')





