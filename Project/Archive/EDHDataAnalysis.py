# Scryfall API Wrapper: https://github.com/NandaScott/Scrython
# EDHRec Wrapper: https://pypi.org/project/pyedhrec/
# Archidekt Wrapper: https://github.com/linkian209/pyrchidekt

import os
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import scrython
from scrython.base import ScrythonRequestHandler
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

app_name = "EDHDataAnalysis"
app_email = "avivg2001@gmail.com"
path_cards_file = "Project\\cards.json"
path_cards_parquet = "Project\\cards.parquet"

# manually increase whenever field cleaning / preprocessing logic changes, to force parquet update
SCHEMA_VERSION = "2"


def fetch_cards():
    ScrythonRequestHandler.set_user_agent(f'{app_name}/1.0 ({app_email})')
    bulk = scrython.bulk_data.ByType(type='oracle_cards')
    bulk.download(filepath=path_cards_file, progress=True)


def clean_cards_df(df: pd.DataFrame) -> pd.DataFrame:
    df['released_at'] = pd.to_datetime(df['released_at'], errors='coerce')

    # power/toughness can be "*" or "1+*" — coerce non-numeric to NaN
    numeric_cols = [
        'power', 'toughness', 'cmc', 'edhrec_rank', 'collector_number',
        'mtgo_id', 'mtgo_foil_id', 'tcgplayer_id', 'cardmarket_id',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'prices' in df.columns:
        prices = pd.json_normalize(df['prices'].tolist())
        prices.index = df.index
        for col in prices.columns:
            df[f'price_{col}'] = pd.to_numeric(prices[col], errors='coerce')
        df = df.drop(columns=['prices'])

    if 'legalities' in df.columns:
        legalities = pd.json_normalize(df['legalities'].tolist())
        legalities.index = df.index
        for col in legalities.columns:
            df[f'legal_{col}'] = legalities[col] == 'legal'
        df = df.drop(columns=['legalities'])

    categorical_cols = ['rarity', 'layout', 'set_type', 'border_color', 'frame', 'lang', 'image_status']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    bool_cols = [
        'reserved', 'game_changer', 'foil', 'nonfoil', 'oversized', 'promo',
        'reprint', 'variation', 'digital', 'full_art', 'textless', 'booster',
        'story_spotlight', 'highres_image',
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('boolean')

    if 'legal_commander' in df.columns and 'games' in df.columns:
        df = df[df['legal_commander'] & df['games'].apply(lambda x: 'paper' in x)]

    return df


def load_cards_df() -> pd.DataFrame:
    fetch_date = None
    schema_ok = False

    if os.path.exists(path_cards_parquet):
        meta = pq.read_metadata(path_cards_parquet).metadata or {}
        schema_ok = meta.get(b'schema_version', b'').decode() == SCHEMA_VERSION
        fetch_date_str = meta.get(b'fetch_date', b'').decode()
        if fetch_date_str:
            fetch_date = datetime.fromisoformat(fetch_date_str)

    if fetch_date is None and os.path.exists(path_cards_file):
        fetch_date = datetime.fromtimestamp(os.path.getmtime(path_cards_file))

    stale = fetch_date is None or (datetime.now() - fetch_date) >= timedelta(weeks=2)

    if stale:
        fetch_cards()
        fetch_date = datetime.now()

    if schema_ok and not stale:
        return pd.read_parquet(path_cards_parquet)

    df = clean_cards_df(pd.read_json(path_cards_file))
    table = pa.Table.from_pandas(df)
    merged_meta = {
        **(table.schema.metadata or {}),
        b'schema_version': SCHEMA_VERSION.encode(),
        b'fetch_date': fetch_date.isoformat().encode(),
    }
    pq.write_table(table.replace_schema_metadata(merged_meta), path_cards_parquet)
    return df


OVERVIEW_COLS = [
    'name', 'cmc', 'color_identity', 'mana_cost', 'type_line', 'set',
    'oracle_text', 'power', 'toughness', 'edhrec_rank', 'produced_mana',
    'game_changer', 'price_usd',
]

def print_df_stats(df: pd.DataFrame) -> None:
    print(f"Cards: {len(df):,}")
    print(f"Fields: {len(df.columns)}\n")

    cols = [c for c in OVERVIEW_COLS if c in df.columns]
    col_width = max(len(c) for c in cols)
    for col in cols:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean() * 100
        print(f"  {col:<{col_width}}  {dtype:<12}  {null_pct:5.1f}% null")


def print_top_edhrec(df: pd.DataFrame, n: int = 10) -> None:
    top = (df.dropna(subset=['edhrec_rank'])
             .sort_values('edhrec_rank')
             .head(n)[['name', 'edhrec_rank', 'type_line', 'cmc']])
    print(top.to_string(index=False))


def plot_cmc_distribution(df: pd.DataFrame, n: int = 1000) -> None:
    top = (df.dropna(subset=['edhrec_rank', 'cmc'])
             .sort_values('edhrec_rank')
             .head(n))
    top['cmc'].value_counts().sort_index().plot(kind='bar')
    plt.title(f'CMC Distribution — Top {n} EDHREC Cards')
    plt.xlabel('CMC')
    plt.ylabel('Card Count')
    plt.tight_layout()
    plt.show()


def plot_color_identity_distribution(df: pd.DataFrame) -> None:
    color_names = {'W': 'White', 'U': 'Blue', 'B': 'Black', 'R': 'Red', 'G': 'Green'}

    def categorize(colors):
        if len(colors) == 0:
            return 'Colorless'
        if len(colors) == 1:
            return color_names[colors[0]]
        return 'Multicolored'

    order = ['White', 'Blue', 'Black', 'Red', 'Green', 'Colorless', 'Multicolored']
    counts = df['color_identity'].apply(categorize).value_counts().reindex(order, fill_value=0)
    counts.plot(kind='bar')
    plt.title('Card Count by Color Identity')
    plt.xlabel('Color')
    plt.ylabel('Card Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_card_json():
    cards_df = load_cards_df()
    print_df_stats(cards_df)
    print_top_edhrec(cards_df)
    plot_color_identity_distribution(cards_df)


if __name__ == "__main__":
    analyze_card_json()
