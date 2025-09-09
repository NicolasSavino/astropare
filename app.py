import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()
ADS_API_KEY = os.getenv('ADS_API_KEY')

def fetch_ads_data(query, rows=100):
    """
    Fetches data from the NASA ADS API for a given query.
    """
    headers = {
        'Authorization': f'Bearer {ADS_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Define the fields you want to retrieve
    # 'title', 'author', 'year', 'pubdate', 'abstract', 'keyword', 'citation_count'
    fields = "title,author,year,pubdate,abstract,keyword,citation_count"

    # Define the query parameters
    params = {
        'q': query,
        'fl': fields,
        'rows': rows,
        'sort': 'date desc' # Get the most recent papers first
    }

    try:
        response = requests.get('https://api.adsabs.harvard.edu/v1/search/query', headers=headers, params=params)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

        # Get the JSON data from the response
        results = response.json()
        docs = results.get('response', {}).get('docs', [])

        # Convert the list of paper dictionaries into a Pandas DataFrame
        df = pd.DataFrame(docs)
        
        print(f"Successfully fetched {len(df)} papers.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# --- Main execution ---
if __name__ == '__main__':
    # Define your search query
    # Example: Search for papers about exoplanets published in the last year
    search_query = 'exoplanet year:2024-2025'
    
    papers_df = fetch_ads_data(query=search_query, rows=200)

    if papers_df is not None:
        # Print the first 5 rows of the DataFrame to see the results
        print("\nFirst 5 papers:")
        print(papers_df.head())

        # Print the columns to see what data we got
        print("\nAvailable columns:")
        print(papers_df.columns)

def clean_and_prepare_data(df):
    """
    Cleans the raw DataFrame from the API.
    - Converts pubdate to datetime objects.
    - Handles missing keywords.
    - Explodes the DataFrame so each keyword has its own row.
    """
    # Make a copy to avoid changing the original DataFrame
    df_clean = df.copy()
    
    # Convert 'pubdate' to datetime objects for time-series analysis
    df_clean['pubdate'] = pd.to_datetime(df_clean['pubdate'])
    
    # Drop rows where there is no abstract or no keywords, as they are not useful for our analysis
    df_clean.dropna(subset=['abstract', 'keyword'], inplace=True)
    
    # Explode the 'keyword' column
    # If a paper has ['A', 'B'], it becomes two rows: one for A, one for B
    df_clean = df_clean.explode('keyword')
    
    # Clean up the keyword strings (lowercase, strip whitespace)
    df_clean['keyword'] = df_clean['keyword'].str.lower().str.strip()
    
    print("Data cleaning and preparation complete.")
    return df_clean

def create_monthly_topic_counts(df):
    """
    Aggregates the data to get a monthly count for each keyword.
    """
    # Set the publication date as the index for time-series resampling
    df.set_index('pubdate', inplace=True)
    
    # Group by keyword, then resample by month and count the occurrences
    # 'M' stands for Month End frequency
    monthly_counts = df.groupby('keyword').resample('M').size().rename('count').reset_index()
    
    print("Monthly topic counts created.")
    return monthly_counts


# --- Main execution ---
if __name__ == '__main__':
    search_query = 'exoplanet year:2024-2025'
    
    raw_papers_df = fetch_ads_data(query=search_query, rows=1000) # Increased rows for better analysis

    if raw_papers_df is not None:
        # Step 2: Clean and prepare the data
        prepared_df = clean_and_prepare_data(raw_papers_df)

        # Step 3: Create the monthly time-series of topic counts
        monthly_topic_df = create_monthly_topic_counts(prepared_df)
        
        if not monthly_topic_df.empty:
            print("\nTop 10 Trending Topics (by monthly count):")
            # Sort to see the keywords with the highest counts in a single month
            print(monthly_topic_df.sort_values('count', ascending=False).head(10))