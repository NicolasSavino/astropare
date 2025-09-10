import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

# --- Configuration and Setup ---
load_dotenv()
ADS_API_KEY = os.getenv('ADS_API_KEY')
st.set_page_config(layout="wide") # Use a wider layout for the dashboard

# --- Data Caching and Fetching ---
# CHANGE 1: Caching for Performance
# This decorator tells Streamlit to store the results of this function.
# If the same function is called with the same inputs, it returns the cached data
# instead of fetching from the API again, making the app much faster.
@st.cache_data
def fetch_ads_data(query: str, rows: int) -> pd.DataFrame | None:
    """Fetches data from the NASA ADS API."""
    headers = {'Authorization': f'Bearer {ADS_API_KEY}'}
    fields = "title,author,year,pubdate,abstract,citation_count"
    params = {'q': query, 'fl': fields, 'rows': rows, 'sort': 'date desc'}
    try:
        response = requests.get('https://api.adsabs.harvard.edu/v1/search/query', headers=headers, params=params)
        response.raise_for_status()
        docs = response.json().get('response', {}).get('docs', [])
        return pd.DataFrame(docs)
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        return None

# --- Analysis Pipeline Functions ---
# CHANGE 2: Robust Topic Assignment Logic
def process_data(df: pd.DataFrame, query_terms: list[str]) -> pd.DataFrame:
    """Cleans data and assigns topics based on query terms in title/abstract."""
    df_clean = df.copy()
    df_clean['pubdate'] = pd.to_datetime(df_clean['pubdate'], errors='coerce')
    df_clean.dropna(subset=['pubdate', 'abstract', 'title'], inplace=True)
    
    df_clean['text_for_search'] = df_clean['title'].str.lower() + " " + df_clean['abstract'].str.lower()
    
    topics = [
        [term for term in query_terms if term in text] 
        for text in df_clean['text_for_search']
    ]
    df_clean['topics'] = topics
    
    df_clean = df_clean[df_clean['topics'].apply(len) > 0].explode('topics')
    return df_clean

def analyze_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full trend analysis pipeline on the processed data."""
    # 1. Create monthly counts
    df.set_index('pubdate', inplace=True)
    monthly_counts = df.groupby('topics').resample('M').size().rename('count').reset_index()
    
    # 2. Calculate momentum features
    monthly_counts = monthly_counts.sort_values(by=['topics', 'pubdate'])
    monthly_counts['velocity'] = monthly_counts.groupby('topics')['count'].diff().fillna(0)
    monthly_counts['acceleration'] = monthly_counts.groupby('topics')['velocity'].diff().fillna(0)
    
    # 3. Find anomalies
    all_anomalies = []
    for topic, group in monthly_counts.groupby('topics'):
        if len(group) < 2: continue
        features = group[['count', 'velocity', 'acceleration']]
        model = IsolationForest(contamination='auto', random_state=42)
        model.fit(features)
        
        group['anomaly_score'] = model.decision_function(features)
        group['is_anomaly'] = model.predict(features)
        
        anomalies = group[group['is_anomaly'] == -1]
        all_anomalies.append(anomalies)

    if not all_anomalies: return pd.DataFrame()
    return pd.concat(all_anomalies).sort_values(by='anomaly_score')

# --- Streamlit User Interface ---

st.title("Astropare 🌠")
st.write("An engine to forecast emerging research trends in astrophysics by detecting anomalous growth in publication volume.")

# Sidebar for controls
st.sidebar.header("Controls")
search_query_input = st.sidebar.text_input(
    "Search Query",
    'jwst year:2020-2025'
)
num_rows = st.sidebar.slider("Number of papers to fetch", 500, 2000, 1500, 100)

if st.sidebar.button("Analyze Trends"):
    if not search_query_input:
        st.warning("Please enter a search query.")
    else:
        query_terms = [term.lower() for term in search_query_input.split() if 'year:' not in term and term.lower() not in ['or', 'and']]
        
        raw_papers_df = fetch_ads_data(query=search_query_input, rows=num_rows)
        
        if raw_papers_df is not None and not raw_papers_df.empty:
            st.success(f"Successfully fetched and cached {len(raw_papers_df)} papers.")
            
            processed_df = process_data(raw_papers_df, query_terms)
            emerging_topics = analyze_trends(processed_df)
            
            st.subheader("🚀 Emerging Topic Forecast")
            
            if not emerging_topics.empty:
                st.write("The following monthly data points show anomalous growth patterns:")
                display_cols = ['topics', 'pubdate', 'count', 'velocity', 'acceleration', 'anomaly_score']
                st.dataframe(emerging_topics[display_cols])

                # CHANGE 3: Interactive Visualization
                st.subheader("Visualize a Trend")
                topic_to_plot = st.selectbox("Select a topic to visualize its timeline:", options=emerging_topics['topics'].unique())
                
                if topic_to_plot:
                    # Get all data for the selected topic, not just the anomalies
                    topic_timeline = processed_df[processed_df['topics'] == topic_to_plot]
                    monthly_timeline = topic_timeline.set_index('pubdate').resample('M').size().rename('count').reset_index()
                    
                    fig = px.line(monthly_timeline, x='pubdate', y='count', title=f"Publication Timeline for '{topic_to_plot}'")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No anomalous trends were detected. Try a broader query or date range.")
        else:
            st.error("Could not retrieve or process data.")