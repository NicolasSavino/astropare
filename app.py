import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

# --- Page Configuration ---
st.set_page_config(
    page_title="Astropare Trend Forecaster",
    page_icon="🌠",
    layout="wide"
)

# --- (All your data functions remain the same) ---
load_dotenv()
ADS_API_KEY = os.getenv('ADS_API_KEY')

@st.cache_data
def fetch_ads_data(query: str, rows: int) -> pd.DataFrame | None:
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

def process_data(df: pd.DataFrame, query_terms: list[str]) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean['pubdate'] = pd.to_datetime(df_clean['pubdate'], errors='coerce')
    df_clean.dropna(subset=['pubdate', 'abstract', 'title'], inplace=True)
    df_clean['text_for_search'] = df_clean['title'].str.lower() + " " + df_clean['abstract'].str.lower()
    topics = [[term for term in query_terms if term in text] for text in df_clean['text_for_search']]
    df_clean['topics'] = topics
    df_clean = df_clean[df_clean['topics'].apply(len) > 0].explode('topics')
    return df_clean

def analyze_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df.set_index('pubdate', inplace=True)
    monthly_counts = df.groupby('topics').resample('M').size().rename('count').reset_index()
    monthly_counts = monthly_counts.sort_values(by=['topics', 'pubdate'])
    monthly_counts['velocity'] = monthly_counts.groupby('topics')['count'].diff().fillna(0)
    monthly_counts['acceleration'] = monthly_counts.groupby('topics')['velocity'].diff().fillna(0)
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
    if not all_anomalies: return pd.DataFrame(), monthly_counts
    final_anomalies = pd.concat(all_anomalies).sort_values(by='anomaly_score')
    return final_anomalies, monthly_counts

# --- Streamlit User Interface ---

st.title("Astropare 🌠")
st.write("An engine to forecast emerging research trends in astrophysics by detecting anomalous growth in publication volume.")

# UI ENHANCEMENT 1: Intuitive Sidebar Controls
st.sidebar.header("Query Controls")
query_keywords = st.sidebar.text_area("Keywords (one per line)", "jwst\nexoplanet")
year_range = st.sidebar.slider("Year Range", 2010, 2025, (2020, 2025))
num_rows = st.sidebar.slider("Number of papers to analyze", 500, 2000, 1500, 100)

# UI ENHANCEMENT 2: Instructions in an expander
with st.sidebar.expander("How to write a query"):
    st.markdown("""
        - **Keywords:** Enter one keyword or phrase per line. The app will search for papers containing these terms in their title or abstract.
        - **Operators:** You can use `OR` to find papers with any of the terms (e.g., `jwst OR exoplanet`).
        - **Date Range:** The `year` filter is automatically constructed from the slider.
    """)

if st.sidebar.button("Analyze Trends"):
    # Construct the query from the new UI controls
    keywords_list = [k.strip().lower() for k in query_keywords.split('\n') if k.strip()]
    query_string = f"({' OR '.join(keywords_list)}) year:{year_range[0]}-{year_range[1]}"
    
    with st.spinner("Fetching and analyzing data... this may take a moment."):
        raw_papers_df = fetch_ads_data(query=query_string, rows=num_rows)

    if raw_papers_df is not None and not raw_papers_df.empty:
        processed_df = process_data(raw_papers_df, keywords_list)
        emerging_topics, full_timeline = analyze_trends(processed_df)
        
        # UI ENHANCEMENT 3: Key Metrics
        st.header("Analysis Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Papers Analyzed", f"{len(processed_df)}")
        col2.metric("Unique Topics Found", f"{processed_df['topics'].nunique()}")
        col3.metric("Anomalous Trends Detected", f"{len(emerging_topics)}")

        # UI ENHANCEMENT 4: Organized Tabs for Results
        tab1, tab2, tab3 = st.tabs(["🚀 Forecast", "📈 Trend Visualizer", "📚 Raw Data"])

        with tab1:
            st.subheader("Emerging Topic Forecast")
            if not emerging_topics.empty:
                st.write("The following monthly data points show anomalous growth patterns, indicating a potential emerging trend.")
                display_cols = ['topics', 'pubdate', 'count', 'velocity', 'acceleration', 'anomaly_score']
                st.dataframe(emerging_topics[display_cols].style.format({'pubdate': '{:%Y-%m}'}))
            else:
                st.info("No anomalous trends were detected. The topics show stable growth.")

        with tab2:
            st.subheader("Visualize a Topic's Full Timeline")
            if not full_timeline.empty:
                topic_to_plot = st.selectbox("Select a topic to visualize:", options=full_timeline['topics'].unique())
                if topic_to_plot:
                    topic_data = full_timeline[full_timeline['topics'] == topic_to_plot]
                    fig = px.line(topic_data, x='pubdate', y='count', title=f"Publication Timeline for '{topic_to_plot}'", markers=True)
                    fig.update_layout(xaxis_title="Month", yaxis_title="Publication Count")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available to plot.")
        
        with tab3:
            st.subheader("Raw Data Fetched from NASA ADS")
            st.write("This is the raw, cached data used for the analysis.")
            st.dataframe(raw_papers_df)

    else:
        st.error("Could not retrieve data. Please check your query or API key.")