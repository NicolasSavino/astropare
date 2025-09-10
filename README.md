Astropare: Predictive Trend Forecasting for Astrophysical Research
Live Application: your-streamlit-app-url.streamlit.app

Astropare is an end-to-end data science application that ingests publication data from the NASA Astrophysics Data System (ADS), engineers time-series momentum features, and leverages an unsupervised machine learning model to forecast emerging research trends.

Abstract
In academic research, particularly in fast-moving fields like astrophysics, the early identification of nascent, high-growth research topics presents a significant strategic advantage. This project addresses that challenge by automating the detection of anomalous growth patterns in scientific literature. The system moves beyond descriptive analytics to provide a predictive forecast, identifying topics with statistically significant positive changes in publication velocity and acceleration.

Technical Architecture
The application is architected as a cohesive data pipeline built with Python and deployed as an interactive web application using Streamlit.

Data Ingestion Layer: A client module interfaces with the NASA ADS REST API. It constructs queries, handles HTTP requests, and serializes the JSON response into a structured Pandas DataFrame. API calls are cached using Streamlit's st.cache_data decorator to optimize performance and minimize redundant API hits.

Data Processing & Feature Engineering Layer: This layer, built with Pandas, is responsible for data cleaning and feature generation. It processes raw bibliographic data to create a time-series representation of topic publication frequency. Key engineered features include:

Velocity: First-order derivative of the monthly publication count (count.diff()).

Acceleration: Second-order derivative of the monthly publication count (velocity.diff()).

Machine Learning & Inference Layer: An unsupervised anomaly detection model (sklearn.ensemble.IsolationForest) is trained on the momentum features for each topic. The model identifies data points that are statistical outliers, flagging them as periods of anomalous, accelerated growth.

Presentation Layer: An interactive dashboard built with Streamlit and Plotly. This layer allows users to dynamically query the system and visualize the results, including the output of the anomaly detection model and time-series plots for forecasted trends.

Machine Learning Pipeline
Data Ingestion: Fetches up to 2,000 records from the NASA ADS API based on a user-defined search query.

Topic Assignment: A robust topic extraction logic parses the title and abstract of each paper to identify and assign relevant topics from the user's query, mitigating inconsistencies in the API's keyword metadata.

Time-Series Aggregation: The data is aggregated into a monthly time-series DataFrame, calculating the publication frequency for each unique topic.

Feature Engineering: The pipeline calculates the first and second-order derivatives (velocity and acceleration) of the time-series for each topic.

Anomaly Detection: An Isolation Forest model is dynamically trained on the feature set (count, velocity, acceleration) to identify and score anomalous data points. The contamination parameter is set to 'auto' for adaptive outlier detection.

Output: The system returns a ranked list of topics exhibiting the most significant anomalous growth, along with their momentum metrics and anomaly scores.

Local Setup and Usage
To run the application locally, clone the repository and install the required dependencies.

# Clone the repository
git clone https://github.com/YourUsername/Astropare.git
cd Astropare

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py

Ensure a .env file is present in the root directory containing your ADS_API_KEY.