import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
import zipfile
import os

# Set wide layout
st.set_page_config(layout="wide")

# Load Data Function
def load_data():
    uploaded_file = st.file_uploader(
        "Upload merged data file (CSV or ZIP containing CSV)", type=['csv','zip']
    )
    if uploaded_file:
        if uploaded_file.name.lower().endswith('.zip'):
            # Handle ZIP containing CSV
            with zipfile.ZipFile(uploaded_file) as zf:
                csvs = [f for f in zf.namelist() if f.lower().endswith('.csv')]
                if not csvs:
                    st.error("No CSV file found inside the ZIP archive.")
                    st.stop()
                with zf.open(csvs[0]) as f:
                    df = pd.read_csv(f)
        else:
            # Direct CSV upload
            df = pd.read_csv(uploaded_file)
    else:
        # Attempt to load local CSV first
        if os.path.exists('humberside-street-merged.csv'):
            df = pd.read_csv('humberside-street-merged.csv')
        # Else try ZIP in repository
        elif os.path.exists('data/humberside-street-merged.zip'):
            with zipfile.ZipFile('data/humberside-street-merged.zip') as zf:
                csvs = [f for f in zf.namelist() if f.lower().endswith('.csv')]
                if not csvs:
                    st.error("No CSV file found inside local ZIP archive.")
                    st.stop()
                with zf.open(csvs[0]) as f:
                    df = pd.read_csv(f)
        else:
            st.error("No data file found. Please upload a CSV or ZIP containing your data.")
            st.stop()

    # Clean and preprocess
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['location'] = df['location'].astype(str).str.strip()
    df = df[df['location'] != '']
    df = df.dropna(subset=['crime_id', 'crime_type'])
    df = df.drop_duplicates(subset=['crime_id']).reset_index(drop=True)
    return df

# Prediction model preparation
@st.cache_data
def train_model(df):
    df_model = df.dropna(subset=['lsoa_code', 'lsoa_name']).copy()
    rare = df_model['crime_type'].value_counts()[df_model['crime_type'].value_counts() < 1000].index
    df_model['crime_type'] = df_model['crime_type'].apply(lambda x: 'Other' if x in rare else x)
    features = ['longitude', 'latitude', 'reported_by', 'falls_within', 'last_outcome_category']
    X = df_model[features]
    y = df_model['crime_type']
    # Encode
    X_enc = X.copy()
    label_encoders = {}
    for col in ['reported_by', 'falls_within', 'last_outcome_category']:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col])
        label_encoders[col] = le
    scaler = StandardScaler()
    X_enc[['longitude', 'latitude']] = scaler.fit_transform(X_enc[['longitude', 'latitude']])
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_enc, y_enc)
    return rf, le_target, scaler, label_encoders, X, df_model

# Main App
def main():
    df = load_data()

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["General Overview", "EDA Analysis", "Crime Prediction"])

    # Tab 1: General Overview
    with tab1:
        st.header("🔎 Predictive Analytics Dashboard for Humberside Street")
        st.markdown(
            """
            This dashboard analyses the Humberside street crime dataset (May 2022 - Apr 2025), showing:
            - Data Overview
            - Correlation Heatmap
            - Interactive Crime Map
            """
        )
        st.subheader("1. Data Overview")
        st.write(df.head())

        st.subheader("2. Correlation Heatmap")
        corr_df = pd.concat([
            df[['latitude', 'longitude']],
            pd.get_dummies(df['crime_type'], prefix='crime'),
            pd.get_dummies(df['last_outcome_category'], prefix='outcome')
        ], axis=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_df.corr(), cmap='coolwarm', center=0, cbar_kws={'shrink': .5}, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)


        zip_path = 'data/crime_map.zip'

        with zipfile.ZipFile(zip_path, 'r') as zf:
            html_files = [f for f in zf.namelist() if f.endswith('.html')]
            if not html_files:
                st.error("No HTML file found inside crime_map.zip.")
            else:
                html_content = zf.read(html_files[0]).decode('utf-8')
                st.components.v1.html(html_content, height=600)

    # Tab 2: EDA Analysis
    with tab2:
        st.header("📊 Exploratory Data Analysis")
        cat_cols = ['reported_by', 'falls_within', 'crime_type', 'last_outcome_category']
        for col in cat_cols:
            st.subheader(f"Count Plot: {col}")
            counts = df[col].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(y=counts.index, x=counts.values, ax=ax)
            ax.set_title(f"Counts of {col}")
            st.pyplot(fig)

            st.subheader(f"Percentage Plot: {col}")
            pct = df[col].value_counts(normalize=True) * 100
            fig, ax = plt.subplots()
            sns.barplot(y=pct.index, x=pct.values, ax=ax)
            ax.set_title(f"Percentage of {col}")
            st.pyplot(fig)

        st.subheader("Top 10 LSOA Names + 'Other'")
        lsoa_counts = df['lsoa_name'].value_counts()
        top10 = lsoa_counts.iloc[:10]
        top10['Other'] = lsoa_counts.iloc[10:].sum()
        fig, ax = plt.subplots()
        sns.barplot(y=top10.index, x=top10.values, ax=ax)
        ax.set_title("Top 10 LSOA Names")
        st.pyplot(fig)

    # Tab 3: Crime Prediction
    with tab3:
        st.header("🔮 Crime Prediction (Next 6 Months)")
        rf, le_target, scaler, label_encoders, X_train, df_model = train_model(df)

        # Simulate future data
        lon_min, lon_max = df_model['longitude'].min(), df_model['longitude'].max()
        lat_min, lat_max = df_model['latitude'].min(), df_model['latitude'].max()
        future_list = []
        for m in range(1, 7):
            random_lons = np.random.uniform(lon_min, lon_max, 5000)
            random_lats = np.random.uniform(lat_min, lat_max, 5000)
            df_f = pd.DataFrame({
                'longitude': random_lons,
                'latitude': random_lats,
                'reported_by': label_encoders['reported_by'].transform([df_model['reported_by'].mode()[0]][0:1] * 5000),
                'falls_within': label_encoders['falls_within'].transform([df_model['falls_within'].mode()[0]][0:1] * 5000),
                'last_outcome_category': label_encoders['last_outcome_category'].transform([df_model['last_outcome_category'].mode()[0]][0:1] * 5000),
                'simulated_month': m
            })
            df_f[['longitude', 'latitude']] = scaler.transform(df_f[['longitude', 'latitude']])
            preds = rf.predict(df_f[X_train.columns])
            df_f['predicted_crime_type'] = le_target.inverse_transform(preds)
            future_list.append(df_f)
        fut_df = pd.concat(future_list, ignore_index=True)

        st.subheader("Predicted Crime Types")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(
            data=fut_df,
            y='predicted_crime_type',
            order=fut_df['predicted_crime_type'].value_counts().index,
            ax=ax
        )
        ax.set_title("Predicted Crimes Next 6 Months")
        st.pyplot(fig)

        st.subheader("Predictions by Month")
        pivot = fut_df.groupby(['simulated_month', 'predicted_crime_type']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title("Predicted Crime Types by Month")
        st.pyplot(fig)

        st.subheader("Future vs Historical")
        num_hist = df_model.shape[0] / 36
        future_tot = fut_df.groupby('simulated_month').size().reset_index(name='future_count')
        future_tot['historical_avg'] = num_hist
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(future_tot['simulated_month'], future_tot['future_count'], marker='o', label='Future')
        ax.plot(future_tot['simulated_month'], future_tot['historical_avg'], linestyle='--', label='Historical Avg')
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Crimes")
        ax.set_title("Future Predictions vs Historical Average")
        ax.legend()
        st.pyplot(fig)

    # Download cleaned data
    st.sidebar.header("Download Data")
    csv_data = df_model.to_csv(index=False).encode()
    st.sidebar.download_button("Download Cleaned Data CSV", data=csv_data, file_name='cleaned_data.csv')

if __name__ == '__main__':
    main()
