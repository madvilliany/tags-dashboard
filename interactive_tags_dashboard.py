import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import combinations

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load the data and convert durations to seconds."""
    df = pd.read_csv(file_path)
    if 'Duration' in df.columns:
        df['Duration (seconds)'] = df['Duration'].apply(convert_duration_to_seconds)
    return df

def convert_duration_to_seconds(duration_str):
    """Convert HH:MM:SS or MM:SS to seconds."""
    try:
        time_parts = list(map(int, duration_str.strip().split(":")))
        if len(time_parts) == 3:  # HH:MM:SS
            return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:  # MM:SS
            return time_parts[0] * 60 + time_parts[1]
    except:
        return None

def calculate_metrics(dataframe):
    """Calculate total and average duration in minutes."""
    if 'Duration (seconds)' in dataframe.columns:
        total_duration_minutes = dataframe['Duration (seconds)'].sum() / 60
        average_duration_minutes = dataframe['Duration (seconds)'].mean() / 60
        return round(total_duration_minutes, 2), round(average_duration_minutes, 2)
    return None, None

def calculate_cooccurrences(df):
    """Calculate co-occurrence frequencies of tags."""
    if 'Title' in df.columns and 'Tag' in df.columns:
        grouped_tags = df.groupby('Title')['Tag'].apply(list)
        cooccurrence_counts = {}
        for tags in grouped_tags:
            for pair in combinations(sorted(set(tags)), 2):  # Avoid duplicate pairs
                cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1
        return pd.DataFrame([{'Tag1': k[0], 'Tag2': k[1], 'Count': v} for k, v in cooccurrence_counts.items()])
    else:
        st.warning("Columns 'Title' and/or 'Tag' are missing. Co-occurrence analysis skipped.")
        return pd.DataFrame()

# Load the data
data_file = "TAGS.csv"  # Replace with your file path
tags_df = load_data(data_file)

# Validate Columns
required_columns = ['Tag', 'Artist', 'Duration (seconds)']
missing_columns = [col for col in required_columns if col not in tags_df.columns]

if missing_columns:
    st.error(f"The dataset is missing the following required columns: {', '.join(missing_columns)}")
else:
    st.image("logo.png", width=200)
    st.title("TAGS Dashboard with Interactive Charts")
    st.sidebar.header("Filter Options")

    # Sidebar Filters
    search_query = st.text_input("Search Tags or Artists:")
    filtered_data = tags_df[
        tags_df['Tag'].str.contains(search_query, na=False, case=False) |
        tags_df['Artist'].str.contains(search_query, na=False, case=False)
    ]

    selected_tags = st.sidebar.multiselect("Select Tags", options=tags_df['Tag'].unique())
    selected_artists = st.sidebar.multiselect("Select Artists", options=tags_df['Artist'].unique())

    # Filter Data
    filtered_data = tags_df[
        (tags_df['Tag'].isin(selected_tags)) &
        (tags_df['Artist'].isin(selected_artists) if selected_artists else True)
    ]

    # Overview Metrics
    st.subheader("Overview Metrics")
    total_duration_minutes, average_duration_minutes = calculate_metrics(tags_df)
    if total_duration_minutes is not None:
        st.metric("Total Tags", len(tags_df['Tag'].unique()))
        st.metric("Total Duration (minutes)", total_duration_minutes)
        st.metric("Average Duration (minutes)", average_duration_minutes)

    # Filtered Metrics
    st.subheader("Filtered Data Metrics")
    filtered_total, filtered_avg = calculate_metrics(filtered_data)
    if filtered_total is not None:
        st.metric("Filtered Total Duration (minutes)", filtered_total)
        st.metric("Filtered Average Duration (minutes)", filtered_avg)

    # Co-occurrence Analysis
    st.subheader("Tag Co-occurrence Analysis")
    cooccurrence_df = calculate_cooccurrences(tags_df)
    if not cooccurrence_df.empty:
        cooccurrence_matrix = cooccurrence_df.pivot(index='Tag1', columns='Tag2', values='Count').fillna(0)
        fig = px.imshow(
            cooccurrence_matrix,
            title="Tag Co-occurrence Heatmap",
            labels={'x': "Tag", 'y': "Tag", 'color': "Co-occurrence Count"},
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig)

    # Top Artists by Duration
    st.subheader("Top Artists by Duration")
    if 'Artist' in tags_df.columns:
        top_artists = tags_df.groupby('Artist')['Duration (seconds)'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_artists, x=top_artists.index, y=top_artists.values, title="Top Artists by Total Duration")
        st.plotly_chart(fig)

    # Filtered Data Display
    st.subheader("Filtered Data")
    if not filtered_data.empty:
        st.dataframe(filtered_data.head(50))  # Limit rows displayed for performance
        csv = filtered_data.to_csv(index=False)
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")
    else:
        st.warning("No data available for the selected filters.")
