import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx

# Set up page configuration
st.set_page_config(
    page_title="TAGS & TRACKS Dashboard",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load CSV data."""
    return pd.read_csv(file_path)

@st.cache_data
def convert_duration_to_seconds(duration_str):
    """Convert HH:MM:SS or MM:SS to seconds."""
    try:
        time_parts = list(map(int, duration_str.strip().split(":")))
        if len(time_parts) == 3:
            return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:
            return time_parts[0] * 60 + time_parts[1]
        return None
    except:
        return None

def plot_tag_relationships(dataframe, max_edges=100):
    """Generate and display a network graph of tag relationships."""
    grouped_tags = dataframe.groupby('Title')['Tag'].apply(list)
    G = nx.Graph()
    edge_count = 0
    for tags in grouped_tags:
        for pair in combinations(tags, 2):
            if edge_count >= max_edges:
                break
            G.add_edge(pair[0], pair[1])
            edge_count += 1

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, with_labels=True, node_size=50, font_size=10)
    st.pyplot(plt)

# Load datasets
tags_file = "TAGS.csv"
tracks_file = "wesn_track_data.csv"

tags_df = load_data(tags_file)
tracks_df = load_data(tracks_file)

# Validate and handle missing columns in TAGS dataset
if 'Duration (seconds)' not in tags_df.columns:
    # Use the existing 'Duration' column to calculate seconds dynamically
    if 'Duration' in tags_df.columns:
        tags_df['Duration (seconds)'] = tags_df['Duration'].apply(convert_duration_to_seconds)
    else:
        tags_df['Duration (seconds)'] = 0

# Calculate metrics dynamically without altering the original format
if tags_df.empty:
    st.warning("The TAGS dataset is empty. No data to analyze.")
    total_duration_tags, avg_duration_tags = 0, 0
else:
    total_duration_tags = tags_df['Duration (seconds)'].sum() / 60  # Total in minutes
    avg_duration_tags = tags_df['Duration (seconds)'].mean() / 60  # Average in minutes

# Validate and handle missing columns in TRACKS dataset
tracks_df.rename(columns={'Song': 'Track', 'Category': 'Genre'}, inplace=True)
if 'Duration (seconds)' not in tracks_df.columns:
    if 'Duration' in tracks_df.columns:
        tracks_df['Duration (seconds)'] = tracks_df['Duration'].apply(convert_duration_to_seconds)
    else:
        tracks_df['Duration (seconds)'] = 0
tracks_df['Duration (seconds)'] = pd.to_numeric(tracks_df['Duration (seconds)'], errors='coerce').fillna(0)
tracks_df['Track'] = tracks_df.get('Track', "Unknown Track")
tracks_df['Genre'] = tracks_df.get('Genre', "Unknown Genre")

# Tabs for dashboard
tab1, tab2, tab3 = st.tabs(["TAGS", "TRACKS", "Data Overview"])

with tab1:
    st.header("TAGS Data")
    st.metric("Total Duration (minutes)", round(total_duration_tags, 2))
    st.metric("Average Duration (minutes)", round(avg_duration_tags, 2))
    st.subheader("Original Duration Format")
    st.dataframe(tags_df[['Title', 'Duration', 'Tag']])

with tab2:
    st.header("TRACKS Data")
    st.write("Add metrics and filtering for TRACKS here...")

with tab3:
    st.header("Data Overview")
    st.subheader("TAGS Dataset")
    st.dataframe(tags_df)
    st.subheader("TRACKS Dataset")
    st.dataframe(tracks_df)
