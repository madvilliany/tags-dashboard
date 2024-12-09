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

# Fix missing or misnamed columns in TAGS dataset
tags_df.rename(columns={'Duration': 'Duration (seconds)'}, inplace=True)
if 'Duration (seconds)' not in tags_df.columns:
    if 'Duration' in tags_df.columns:
        tags_df['Duration (seconds)'] = tags_df['Duration'].apply(convert_duration_to_seconds)
    else:
        tags_df['Duration (seconds)'] = 0

# Fix missing or misnamed columns in TRACKS dataset
tracks_df.rename(columns={'Song': 'Track', 'Category': 'Genre'}, inplace=True)

# Handle missing columns in TRACKS dataset
if 'Duration (seconds)' not in tracks_df.columns:
    if 'Duration' in tracks_df.columns:
        tracks_df['Duration (seconds)'] = tracks_df['Duration'].apply(convert_duration_to_seconds)
    else:
        tracks_df['Duration (seconds)'] = 0  # Default to 0 if missing
if 'Track' not in tracks_df.columns:
    tracks_df['Track'] = "Unknown Track"
if 'Genre' not in tracks_df.columns:
    tracks_df['Genre'] = "Unknown Genre"

# Validate datasets
tags_required_columns = ['Tag', 'Artist', 'Duration (seconds)', 'Title']
tracks_required_columns = ['Track', 'Artist', 'Genre', 'Duration (seconds)']

missing_columns_tags = [col for col in tags_required_columns if col not in tags_df.columns]
missing_columns_tracks = [col for col in tracks_required_columns if col not in tracks_df.columns]

if missing_columns_tags:
    st.error(f"TAGS dataset is missing columns: {', '.join(missing_columns_tags)}")
if missing_columns_tracks:
    st.error(f"TRACKS dataset is missing columns: {', '.join(missing_columns_tracks)}")
else:
    # Dashboard Title and Introduction
    st.title("🎵 TAGS & TRACKS Dashboard")
    st.markdown("Explore, filter, and visualize your TAGS and TRACKS data.")

    # Tabs for Navigation
    tab1, tab2 = st.tabs(["TAGS", "TRACKS"])

    # TAGS Section
    with tab1:
        st.header("TAGS Data")
        total_tags = len(tags_df['Tag'].unique())
        total_duration_tags = tags_df['Duration (seconds)'].sum() / 60
        avg_duration_tags = tags_df['Duration (seconds)'].mean() / 60

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tags", total_tags)
        with col2:
            st.metric("Total Duration (minutes)", round(total_duration_tags, 2))
        with col3:
            st.metric("Average Duration (minutes)", round(avg_duration_tags, 2))

        st.subheader("Tag Relationships")
        max_edges = st.slider("Max edges in the network graph", 10, 500, 100)
        plot_tag_relationships(tags_df, max_edges=max_edges)

    # TRACKS Section
    with tab2:
        st.header("TRACKS Data")
        total_tracks = len(tracks_df)
        avg_duration_tracks = tracks_df["Duration (seconds)"].mean() / 60
        total_duration_tracks = tracks_df["Duration (seconds)"].sum() / 60

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tracks", total_tracks)
        with col2:
            st.metric("Average Track Duration (minutes)", round(avg_duration_tracks, 2))
        with col3:
            st.metric("Total Track Duration (minutes)", round(total_duration_tracks, 2))

        st.sidebar.header("Track Filters")
        selected_artist = st.sidebar.multiselect("Filter by Artist", tracks_df["Artist"].unique())
        selected_genre = st.sidebar.multiselect("Filter by Genre", tracks_df["Genre"].unique())

        filtered_tracks = tracks_df[
            (tracks_df["Artist"].isin(selected_artist) if selected_artist else True) &
            (tracks_df["Genre"].isin(selected_genre) if selected_genre else True)
        ]

        st.subheader("Top Tracks by Duration")
        top_tracks = filtered_tracks.sort_values("Duration (seconds)", ascending=False).head(10)
        if not top_tracks.empty:
            fig = px.bar(
                top_tracks,
                x="Track",
                y="Duration (seconds)",
                color="Artist",
                title="Top Tracks by Duration",
                labels={"Track": "Track Name", "Duration (seconds)": "Duration"}
            )
            st.plotly_chart(fig)
        else:
            st.warning("No tracks found for the selected filters.")
