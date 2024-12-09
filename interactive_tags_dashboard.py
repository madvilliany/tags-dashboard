import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Page Configuration
st.set_page_config(
    page_title="TAGS Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
@st.cache_data
def load_data(file_path):
    """Load the dataset and convert durations to seconds."""
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

def plot_tag_diversity(dataframe):
    """Plot tag diversity by unique artists."""
    tag_diversity = dataframe.groupby('Tag')['Artist'].nunique().sort_values(ascending=False)
    fig = px.bar(
        tag_diversity, 
        x=tag_diversity.index, 
        y=tag_diversity.values,
        title="Tag Diversity by Unique Artists",
        labels={"x": "Tag", "y": "Unique Artists"}
    )
    st.plotly_chart(fig)

def plot_top_artists(dataframe, n=10):
    """Plot top N artists by total duration."""
    top_artists = dataframe.groupby('Artist')['Duration (seconds)'].sum().sort_values(ascending=False).head(n)
    fig = px.bar(
        top_artists, 
        x=top_artists.index, 
        y=top_artists.values, 
        title="Top Artists by Total Duration",
        labels={"x": "Artist", "y": "Total Duration (seconds)"}
    )
    st.plotly_chart(fig)

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

# Load Data
data_file = "TAGS.csv"  # Replace with your file path
tags_df = load_data(data_file)

# Validate Columns
required_columns = ['Tag', 'Artist', 'Duration (seconds)', 'Title']
missing_columns = [col for col in required_columns if col not in tags_df.columns]

if missing_columns:
    st.error(f"The dataset is missing the following required columns: {', '.join(missing_columns)}")
else:
    # App Title and Description
    st.title("🎵 TAGS Dashboard")
    st.markdown("""
        Welcome to the TAGS Dashboard! Explore, filter, and visualize your data. Use the sidebar for filtering and customizations.
    """)

    # Sidebar Filters
    st.sidebar.header("Filter Options")
    selected_tags = st.sidebar.multiselect("Select Tags", options=tags_df['Tag'].unique())
    selected_artists = st.sidebar.multiselect("Select Artists", options=tags_df['Artist'].unique())

    # Filter Data
    filtered_data = tags_df[
        (tags_df['Tag'].isin(selected_tags)) &
        (tags_df['Artist'].isin(selected_artists) if selected_artists else True)
    ]

    # Metrics Section
    total_duration_minutes, average_duration_minutes = calculate_metrics(tags_df)
    filtered_total, filtered_avg = calculate_metrics(filtered_data)

    st.subheader("Overview Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tags", len(tags_df['Tag'].unique()))
    with col2:
        st.metric("Total Duration (minutes)", total_duration_minutes)
    with col3:
        st.metric("Average Duration (minutes)", average_duration_minutes)

    st.subheader("Filtered Data Metrics")
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Filtered Total Duration (minutes)", filtered_total)
    with col5:
        st.metric("Filtered Average Duration (minutes)", filtered_avg)

    # Visualizations
    st.subheader("Tag Diversity")
    plot_tag_diversity(tags_df)

    st.subheader("Top Artists by Duration")
    plot_top_artists(tags_df)

    st.subheader("Tag Relationships")
    max_edges = st.slider("Max edges in the network graph", 10, 500, 100)
    plot_tag_relationships(tags_df, max_edges=max_edges)

    # Filtered Data Display
    st.subheader("Filtered Data")
    num_rows = st.slider("Number of rows to display", min_value=10, max_value=100, value=50)
    st.dataframe(filtered_data.head(num_rows))

    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_data_{len(filtered_data)}_rows.csv",
        mime="text/csv"
    )


import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="TAGS & TRACKS Dashboard",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_track_data(file_path):
    df = pd.read_csv(file_path)
    if 'Duration' in df.columns:
        df['Duration (seconds)'] = df['Duration'].apply(convert_duration_to_seconds)
    return df

def convert_duration_to_seconds(duration_str):
    try:
        time_parts = list(map(int, duration_str.strip().split(":")))
        if len(time_parts) == 3:
            return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:
            return time_parts[0] * 60 + time_parts[1]
    except:
        return None

# Load Data
tags_file = "TAGS.csv"
tracks_file = "wesn_track_data_120124.csv"

tags_df = load_data(tags_file)
tracks_df = load_track_data(tracks_file)

# Validate Columns for Tags and Tracks
tags_required_columns = ['Tag', 'Artist', 'Duration (seconds)', 'Title']
tracks_required_columns = ['Track', 'Artist', 'Genre', 'Duration (seconds)']

tags_missing_columns = [col for col in tags_required_columns if col not in tags_df.columns]
tracks_missing_columns = [col for col in tracks_required_columns if col not in tracks_df.columns]

if tags_missing_columns:
    st.error(f"The tags dataset is missing the following columns: {', '.join(tags_missing_columns)}")
if tracks_missing_columns:
    st.error(f"The tracks dataset is missing the following columns: {', '.join(tracks_missing_columns)}")
else:
    # Main Layout
    st.title("🎵 TAGS & TRACKS Dashboard")
    st.markdown("Explore, filter, and visualize your TAGS and TRACKS data.")
    
    tab1, tab2 = st.tabs(["TAGS", "TRACKS"])

    # TAGS Section
    with tab1:
        st.header("TAGS Data")
        # Add existing tags analysis and visualizations here

    # TRACKS Section
    with tab2:
        st.header("TRACKS Data")

        # Track Metrics
        total_tracks = len(tracks_df)
        avg_track_duration = tracks_df['Duration (seconds)'].mean() / 60 if 'Duration (seconds)' in tracks_df else None
        total_track_duration = tracks_df['Duration (seconds)'].sum() / 60 if 'Duration (seconds)' in tracks_df else None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tracks", total_tracks)
        with col2:
            st.metric("Average Track Duration (minutes)", round(avg_track_duration, 2) if avg_track_duration else "N/A")
        with col3:
            st.metric("Total Track Duration (minutes)", round(total_track_duration, 2) if total_track_duration else "N/A")

        # Filters
        st.sidebar.header("Track Filters")
        selected_artist = st.sidebar.multiselect("Select Artist(s)", options=tracks_df['Artist'].unique())
        selected_genre = st.sidebar.multiselect("Select Genre(s)", options=tracks_df['Genre'].unique())

        filtered_tracks = tracks_df[
            (tracks_df['Artist'].isin(selected_artist) if selected_artist else True) &
            (tracks_df['Genre'].isin(selected_genre) if selected_genre else True)
        ]

        # Visualizations
        st.subheader("Top Tracks by Duration")
        top_tracks = filtered_tracks.sort_values('Duration (seconds)', ascending=False).head(10)
        if not top_tracks.empty:
            fig = px.bar(
                top_tracks,
                x='Track',
                y='Duration (seconds)',
                color='Artist',
                title="Top 10 Tracks by Duration",
                labels={"Track": "Track Name", "Duration (seconds)": "Duration (seconds)", "Artist": "Artist"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No tracks found for the selected filters.")

        st.subheader("Tracks by Genre")
        genre_count = filtered_tracks['Genre'].value_counts()
        if not genre_count.empty:
            fig = px.pie(
                names=genre_count.index,
                values=genre_count.values,
                title="Track Distribution by Genre"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No genres available for the selected filters.")

        # Filtered Track Data Display
        st.subheader("Filtered Tracks")
        num_rows = st.slider("Number of rows to display", min_value=10, max_value=100, value=20)
        st.dataframe(filtered_tracks.head(num_rows))

        # Download Filtered Tracks
        csv = filtered_tracks.to_csv(index=False)
        st.download_button(
            label="Download Filtered Tracks as CSV",
            data=csv,
            file_name=f"filtered_tracks_{len(filtered_tracks)}_rows.csv",
            mime="text/csv"
        )
