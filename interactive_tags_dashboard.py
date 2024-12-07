import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Duration (seconds)'] = df['Duration'].apply(convert_duration_to_seconds)
    return df

def convert_duration_to_seconds(duration_str):
    try:
        time_parts = list(map(int, duration_str.strip().split(":")))
        if len(time_parts) == 3:  # HH:MM:SS
            return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:  # MM:SS
            return time_parts[0] * 60 + time_parts[1]
    except:
        return None

# Load your data file
data_file = "TAGS.csv"  # Replace with your file path
tags_df = load_data(data_file)

# Dashboard layout
st.title("TAGS Dashboard with Interactive Charts")
st.sidebar.header("Filter Options")

# Filter by Tag
unique_tags = tags_df['Tag'].unique()
selected_tags = st.sidebar.multiselect("Select Tags", unique_tags, default=unique_tags[:5])

# Filter by Artist
unique_artists = tags_df['Artist'].unique()
selected_artists = st.sidebar.multiselect("Select Artists", unique_artists)

# Filtered Data
filtered_data = tags_df[
    (tags_df['Tag'].isin(selected_tags)) & 
    (tags_df['Artist'].isin(selected_artists) if selected_artists else True)
]

st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Overview Metrics
st.subheader("Overview Metrics")
st.metric("Total Tags", len(tags_df['Tag'].unique()))
st.metric("Total Duration (s)", tags_df['Duration (seconds)'].sum())
st.metric("Average Duration (s)", tags_df['Duration (seconds)'].mean())

# Interactive Chart: Top Tags by Duration
st.subheader("Top Tags by Total Duration")
top_tags = (
    tags_df.groupby('Tag')['Duration (seconds)']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
fig = px.bar(
    top_tags,
    x="Tag",
    y="Duration (seconds)",
    title="Top 10 Tags by Total Duration",
    labels={"Duration (seconds)": "Total Duration (s)"},
)
st.plotly_chart(fig)

# Interactive Chart: Top Tags by Artist Count
st.subheader("Top Tags by Artist Count")
artist_counts = tags_df.groupby('Tag')['Artist'].nunique().reset_index()
artist_counts = artist_counts.sort_values(by="Artist", ascending=False).head(10)
fig = px.bar(
    artist_counts,
    x="Tag",
    y="Artist",
    title="Top 10 Tags by Unique Artists",
    labels={"Artist": "Unique Artists"},
)
st.plotly_chart(fig)

# Interactive Chart: Duration Distribution
st.subheader("Duration Distribution Across Tags")
fig = px.box(
    tags_df,
    x="Tag",
    y="Duration (seconds)",
    title="Duration Distribution by Tag",
    labels={"Duration (seconds)": "Duration (s)"},
)
st.plotly_chart(fig)

# Artist Listing per Tag
st.subheader("Artists Associated with Selected Tags")
for tag in selected_tags:
    # Convert all artist values to strings to avoid errors
    artists = tags_df[tags_df['Tag'] == tag]['Artist'].dropna().unique()
    st.write(f"**{tag}**: ", ", ".join(map(str, artists)))
