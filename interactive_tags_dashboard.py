import streamlit as st
import pandas as pd
import plotly.express as px


# Trending Tags Over Time
if 'Date' in tags_df.columns:
    tags_df['Date'] = pd.to_datetime(tags_df['Date'], errors='coerce')
    if tags_df['Date'].notna().any():
        tag_trends = tags_df.groupby(['Tag', 'Date']).size().reset_index(name='Count')
        fig = px.line(tag_trends, x='Date', y='Count', color='Tag', title='Trending Tags Over Time')
        st.plotly_chart(fig)
    else:
        st.warning("The 'Date' column contains no valid dates. Skipping trending tags analysis.")
else:
    st.warning("The 'Date' column is missing. Skipping trending tags analysis.")

required_columns = ['Tag', 'Artist', 'Duration (seconds)', 'Date']
missing_columns = [col for col in required_columns if col not in tags_df.columns]

if missing_columns:
    st.warning(f"The dataset is missing the following required columns: {', '.join(missing_columns)}")
else:
    tags_df['Date'] = pd.to_datetime(tags_df['Date'], errors='coerce')


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
st.image("logo.png", width=200)


# Dashboard layout
st.title("TAGS Dashboard with Interactive Charts")
st.sidebar.header("Filter Options")

search_query = st.text_input("Search Tags or Artists:")
filtered_data = tags_df[
    tags_df['Tag'].str.contains(search_query, na=False, case=False) |
    tags_df['Artist'].str.contains(search_query, na=False, case=False)
]

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

# Overview Metrics (convert seconds to minutes)
total_duration_minutes = tags_df['Duration (seconds)'].sum() / 60
average_duration_minutes = tags_df['Duration (seconds)'].mean() / 60

st.subheader("Overview Metrics")
st.metric("Total Tags", len(tags_df['Tag'].unique()))
st.metric("Total Duration (minutes)", round(total_duration_minutes, 2))
st.metric("Average Duration (minutes)", round(average_duration_minutes, 2))

# Visualize Trending Tags Over Time
fig = px.line(tag_trends, x='Date', y='Count', color='Tag', title='Trending Tags Over Time')
st.plotly_chart(fig)
top_artists = tags_df.groupby('Artist')['Duration (seconds)'].sum().sort_values(ascending=False).head(10)
fig = px.bar(top_artists, x=top_artists.index, y=top_artists.values, title="Top Artists by Total Duration")
st.plotly_chart(fig)

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


if st.checkbox("Show Artist Details"):
    selected_artist = st.selectbox("Select an Artist", tags_df['Artist'].dropna().unique())
    artist_data = tags_df[tags_df['Artist'] == selected_artist]
    total_artist_duration = artist_data['Duration (seconds)'].sum() / 60
    st.metric(f"Total Duration for {selected_artist} (minutes)", round(total_artist_duration, 2))
    st.dataframe(artist_data)



from itertools import combinations

# Overview Metrics for Filtered Data
filtered_duration_minutes = filtered_data['Duration (seconds)'].sum() / 60
filtered_average_duration = filtered_data['Duration (seconds)'].mean() / 60

st.metric("Filtered Total Duration (minutes)", round(filtered_duration_minutes, 2))
st.metric("Filtered Average Duration (minutes)", round(filtered_average_duration, 2))

#fig = px.bar(data, x="X", y="Y", color="Z", color_continuous_scale="Viridis")

st.write("Welcome to the W-ESN TAGS Dashboard! Use the filters to explore tag data.")

csv = filtered_data.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv'
)

st.metric("Total Tags", len(tags_df['Tag'].unique()), help="Total number of unique tags in the dataset.")
st.metric("Total Duration (minutes)", round(total_duration_minutes, 2), help="Sum of all track durations in minutes.")

from itertools import combinations

def calculate_cooccurrences(df):
    grouped_tags = df.groupby('Title')['Tag'].apply(list)
    cooccurrence_counts = {}
    for tags in grouped_tags:
        for pair in combinations(sorted(set(tags)), 2):
            cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1
    return pd.DataFrame([
        {'Tag1': k[0], 'Tag2': k[1], 'Count': v} for k, v in cooccurrence_counts.items()
    ])

if 'Tag' in tags_df.columns and 'Artist' in tags_df.columns:
    filtered_data = tags_df[
        (tags_df['Tag'].isin(selected_tags)) & 
        (tags_df['Artist'].isin(selected_artists) if selected_artists else True)
    ]
else:
    st.warning("Columns 'Tag' or 'Artist' are missing. Filtering cannot be applied.")
    filtered_data = pd.DataFrame()  # Empty DataFrame


if 'Duration (seconds)' in tags_df.columns:
    total_duration_minutes = tags_df['Duration (seconds)'].sum() / 60
    average_duration_minutes = tags_df['Duration (seconds)'].mean() / 60
    st.metric("Total Duration (minutes)", round(total_duration_minutes, 2))
    st.metric("Average Duration (minutes)", round(average_duration_minutes, 2))
else:
    st.warning("The 'Duration (seconds)' column is missing. Duration metrics cannot be calculated.")


# Consolidate Filtered Data
if 'Tag' in tags_df.columns and 'Artist' in tags_df.columns:
    filtered_data = tags_df[
        (tags_df['Tag'].isin(selected_tags)) & 
        (tags_df['Artist'].isin(selected_artists) if selected_artists else True)
    ]
else:
    st.warning("Columns 'Tag' or 'Artist' are missing. Filtering cannot be applied.")
    filtered_data = pd.DataFrame()

# Consolidate Metrics
if 'Duration (seconds)' in filtered_data.columns:
    filtered_duration_minutes = filtered_data['Duration (seconds)'].sum() / 60
    filtered_average_duration = filtered_data['Duration (seconds)'].mean() / 60
    st.metric("Filtered Total Duration (minutes)", round(filtered_duration_minutes, 2))
    st.metric("Filtered Average Duration (minutes)", round(filtered_average_duration, 2))
else:
    st.warning("The 'Duration (seconds)' column is missing. Metrics cannot be calculated.")

if 'Title' in tags_df.columns:
    cooccurrence_df = calculate_cooccurrences(tags_df)
    cooccurrence_matrix = cooccurrence_df.pivot(index='Tag1', columns='Tag2', values='Count').fillna(0)
    fig = px.imshow(
        cooccurrence_matrix,
        title="Tag Co-occurrence Heatmap",
        labels={'x': "Tag", 'y': "Tag", 'color': "Co-occurrence Count"},
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig)
else:
    st.warning("The 'Title' column is missing. Co-occurrence analysis cannot be performed.")

st.subheader("Filtered Data")
if not filtered_data.empty:
    st.dataframe(filtered_data.head(100))  # Display only the first 100 rows
else:
    st.warning("No data available for the selected filters.")

def display_metrics(dataframe):
    total_duration_minutes = dataframe['Duration (seconds)'].sum() / 60
    average_duration_minutes = dataframe['Duration (seconds)'].mean() / 60
    st.metric("Total Duration (minutes)", round(total_duration_minutes, 2))
    st.metric("Average Duration (minutes)", round(average_duration_minutes, 2))

if not filtered_data.empty:
    display_metrics(filtered_data)
