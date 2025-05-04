import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
@st.cache_data
def load_data():
    # 1. Load RFM
    rfm = pd.read_csv("data/processed/rfm.csv")

    # 2. Scale and cluster
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(X).astype(str)

    return rfm


@st.cache_data
def get_profiles(df):
    # Compute profile stats
    profile = (
        df.groupby("Segment")[["Recency","Frequency","Monetary"]]
        .mean()
        .round(1)
        .reset_index()
    )
    profile["Count"] = df["Segment"].value_counts().loc[profile["Segment"]].values
    return profile

df = load_data()
profiles = get_profiles(df)

st.title("📊 Customer Segmentation Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
segments = df["Segment"].unique().tolist()
sel = st.sidebar.multiselect("Select segment(s)", segments, default=segments)

filtered = df[df["Segment"].isin(sel)]
profile_filtered = profiles[profiles["Segment"].isin(sel)]

# Display overall metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df["CustomerID"].unique()))
col2.metric("Total Segments", len(segments))
col3.metric("Displayed Customers", len(filtered["CustomerID"].unique()))

st.markdown("## Segment Profiles")
st.dataframe(profile_filtered.set_index("Segment"))

# Plot segment distribution
st.markdown("## Segment Distribution")
fig, ax = plt.subplots()
counts = filtered["Segment"].value_counts()
ax.bar(counts.index, counts.values)
ax.set_ylabel("Number of Customers")
ax.set_xlabel("Segment")
ax.set_title("Customers per Segment")
st.pyplot(fig)

# 2D scatter of Recency vs Monetary colored by Segment
st.markdown("## Recency vs. Monetary by Segment")
fig2, ax2 = plt.subplots()
for seg_name, group in filtered.groupby("Segment"):
    ax2.scatter(group["Recency"], group["Monetary"], label=seg_name, alpha=0.6)
ax2.set_xlabel("Recency (days)")
ax2.set_ylabel("Monetary (£)")
ax2.legend()
st.pyplot(fig2)

st.markdown("## Frequency Histogram")
fig3, ax3 = plt.subplots()
filtered["Frequency"].hist(bins=20, ax=ax3)
ax3.set_xlabel("Frequency (orders)")
ax3.set_ylabel("Count")
st.pyplot(fig3)

st.markdown("—")
st.markdown("Built with ❤️ using Streamlit")


# Sidebar filters
st.sidebar.header("Filter Options")
min_recency, max_recency = st.sidebar.slider(
    "Recency Range",
    int(df['Recency'].min()),
    int(df['Recency'].max()),
    (int(df['Recency'].min()), int(df['Recency'].max()))
)
min_frequency, max_frequency = st.sidebar.slider(
    "Frequency Range",
    int(df['Frequency'].min()),
    int(df['Frequency'].max()),
    (int(df['Frequency'].min()), int(df['Frequency'].max()))
)
min_monetary, max_monetary = st.sidebar.slider(
    "Monetary Range",
    int(df['Monetary'].min()),
    int(df['Monetary'].max()),
    (int(df['Monetary'].min()), int(df['Monetary'].max()))
)

# Apply filters
filtered_df = df[
    (df['Recency'] >= min_recency) & (df['Recency'] <= max_recency) &
    (df['Frequency'] >= min_frequency) & (df['Frequency'] <= max_frequency) &
    (df['Monetary'] >= min_monetary) & (df['Monetary'] <= max_monetary)
]


import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap of RFM Values by Segment
st.markdown("### Heatmap of RFM Values by Segment")
rfm_pivot = df.pivot_table(index='Segment', values=['Recency', 'Frequency', 'Monetary'], aggfunc='mean')
fig, ax = plt.subplots()
sns.heatmap(rfm_pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
st.pyplot(fig)

# Boxplots of RFM Distributions by Segment
st.markdown("### Boxplots of RFM Distributions by Segment")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(x='Segment', y='Recency', data=df, ax=axes[0])
sns.boxplot(x='Segment', y='Frequency', data=df, ax=axes[1])
sns.boxplot(x='Segment', y='Monetary', data=df, ax=axes[2])
st.pyplot(fig)






st.markdown("### Download Filtered Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name='filtered_data.csv',
    mime='text/csv'
)
