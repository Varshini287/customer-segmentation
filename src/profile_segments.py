import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load RFM data
rfm = pd.read_csv("data/processed/rfm.csv")

# 2. Scale features
features = ['Recency','Frequency','Monetary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# 3. Fit K-Means (choose your k)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. Profile clusters
profile = (
    rfm
    .groupby('Cluster')[features]
    .mean()
    .round(1)
)
profile['Count'] = rfm['Cluster'].value_counts().sort_index().values

print("Cluster profile:\n", profile)

# 5. (Optional) Save cluster assignments
rfm[['CustomerID','Cluster']].to_csv("data/processed/customer_segments.csv", index=False)
print("\n✅ Saved customer_segments.csv")
