import pinecone
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


regions = ["Afghanistan", "Azerbaijan", "Egypt", "France", "Germany", "Iraq", "Israel","Italy","Japan","Russian Federation","Serbia","Turkey","Ukraine","United Arab Emirates","United Kingdom","United States"];

for region in regions:

  response = index.query(
      vector=[0] * 384,
      top_k=10000,
      filter={"region": {"$eq": region}},
      include_metadata=True
  )

  # Extract the data
  top_10000_vectors = response['matches']

  hashtags_by_region = {}

  for vector in top_10000_vectors:
      region = vector["metadata"].get("region", "Unknown")
      hashtags = vector["metadata"].get("hashtags", [])
      if region not in hashtags_by_region:
          hashtags_by_region[region] = []
      hashtags_by_region[region].extend(hashtags)

  # Step 2: Count hashtag frequencies for each region
  hashtag_counts_by_region = {region: Counter(hashtags) for region, hashtags in hashtags_by_region.items()}

  # Step 3: Visualize the data
  for region, hashtag_counts in hashtag_counts_by_region.items():
      # Convert the hashtag counts to a DataFrame for easier plotting
      df = pd.DataFrame(hashtag_counts.items(), columns=["Hashtag", "Count"]).sort_values(by="Count", ascending=False)

      # Plot the top hashtags for this region
      plt.figure(figsize=(10, 6))
      plt.bar(df["Hashtag"][:10], df["Count"][:10], color="skyblue")
      plt.title(f"Top Hashtags in {region}", fontsize=16)
      plt.xlabel("Hashtags", fontsize=12)
      plt.ylabel("Frequency", fontsize=12)
      plt.xticks(rotation=45, fontsize=10)
      plt.tight_layout()
      plt.show()
