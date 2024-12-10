import pinecone
import pandas as pd
import matplotlib.pyplot as plt

response = index.query(
    vector=[9] * 384, 
    top_k=10000,
    include_metadata=True
)

# Convert response to DataFrame
data = [
    {
        "id": match["id"],
        **match["metadata"]
    }
    for match in response["matches"]
]
df = pd.DataFrame(data)

# Count tweets by phases of day
tweet_frequency = df['phases_of_day'].value_counts().reset_index()
tweet_frequency.columns = ['phases_of_day', 'tweet_count']

print(tweet_frequency)

# Calculate total number of tweets for each region
tweets_by_region = df['region'].value_counts().reset_index()

# Rename the columns for better clarity
tweets_by_region.columns = ['region', 'tweet_count']

# Display the result
print(tweets_by_region)


# Assuming retweet_count and like_count are part of metadata
if 'retweet_count' in df.columns and 'like_count' in df.columns:
    # Group by phases of day and calculate average engagement
    engagement_by_phase = df.groupby('phases_of_day')[['retweet_count', 'like_count']].mean().reset_index()

    print(engagement_by_phase)

# Pie-chart

# Calculate tweet frequency by phases of day
tweet_frequency = df['phases_of_day'].value_counts().reset_index()
tweet_frequency.columns = ['phases_of_day', 'tweet_count']

# Plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    tweet_frequency['tweet_count'],
    labels=tweet_frequency['phases_of_day'],
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired.colors
)

plt.title("Tweet Frequency by Phases of Day", fontsize=16)
plt.show()


# Get the top 3 countries based on tweet count
top_3_countries = tweets_by_region.head(3)['region'].tolist()

# Filter the data to only include the top 4 countries
filtered_df = df[df['region'].isin(top_3_countries)]

# Set up the figure for multiple pie charts (one for each country)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes = axes.flatten()

# Iterate over the top 3 countries and plot a pie chart for each
for i, country in enumerate(top_3_countries):
    # Filter data for the specific country
    country_data = filtered_df[filtered_df['region'] == country]

    # Count tweet frequency for each phase of day
    phase_counts = country_data['phases_of_day'].value_counts()

    # Plot pie chart
    axes[i].pie(
        phase_counts,
        labels=phase_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Paired.colors
    )
    axes[i].set_title(f"Tweet Frequency by Phases of Day for {country}", fontsize=14)

# Display the pie charts
plt.tight_layout()
plt.show()
