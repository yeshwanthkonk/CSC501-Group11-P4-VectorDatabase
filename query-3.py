import pandas as pd
import matplotlib.pyplot as plt

# Example data for all tweets (combine data into a DataFrame)
data = {
    "Region": [
        "United States", "United Kingdom", "Russian Federation", "Azerbaijan",
        "Israel", "Iraq", "United Arab Emirates", "Ukraine", "Afghanistan", "Egypt", "Germany"
    ],
    "#BREAKING: Al Arabiya sources: #Russian ambassador in #Khartoum found drowned in the swimming pool at home": [6869, 2948, 26, 101, 5, 10, 6, 5, 8, 7, 4],
    "Dianne Feinstein: No Evidence Confirms Trump-Russia Collusion #MAGA #WakeUpAmerica": [9890, 16, 2, 59, 0, 1, 5, 7, 10, 0, 0],
    "RT mitchellvii: Trump just spoke to Putin about Syria. I previously said Trump hit Syria to prepare for 'safe zones.'": [7486, 2256, 18, 159, 20, 9, 11, 6, 11, 3, 1],
    "Fremont Teachers Protest Against Trump Education Pick DeVos": [9961, 2, 0, 27, 0, 0, 2, 1, 2, 0, 0],
    "Putin says US will have to shed 755 from diplomatic staff": [8910, 763, 38, 198, 11, 10, 20, 5, 19, 0, 0]
}

df = pd.DataFrame(data)

# Remove the "Unknown" region
df = df[df["Region"] != "Unknown"]

# Set Region as the index for better plotting
df.set_index("Region", inplace=True)

# Plot the line chart
plt.figure(figsize=(14, 8))  # Increased figure size
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

# Add chart details
plt.title("Tweet Similarity Variations by Region", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, fontsize=10, ha='right')  # Rotated and aligned to the right

# Place legend at the right side of the chart
plt.legend(title="Tweets", fontsize=6, bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)

plt.tight_layout()

# Show the chart
plt.show()
