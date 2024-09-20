import pandas as pd
import json
import feather
# Load the spaCy English model
''''''# Load the Amazon reviews from json file
path = 'Video_Games.json'
with open(path, 'r') as f:
    data = [json.loads(line) for line in f]

data_frame = pd.DataFrame.from_records(data)

# Rename overall -> stars
data_frame = data_frame.rename(columns={'overall': 'stars'})
data_frame.to_csv('Hole_Data_Frame.csv')

print(data_frame.columns)

# Extract the reviews and rating to a separate DataFrame
reviews_df = data_frame[['reviewText', 'stars', 'summary', 'asin']]
# Save the reviews,stars to a separate feather file
reviews_df.to_csv('Video_Games_reviews.csv')


# Load amazon reviews dataset
path_to_feather = "Video_Games_reviews.feather"
# Assign values to [data] pandas Dataframe
data = pd.read_feather(path_to_feather, use_threads=True)
# Reset the index
data.reset_index(drop=True, inplace=True)
data.to_csv("dataset.csv", index=False)
print(data.shape)
# Split the dataframe into three parts
total_rows = len(data)
split_size = total_rows // 3


df_part1 = data[:split_size].reset_index(drop=True)
df_part2 = data[split_size:2*split_size].reset_index(drop=True)
df_part3 = data[2*split_size:].reset_index(drop=True)


print(df_part1.shape, df_part2.shape, df_part3.shape)
# Save the three parts to separate feather files
df_part1.to_feather('part1.feather')
df_part2.to_feather('part2.feather')
df_part3.to_feather('part3.feather')

