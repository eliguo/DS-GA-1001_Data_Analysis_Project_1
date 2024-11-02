#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal


# In[2]:


df = pd.read_csv('movieReplicationSet.csv')
df


# ## Q1 - Are movies that are more popular  rated higher than movies that are less popular? 
# 
# ##### [Hint: You can do a median-split of popularity to determinehigh vs. low popularitymovies]

# In[3]:


# approach one

# get subset rating dataset
df_rating = df.iloc[:, :400]

# find the count of each movie
rating_counts = df_rating.count()
df_counts = pd.DataFrame({'movie': df.columns[:400], 'rating_count': rating_counts})

median_count = rating_counts.median()

# Separate the ratings into two different groups based on popularity
mask = df_counts['rating_count'] > median_count
high_popularity_movies = df_counts[mask]['movie'].values
low_popularity_movies = df_counts[~mask]['movie'].values

high_pop_ratings = df[high_popularity_movies].median()
low_pop_ratings = df[low_popularity_movies].median()

statistic, p_value = mannwhitneyu(high_pop_ratings, low_pop_ratings, alternative='greater')

print(f'Mann-Whitney U statistic: {statistic}, with p-value: {p_value}')

if p_value <= 0.005:
    print("The differences in the ratings between high popularity group and low popularity group are unlikely to be due to chance alone. Therefore, movies that are more popular are rated higher than movies that are less popular.")
else:
    print("There is no significant difference in ratings between the high popularity group and low popularity group.")


combined_ratings_df = pd.DataFrame({
    'Rating': np.concatenate([high_pop_ratings.values, low_pop_ratings.values]),
    'Popularity': ['High'] * len(high_pop_ratings) + ['Low'] * len(low_pop_ratings)
})

plt.figure(figsize=(10, 8))

# Set up the figure and axes for histogram and box plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Histogram
ax1.hist(high_pop_ratings.dropna(), bins=6, alpha=0.7, color='royalblue', label='Popular Movies', edgecolor='white')
ax1.hist(low_pop_ratings.dropna(), bins=6, alpha=0.7, color='coral', label='Not Popular Movies', edgecolor='white')
ax1.set_title('Histogram of Movie Ratings by Popularity')
ax1.set_xlabel('Ratings')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(visible=True, linestyle="--", alpha=0.7)

# Box plot 
custom_palette = {"Low": "coral", "High": "royalblue"}
sns.boxplot(x='Popularity', y='Rating', data=combined_ratings_df, palette=custom_palette, width=0.5, ax=ax2)
ax2.set_title('Box Plot of Movie Ratings by Popularity')
ax2.set_xlabel('Popularity')
ax2.set_ylabel('Ratings')
ax2.grid(visible=True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# In[4]:


high_popularity_columns = df_rating.columns[mask]
low_popularity_columns = df_rating.columns[~mask]

# Separate ratings into two groups and flatten, removing NaNs
high_popularity_ratings_clean = df_rating[high_popularity_columns].values.flatten()
low_popularity_ratings_clean = df_rating[low_popularity_columns].values.flatten()

high_popularity_ratings_clean = high_popularity_ratings_clean[~np.isnan(high_popularity_ratings_clean)]
low_popularity_ratings_clean = low_popularity_ratings_clean[~np.isnan(low_popularity_ratings_clean)]


# Mann-Whitney U test
statistic, p_value = mannwhitneyu(low_popularity_ratings_clean, high_popularity_ratings_clean, alternative='greater')

# Print the results
print(f'Mann-Whitney U statistic: {statistic}, with p-value: {p_value}')
if p_value <= 0.005:
    print("Significant difference: More popular movies are rated higher.")
else:
    print("No significant difference in ratings between high and low popularity groups.")

# Prepare data for plotting
high_popularity_df = df_rating[high_popularity_columns].melt(var_name='Movie', value_name='Rating')
high_popularity_df['popularity'] = 'high'

low_popularity_df = df_rating[low_popularity_columns].melt(var_name='Movie', value_name='Rating')
low_popularity_df['popularity'] = 'low'



# Combine both DataFrames and drop NaNs
combined_ratings_df = pd.concat([high_popularity_df,low_popularity_df], ignore_index=True)
combined_ratings_df = combined_ratings_df.dropna()

# Set up the figure and axes
fig, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Histogram f
ax3.hist(high_popularity_ratings_clean, bins=8, alpha=0.7, color='royalblue', label='Popular Movies', edgecolor='white')
ax3.hist(low_popularity_ratings_clean, bins=8, alpha=0.7, color='coral', label='Not Popular Movies', edgecolor='white')
ax3.set_title('Histogram of Movie Ratings for Popularity Groups')
ax3.set_xlabel('Ratings')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(visible=True, linestyle="--", alpha=0.7)

# Box plot for popularity groups
custom_palette = {"low": "coral", "high": "royalblue"}
sns.boxplot(x='popularity', y='Rating', palette=custom_palette, data=combined_ratings_df, width=0.5, ax=ax4)
ax4.set_title('Box Plot of Movie Ratings by Popularity')
ax4.set_xlabel('Popularity')
ax4.set_ylabel('Ratings')
ax4.grid(visible=True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# ## Q2 - Are movies that are newer rated differently than movies that are older? 
# ##### [Hint: Do a median split of year of release to contrast movies in terms of whether they are old or new

# In[5]:


# approach 1

df_rating = df.iloc[:, :400]

# Find the median year from the column names
year = df_rating.columns.str.extract(r'\((\d+)\)')[0]  # Extract year
year = pd.to_numeric(year, errors='coerce') 

median_year = year.median()

# Use the mask to separate the column names
mask = year >= median_year
old_columns = df_rating.columns[~mask.fillna(False)] 
new_columns = df_rating.columns[mask.fillna(False)] 

# Separate the rating into two different groups
old_ratings = df_rating[old_columns].median()
new_ratings = df_rating[new_columns].median()

# Conduct Mann-Whitney U test to check for differences
statistic, p_value = mannwhitneyu(old_ratings, new_ratings, alternative='two-sided')

# Print the results
print(f'Mann-Whitney U statistic: {statistic}, with p-value: {p_value}')

if p_value <= 0.005:
    print("The difference in the ratings between new movies and old movies is unlikely due to chance alone. Movies that are newer rated differently than movies that are older.")
else:
    print("There is no significant difference in ratings between new movies and old movies.")

# Prepare data for plotting
combined_ratings_df = pd.DataFrame({
    'Rating': np.concatenate([new_ratings.values, old_ratings.values]),
    'Time': ['old'] * len(old_ratings) + ['new'] * len(new_columns)
})

plt.figure(figsize=(10, 8))

# Set up the figure and axes for histogram and box plot
fig, (ax5, ax6) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Histogram 
ax5.hist(old_ratings.dropna(), bins=6, alpha=0.7, color='royalblue', label='old', edgecolor='white')
ax5.hist(new_ratings.dropna(), bins=8, alpha=0.7, color='coral', label='new', edgecolor='white')
ax5.set_title('Histogram of Movie Ratings by Time')
ax5.set_xlabel('Ratings')
ax5.set_ylabel('Frequency')
ax5.legend()
ax5.grid(visible=True, linestyle="--", alpha=0.7)

# Box plot
sns.boxplot(x='Time', y='Rating', data=combined_ratings_df, palette="tab10", width=0.5, ax=ax6)
ax6.set_title('Box Plot of Movie Ratings by Time')
ax6.set_xlabel('Time')
ax6.set_ylabel('Ratings')
ax6.grid(visible=True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# In[6]:


# approach 2

# Use the mask to separate the column names
old_columns = df_rating.columns[~mask.fillna(False)]
new_columns = df_rating.columns[mask.fillna(False)]    

# Separate the rating into two different groups
old_ratings = df_rating[old_columns]
new_ratings = df_rating[new_columns]  

old_ratings_clean = old_ratings.values.flatten()
new_ratings_clean = new_ratings.values.flatten()

old_ratings_clean = old_ratings_clean[~np.isnan(old_ratings_clean)]
new_ratings_clean = new_ratings_clean[~np.isnan(new_ratings_clean)]

# Conduct Mann-Whitney U test to check for differences
statistic, p_value = mannwhitneyu(old_ratings_clean, new_ratings_clean, alternative='two-sided')

# Print the results
print(f'Mann-Whitney U statistic: {statistic}, with p-value: {p_value}')

if p_value <= 0.005:
    print("The difference in the ratings between new movies and old movies is unlikely due to chance alone. Movies that are newer rated differently than movies that are older.")
else:
    print("There is no significant difference in ratings between new movies and old movies.")

new_ratings_df = df[new_columns].melt(var_name='Movie', value_name='Rating')
new_ratings_df['time'] = 'new'

old_ratings_df = df[old_columns].melt(var_name='Movie', value_name='Rating')
old_ratings_df['time'] = 'old'

# Combine both DataFrames
combined_ratings_df = pd.concat([new_ratings_df, old_ratings_df], ignore_index=True)
combined_ratings_df = combined_ratings_df.dropna()

fig, (ax7, ax8) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Histogram 
ax7.hist(new_ratings_clean, bins=8, alpha=0.7, color='royalblue', label='New Movies', edgecolor='white')
ax7.hist(old_ratings_clean, bins=8, alpha=0.7, color='coral', label='Old Movies', edgecolor='white')
ax7.set_title('Histogram of Movie Ratings for Old and New Movies')
ax7.set_xlabel('Ratings')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(visible=True, linestyle="--", alpha=0.7)

# Box plot
sns.boxplot(x='time', y='Rating', data=combined_ratings_df, palette="tab10", width=0.5, ax=ax8)
ax8.set_title('Box Plot of Movie Ratings by New/Old')
ax8.set_xlabel('Movie Release Time')
ax8.set_ylabel('Ratings')
ax8.grid(visible=True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


#  ## Q3 - Is enjoyment of ‘Shrek(2001)’ gendered, i.e. do male and female viewers rate it differently?

# In[7]:


df_Shrek = df[['Shrek (2001)','Gender identity (1 = female; 2 = male; 3 = self-described)']]
df_Shrek_female = df_Shrek['Shrek (2001)'][df_Shrek['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]
df_Shrek_male = df_Shrek['Shrek (2001)'][df_Shrek['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]
df_Shrek_other = df_Shrek['Shrek (2001)'][df_Shrek['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 3]

female_clean = df_Shrek_female.dropna()
male_clean = df_Shrek_male.dropna()
other_clean = df_Shrek_other.dropna()

statistic, p_value = kruskal(female_clean, male_clean, other_clean)

# Print the results
print(f'Kruskal-Wallis test statistic: {statistic}, with p-value: {p_value}')

if p_value <= 0.005:
    print("The difference in the ratings between new movies and old movies is unlikely due to chance alone. Movies that are newer rated differently than movies that are older.")
else:
    print("There is no significant difference in ratings between new movies and old movies.")

# Set up the figure and axes
fig, (ax9, ax10) = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Histogram
ax9.hist(female_clean, bins=8, alpha=0.7, color='royalblue', label='Female', edgecolor='white')
ax9.hist(male_clean, bins=7, alpha=0.7, color='coral', label='Male', edgecolor='white')
ax9.hist(other_clean, bins=7, alpha=0.7, color = 'teal', label='other', edgecolor='white')
ax9.set_title('Histogram of Movie Ratings for Old and New Movies')
ax9.set_xlabel('Ratings')
ax9.set_ylabel('Frequency')
ax9.legend()
ax9.grid(visible=True, linestyle="--", alpha=0.7)

# Box plot
sns.boxplot(x='Gender identity (1 = female; 2 = male; 3 = self-described)', y='Shrek (2001)', data=df_Shrek, palette="tab10", width=0.5, ax=ax10)
ax10.set_title('Box Plot of Movie Ratings by Gender')
ax10.set_xlabel('Movie Release Time')
ax10.set_ylabel('Ratings')
ax10.grid(visible=True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# ## What proportion of movies are rated differently by male and female viewer

# In[8]:


different_num = 0

for i in range(0,400):
    df_ = df.iloc[:,[i,474]]
    column_title = df.columns[i]
    df_female = df_[column_title][df_['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 1]
    df_male = df_[column_title][df_['Gender identity (1 = female; 2 = male; 3 = self-described)'] == 2]

    female_clean = df_female.dropna()
    male_clean = df_male.dropna()

    # Conduct Mann-Whitney U test to check for differences
    statistic, p_value = mannwhitneyu(female_clean, male_clean, alternative='two-sided')

    # Print the results
    #print(f'Mann-Whitney U statistic: {statistic}, with p-value: {p_value}')

    if p_value <= 0.005:
        different_num += 1
        
proportion_different = different_num / 400
print(f'proportion of movies are rated differently by male and female viewer: {proportion_different}')


# In[ ]:





# In[ ]:




