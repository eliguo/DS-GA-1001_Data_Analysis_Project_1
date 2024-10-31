import pandas as pd
import sqlite3
from scipy.stats import ks_2samp, kruskal, chi2_contingency

# constants
FILE_PATH = 'movieReplicationSet.csv'
DB_PATH = 'movieReplicationSet.db'
TABLE_NAME = 'movie_ratings'
ALPHA = 0.005

# load data
df = pd.read_csv(FILE_PATH)

# create database and insert data
conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
conn.commit()

# function to get ratings for a specific movie
def get_movie_ratings(movie, conn, table=TABLE_NAME):
    query = f"SELECT `{movie}` FROM {table} WHERE `{movie}` IS NOT NULL"
    return pd.read_sql(query, conn)[movie]

# Question 9
home_along_ratings = get_movie_ratings('Home Alone (1990)', conn)
finding_nemo_ratings = get_movie_ratings('Finding Nemo (2003)', conn)

# Kolmogorov-Smirnov (KS) test
ks_stat, ks_p = ks_2samp(home_along_ratings, finding_nemo_ratings)

print(f"Kolmogorov-Smirnov (KS) test between uniform and normal distributions\nD: {ks_stat}\nP-value: {ks_p}")
print("Conclusion:", "Significant difference\n" if ks_p <= ALPHA else "No significant difference\n")

# Question 10
# define franchise keywords
franchises = ['Star Wars', 'Harry Potter', 'The Matrix', 'Indiana Jones', 'Jurassic Park', 'Pirates of the Caribbean', 'Toy Story', 'Batman']

# count for franchises of inconsistent quality
count = 0

# loop through each franchise
for franchise in franchises:
    # find all movies in the franchise
    franchise_movies = [col for col in df.columns if franchise in col]
    # print(len(franchise_movies))
    # collect ratings for each movie in the franchise
    franchise_ratings = []
    for movie in franchise_movies:
        movie_ratings = get_movie_ratings(movie, conn)
        franchise_ratings.append(movie_ratings)
    
    # Kruskal-Wallis test
    kw_stat, kw_p = kruskal(*franchise_ratings)
    print(f"Kruskal-Wallis test or {franchise}\nH: {kw_stat}\nP-value: {kw_p}")

    if kw_p < ALPHA:
        print(f"The ratings for movies in the {franchise} franchise show a significant difference.\n")
        count += 1
    else:
        print(f"No significant difference in ratings for movies in the {franchise} franchise.\n")

# print the total count of inconsistent franchises
print(f"{count} of the franchises are of inconsistent quality")

# Extra Credit - "Is 'I have cried during a movie' gendered?"

CRYIED_COL = 'I have cried during a movie'
GENDER_COL = 'Gender identity (1 = female; 2 = male; 3 = self-described)'

# get data
data = pd.read_sql(f"SELECT `{CRYIED_COL}`, `{GENDER_COL}` FROM {TABLE_NAME}", conn).dropna()

# create a contingency table
contingency_table = pd.crosstab(data[CRYIED_COL], data[GENDER_COL])

# chi-squared test
chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)

# print results
print(f"Chi-squared Test\nChi2: {chi2_stat}\nP-value: {chi2_p}")
print("Conclusion:", "Significant association\n" if chi2_p <= ALPHA else "No significant association\n")

conn.close()
