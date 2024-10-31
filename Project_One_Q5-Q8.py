import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

mr_set = pd.read_csv("movieReplicationSet.csv")

# This is Question 5
print("Question 5: Do people who are only children enjoy 'The Lion King (1994)' more than people with siblings?")
signi_level = 0.005
# Firstly extract the columns we need to use.
lion_k = mr_set.loc[:, ["The Lion King (1994)", "Are you an only child? (1: Yes; 0: No; -1: Did not respond)"]]
# Split the data into two categories and only keep the series of the ratings, and also dropna.
lion_k_OC = lion_k.query("`Are you an only child? (1: Yes; 0: No; -1: Did not respond)` == 1").dropna()["The Lion King (1994)"]
lion_k_HS = lion_k.query("`Are you an only child? (1: Yes; 0: No; -1: Did not respond)` == 0").dropna()["The Lion King (1994)"]
# Lastly do the ONE-sided independent samples t-test.
stat, pval = stats.ttest_ind(lion_k_OC, lion_k_HS, alternative="greater")
whether_significant = 'significant!' if pval < signi_level else 'not significant!'
print(f"The p-value for the t-test is {pval}, which means that the result is {whether_significant}")
if pval >= signi_level:
    print("Conclusion: people who are only children do not enjoy 'The Lion King (1994)' more than people with siblings")
else:
    print("Conclusion: people who are only children enjoy 'The Lion King (1994)' more than people with siblings")
print("\n")

# Question 6
print("Question 6: What proportion of movies exhibit an “only child effect”, i.e. are rated different \
by viewers with siblings vs. those without?")
# In this question we may firstly split the whole dataset into to categories, then get the movie names.
data_child_OC = mr_set.query("`Are you an only child? (1: Yes; 0: No; -1: Did not respond)` == 1")
data_child_HS = mr_set.query("`Are you an only child? (1: Yes; 0: No; -1: Did not respond)` == 0")
movie_names = list(mr_set.columns[:400])
count_sig, count_not_sig = 0, 0
for each_movie in movie_names:
    movie_OC = data_child_OC[each_movie].dropna()
    movie_HS = data_child_HS[each_movie].dropna()
    each_stat, each_pval = stats.ttest_ind(movie_OC, movie_HS)
    if each_pval < signi_level:
        count_sig += 1
    else:
        count_not_sig += 1
print(f"Among the 400 movies, {count_sig/400 :.2%}({count_sig} movies) exhibit an “only child effect”, \
and {count_not_sig/400 :.2%}({count_not_sig} movies) do not exhibit an “only child effect”")
print("\n")

# Question 7
print("Question 7: Do people who like to watch movies socially enjoy 'The Wolf of Wall Street (2013)' \
more than those who prefer to watch them alone?")
# Repeaing my code for Q5
# Firstly extract the columns we need to use.
wolf_ws = mr_set.loc[:, ["The Wolf of Wall Street (2013)", "Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)"]]
# Split the data into two categories and only keep the series of the ratings, and also dropna.
wolf_ws_AL = wolf_ws.query("`Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)` == 1").dropna()["The Wolf of Wall Street (2013)"]
wolf_ws_SO = wolf_ws.query("`Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)` == 0").dropna()["The Wolf of Wall Street (2013)"]
# Lastly do the ONE-sided independent samples t-test.
stat, pval = stats.ttest_ind(wolf_ws_SO, wolf_ws_AL, alternative="greater")
whether_significant = 'significant!' if pval < signi_level else 'not significant!'
print(f"The p-value for the t-test is {pval}, which means that the result is {whether_significant}")
if pval >= signi_level:
    print("conclusion: people who like to watch movies socially do not enjoy 'The Wolf of Wall Street (2013)' \
more than those who prefer to watch them alone.")
else:
    print("Conclusion: people who like to watch movies socially enjoy 'The Wolf of Wall Street (2013)' \
more than those who prefer to watch them alone")
print("\n")

# Question 8
print("Question 8: What proportion of movies exhibit such a “social watching” effect?")
# In this question we may firstly split the whole dataset into to categories, then get the movie names.
data_watch_AL = mr_set.query("`Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)` == 1")
data_watch_SO = mr_set.query("`Movies are best enjoyed alone (1: Yes; 0: No; -1: Did not respond)` == 0")
movie_names = list(mr_set.columns[:400])
# Go over all movies and do an independent samples t-test for each of them.
signi_level = 0.005
count_sig, count_not_sig = 0, 0
for each_movie in movie_names:
    movie_watch_AL = data_watch_AL[each_movie].dropna()
    movie_watch_SO = data_watch_SO[each_movie].dropna()
    each_stat, each_pval = stats.ttest_ind(movie_watch_AL, movie_watch_SO)
    if each_pval < signi_level:
        count_sig += 1
    else:
        count_not_sig += 1
print(f"Among the 400 movies, {count_sig/400 :.2%}({count_sig} movies) exhibit a “social watching”, \
and {count_not_sig/400 :.2%}({count_not_sig} movies) do not exhibit a “social watching”")











