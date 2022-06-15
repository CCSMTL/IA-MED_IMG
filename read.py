import pandas as pd
import matplotlib.pyplot as plt
combined_df=pd.read_csv("combined_csv.csv")


combined_df.rename({'Finding Labels': 'classes'},
                       axis=1, inplace=True)

#print(combined_df["classes"])
combined_df["special"] = ""
combined_df["NoFinding"] = ""

combined_df["special"]=(combined_df['classes'].str.contains(r'[@#&$%+-/*|]') )
combined_df["NoFinding"]=(combined_df['classes'].str.contains(r'No Finding') )

combined_df.drop(combined_df[combined_df['special'] == True].index , inplace=True)
combined_df.drop(combined_df[combined_df['NoFinding'] == True].index , inplace=True)

# Plot the most repeated classes

import seaborn as sns


a=combined_df.groupby('classes').count().sort_values(by='Patient ID',ascending=True).apply(lambda x:
                                                 x / 30963).round(decimals = 2)



a.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(25,10))

sns.barplot(x='classes', y='Patient ID', data=a).plot(ax=ax)

ax.set_xlabel("Most Frequented Classes with  Outliers",fontsize=30)

ax.set_ylabel("Frequency",fontsize=30)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)

ax.bar_label(ax.containers[0])

plt.savefig('Most_Frequented_Classes111.png')


# HISTOGRAM
combined_df.rename({'Finding Labels': 'classes'},
                       axis=1, inplace=True)

#Remove outliers and No finding labels
combined_df["special"] = ""
combined_df["NoFinding"] = ""

combined_df["special"]=(combined_df['classes'].str.contains(r'[@#&$%+-/*|]') )
combined_df["NoFinding"]=(combined_df['classes'].str.contains(r'No Finding') )

combined_df.drop(combined_df[combined_df['special'] == True].index , inplace=True)
combined_df.drop(combined_df[combined_df['NoFinding'] == True].index , inplace=True)

# Plot the most repeated classes

import seaborn as sns

a=combined_df.groupby('classes').count().sort_values(by='Patient ID',ascending=True)

a.head(15)

a.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(25,10))

sns.barplot(x='classes', y='Patient ID', data=a).plot(ax=ax)

ax.set_xlabel("Most Frequented Classes with out Outliers",fontsize=30)

ax.set_ylabel("Frequency",fontsize=30)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)

ax.bar_label(ax.containers[0])


#AUC Curve





#Acuuracy