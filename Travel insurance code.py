import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter('ignore')

df = pd.read_csv('TravelInsurancePrediction.csv')
df.head()

df.isna().sum()

df.drop(['Unnamed: 0', 'GraduateOrNot'], axis = 1, inplace = True)              

df.head()

df.describe()

numeric_df = df.select_dtypes(include = 'number')

fig,ax = plt.subplots(figsize= (10,6))
sns.heatmap(numeric_df.corr(), annot = True, fmt = '.1g', cmap = 'viridis', cbar = True)
plt.title("Correlation matrix between numeric values")
plt.show()

plt.style.use('classic')

unique_ages = df['Age'].nunique()
palette = sns.color_palette('husl',unique_ages)

fig,ax = plt.subplots(figsize= (10,6))
sns.countplot(x = 'Age', data = df, ax = ax, palette = palette, order = sorted(df['Age'].unique()))
ax.set_title('Age distribution', fontsize = 14)
plt.tight_layout()
plt.show()

plt.style.use('classic')

unique_families = df['FamilyMembers'].nunique()
palette = sns.color_palette('husl',unique_families)

fig,ax = plt.subplots(figsize= (10,6))
sns.countplot(x = 'FamilyMembers', data = df, ax = ax, palette = palette, order = sorted(df['FamilyMembers'].unique()))
ax.set_title('Distribution of family members', fontsize = 14)
plt.tight_layout()
plt.show()

plt.style.use('classic')

fig,ax = plt.subplots(figsize = (10,6))
sns.histplot(df['AnnualIncome'], kde = True, color = 'green', ax = ax)

mean_income = df['AnnualIncome'].mean()
ax.axvline(mean_income, color = 'red', linestyle = '--', linewidth = 2 , label = 'Mean')
ax.set_title(f'Annual income distribution [ mean : {mean_income:.2f}]', fontsize = 20)
ax.legend()
plt.tight_layout()
plt.show()

plt.style.use('classic')

disease_counts = df['ChronicDiseases'].value_counts()
labels = ['Non chronic', 'Chronic']

fig,ax = plt.subplots(figsize = (10,6))
wedges, texts, autotexts = ax.pie( x = disease_counts,
                                   labels = labels,
                                   colors = ['crimson','firebrick'],
                                   shadow = True,
                                   explode = (0,0.1),
                                   autopct ="%1.1f%%",
                                   startangle = 140,
                                   textprops = dict(color = 'white', fontsize = 12))

ax.set_title("Distribution off chronic disease", fontsize = 20)
plt.setp(autotexts, weight = True)
plt.tight_layout()
plt.show()

plt.style.use('classic')

fig,ax = plt.subplots(figsize= (10,6))
sns.countplot(x = 'FrequentFlyer', data = df, ax = ax, palette = 'crest')
ax.set_title('Frequent flyer status count', fontsize = 14)
plt.tight_layout()
plt.show()

plt.style.use('classic')

disease_counts = df['TravelInsurance'].value_counts()
labels = ["Don't have travel insurance ", 'Have travel insurance']

fig,ax = plt.subplots(figsize = (10,6))
ax.pie( x = disease_counts,
            labels = labels,
            colors = ['darkorange','firebrick'],
            shadow = True,
            explode = (0,0.1),
            autopct ="%1.1f%%",
            startangle = 90,
            )

ax.set_title("Distribution of Travel insurance", fontsize = 20)
ax.axis('equal')
plt.tight_layout()
plt.show()

df['FrequentFlyer'] = df['FrequentFlyer'].map({'Yes' : 1, 'No' : 0})
df['EverTravelledAbroad'] = df['EverTravelledAbroad'].map({'Yes' : 1, 'No' : 0})
df['Employment Type'] = df['Employment Type'].map({'Government Sector' : 1 , 'Private Sector/Self Employed' : 0 })
df.head()

X = df.drop('TravelInsurance', axis = 1)
X.head()

y = df['TravelInsurance']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
len(X_train), len(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

RandomForestClassifier = rf.score(X_test, y_test)
RandomForestClassifier

print("Accuracy of RandomForestClassifier model : ", RandomForestClassifier*100)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap' : [True],
    'max_depth' : [80,90,100,110,120],
    'max_features' : [2,3],
    'min_samples_leaf' : [3,4,5],
    'min_samples_split' : [8,10,12],
    'criterion' : ['gini', 'entropy'],
    'n_estimators' : [100,200,300,1000]
}

grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search_rf.fit(X_train, y_train)

grid_search_rf.best_params_

grid_search_rf.best_score_

grid_search_rf_predict = grid_search_rf.predict(X_test)

`int('Improvement in random forest classifier after gridsearchcv : { :0.2f}%.'.format(100 * (grid_search_rf_best_score_ - RandomForestClassifierScore) / RandomForestClassifierScore))

from sklearn.metrics import classification_report

print(classification_report(y_test, grid_search_rf_predict))
