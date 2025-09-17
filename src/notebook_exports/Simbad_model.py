#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import auth

auth.authenticate_user()

import gspread
from google.auth import default

creds, _ = default()

gc = gspread.authorize(creds)


# In[2]:


worksheet = gc.open("eFEDS_VLASS_Simbad").sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

index = [i[0] for i in rows[1:]]
adjusted_rows = [i[1:] for i in rows[1:]]

import pandas as pd

df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)

df.head()


# In[3]:


df.describe()


# In[4]:


df.shape


# In[5]:


import numpy as np

df = df.replace("", np.nan)


# In[6]:


df.isnull().sum()


# In[7]:


df[df.columns[:-5]] = df[df.columns[:-5]].apply(pd.to_numeric)
df[df.columns[-4:-1]] = df[df.columns[-4:-1]].apply(pd.to_numeric)
df.dtypes


# In[8]:


df = df[df.CTP_quality > 2]
df.head()


# In[9]:


df.shape


# In[10]:


def is_AGN(classification, ref):
    value = "Unknown"
    try:
        if (
            classification
            in ["QSO", "Seyfert_1", "Seyfert_2", "BLLac", "Blazar", "RadioG", "AGN"]
            and ref >= 3
        ):
            return True
        elif pd.isna(classification):
            return value
        elif ref < 3 or "Candidate" in classification:
            return value
        else:
            return False
    except Exception as e:
        print(e)
        print(classification, ref)
        input()


df["is_AGN"] = df.apply(lambda x: is_AGN(x.main_type, x.nbref), axis=1)
df.is_AGN.value_counts()


# In[11]:


df.main_type.unique()


# In[12]:


df = df[
    (abs(df["GaiaEDR3_parallax"] / df["GaiaEDR3_parallax_error"]) <= 3)
    | (df["GaiaEDR3_parallax"].isnull())
]
df.shape


# In[13]:


df = df[
    (abs(df.GaiaEDR3_pmra / df.GaiaEDR3_pmra_error) <= 3) | df.GaiaEDR3_pmra.isnull()
]
df.shape


# In[14]:


df = df[
    (abs(df.GaiaEDR3_pmdec / df.GaiaEDR3_pmdec_error) <= 3) | df.GaiaEDR3_pmdec.isnull()
]
df.shape


# In[15]:


from sklearn import metrics


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


# In[16]:


df["log_FLUX"] = np.log10(df["ERO_ML_FLUX"])
df.head()


# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

input_labels = ["LS8_g", "LS8_r", "LS8_z", "W1", "W2", "W3", "W4", "log_FLUX", "is_AGN"]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Regression_model = RandomForestClassifier(n_estimators=500, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred = Regression_model.predict(X_test)
model_stats(y_test, y_pred)


# In[18]:


from sklearn.ensemble import HistGradientBoostingClassifier

known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Hist_tree = HistGradientBoostingClassifier(early_stopping=True).fit(X_train, y_train)
y_pred2 = Hist_tree.predict(X_test)

model_stats(y_test, y_pred2)


# In[19]:


def is_AGN(x):
    value = "Unknown"
    try:
        if (
            x.main_type
            in ["QSO", "Seyfert_1", "Seyfert_2", "BLLac", "Blazar", "RadioG", "AGN"]
            and x.nbref >= 3
        ):
            return True
        elif x.CTP_Classification == "SECURE GALACTIC":
            return False
        elif (
            abs(x.GaiaEDR3_pmra / x.GaiaEDR3_pmra_error) >= 3
            or abs(x.GaiaEDR3_pmdec / x.GaiaEDR3_pmdec_error) >= 3
            or abs(x.GaiaEDR3_parallax / x.GaiaEDR3_parallax_error) >= 3
        ):
            return False
        elif pd.isna(x.main_type):
            return value
        elif x.nbref < 3 or "Candidate" in x.main_type:
            return value
        else:
            return False
    except Exception as e:
        print(e)
        print(x)
        input()


df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)
df = df.replace("", np.nan)
df[df.columns[:-5]] = df[df.columns[:-5]].apply(pd.to_numeric)
df[df.columns[-4:-1]] = df[df.columns[-4:-1]].apply(pd.to_numeric)
df = df[df.CTP_quality > 2]
df["is_AGN"] = df.apply(lambda x: is_AGN(x), axis=1)
df.is_AGN.value_counts()


# In[20]:


def classifier(classification):
    if classification == "SECURE GALACTIC":
        return 0
    elif classification == "SECURE EXTRAGALACTIC":
        return 2
    else:
        return 1


df["classification"] = df.apply(lambda x: classifier(x.CTP_Classification), axis=1)
df.classification.value_counts()


# In[21]:


df["log_FLUX"] = np.log10(df["ERO_ML_FLUX"])
input_labels = [
    "LS8_g",
    "LS8_r",
    "LS8_z",
    "W1",
    "W2",
    "W3",
    "W4",
    "log_FLUX",
    "classification",
    "is_AGN",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Regression_model = RandomForestClassifier(n_estimators=500, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred = Regression_model.predict(X_test)
model_stats(y_test, y_pred)


# In[ ]:


known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Hist_tree = HistGradientBoostingClassifier(early_stopping=True).fit(X_train, y_train)
y_pred2 = Hist_tree.predict(X_test)

model_stats(y_test, y_pred2)


# In[ ]:


metrics_dic = {"rate": [], "iter": [], "accuracy": []}
for rate in np.linspace(0.001, 1, 100):
    for iter in range(100, 1000, 100):
        Hist_tree = HistGradientBoostingClassifier(
            early_stopping=True, learning_rate=rate, max_iter=iter
        ).fit(X_train, y_train)
        y_pred = Hist_tree.predict(X_test)
        metrics_dic["rate"].append(rate)
        metrics_dic["iter"].append(iter)
        metrics_dic["accuracy"].append(metrics.accuracy_score(y_test, y_pred))

metrics_df = pd.DataFrame(metrics_dic)

import seaborn as sns

sns.lineplot(data=metrics_df, x="rate", y="accuracy", hue="iter")


# In[ ]:


metrics_df.sort_values("accuracy").tail()


# In[ ]:


from sklearn.model_selection import GridSearchCV

model = HistGradientBoostingClassifier()
space = dict()
space["learning_rate"] = np.linspace(0.001, 1, 100)
space["max_iter"] = range(100, 1000, 100)
space["early_stopping"] = [True]
space["max_leaf_nodes"] = range(1, 50, 10)


search = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1)
result = search.fit(X, y)

print("Best Score: %s" % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)


# In[23]:


known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

metrics_dic = {"samples": [], "estimators": [], "accuracy": []}
for estimators in range(100, 1000, 100):
    for samples in range(2, 10):
        forest = RandomForestClassifier(
            n_estimators=estimators, min_samples_split=samples
        ).fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        metrics_dic["samples"].append(samples)
        metrics_dic["estimators"].append(estimators)
        metrics_dic["accuracy"].append(metrics.accuracy_score(y_test, y_pred))

metrics_df = pd.DataFrame(metrics_dic)

import seaborn as sns

sns.lineplot(data=metrics_df, x="estimators", y="accuracy", hue="samples")


# In[26]:


sns.lineplot(
    data=metrics_df, x="estimators", y="accuracy", hue=metrics_df["samples"].astype(str)
)


# In[28]:


metrics_df.sort_values("accuracy").tail()


# In[31]:


model = RandomForestClassifier()
space = dict()
space["min_samples_split"] = range(2, 15)
space["n_estimators"] = range(500, 1200, 100)

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1)
result = search.fit(X, y)

print("Best Score: %s" % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)
