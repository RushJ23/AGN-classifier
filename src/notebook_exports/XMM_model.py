#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.colab import auth

auth.authenticate_user()

import gspread
from google.auth import default

creds, _ = default()

gc = gspread.authorize(creds)


# In[3]:


worksheet = gc.open("xmm_with_redshift").sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

index = [i[0] for i in rows[1:]]
adjusted_rows = [i[1:] for i in rows[1:]]

import pandas as pd

df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)

df.head()


# In[4]:


import numpy as np

df = df.replace("", np.nan)


# In[5]:


for i in range(len(df.columns)):
    try:
        df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric)
    except:
        print(i)
        pass
df.dtypes


# In[6]:


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


# In[7]:


df = df[(abs(df["parallax"] / df["parallax_error"]) <= 3) | (df["parallax"].isnull())]
df.shape


# In[8]:


df = df[(abs(df.pmra / df.pmra_error) <= 3) | df.pmra.isnull()]
df.shape


# In[9]:


df = df[(abs(df.pmdec / df.pmdec_error) <= 3) | df.pmdec.isnull()]
df.shape


# In[10]:


from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[11]:


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


# In[12]:


df["log_flux"] = np.log10(df["SC_EP_8_FLUX"])
input_labels = [
    "gmag",
    "ymag",
    "rmag",
    "zmag",
    "imag",
    "W1mag",
    "W2mag",
    "W3mag",
    "W4mag",
    "log_flux",
    "is_AGN",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[13]:


from sklearn.ensemble import RandomForestClassifier

Regression_model = RandomForestClassifier(n_estimators=200, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred = Regression_model.predict(X_test)
model_stats(y_test, y_pred)


# In[14]:


from sklearn.ensemble import HistGradientBoostingClassifier

known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Hist_tree = HistGradientBoostingClassifier(early_stopping=True).fit(X_train, y_train)
y_pred2 = Hist_tree.predict(X_test)

model_stats(y_test, y_pred2)


# In[15]:


df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)
df = df.replace("", np.nan)
for i in range(len(df.columns)):
    try:
        df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric)
    except:
        print(i)
        pass


# In[16]:


def is_AGN(x):
    value = "Unknown"
    if (
        x.main_type
        in ["QSO", "Seyfert_1", "Seyfert_2", "BLLac", "Blazar", "RadioG", "AGN"]
        and x.nbref >= 3
    ):
        return True
    elif (
        x.parallax / x.parallax_error > 3
        or x.pmra / x.pmra_error > 3
        or x.pmdec / x.pmdec_error > 3
    ):
        return False
    elif pd.isna(x.main_type):
        return value
    elif x.nbref < 3 or "Candidate" in x.main_type:
        return value
    else:
        return False


df["is_AGN"] = df.apply(lambda x: is_AGN(x), axis=1)
df.is_AGN.value_counts()


# In[17]:


def classifier(x):
    if (
        x.parallax / x.parallax_error > 3
        or x.pmra / x.pmra_error > 3
        or x.pmdec / x.pmdec_error > 3
    ):
        return 0
    elif (
        x.parallax / x.parallax_error > 1
        or x.pmra / x.pmra_error > 1
        or x.pmdec / x.pmdec_error > 1
    ):
        return 1
    else:
        return 2


df["classification"] = df.apply(lambda x: classifier(x), axis=1)
df.classification.value_counts()


# In[18]:


df["log_flux"] = np.log10(df["SC_EP_8_FLUX"])
input_labels = [
    "gmag",
    "ymag",
    "rmag",
    "zmag",
    "imag",
    "W1mag",
    "W2mag",
    "W3mag",
    "W4mag",
    "log_flux",
    "is_AGN",
    "classification",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[19]:


Regression_model = RandomForestClassifier(n_estimators=200, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred = Regression_model.predict(X_test)
model_stats(y_test, y_pred)


# In[20]:


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

sns.lineplot(
    data=metrics_df, x="rate", y="accuracy", hue=metrics_df["iter"].astype(str)
)


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


# In[ ]:


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


# In[ ]:


sns.lineplot(
    data=metrics_df, x="estimators", y="accuracy", hue=metrics_df["samples"].astype(str)
)


# In[ ]:


metrics_df.sort_values("accuracy").tail()


# In[ ]:


known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)

model = RandomForestClassifier()
space = dict()
space["min_samples_split"] = range(2, 15)
space["n_estimators"] = range(500, 1200, 100)

from sklearn.model_selection import GridSearchCV

search = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1)
result = search.fit(X, y)

print("Best Score: %s" % result.best_score_)
print("Best Hyperparameters: %s" % result.best_params_)
