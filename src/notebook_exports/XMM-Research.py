#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import auth

auth.authenticate_user()

import gspread
from google.auth import default

creds, _ = default()

gc = gspread.authorize(creds)


# In[ ]:


worksheet = gc.open("xmm_panstarrs_gaia_wise_simbad").sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

index = [i[0] for i in rows[1:]]
adjusted_rows = [i[1:] for i in rows[1:]]

import pandas as pd

df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)

df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


import numpy as np

df = df.replace("", np.nan)


# In[ ]:


df.isnull().sum()


# In[ ]:


for i in range(len(df.columns)):
    try:
        df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric)
    except:
        print(i)
        pass
df.dtypes


# In[ ]:


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


# In[ ]:


df.main_type.unique()


# In[ ]:


df = df[(abs(df["parallax"] / df["parallax_error"]) <= 3) | (df["parallax"].isnull())]
df.shape


# In[ ]:


df = df[(abs(df.pmra / df.pmra_error) <= 3) | df.pmra.isnull()]
df.shape


# In[ ]:


df = df[(abs(df.pmdec / df.pmdec_error) <= 3) | df.pmdec.isnull()]
df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.is_AGN.value_counts()


# In[ ]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


# In[ ]:


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
    "SC_EP_8_FLUX",
    "is_AGN",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


logreg_model = LogisticRegression(max_iter=500)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logreg_model.fit(X_train, y_train)

y_pred1 = logreg_model.predict(X_test)
model_stats(y_test, y_pred1)


# In[ ]:


np.unique(y_pred1, return_counts=True)


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg_model, X, y, cv=5)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

Regression_model = RandomForestClassifier(n_estimators=200, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred2 = Regression_model.predict(X_test)
model_stats(y_test, y_pred2)


# In[ ]:


scores = cross_val_score(Regression_model, X, y, cv=5)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


from sklearn.ensemble import HistGradientBoostingClassifier

known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Hist_tree = HistGradientBoostingClassifier(early_stopping=True).fit(X_train, y_train)
y_pred6 = Hist_tree.predict(X_test)

model_stats(y_test, y_pred6)


# In[ ]:


scores = cross_val_score(Hist_tree, X, y, cv=5)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


known_df = df[df["is_AGN"] == "Unknown"]
input_df = known_df[input_labels]
X = input_df.drop("is_AGN", axis=1)

unknown_pred = Hist_tree.predict(X)
known_df["predicted"] = unknown_pred
known_df.predicted.value_counts()


# In[ ]:


known_df = df[df["is_AGN"] == "Unknown"]
input_df = known_df[input_labels].dropna()
X = input_df.drop("is_AGN", axis=1)

unknown_pred = Regression_model.predict(X)
input_df["predicted"] = unknown_pred
input_df.predicted.value_counts()


# In[ ]:


import seaborn as sns

sns.histplot(data=df, x="gmag")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

plt.hist(np.log10(df["SC_EP_8_FLUX"].values))
