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


# In[ ]:


# In[3]:


df.describe().transpose()


# In[ ]:


df.shape


# In[ ]:


import numpy as np

df = df.replace("", np.nan)


# In[ ]:


df.isnull().sum()


# In[ ]:


df[df.columns[:-5]] = df[df.columns[:-5]].apply(pd.to_numeric)
df[df.columns[-4:-1]] = df[df.columns[-4:-1]].apply(pd.to_numeric)
df.dtypes


# In[ ]:


df = df[df.CTP_quality > 2]
df.head()


# In[ ]:


df.shape


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


df = df[
    (abs(df["GaiaEDR3_parallax"] / df["GaiaEDR3_parallax_error"]) <= 3)
    | (df["GaiaEDR3_parallax"].isnull())
]
df.shape


# In[ ]:


df = df[
    (abs(df.GaiaEDR3_pmra / df.GaiaEDR3_pmra_error) <= 3) | df.GaiaEDR3_pmra.isnull()
]
df.shape


# In[ ]:


df = df[
    (abs(df.GaiaEDR3_pmdec / df.GaiaEDR3_pmdec_error) <= 3) | df.GaiaEDR3_pmdec.isnull()
]
df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.is_AGN.value_counts()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["FUV"] - df["NUV"]),
    hue=df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["FUV"] - df["NUV"]),
    hue=df["is_AGN"],
    palette="bright",
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["W2"] - df["W1"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["W1"] - df["W2"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["LS8_g"] - df["LS8_r"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["LS8_g"] - df["LS8_r"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["VISTA_J"] - df["VISTA_H"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["VISTA_J"] - df["VISTA_H"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["W1"] - df["W4"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["W1"] - df["W4"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["LS8_g"] - df["LS8_z"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["LS8_g"] - df["LS8_z"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["W3"] - df["W2"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["W3"] - df["W2"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))
plot1 = sns.scatterplot(
    x=df["ERO_ML_FLUX"],
    y=(df["VISTA_H"] - df["VISTA_Ks"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[0],
    s=4,
)
plot2 = sns.scatterplot(
    x=df["Total_flux"],
    y=(df["VISTA_H"] - df["VISTA_Ks"]),
    hue=df["is_AGN"].apply(lambda x: True if x == True else False),
    palette=sns.color_palette("bright", as_cmap=True),
    ax=ax[1],
    s=4,
)
plot1.set(xscale="log")
plot2.set(xscale="log")


# In[ ]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

input_labels = [
    "LS8_g",
    "LS8_r",
    "LS8_z",
    "W1",
    "W2",
    "W3",
    "W4",
    "ERO_ML_FLUX",
    "is_AGN",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)

X.shape


# In[ ]:


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


# In[ ]:


logreg_model = LogisticRegression(max_iter=200)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logreg_model.fit(X_train, y_train)

y_pred1 = logreg_model.predict(X_test)
model_stats(y_test, y_pred1)


# In[ ]:


y_pred1


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg_model, X, y, cv=5)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)

X.shape


# In[ ]:


from xgboost import XGBRegressor, cv

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

XGB_model = XGBRegressor(
    objective="binary:hinge",
    eval_metric=metrics.accuracy_score,
    eta=0.2,
    early_stopping_rounds=5,
)

XGB_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
y_pred2 = XGB_model.predict(X_test)

model_stats(y_test, y_pred2)


# In[ ]:


y_pred2


# In[ ]:


scores = cross_val_score(
    XGB_model,
    X,
    y,
    cv=5,
    fit_params={"eval_set": [(X_test, y_test)], "verbose": False},
    scoring="accuracy",
)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


colors_df = pd.DataFrame(
    {
        "W2-W1": df["W2"] - df["W1"],
        "g-r": df["LS8_g"] - df["LS8_r"],
        "J-H": df["VISTA_J"] - df["VISTA_H"],
        "g-z": df["LS8_g"] - df["LS8_z"],
        "ERO_flux": df["ERO_ML_FLUX"],
        "is_AGN": df["is_AGN"],
    }
)

colors_df.head()


# In[ ]:


print(colors_df.isnull().sum())
print(colors_df.shape)


# In[ ]:


known_colors_df = colors_df[colors_df["is_AGN"] != "Unknown"]
input_colors_df = known_colors_df.dropna()
y = input_colors_df["is_AGN"].astype(int)
X = input_colors_df.drop("is_AGN", axis=1)

X.shape


# In[ ]:


logreg_model2 = LogisticRegression(max_iter=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
logreg_model2.fit(X_train, y_train)

y_pred3 = logreg_model2.predict(X_test)
model_stats(y_test, y_pred3)


# In[ ]:


y_pred3


# In[ ]:


scores = cross_val_score(logreg_model2, X, y, cv=5)
print("Cross-Validation Accuracy Scores", scores)

# Non colors model score[0.96022727 0.98863636 0.97159091 0.96022727 0.96      ]


# In[ ]:


known_colors_df = colors_df[colors_df["is_AGN"] != "Unknown"]
known_colors_df = known_colors_df
y = known_colors_df["is_AGN"].astype(int)
X = known_colors_df.drop("is_AGN", axis=1)

X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

XGB_model2 = XGBRegressor(
    objective="binary:hinge",
    eval_metric=metrics.accuracy_score,
    eta=0.05,
    early_stopping_rounds=50,
)

XGB_model2.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred4 = XGB_model2.predict(X_test)

model_stats(y_test, y_pred4)


# In[ ]:


y_pred4


# In[ ]:


scores = cross_val_score(
    XGB_model2,
    X,
    y,
    cv=5,
    fit_params={"eval_set": [(X_test, y_test)], "verbose": False},
    scoring="accuracy",
)
print("Cross-Validation Accuracy Scores", scores)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

input_labels = [
    "LS8_g",
    "LS8_r",
    "LS8_z",
    "W1",
    "W2",
    "W3",
    "W4",
    "ERO_ML_FLUX",
    "is_AGN",
]
known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Regression_model = RandomForestClassifier(n_estimators=200, random_state=0)
Regression_model.fit(X_train, y_train)

y_pred5 = Regression_model.predict(X_test)
model_stats(y_test, y_pred5)


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
known_df.sort_values("SPECZ_REDSHIFT")[~(known_df.SPECZ_REDSHIFT.isnull())].head(30)


# In[ ]:


known_df.sort_values("SPECZ_REDSHIFT")[~(known_df.SPECZ_REDSHIFT.isnull())].tail(30)


# In[ ]:


known_df.predicted.value_counts()


# In[ ]:


known_df = df[df["is_AGN"] == "Unknown"]
input_df = known_df[input_labels].dropna()
X = input_df.drop("is_AGN", axis=1)

unknown_pred = Regression_model.predict(X)
input_df["predicted"] = unknown_pred
redshift = []
for i in range(len(input_df)):
    redshift.append(df.SPECZ_REDSHIFT[df.index == input_df.index[i]].values[0])
input_df["REDSHIFT"] = redshift
input_df.sort_values("REDSHIFT")[~(input_df.REDSHIFT.isnull())].head(30)


# In[ ]:


input_df.sort_values("REDSHIFT")[~(input_df.REDSHIFT.isnull())].tail(30)


# In[ ]:


input_df.predicted.value_counts()
