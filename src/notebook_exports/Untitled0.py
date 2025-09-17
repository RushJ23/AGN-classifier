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


worksheet = gc.open("xmm_with_redshift").sheet1

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

index = [i[0] for i in rows[1:]]
adjusted_rows = [i[1:] for i in rows[1:]]

import pandas as pd

df = pd.DataFrame.from_records(adjusted_rows, columns=rows[0][1:], index=index)

df.head()


# In[ ]:


import numpy as np

df = df.replace("", np.nan)
for i in range(len(df.columns)):
    try:
        df[df.columns[i]] = df[df.columns[i]].apply(pd.to_numeric)
    except:
        print(i)
        pass


# In[ ]:


print(df.zsp.max(), df.zsp.min())
# It is now 13.721 Gyr since the Big Bang.
# The age at redshift z was 0.769 Gyr.
# The light travel time was 12.952 Gyr.


# In[ ]:


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


# In[ ]:


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


# In[ ]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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


# In[ ]:


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


# In[ ]:


Regression_model = RandomForestClassifier(n_estimators=500, min_samples_split=8)
Regression_model.fit(X_train, y_train)

y_pred = Regression_model.predict(X_test)
model_stats(y_test, y_pred)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=[False, True]
)
cm_display.plot()
plt.show()


# In[ ]:


unknown_df = df[df["is_AGN"] == "Unknown"]
unknown_df = unknown_df[input_labels]
unknown_df = unknown_df.drop("is_AGN", axis=1)
unknown_df = unknown_df.dropna()
predicted = Regression_model.predict(unknown_df)
unknown_df["predicted"] = predicted
sns.set(font_scale=3)

fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_flux"],
    y=(known_df["W1mag"] - known_df["W2mag"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="W1-W2", title="Known dataset")
plot2 = sns.scatterplot(
    x=unknown_df["log_flux"],
    y=(unknown_df["W1mag"] - unknown_df["W2mag"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="W1-W2", title="Predicted dataset")


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_flux"],
    y=(known_df["gmag"] - known_df["rmag"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="g-r", title="Known datset")
plot2 = sns.scatterplot(
    x=unknown_df["log_flux"],
    y=(unknown_df["gmag"] - unknown_df["rmag"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="g-r", title="Predicted dataset")


# In[ ]:


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


# In[ ]:


from scipy import interpolate

metrics_df["smooth_accuracy"] = 0
for i in metrics_df.samples.unique():
    spline = interpolate.UnivariateSpline(
        metrics_df[metrics_df.samples == i]["estimators"],
        metrics_df[metrics_df.samples == i]["accuracy"],
    )
    metrics_df.loc[metrics_df.samples == i, "smooth_accuracy"] = spline(
        metrics_df.loc[metrics_df.samples == i, "estimators"]
    )

sns.lineplot(
    data=metrics_df,
    x="estimators",
    y="smooth_accuracy",
    hue=metrics_df["samples"].astype(str),
)


# In[ ]:


known_df = df[df["is_AGN"] != "Unknown"]
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = HistGradientBoostingClassifier(
    early_stopping=True, learning_rate=0.0111, max_iter=700, max_leaf_nodes=41
)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=[False, True]
)
cm_display.plot()
plt.show()


# In[ ]:


model_stats(y_test, y_pred)


# In[ ]:


unknown_df = df[df["is_AGN"] == "Unknown"]
unknown_df = unknown_df[input_labels]
unknown_df = unknown_df.drop("is_AGN", axis=1)
predicted = model.predict(unknown_df)
unknown_df["predicted"] = predicted
sns.set(font_scale=2.5)

fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_flux"],
    y=(known_df["W1mag"] - known_df["W2mag"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="W1-W2", title="Known dataset")
plot2 = sns.scatterplot(
    x=unknown_df["log_flux"],
    y=(unknown_df["W1mag"] - unknown_df["W2mag"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="W1-W2", title="Predicted dataset")


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_flux"],
    y=(known_df["gmag"] - known_df["rmag"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="g-r", title="Known datset")
plot2 = sns.scatterplot(
    x=unknown_df["log_flux"],
    y=(unknown_df["gmag"] - unknown_df["rmag"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="g-r", title="Predicted dataset")


# In[ ]:


sns.histplot(data=df, x="log_flux")


# In[ ]:


sns.histplot(data=df, x="W1mag")


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


# In[ ]:


metrics_df["smooth_accuracy"] = 0
for i in metrics_df.iter.unique():
    spline = interpolate.UnivariateSpline(
        metrics_df[metrics_df.iter == i]["rate"],
        metrics_df[metrics_df.iter == i]["accuracy"],
    )
    metrics_df.loc[metrics_df.iter == i, "smooth_accuracy"] = spline(
        metrics_df.loc[metrics_df.iter == i, "rate"]
    )

sns.lineplot(
    data=metrics_df, x="rate", y="smooth_accuracy", hue=metrics_df["iter"].astype(str)
)


# In[ ]:


red_df = df[~df.zsp.isnull()]
red_df = red_df[red_df.is_AGN == "Unknown"]
red_df = red_df.drop("is_AGN", axis=1)
red_df["predicted"] = model.predict(
    red_df[
        [
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
            "classification",
        ]
    ]
).astype(bool)
red_df.head()
