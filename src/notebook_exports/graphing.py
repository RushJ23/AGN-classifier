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


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[3]:


import numpy as np

df = df.replace("", np.nan)


# In[ ]:


df.isnull().sum()


# In[4]:


df[df.columns[:-5]] = df[df.columns[:-5]].apply(pd.to_numeric)
df[df.columns[-4:-1]] = df[df.columns[-4:-1]].apply(pd.to_numeric)
df.dtypes


# In[5]:


df = df[df.CTP_quality > 2]
df.head()


# In[6]:


df.shape


# In[7]:


print(
    df.LS8_g.median(),
    df.LS8_r.median(),
    df.LS8_z.median(),
    df.W1.median(),
    df.W2.median(),
    df.W3.median(),
    df.W4.median(),
    df.ERO_ML_FLUX.median(),
)


# In[ ]:


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


df["is_AGN"] = df.apply(lambda x: is_AGN(x), axis=1)
df.is_AGN.value_counts()


# In[ ]:


df["log_FLUX"] = np.log10(df["ERO_ML_FLUX"])


# In[ ]:


def classifier(classification):
    if classification == "SECURE GALACTIC":
        return 0
    elif classification == "SECURE EXTRAGALACTIC":
        return 2
    else:
        return 1


df["classification"] = df.apply(lambda x: classifier(x.CTP_Classification), axis=1)
df.classification.value_counts()


# In[ ]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

model = HistGradientBoostingClassifier(
    early_stopping=True, learning_rate=0.0211, max_iter=100, max_leaf_nodes=21
)
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
known_df = known_df[input_labels]
y = known_df["is_AGN"].astype(int)
X = known_df.drop("is_AGN", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=[False, True]
)
cm_display.plot()
plt.show()


# In[ ]:


def model_stats(y_test, y_pred):
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))


model_stats(y_test, y_pred)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=3)
unknown_df = df[df["is_AGN"] == "Unknown"]
unknown_df = unknown_df[input_labels]
unknown_df = unknown_df.drop("is_AGN", axis=1)
predicted = model.predict(unknown_df)
unknown_df["predicted"] = predicted

fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_FLUX"],
    y=(known_df["W1"] - known_df["W2"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="W1-W2", title="Known dataset")
plot2 = sns.scatterplot(
    x=unknown_df["log_FLUX"],
    y=(unknown_df["W1"] - unknown_df["W2"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="W1-W2", title="Predicted dataset")


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


fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_FLUX"],
    y=(known_df["LS8_g"] - known_df["LS8_r"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="g-r", title="Known datset")
plot2 = sns.scatterplot(
    x=unknown_df["log_FLUX"],
    y=(unknown_df["LS8_g"] - unknown_df["LS8_r"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="g-r", title="Predicted dataset")


# In[ ]:


sns.histplot(data=df, x="log_FLUX")


# In[ ]:


sns.histplot(data=df, x="W1")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

known_df = df[df["is_AGN"] != "Unknown"]
input_df = known_df[input_labels].dropna()
y = input_df["is_AGN"].astype(int)
X = input_df.drop("is_AGN", axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

Regression_model = RandomForestClassifier(
    n_estimators=800, random_state=0, min_samples_split=9
)
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

fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_FLUX"],
    y=(known_df["W1"] - known_df["W2"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="W1-W2", title="Known dataset")
plot2 = sns.scatterplot(
    x=unknown_df["log_FLUX"],
    y=(unknown_df["W1"] - unknown_df["W2"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="W1-W2", title="Predicted dataset")


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(20, 20))
plot1 = sns.scatterplot(
    x=known_df["log_FLUX"],
    y=(known_df["LS8_g"] - known_df["LS8_r"]),
    hue=known_df["is_AGN"],
    palette="bright",
    ax=ax[0],
    s=4,
)
plot1.set(ylabel="g-r", title="Known datset")
plot2 = sns.scatterplot(
    x=unknown_df["log_FLUX"],
    y=(unknown_df["LS8_g"] - unknown_df["LS8_r"]),
    hue=unknown_df["predicted"].astype(bool),
    palette="bright",
    ax=ax[1],
    s=4,
)
plot2.set(ylabel="g-r", title="Predicted dataset")


# In[ ]:


red_df = df[~df.SPECZ_REDSHIFT.isnull()]
red_df = red_df[red_df.is_AGN == "Unknown"]
red_df = red_df.drop("is_AGN", axis=1)
red_df["predicted"] = model.predict(
    red_df[
        [
            "LS8_g",
            "LS8_r",
            "LS8_z",
            "W1",
            "W2",
            "W3",
            "W4",
            "log_FLUX",
            "classification",
        ]
    ]
).astype(bool)
red_df.head()


# In[ ]:


sns.histplot(data=red_df, x="SPECZ_REDSHIFT", hue="predicted", multiple="stack")


# In[ ]:


from astropy.convolution import Box1DKernel

box1d = Box1DKernel(100)
sns.lineplot(
    x=metrics_df["rate"], y=box1d(metrics_df["accuracy"]), hue=metrics_df["iter"]
)


# In[ ]:


metrics_df["smooth_accuracy"] = 0


# In[ ]:


metrics_df.head()


# In[ ]:


from scipy.ndimage import gaussian_filter1d

for i in metrics_df.iter.unique():
    smoothed = gaussian_filter1d(metrics_df[metrics_df.iter == i].accuracy, sigma=2)
    plt.plot(metrics_df[metrics_df.iter == i]["rate"], smoothed)

plt.show()


# In[ ]:


from scipy import interpolate

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
