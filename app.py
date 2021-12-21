import datetime
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve

url = "https://data.drees.solidarites-sante.gouv.fr/api/records/1.0/download/?dataset=covid-19-resultats-par-age-issus-des-appariements-entre-si-vic-si-dep-et-vac-si"
url2 = "https://data.drees.solidarites-sante.gouv.fr/explore/dataset/covid-19-resultats-par-age-issus-des-appariements-entre-si-vic-si-dep-et-vac-si/information/"


def transform_status(s):
    if "Non-vaccinés" in s:
        return "[0]. Non vaccinés"
    if "récente" in s:
        return "[1]. Première dose (≤ 14 jours)"
    if "efficace" in s:
        return "[2]. Première dose (> 14 jours)"
    if "Complet" in s and "sans rappel" in s:
        return "[3]. Schéma vaccinal complet sans rappel"
    if "Complet" in s and "avec rappel" in s:
        return "[4]. Schéma vaccinal complet avec rappel"


def transform_age(s):
    return {
        "[20,39]": "20-39 ans",
        "[40,59]": "40-59 ans",
        "[60,79]": "60-79 ans",
        "[80;+]": "80 ans et plus",
    }[s]


@st.cache
def get_and_preprocess_data(today):
    urlretrieve(url, filename="data.csv")
    df = pd.read_csv("data.csv", sep=";")

    # Remove the 0-19 age range
    df = df[df["age"] != "[0,19]"]

    # Keep only the 15 last days available
    dates_kept = sorted(list(set(df["date"])))[-15:]
    df = df[df["date"].isin(dates_kept)]

    # Relabel the vaccination status and the age columns
    df["vac_statut"] = df["vac_statut"].map(transform_status)
    df["age"] = df["age"].map(transform_age)

    # Agregate the data across all dates
    agg_dict = {
        col: "sum"
        for col in list(df.columns)
        if "hc" in col or "nb" in col or "sc" in col or "dc" in col
    }
    agg_dict["effectif"] = "mean"
    df = (
        df.groupby(by=["age", "vac_statut", "date"])
        .sum()
        .groupby(by=["age", "vac_statut"])
        .agg(agg_dict)
    )

    # Compute the relative numbers of hospital admissions, ICU admissions and deaths
    df = df.loc[:, ["hc_pcr", "sc_pcr", "dc_pcr", "effectif"]]
    df["hc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.hc_pcr / x.effectif, axis=1)
    df["sc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.sc_pcr / x.effectif, axis=1)
    df["dc_pcr_per_1M"] = df.apply(lambda x: x.dc_pcr / x.effectif, axis=1)
    df = df.reset_index(level=["age", "vac_statut"]).rename(columns={"age": "Âge"})
    return df, dates_kept


today = str(datetime.date.today())
today = f"{today[-2:]}/{today[5:7]}/{today[:4]}"
df, dates_kept = get_and_preprocess_data(today)
earliest, latest = dates_kept[0], dates_kept[-1]
earliest = f"{earliest[-2:]}/{earliest[5:7]}"
latest = f"{latest[-2:]}/{latest[5:7]}"


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def create_fig(key, title, today):
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="Âge",
        y=key,
        hue="vac_statut",
        palette=sns.color_palette("rocket"),
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [x[5:] for x in labels], title="Statut vaccinal")
    ax.set(
        ylabel="Cas par million de personnes", title=title,
    )
    sns.despine()
    return fig


st.sidebar.header("COVID-19 : cas graves en fonction de l'âge et du statut vaccinal")

st.sidebar.markdown(
    f"""
Les 3 graphiques de cette page montrent les nombres, par millions de personnes, des :

- **hospitalisations**
- entrées en **soins critiques**
- **décès**

... au cours des 15 derniers jours pour lesquels les données nationales sont disponibles ({earliest}-{latest}) et en fonction de l'**âge** et du **statut vaccinal**.

Ces graphiques sont mis à jour quotidiennement à partir des données de la [DREES]({url2}).
"""
)

for key, title in [
    ("hc_pcr_per_1M", "Hospitalisations avec test PCR positif"),
    ("sc_pcr_per_1M", "Entrées en soins critiques avec test PCR positif"),
    ("dc_pcr_per_1M", "Décès avec test PCR positif"),
]:
    st.pyplot(create_fig(key, title, today))

st.markdown(
    f"Date de mise à jour : {today}, source : [DREES]({url2}), code : [GitHub](https://github.com/vivien000/vaccine-stats)"
)
