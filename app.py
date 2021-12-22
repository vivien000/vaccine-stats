import datetime
import streamlit as st
import numpy as np
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


@st.cache(show_spinner=False)
def get_and_preprocess_data(today):
    urlretrieve(url, filename="data.csv")
    df = pd.read_csv("data.csv", sep=";")

    # Remove the 0-19 age range
    df = df[df["age"] != "[0,19]"]

    # Keep only the last 15 days available
    dates_kept = sorted(list(set(df["date"])))[-15:]
    df = df[df["date"].isin(dates_kept)]

    # Relabel the vaccination status and the age columns
    df["vac_statut"] = df["vac_statut"].map(transform_status)
    df["age"] = df["age"].map(transform_age)

    # Aggregate the data across all dates
    agg_dict = {"hc_pcr": "sum", "sc_pcr": "sum", "dc_pcr": "sum", "effectif": "mean"}
    df = (
        df.groupby(by=["age", "vac_statut", "date"])
        .sum()
        .groupby(by=["age", "vac_statut"])
        .agg(agg_dict)
    )

    # Compute the relative numbers of hospital admissions, ICU admissions and deaths
    df["hc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.hc_pcr / x.effectif, axis=1)
    df["sc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.sc_pcr / x.effectif, axis=1)
    df["dc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.dc_pcr / x.effectif, axis=1)
    df = df.reset_index(level=["age", "vac_statut"]).rename(columns={"age": "Âge"})
    return df, dates_kept


today = str(datetime.date.today())
today = f"{today[-2:]}/{today[5:7]}/{today[:4]}"
df, dates_kept = get_and_preprocess_data(today)
earliest, latest = dates_kept[0], dates_kept[-1]
earliest = f"{earliest[-2:]}/{earliest[5:7]}"
latest = f"{latest[-2:]}/{latest[5:7]}"


@st.cache(hash_funcs={matplotlib.figure.Figure: hash}, show_spinner=False)
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
Les 3 graphiques de cette page montrent les nombres, par million de personnes, des :

- **hospitalisations**
- entrées en **soins critiques**
- **décès**

... **au cours des 15 derniers jours** pour lesquels les données nationales sont disponibles ({earliest}-{latest}) et en fonction de l'**âge** et du **statut vaccinal**.

Ces graphiques sont mis à jour quotidiennement à partir des données de la [DREES]({url2}).
"""
)

for key, title in [
    ("hc_pcr_per_1M", "Hospitalisations avec test PCR positif"),
    ("sc_pcr_per_1M", "Entrées en soins critiques avec test PCR positif"),
    ("dc_pcr_per_1M", "Décès avec test PCR positif"),
]:
    st.pyplot(create_fig(key, title, today))

sum_by_vac_status = df.groupby(by=["Âge"]).sum()[
    ["effectif", "hc_pcr", "sc_pcr", "dc_pcr"]
]
age_range_sizes = sum_by_vac_status["effectif"]
observed = sum_by_vac_status.sum()[["hc_pcr", "sc_pcr", "dc_pcr"]]
non_vac_rates = (
    df[df["vac_statut"] == "[0]. Non vaccinés"]
    .groupby(by=["Âge"])
    .sum()[["hc_pcr_per_1M", "sc_pcr_per_1M", "dc_pcr_per_1M"]]
)

counterfactual = {
    "hc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["hc_pcr_per_1M"]) / 1e6,
    "sc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["sc_pcr_per_1M"]) / 1e6,
    "dc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["dc_pcr_per_1M"]) / 1e6,
}

st.markdown(
    f"""
À partir des données précédentes et connaissant l'effectif de chaque classe d'âge, il est possible de comparer la situation réellement observée et une situation hypothétique où chaque personne aurait les mêmes risques que les personnes non vaccinées de sa catégorie d'âge. **Le nombre de cas dans cette situation hypothétique est toutefois sous-estimé car il ne tient pas compte de l'effet de la couverture vaccinale sur la transmission**.

<style type="text/css">
table {{margin: auto}}
td {{width: 33.3%;text-align:center;vertical-align:middle}}
.bold {{font-weight: bold}}
</style>
<table>
<tbody>
  <tr style="border: none">
    <td style="border: none"></td>
    <td colspan="2"><b>Nombre de cas quotidiens avec test PCR positif</b><br>(période du {earliest} au {latest}, 20 ans ou plus)</td>
  </tr>
  <tr style="border: none" class="bold">
    <td style="border: none"></td>
    <td>Situation observée<br></td>
    <td>Situation hypothétique sans vaccins</td>
  </tr>
  <tr class="background-color">
    <td class="bold">Hospitalisations</td>
    <td>{int(observed['hc_pcr']/15)}</td>
    <td>{int(counterfactual['hc_pcr']/15)}</td>
  </tr>
  <tr>
    <td class="bold">Entrées en soins critiques</td>
    <td>{int(observed['sc_pcr']/15)}</td>
    <td>{int(counterfactual['sc_pcr']/15)}</td>
  </tr>
  <tr class="background-color">
    <td class="bold">Décès</td>
    <td>{int(observed['dc_pcr']/15)}</td>
    <td>{int(counterfactual['dc_pcr']/15)}</td>
  </tr>
</tbody>
</table>
<br>

Date de mise à jour : {today}, source : [DREES]({url2}), code : [GitHub](https://github.com/vivien000/vaccine-stats)

""",
    unsafe_allow_html=True,
)
