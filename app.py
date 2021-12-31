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

today = str(datetime.date.today())
today = f"{today[-2:]}/{today[5:7]}/{today[:4]}"

query_params = st.experimental_get_query_params()
lang2 = "en" if "lang" in query_params and "en" in query_params["lang"] else "fr"
legend_title = {"fr": "Statut vaccinal", "en": "Vaccine status"}
xaxis_title = {"fr": "Âge", "en": "Age"}
yaxis_title = {"fr": "Cas par million de personnes", "en": "Cases per million people"}

lang_selection = st.selectbox("", ["🇫🇷", "🇬🇧"], index=0 if lang2 == "fr" else 1)
lang = "fr" if lang_selection == "🇫🇷" else "en"


def transform_status(s, lang):
    if lang == "fr":
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
    else:
        if "Non-vaccinés" in s:
            return "[0]. Not vaccinated"
        if "récente" in s:
            return "[1]. First dose (≤ 14 days)"
        if "efficace" in s:
            return "[2]. First dose (> 14 jours)"
        if "Complet" in s and "sans rappel" in s:
            return "[3]. Fully vaccinated (without booster)"
        if "Complet" in s and "avec rappel" in s:
            return "[4]. Fully vaccinated (with booster)"


def transform_age(s, lang):
    if lang == "fr":
        d = {
            "[20,39]": "20-39 ans",
            "[40,59]": "40-59 ans",
            "[60,79]": "60-79 ans",
            "[80;+]": "80 ans et plus",
        }
    else:
        d = {
            "[20,39]": "20-39 years old",
            "[40,59]": "40-59 years old",
            "[60,79]": "60-79 years old",
            "[80;+]": "80+ years old",
        }
    return d[s]


@st.cache(show_spinner=False)
def get_data(today):
    urlretrieve(url, filename="data.csv")
    df = pd.read_csv("data.csv", sep=";", on_bad_lines="skip")

    # Remove the 0-19 age range
    df = df[df["age"] != "[0,19]"]

    # Keep only the last 15 days available
    dates_kept = sorted(list(set(df["date"])))[-15:]
    df = df[df["date"].isin(dates_kept)]

    return df, dates_kept


df_initial, dates_kept = get_data(today)
earliest, latest = dates_kept[0], dates_kept[-1]
earliest = f"{earliest[-2:]}/{earliest[5:7]}"
latest = f"{latest[-2:]}/{latest[5:7]}"


@st.cache(show_spinner=False)
def preprocess_data(today, df_initial, lang):
    # Relabel the vaccination status and the age columns
    df = df_initial.copy()
    df["vac_statut"] = df["vac_statut"].map(lambda s: transform_status(s, lang))
    df["age"] = df["age"].map(lambda s: transform_age(s, lang))

    # Aggregate the data across all dates
    agg_dict = {"hc_pcr": "sum", "sc_pcr": "sum", "dc_pcr": "sum", "effectif": "mean"}
    df = (
        df.groupby(by=["age", "vac_statut", "date"])
        .sum()
        .groupby(by=["age", "vac_statut"])
        .agg(agg_dict)
    )
    df = df.reset_index(level=["age", "vac_statut"])

    # Compute the relative numbers of hospital admissions, ICU admissions and deaths
    df["hc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.hc_pcr / x.effectif, axis=1)
    df["sc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.sc_pcr / x.effectif, axis=1)
    df["dc_pcr_per_1M"] = df.apply(lambda x: 1e6 * x.dc_pcr / x.effectif, axis=1)

    return df


df = preprocess_data(today, df_initial, lang)

sidebar_title = {
    "fr": "COVID-19 : cas graves en fonction de l'âge et du statut vaccinal en France",
    "en": "COVID-19: severe cases by age and vaccine status in France",
}

sidebar_text = {
    "fr": f"""
Les 3 graphiques de cette page montrent les nombres, par million de personnes, des :

- **hospitalisations**
- entrées en **soins critiques**
- **décès**

... **au cours des 15 derniers jours** pour lesquels les données nationales sont disponibles ({earliest}-{latest}) et en fonction de l'**âge** et du **statut vaccinal**.

Ces graphiques sont mis à jour quotidiennement à partir des données de la [DREES]({url2}).
""",
    "en": f"""
The 3 charts on this page show the relative numbers (per million of people, by **age** and **vaccine status**) of:

- **hospital admissions**
- **ICU admissions**
- **deaths**

... **during the last 15 days** for which national data are available ({earliest}-{latest}).

The charts are updated daily with [DREES]({url2}) data.
""",
}

titles = {
    0: {
        "fr": r"$\bf{Hospitalisations}$"
        + f" avec test PCR positif du {earliest} au {latest}",
        "en": r"$\bf{Hospital\ admissions}$"
        + f" with positive PCR test from {earliest} until {latest}",
    },
    1: {
        "fr": r"$\bf{Entrées\ en\ soins\ critiques}$"
        + f" avec test PCR positif du {earliest} au {latest}",
        "en": r"$\bf{ICU\ admissions}$"
        + f" with positive PCR test from {earliest} until {latest}",
    },
    2: {
        "fr": r"$\bf{Décès}$" + f" avec test PCR positif du {earliest} au {latest}",
        "en": r"$\bf{Deaths}$"
        + f" with positive PCR test from {earliest} until {latest}",
    },
}

table_title = {
    "fr": f"<b>Nombre de cas quotidiens avec test PCR positif</b><br>(période du {earliest} au {latest}, 20 ans ou plus)",
    "en": f"<b>Daily number of cases with positive PCR test</b><br>(period from {earliest} until {latest}, 20 years old or more)",
}

table_columns = {
    0: {"fr": "Situation observée", "en": "Actual situation"},
    1: {
        "fr": "Situation hypothétique sans vaccins",
        "en": "Hypothetical situation without vaccines",
    },
}

table_rows = {
    0: {"fr": "Hospitalisations", "en": "Hospital admissions"},
    1: {"fr": "Entrées en soins critiques", "en": "ICU admissions"},
    2: {"fr": "Décès", "en": "Deaths"},
}

table_text = {
    0: {
        "fr": "À partir des données précédentes et connaissant l'effectif de chaque classe d'âge, il est possible de comparer la situation réellement observée et une situation hypothétique où chaque personne aurait les mêmes risques que les personnes non vaccinées de sa catégorie d'âge. **Le nombre de cas dans cette situation hypothétique est toutefois sous-estimé car il ne tient pas compte de l'effet de la couverture vaccinale sur la transmission**.",
        "en": "With these data and the size of the each age group, it's possible to compare the actual situation with a hypothetical situation in which everyone had the same risks as the risks as the non-vaccinated persons of her age group. **The number of cases in this hypothetical situation is however underestimated because it doesn't take into account the impact of vaccination on transmission.**",
    },
    1: {
        "fr": f"Date de mise à jour : {today}, source : [DREES]({url2}), code : [GitHub](https://github.com/vivien000/vaccine-stats)",
        "en": f"Updated: {today}, source: [DREES]({url2}), code: [GitHub](https://github.com/vivien000/vaccine-stats)",
    },
}

# @st.cache(hash_funcs={matplotlib.figure.Figure: hash}, show_spinner=False)
def create_fig(key, title, lang, today):
    sns.set_context("paper")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="age",
        y=key,
        hue="vac_statut",
        palette=sns.color_palette("rocket"),
        ax=ax,
    )
    sns.despine()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [x[5:] for x in labels], title=legend_title[lang])
    ax.set(
        ylabel=yaxis_title[lang], xlabel=xaxis_title[lang], title=title,
    )
    return fig


st.sidebar.header(sidebar_title[lang])
st.sidebar.markdown(sidebar_text[lang])

st.markdown(
    """
    <style>
        div.stSelectbox{
            width: 15% !important;
        }
        .element-container {
            display: flex;
            justify-content: flex-end;
            flex-direction: row;
        }
    </style>""",
    unsafe_allow_html=True,
)

for key, title in [
    ("hc_pcr_per_1M", titles[0][lang]),
    ("sc_pcr_per_1M", titles[1][lang]),
    ("dc_pcr_per_1M", titles[2][lang]),
]:
    st.pyplot(create_fig(key, title, lang, today))


@st.cache(show_spinner=False)
def compute_daily_cases(today):
    sum_by_vac_status = df.groupby(by=["age"]).sum()[
        ["effectif", "hc_pcr", "sc_pcr", "dc_pcr"]
    ]
    age_range_sizes = sum_by_vac_status["effectif"]
    observed = sum_by_vac_status.sum()[["hc_pcr", "sc_pcr", "dc_pcr"]]
    non_vac_rates = (
        df[df["vac_statut"].map(lambda s: s[1]) == "0"]
        .groupby(by=["age"])
        .sum()[["hc_pcr_per_1M", "sc_pcr_per_1M", "dc_pcr_per_1M"]]
    )
    counterfactual = {
        "hc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["hc_pcr_per_1M"])
        / 1e6,
        "sc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["sc_pcr_per_1M"])
        / 1e6,
        "dc_pcr": np.dot(np.array(age_range_sizes), non_vac_rates["dc_pcr_per_1M"])
        / 1e6,
    }
    return observed, counterfactual


observed, counterfactual = compute_daily_cases(today)

st.markdown(
    f"""
{table_text[0][lang]}

<style type="text/css">
table {{margin: auto}}
td {{width: 33.3%;text-align:center;vertical-align:middle}}
.bold {{font-weight: bold}}
</style>
<table>
<tbody>
  <tr style="border: none">
    <td style="border: none"></td>
    <td colspan="2">{table_title[lang]}</td>
  </tr>
  <tr style="border: none" class="bold">
    <td style="border: none"></td>
    <td>{table_columns[0][lang]}</td>
    <td>{table_columns[1][lang]}</td>
  </tr>
  <tr class="background-color">
    <td class="bold">{table_rows[0][lang]}</td>
    <td>{int(observed['hc_pcr']/15)}</td>
    <td>{int(counterfactual['hc_pcr']/15)}</td>
  </tr>
  <tr>
    <td class="bold">{table_rows[1][lang]}</td>
    <td>{int(observed['sc_pcr']/15)}</td>
    <td>{int(counterfactual['sc_pcr']/15)}</td>
  </tr>
  <tr class="background-color">
    <td class="bold">{table_rows[2][lang]}</td>
    <td>{int(observed['dc_pcr']/15)}</td>
    <td>{int(counterfactual['dc_pcr']/15)}</td>
  </tr>
</tbody>
</table>
<br>

{table_text[1][lang]}

""",
    unsafe_allow_html=True,
)
