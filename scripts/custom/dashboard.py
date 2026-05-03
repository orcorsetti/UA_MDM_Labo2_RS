import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Miami RE – EDA Dashboard", layout="wide")
sns.set_style("whitegrid")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tabular"

@st.cache_data
def load_data():
    train = pd.read_csv(DATA_DIR / "train_processed.csv")
    test  = pd.read_csv(DATA_DIR / "test_processed.csv")
    return train, test

train, test = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Miami RE — EDA")
section = st.sidebar.radio("Sección", [
    "Resumen del Dataset",
    "Target: Precio",
    "Tipo de Propiedad",
    "Variables Numéricas",
    "Variables Booleanas",
    "Geografía",
    "Correlaciones",
    "Texto",
])

home_types = ["Todos"] + sorted(train["homeType"].dropna().unique().tolist())
ht_filter = st.sidebar.selectbox("Filtrar por homeType", home_types)
if ht_filter != "Todos":
    df = train[train["homeType"] == ht_filter].copy()
else:
    df = train.copy()

st.sidebar.markdown(f"**Registros visibles:** {len(df):,}")

# ── Helper ───────────────────────────────────────────────────────────────────
def fmt_usd(x, pos=None):
    if x >= 1e6: return f"${x/1e6:.1f}M"
    if x >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"

def apply_usd_y(ax): ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
def apply_usd_x(ax): ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))

# ═══════════════════════════════════════════════════════════════════════════════
if section == "Resumen del Dataset":
    st.title("Resumen del Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filas train", f"{len(train):,}")
    col2.metric("Filas test",  f"{len(test):,}")
    col3.metric("Features",    str(train.shape[1]))
    col4.metric("Precio mediano", f"${train['lastSoldPrice_hpi_adjusted'].median():,.0f}")

    st.subheader("Primeras filas")
    st.dataframe(train.head(10))

    st.subheader("Estadísticas descriptivas")
    num_cols = train.select_dtypes("number").columns.tolist()
    st.dataframe(train[num_cols].describe().T.style.format("{:.2f}"))

    st.subheader("Valores nulos")
    nulos = train.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(nulos):
        fig, ax = plt.subplots(figsize=(10, 4))
        nulos.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Valores nulos por columna")
        ax.set_ylabel("Cantidad")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(pd.DataFrame({"Nulos": nulos, "% del total": (nulos/len(train)*100).round(2)}))
    else:
        st.success("Sin valores nulos en el dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Target: Precio":
    st.title("Variable Target: Precio de Venta")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["lastSoldPrice_hpi_adjusted"].plot(kind="hist", bins=60, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
    axes[0].set_title("Distribución del Precio (original)")
    df["log_price"].plot(kind="hist", bins=60, ax=axes[1], color="coral", edgecolor="white")
    axes[1].set_title("Distribución de log_price")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Media",   f"${df['lastSoldPrice_hpi_adjusted'].mean():,.0f}")
    col2.metric("Mediana", f"${df['lastSoldPrice_hpi_adjusted'].median():,.0f}")
    col3.metric("Mínimo",  f"${df['lastSoldPrice_hpi_adjusted'].min():,.0f}")
    col4.metric("Máximo",  f"${df['lastSoldPrice_hpi_adjusted'].max():,.0f}")

    st.subheader("Percentiles del precio")
    percs = [1,5,10,25,50,75,90,95,99]
    pvals = np.percentile(df["lastSoldPrice_hpi_adjusted"].dropna(), percs)
    st.dataframe(pd.DataFrame({"Percentil": percs, "Precio": [f"${v:,.0f}" for v in pvals]}))

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Tipo de Propiedad":
    st.title("Tipo de Propiedad (homeType)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train["homeType"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Cantidad por Tipo")
    axes[0].tick_params(axis="x", rotation=30)

    ht_price = train.groupby("homeType")["lastSoldPrice_hpi_adjusted"].median().sort_values()
    ht_price.plot(kind="barh", ax=axes[1], color="coral", edgecolor="white")
    apply_usd_x(axes[1])
    axes[1].set_title("Precio Mediano por Tipo")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Boxplot por tipo (escala log)")
    order = train.groupby("homeType")["lastSoldPrice_hpi_adjusted"].median().sort_values().index
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.boxplot(x="homeType", y="lastSoldPrice_hpi_adjusted", data=train, order=order, palette="Set2", ax=ax2)
    ax2.set_yscale("log")
    apply_usd_y(ax2)
    ax2.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Variables Numéricas":
    st.title("Variables Numéricas")

    num_var = st.selectbox("Variable", ["livingArea","bedrooms","bathrooms","yearBuilt",
                                         "photoCount","taxAssessedValue","last_listing_price",
                                         "avg_school_rating","lotAreaValue","hoa_fee_monthly"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[num_var].dropna().plot(kind="hist", bins=50, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title(f"Distribución de {num_var}")

    axes[1].scatter(df[num_var], df["lastSoldPrice_hpi_adjusted"], alpha=0.1, s=5, color="coral")
    apply_usd_y(axes[1])
    axes[1].set_title(f"Precio vs {num_var}")
    axes[1].set_xlabel(num_var)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"**Correlación con precio:** `{df[num_var].corr(df['lastSoldPrice_hpi_adjusted']):.3f}`")

    if num_var == "bedrooms":
        st.subheader("Precio mediano por habitaciones")
        bp = df.groupby("bedrooms")["lastSoldPrice_hpi_adjusted"].median()
        bp = bp[bp.index <= 10]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        bp.plot(kind="bar", ax=ax3, color="coral", edgecolor="white")
        apply_usd_y(ax3)
        ax3.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Variables Booleanas":
    st.title("Variables Booleanas")

    bool_cols = ["has_hoa","has_pool","has_garage","has_waterfront",
                 "tag_price_cut","tag_new_construction","tag_foreclosure"]

    st.subheader("Tasas de presencia")
    bool_rates = train[bool_cols].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    bool_rates.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    for i,v in enumerate(bool_rates): ax.text(i, v+0.005, f"{v:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Proporción")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Tasa de Presencia de Atributos Booleanos")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Precio mediano: Con vs Sin atributo")
    fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, col in enumerate(bool_cols):
        g = df.groupby(col)["lastSoldPrice_hpi_adjusted"].median()
        g.index = ["No","Sí"]
        g.plot(kind="bar", ax=axes[i], color=["steelblue","coral"], edgecolor="white")
        apply_usd_y(axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=0)
    axes[-1].set_visible(False)
    plt.suptitle("Precio Mediano: Con vs Sin Atributo", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Tabla resumen")
    rows = []
    for col in bool_cols:
        g = df.groupby(col)["lastSoldPrice_hpi_adjusted"].agg(["median","count"])
        rows.append({"Atributo": col,
                     "Precio (Sin)": f"${g.loc[0,'median']:,.0f}" if 0 in g.index else "-",
                     "Precio (Con)": f"${g.loc[1,'median']:,.0f}" if 1 in g.index else "-",
                     "N (Sin)": int(g.loc[0,"count"]) if 0 in g.index else 0,
                     "N (Con)": int(g.loc[1,"count"]) if 1 in g.index else 0})
    st.dataframe(pd.DataFrame(rows))

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Geografía":
    st.title("Análisis Geográfico")

    top_n = st.slider("Top N ZIP codes", 10, 30, 20)
    top_zips = df["zipcode"].value_counts().head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    top_zips.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title(f"Top {top_n} ZIP Codes")
    axes[0].tick_params(axis="x", rotation=45)

    zip_price = df.groupby("zipcode")["lastSoldPrice_hpi_adjusted"].median()
    zip_price.loc[top_zips.index].sort_values().plot(kind="barh", ax=axes[1], color="coral", edgecolor="white")
    apply_usd_x(axes[1])
    axes[1].set_title("Precio Mediano por ZIP")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Mapa de precios (lat/lon)")
    n_sample = st.slider("Muestra de puntos", 500, 5000, 3000)
    sample = df.dropna(subset=["latitude","longitude","lastSoldPrice_hpi_adjusted"])
    sample = sample.sample(min(n_sample, len(sample)), random_state=42)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sc = ax2.scatter(sample["longitude"], sample["latitude"],
                     c=np.log1p(sample["lastSoldPrice_hpi_adjusted"]),
                     cmap="RdYlGn", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax2, label="log_price")
    ax2.set_title("Distribución Geográfica de Precios")
    ax2.set_xlabel("Longitud"); ax2.set_ylabel("Latitud")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Correlaciones":
    st.title("Matriz de Correlación")

    num_cols = ["bedrooms","bathrooms","livingArea","yearBuilt","lotAreaValue",
                "photoCount","taxAssessedValue","propertyTaxRate","last_listing_price",
                "avg_school_rating","has_hoa","hoa_fee_monthly","has_pool",
                "has_garage","has_waterfront","property_age","bath_to_bed_ratio",
                "desc_length","desc_word_count","lastSoldPrice_hpi_adjusted"]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de Correlación")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Top correlaciones con el precio")
    price_corr = corr["lastSoldPrice_hpi_adjusted"].drop("lastSoldPrice_hpi_adjusted")\
                      .sort_values(key=abs, ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    price_corr.plot(kind="barh", ax=ax2, color=["coral" if v > 0 else "steelblue" for v in price_corr])
    ax2.set_title("Correlación con lastSoldPrice_hpi_adjusted")
    ax2.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Correlaciones por tipo de propiedad")
    ht_sel = st.selectbox("Tipo", train["homeType"].value_counts().head(5).index.tolist())
    subset = train[train["homeType"] == ht_sel]
    corr_sub = subset[num_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, vmin=-1, vmax=1, ax=ax3)
    ax3.set_title(f"Correlación — {ht_sel}")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Texto":
    st.title("Análisis de Texto: Description")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df["desc_length"].plot(kind="hist", bins=50, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Longitud de Descripción (caracteres)")
    axes[1].scatter(df["desc_word_count"], df["lastSoldPrice_hpi_adjusted"],
                    alpha=0.05, s=5, color="coral")
    apply_usd_y(axes[1])
    axes[1].set_title("Precio vs Palabras en la Descripción")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Precio mediano según features textuales")
    text_cols = ["desc_is_boilerplate","desc_mentions_renovated",
                 "desc_mentions_pool","desc_mentions_view","desc_mentions_new"]
    fig2, axes2 = plt.subplots(1, len(text_cols), figsize=(18, 5))
    for ax, col in zip(axes2, text_cols):
        g = df.groupby(col)["lastSoldPrice_hpi_adjusted"].median()
        g.index = ["No","Sí"]
        g.plot(kind="bar", ax=ax, color=["steelblue","coral"], edgecolor="white")
        apply_usd_y(ax)
        ax.set_title(col.replace("desc_","").replace("_"," ").title(), fontsize=10)
        ax.tick_params(axis="x", rotation=0)
    plt.suptitle("Precio Mediano según Features Textuales", fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Boilerplate vs precio")
    bp_rate = df["desc_is_boilerplate"].mean()
    st.metric("% descripciones boilerplate", f"{bp_rate:.1%}")
    st.markdown("""
    ~52% de las descripciones son plantillas genéricas del tipo *"This X square foot home..."*.
    Usar `desc_is_boilerplate` como feature puede ayudar al modelo a capturar la calidad del listado.
    """)
