"""
Segmentación de clientes con FAMD + clustering aglomerativo
============================================================

Pipeline de segmentación no supervisada sobre datos mixtos (variables
numéricas y categóricas), replicando la metodología que uso en proyectos
reales de segmentación empresarial:

    1. Carga y limpieza de datos.
    2. Reducción de dimensionalidad con FAMD (Factor Analysis of Mixed Data),
       que maneja simultáneamente variables numéricas y categóricas.
    3. Clustering aglomerativo (Ward) sobre las coordenadas factoriales.
    4. Selección del número de segmentos por coeficiente de silueta.
    5. Perfilado cualitativo de cada segmento.

Dataset: IBM Telco Customer Churn (público). Churn NO entra al modelo;
se usa solo como validación externa de los segmentos.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
N_COMPONENTS = 5
BASE = Path(__file__).parent
FIG = BASE / "figuras"
OUT = BASE / "resultados"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

sns_palette = plt.cm.Set2.colors


# --------------------------------------------------------------------------- #
# 1. Carga y limpieza
# --------------------------------------------------------------------------- #
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(BASE / "data" / "Telco-Customer-Churn.csv")

    # TotalCharges trae espacios en blanco -> a numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    # SeniorCitizen viene como 0/1 -> es categórica
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables para el modelo (sin ID ni la etiqueta churn)."""
    numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
    categoricas = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    X = df[numericas + categoricas].copy()
    for c in categoricas:
        X[c] = X[c].astype("category")
    return X


# --------------------------------------------------------------------------- #
# 2. FAMD
# --------------------------------------------------------------------------- #
def aplicar_famd(X: pd.DataFrame):
    famd = prince.FAMD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    famd = famd.fit(X)
    coords = famd.row_coordinates(X)
    inercia = famd.eigenvalues_summary
    print("\n== FAMD: % de varianza explicada por componente ==")
    print(inercia)
    return famd, coords


# --------------------------------------------------------------------------- #
# 3-4. Clustering aglomerativo + selección de k
# --------------------------------------------------------------------------- #
def elegir_k(coords: pd.DataFrame, k_min=2, k_max=8):
    """Selecciona k. La silueta suele maximizarse en k=2 (división trivial
    grande/chico), poco útil para activar estrategias; por eso se elige el
    mejor k entre k>=3, que balancea cohesión y granularidad accionable."""
    X = coords.values
    resultados = {}
    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        resultados[k] = silhouette_score(X, labels)
    k_estadistico = max(resultados, key=resultados.get)
    accionables = {k: v for k, v in resultados.items() if k >= 3}
    k_negocio = max(accionables, key=accionables.get)

    plt.figure(figsize=(7, 4))
    plt.plot(list(resultados), list(resultados.values()), "o-", color="#0f6e7a")
    plt.axvline(k_negocio, ls="--", color="#b3541e",
                label=f"k accionable = {k_negocio}")
    plt.title("Selección de número de segmentos · coeficiente de silueta")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "01_silueta.png", dpi=150)
    plt.close()
    print(f"\n== Silueta por k ==\n{resultados}")
    print(f"-> máximo estadístico en k={k_estadistico} (división trivial); "
          f"k accionable elegido = {k_negocio}")
    return k_negocio


def dendrograma(coords: pd.DataFrame):
    Z = linkage(coords.values, method="ward")
    plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5, color_threshold=0,
               above_threshold_color="#0f6e7a")
    plt.title("Dendrograma · clustering aglomerativo (Ward)")
    plt.xlabel("Clientes (agrupados)")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig(FIG / "02_dendrograma.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 5. Perfilado + visualización
# --------------------------------------------------------------------------- #
def scatter_segmentos(coords, labels):
    labels = np.asarray(labels)
    xy = coords.values
    plt.figure(figsize=(7, 6))
    for c in sorted(set(labels)):
        m = labels == c
        plt.scatter(xy[m, 0], xy[m, 1],
                    s=8, alpha=0.5, label=f"Segmento {c}",
                    color=sns_palette[c % len(sns_palette)])
    plt.title("Segmentos en el espacio FAMD (dim. 1 y 2)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(FIG / "03_segmentos_famd.png", dpi=150)
    plt.close()


def perfilar(df, labels):
    df = df.copy()
    df["segmento"] = labels
    n = len(df)

    # Tamaño
    tam = df["segmento"].value_counts().sort_index()
    resumen = pd.DataFrame({
        "n_clientes": tam,
        "% base": (tam / n * 100).round(1),
    })
    # Numéricas (mediana)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        resumen[f"{col}_mediana"] = df.groupby("segmento")[col].median().round(1)
    # Churn como validación externa
    resumen["churn_%"] = (
        df.assign(c=(df["Churn"] == "Yes")).groupby("segmento")["c"].mean() * 100
    ).round(1)
    # Moda de variables clave
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        resumen[f"{col}_moda"] = df.groupby("segmento")[col].agg(
            lambda s: s.mode().iloc[0])

    resumen.to_csv(OUT / "perfil_segmentos.csv")
    print("\n== Perfil de segmentos ==")
    print(resumen.to_string())

    # Gráfico churn por segmento (validación de negocio)
    plt.figure(figsize=(7, 4))
    plt.bar(resumen.index.astype(str), resumen["churn_%"],
            color=[sns_palette[i % len(sns_palette)] for i in resumen.index])
    plt.title("Tasa de fuga (churn) por segmento · validación externa")
    plt.xlabel("Segmento")
    plt.ylabel("% churn")
    plt.tight_layout()
    plt.savefig(FIG / "04_churn_por_segmento.png", dpi=150)
    plt.close()
    return resumen


# --------------------------------------------------------------------------- #
def main():
    df = cargar_datos()
    print(f"Clientes analizados: {len(df):,} · variables: {df.shape[1]}")
    X = preparar_features(df)

    famd, coords = aplicar_famd(X)
    k = elegir_k(coords)
    dendrograma(coords)

    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
        coords.values)
    scatter_segmentos(coords, labels)
    resumen = perfilar(df, labels)

    # Guardar resumen en markdown para el README
    with open(OUT / "perfil_segmentos.md", "w") as f:
        f.write(f"# Perfil de los {k} segmentos\n\n")
        f.write(f"Clientes analizados: **{len(df):,}** · {k} segmentos.\n\n")
        f.write(resumen.to_markdown())
    print("\nFiguras en figuras/ · resultados en resultados/")


if __name__ == "__main__":
    main()
