# pip install numpy scikit-learn xgboost

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo dos gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Carregar a base novamente, se necessário
df = pd.read_csv("china_cancer_patients_synthetic.csv")

# Tratamento básico
df.replace("None", pd.NA, inplace=True)

# -----------------------------
# 1. Distribuição da variável alvo
# -----------------------------
sns.countplot(x="SurvivalStatus", data=df, palette="pastel")
plt.title("Distribuição dos Pacientes por Status de Sobrevivência")
plt.xlabel("Status de Sobrevivência")
plt.ylabel("Quantidade")
plt.show()

# -----------------------------
# 2. Distribuição de idade
# -----------------------------
sns.histplot(df["Age"], bins=30, kde=True, color="skyblue")
plt.title("Distribuição de Idade dos Pacientes")
plt.xlabel("Idade")
plt.show()

# -----------------------------
# 3. Boxplot: Idade x Sobrevivência
# -----------------------------
sns.boxplot(x="SurvivalStatus", y="Age", data=df, palette="Set2")
plt.title("Idade por Status de Sobrevivência")
plt.xlabel("Status")
plt.ylabel("Idade")
plt.show()

# -----------------------------
# 4. Tipo de tumor mais comum
# -----------------------------
sns.countplot(y="TumorType", data=df, order=df["TumorType"].value_counts().index, palette="muted")
plt.title("Frequência por Tipo de Tumor")
plt.xlabel("Número de Casos")
plt.ylabel("Tipo de Tumor")
plt.show()

# -----------------------------
# 5. Estágios do câncer por status de sobrevivência
# -----------------------------
sns.countplot(x="CancerStage", hue="SurvivalStatus", data=df, palette="Set1")
plt.title("Estágio do Câncer por Status de Sobrevivência")
plt.xlabel("Estágio")
plt.ylabel("Quantidade")
plt.legend(title="Status")
plt.show()

# -----------------------------
# 6. Dispersão: Tamanho do tumor x Idade
# -----------------------------
sns.scatterplot(x="Age", y="TumorSize", hue="SurvivalStatus", data=df, alpha=0.6)
plt.title("Tamanho do Tumor vs. Idade")
plt.xlabel("Idade")
plt.ylabel("Tamanho do Tumor (cm)")
plt.show()

# -----------------------------
# 7. Correlação entre variáveis numéricas
# -----------------------------
# Selecionar apenas colunas numéricas
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
correlation = df[num_cols].corr()

# Mapa de calor (heatmap)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação (Variáveis Numéricas)")
plt.show()
