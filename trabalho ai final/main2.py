import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

# Import classifier tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"output_{now}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_path = 'china_cancer_patients_synthetic.csv'
Dataframe = pd.read_csv(data_path, encoding='ascii', delimiter=',')


# Display the first few records to inspect the data
print('Data Loaded Successfully.')
Dataframe.head()

#============================================================
# LIMPAR DADOS
#============================================================
# Vamos inspecionar a estrutura do dataframe
Dataframe.info()

# Verifica valores ausentes
valores_perdidos = Dataframe.isnull().sum()
print('Valores ausentes em cada coluna:')
print(valores_perdidos)

# Converte colunas de data para o formato datetime. Isso é crucial para evitar bugs sutis ao trabalhar com recursos relacionados a tempo.
Colunas = ['DiagnosisDate', 'SurgeryDate']
for col in Colunas:
    try:
        Dataframe[col] = pd.to_datetime(Dataframe[col])
        print(f"Convertido {col} para datetime.")
    except Exception as e:
        print(f"Erro ao converter {col}: {e}")

# Verifica os tipos de dados atualizados
Dataframe.info()

def analisar(escolha = False):
    if not escolha:
        return
    Dataframe_numerico = Dataframe.select_dtypes(include=[np.number])
    print(f"Colunas numéricas para análise: {list(Dataframe_numerico.columns)}")

    # Plota histogramas para cada coluna numérica
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(Dataframe_numerico.columns):
        plt.subplot(3, 3, i+1)
        sns.histplot(Dataframe_numerico[col].dropna(), kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"))
    plt.close()

    # Exibe o mapa de calor de correlação se houver pelo menos quatro colunas numéricas
    if len(Dataframe_numerico.columns) >= 4:
        plt.figure(figsize=(10, 8))
        corr = Dataframe_numerico.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Mapa de Calor de Correlação das Variáveis Numéricas')
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
        plt.close()
    else:
        print('Não há colunas numéricas suficientes para o mapa de calor de correlação.')

    # Explora algumas variáveis categóricas plotando distribuições de contagem
    categorias = ['Gender', 'SurvivalStatus', 'TumorType', 'CancerStage', 'SmokingStatus', 'AlcoholUse']

    plt.figure(figsize=(15, 12))
    for i, col in enumerate(categorias):
        plt.subplot(3, 2, i+1)
        sns.countplot(data=Dataframe, x=col, order=Dataframe[col].value_counts().index)
        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categorical_counts.png"))
    plt.close()

    # Gráfico separado para DiasAteACirurgia
    plt.figure(figsize=(12, 6))
    dias_counts = Dataframe['DiasAteACirurgia'].value_counts().sort_index()
    plt.plot(dias_counts.index, dias_counts.values, marker='o')
    nulos = Dataframe['DiasAteACirurgia'].isnull().sum()
    plt.title('Distribuição dos dias até a cirurgia')
    plt.figtext(0.60, 0.01, f"Sem cirurgia: {nulos}")
    plt.xlabel('DiasAteACirurgia')
    plt.ylabel('Contagem')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "surgery.png"))
    plt.close()

    # Exibe a quantidade de valores ausentes
    #print(f"Valores ausentes em DiasAteACirurgia: {nulos}")

    # Pair Plot das variáveis numéricas selecionadas
    featuresEscolidas = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths', 'DiasAteACirurgia']
    sns.pairplot(Dataframe[featuresEscolidas].dropna())
    plt.suptitle('Pair Plot das Variáveis Numéricas Selecionadas', y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, "Selected features.png"))
    plt.close()

    # Boxplot: distribuição da idade por status de sobrevivência
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='SurvivalStatus', y='Age', data=Dataframe)
    plt.title('Distribuição da Idade por Status de Sobrevivência')
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_age_survival.png"))
    plt.close()

    # Violin Plot para comparar a distribuição do tamanho do tumor por estágio do câncer
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='CancerStage', y='TumorSize', data=Dataframe)
    plt.title('Tamanho do Tumor por Estágio do Câncer')
    plt.savefig(os.path.join(OUTPUT_DIR, "violinplot_tumorsize_stage.png"))
    plt.close()

#============================================================
# ANÁLISE DOS DADOS
#============================================================
analisar(True)

#=============================================
# Prever mortalidade
#=============================================

# Cria uma cópia do dataframe para modelagem
Modelo_epico = Dataframe.copy()

# Lista de variáveis utilizadas para predição. Seleciona colunas relevantes.
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths']

# Para variáveis categóricas, podemos incluir algumas selecionadas. Aqui, incluímos Gender e CancerStage 'DiasAteACirurgia'
features += ['Gender', 'CancerStage']

# A variável alvo é SurvivalStatus. Precisamos codificá-la como 0 e 1.
alvo = 'SurvivalStatus'

# Remove linhas com valores ausentes nas features
Modelo_epico = Modelo_epico[features + [alvo]].dropna()

# Codifica variáveis categóricas
categorias_de_features = ['Gender', 'CancerStage']
le_dict = {}
for col in categorias_de_features:
    le = LabelEncoder()
    Modelo_epico[col] = le.fit_transform(Modelo_epico[col])
    le_dict[col] = le

# Codifica a variável alvo
alvo_le = LabelEncoder()
Modelo_epico[alvo] = alvo_le.fit_transform(Modelo_epico[alvo])

# Arrays de features e alvo
X = Modelo_epico[features]
y = Modelo_epico[alvo]

# Normaliza as variáveis numéricas para melhor desempenho do modelo
scaler = StandardScaler()
numeric_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths']
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa e treina o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Faz previsões no conjunto de teste e calcula a acurácia
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia da predição: {accuracy:.4f}")


# Plota a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=alvo_le.inverse_transform([0, 1]),
    yticklabels=alvo_le.inverse_transform([0, 1])
)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# Calcula a curva ROC e a AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
plt.close()

with open(os.path.join(OUTPUT_DIR, "hiperparametros.txt"), "w", encoding="utf-8") as f:
    f.write("Modelo: LogisticRegression\n")
    f.write("Hiperparâmetros utilizados:\n")
    f.write(str(model.get_params()))
pass