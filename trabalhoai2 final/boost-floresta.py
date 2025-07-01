import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importa XGBoost
from xgboost import XGBClassifier

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"output_xgb_{now}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_path = 'china_cancer_patients_synthetic.csv'
Dataframe = pd.read_csv(data_path, encoding='ascii', delimiter=',')

print('Data Loaded Successfully.')
Dataframe.head()

#============================================================
# LIMPAR DADOS
#============================================================
Dataframe.info()

valores_perdidos = Dataframe.isnull().sum()
print('Valores ausentes em cada coluna:')
print(valores_perdidos)

Colunas = ['DiagnosisDate', 'SurgeryDate']
for col in Colunas:
    try:
        Dataframe[col] = pd.to_datetime(Dataframe[col])
        print(f"Convertido {col} para datetime.")
    except Exception as e:
        print(f"Erro ao converter {col}: {e}")

Dataframe.info()

def analisar(escolha = False):
    if not escolha:
        return
    Dataframe_numerico = Dataframe.select_dtypes(include=[np.number])
    print(f"Colunas numéricas para análise: {list(Dataframe_numerico.columns)}")

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(Dataframe_numerico.columns):
        plt.subplot(3, 3, i+1)
        sns.histplot(Dataframe_numerico[col].dropna(), kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"))
    plt.close()

    if len(Dataframe_numerico.columns) >= 4:
        plt.figure(figsize=(10, 8))
        corr = Dataframe_numerico.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Mapa de Calor de Correlação das Variáveis Numéricas')
        plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
        plt.close()
    else:
        print('Não há colunas numéricas suficientes para o mapa de calor de correlação.')

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

    featuresEscolidas = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths', 'DiasAteACirurgia']
    sns.pairplot(Dataframe[featuresEscolidas].dropna())
    plt.suptitle('Pair Plot das Variáveis Numéricas Selecionadas', y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, "Selected features.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='SurvivalStatus', y='Age', data=Dataframe)
    plt.title('Distribuição da Idade por Status de Sobrevivência')
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplot_age_survival.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='CancerStage', y='TumorSize', data=Dataframe)
    plt.title('Tamanho do Tumor por Estágio do Câncer')
    plt.savefig(os.path.join(OUTPUT_DIR, "violinplot_tumorsize_stage.png"))
    plt.close()

analisar(True)

#=============================================
# Prever mortalidade
#=============================================

Modelo_epico = Dataframe.copy()
features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths']
features += ['Gender', 'CancerStage']
alvo = 'SurvivalStatus'

Modelo_epico = Modelo_epico[features + [alvo]].dropna()

categorias_de_features = ['Gender', 'CancerStage']
le_dict = {}
for col in categorias_de_features:
    le = LabelEncoder()
    Modelo_epico[col] = le.fit_transform(Modelo_epico[col])
    le_dict[col] = le

alvo_le = LabelEncoder()
Modelo_epico[alvo] = alvo_le.fit_transform(Modelo_epico[alvo])

X = Modelo_epico[features]
y = Modelo_epico[alvo]

scaler = StandardScaler()
numeric_features = ['Age', 'TumorSize', 'ChemotherapySessions', 'RadiationSessions', 'FollowUpMonths']
X[numeric_features] = scaler.fit_transform(X[numeric_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Classifier
model = XGBClassifier(random_state=42, n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia da predição: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0
especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

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

plt.figtext(
    1.05, 0.5,
    f"Acurácia: {accuracy:.2f}\nSensibilidade: {sensibilidade:.2f}\nEspecificidade: {especificidade:.2f}",
    fontsize=12, va='center'
)

plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches='tight')
plt.close()

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

# Curva de overfitting (validation curve) para XGBoost
param_range = range(1, 21)
train_scores, test_scores = validation_curve(
    XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
    X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(param_range, train_mean, label="Acurácia Treino", marker='o')
plt.plot(param_range, test_mean, label="Acurácia Teste (Validação)", marker='o')
plt.xlabel("Profundidade Máxima das Árvores (max_depth)")
plt.ylabel("Acurácia")
plt.title("Curva de Overfitting - XGBoost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "overfitting_curve.png"))
plt.close()

# Cross-validation
cv_scores = cross_val_score(
    XGBClassifier(random_state=42, n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
    X, y,
    cv=5,
    scoring='accuracy'
)
print(f"Acurácia média (cross-validation 5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_pred_train = model.predict(X_train)
f1_train = f1_score(y_train, y_pred_train)
f1 = f1_score(y_test, y_pred)

with open(os.path.join(OUTPUT_DIR, "hiperparametros.txt"), "w", encoding="utf-8") as f:
    f.write("Modelo: XGBClassifier\n")
    f.write("Hiperparâmetros utilizados:\n")
    f.write(str(model.get_params()) + "\n")
    f.write(f"F1 Score (treino): {f1_train:.4f}\n")
    f.write(f"F1 Score (teste): {f1:.4f}\n")

# Após o treinamento do modelo
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features_importantes = [features[i] for i in indices[:5]]
importancias_top5 = [(features[i], importances[i]) for i in indices[:5]]

with open(os.path.join(OUTPUT_DIR, "hiperparametros.txt"), "a", encoding="utf-8") as f:
    f.write("Top 5 variáveis mais importantes:\n")
    for nome, valor in importancias_top5:
        f.write(f"{nome}: {valor:.4f}\n")
pass