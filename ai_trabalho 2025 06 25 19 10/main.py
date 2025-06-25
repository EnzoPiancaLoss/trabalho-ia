# -*- coding: utf-8 -*-
"""
Projeto de Inteligência Artificial: Análise de Sobrevivência de Pacientes com Câncer na China
Versão Aprimorada com Análise de TreatmentType
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
from xgboost import XGBClassifier
import time
import datetime
import os

#Ele le esses parametros
#features = [
#     'Age', 'TumorSize', 'Metastasis', 'CancerStageNumeric', 'AdvancedStage',
#     'Received_Radiation', 'Received_Chemotherapy', 'Received_Targeted Therapy', 'Received_Surgery',
#     'Chemo_Intensity', 'Radiation_Intensity', 'Comorbidity_Count'
# ]
# #
#


start_time = time.time()

# Crie a pasta de saída única para cada execução
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = f"output-{now}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hiperparâmetros dos modelos
AUMENTO = [4]

DT_PARAMS = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_leaf': [1, 2, 5]
}

# DT_PARAMS = { #Arvore de decisao
#     'max_depth': [12, 20, 30, 40], #Limita a profundidade da árvore. Valores maiores podem causar overfitting.
#     'min_samples_leaf': [10, 20, 30, 40] #Controlam o tamanho mínimo dos nós/folhas, ajudam a evitar overfitting.
# }


RF_PARAMS = {
    'n_estimators': [200, 400],
    'max_depth': [20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}



# RF_PARAMS = { #Random florest
#     'n_estimators': [200, 400], #Número de árvores (quanto maior, mais lento, mas pode melhorar o resultado).
#     'max_depth': [8, 12], #Limita a profundidade da árvore. Valores maiores podem causar overfitting.
#     'min_samples_split': [5, 10], #Controlam o tamanho mínimo dos nós/folhas, ajudam a evitar overfitting.
#     'min_samples_leaf': [10, 20]
# }

XGB_PARAMS = {
    'n_estimators': [300, 500],
    'max_depth': [5, 8],
    'learning_rate': [0.01],
    'subsample': [0.8]
}



# XGB_PARAMS = { #XGboost
#     'n_estimators': [100, 200],
#     'max_depth': [5, 7], #Limita a profundidade da árvore. Valores maiores podem causar overfitting.
#     'learning_rate': [0.01, 0.1], #Taxa de aprendizado. Valores menores geralmente melhoram o resultado, mas exigem mais árvores.
#     'subsample': [0.8, 1.0] #Fração de amostras usadas em cada árvore (ajuda a regularizar).
# }


# ======================
# 1. CARREGAR OS DADOS
# ======================
print("Carregando dados...")
# Carregar o novo arquivo
df = pd.read_excel('archive/InteligenciaArtificialFiltrado.xlsx')
df.columns = df.columns.str.strip()  # Remove espaços extras dos nomes das colunas

# Filtrar apenas pacientes que fizeram cirurgia
df_cirurgia = df[df['SurgeryDate'].notnull() & (df['SurgeryDate'] != 'N/A')]

# (Opcional) Filtrar apenas estágios III e IV, se desejar:
df_cirurgia = df_cirurgia[df_cirurgia['CancerStage'].isin(['III', 'IV'])]

print(df_cirurgia['SurvivalStatus'].value_counts(normalize=True))

# ======================
# 2. PRÉ-PROCESSAMENTO AVANÇADO
# ======================
print("\nRealizando pré-processamento avançado...")

# Mapeamento de estágios cancerígenos
stage_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df_cirurgia['CancerStageNumeric'] = df_cirurgia['CancerStage'].map(stage_mapping)

# Criar variável binária para estágios avançados
df_cirurgia['AdvancedStage'] = df_cirurgia['CancerStage'].apply(lambda x: 1 if x in ['III', 'IV'] else 0)

# Tratamento de TreatmentType
treatment_types = ['Radiation', 'Chemotherapy', 'Targeted Therapy', 'Surgery']
for t in treatment_types:
    df_cirurgia[f'Received_{t}'] = df_cirurgia['TreatmentType'].apply(lambda x: 1 if t in str(x) else 0)

# Combinar com sessões de tratamento
df_cirurgia['Chemo_Intensity'] = df_cirurgia.apply(lambda row: row['ChemotherapySessions'] if row['Received_Chemotherapy'] else 0, axis=1)
df_cirurgia['Radiation_Intensity'] = df_cirurgia.apply(lambda row: row['RadiationSessions'] if row['Received_Radiation'] else 0, axis=1)

# Variável alvo
df_cirurgia['SurvivalStatus'] = df_cirurgia['SurvivalStatus'].apply(lambda x: 1 if x == 'Deceased' else 0)

# Tratar valores missing
df_cirurgia.fillna({
    'Comorbidities': 'None',
    'GeneticMutation': 'Unknown',
    'AlcoholUse': 'Unknown',
    'SmokingStatus': 'Unknown',
    'TreatmentType': 'None'
}, inplace=True)

# Engenharia de features: Comorbidades
#df_cirurgia['Comorbidity_Count'] = df_cirurgia['Comorbidities'].apply(lambda x: len(str(x).split(',')) if x != 'None' else 0)

# Converter variáveis categóricas para numéricas
if df_cirurgia['Metastasis'].dtype == object:
    df_cirurgia['Metastasis'] = df_cirurgia['Metastasis'].map({'No': 0, 'Yes': 1})

# Converter DiasAteACirurgia para numérico (forçando erros para NaN)
df_cirurgia['DiasAteACirurgia'] = pd.to_numeric(df_cirurgia['DiasAteACirurgia'], errors='coerce')

# Remover linhas com DiasAteACirurgia inválido (NaN)
df_cirurgia = df_cirurgia[df_cirurgia['DiasAteACirurgia'].notna()]

# Criar coluna binária para câncer de pulmão
df_cirurgia['Is_Lung_Cancer'] = df_cirurgia['TumorType'].str.lower().str.contains('pulm', na=False).astype(int)

# Criar coluna binária para já ter fumado (Current ou Former)
df_cirurgia['Has_Smoked'] = df_cirurgia['SmokingStatus'].isin(['Current', 'Former']).astype(int)

# Nova feature: 1 se câncer de pulmão E já fumou, 0 caso contrário
df_cirurgia['LungCancer_and_Smoked'] = (
    df_cirurgia['TumorType'].str.lower().str.contains('pulm', na=False).astype(int) &
    df_cirurgia['SmokingStatus'].isin(['Current', 'Former']).astype(int)
)

# Feature de interação numérica: mais peso para fumantes com câncer de pulmão
df_cirurgia['LungCancer_SmokerNum'] = (
    df_cirurgia['TumorType'].str.lower().str.contains('pulm', na=False).astype(int) *
    df_cirurgia['SmokingStatus'].replace({'Never': 0, 'Former': 2, 'Current': 2})
)

# Converter FollowUpMonths e TumorSize para os tipos corretos
if 'FollowUpMonths' in df_cirurgia.columns:
    df_cirurgia['FollowUpMonths'] = pd.to_numeric(df_cirurgia['FollowUpMonths'], errors='coerce')
else:
    print("Coluna 'FollowUpMonths' não encontrada!")

if 'TumorSize' in df_cirurgia.columns:
    df_cirurgia['TumorSize'] = df_cirurgia['TumorSize'].astype(str).str.replace(',', '.')
    df_cirurgia['TumorSize'] = pd.to_numeric(df_cirurgia['TumorSize'], errors='coerce')
else:
    print("Coluna 'TumorSize' não encontrada!")

# Remover linhas com valores inválidos nessas colunas, se necessário
df_cirurgia = df_cirurgia[df_cirurgia['FollowUpMonths'].notna() & df_cirurgia['TumorSize'].notna()]

# Selecionar features relevantes (incluindo Gender e as novas features)
features = [
    'Gender',
    'Age',
    'TumorType',
    'CancerStage',
    'Metastasis',
    'TreatmentType',
    'ChemotherapySessions',
    'RadiationSessions',
    'SmokingStatus',
    'DiasAteACirurgia',
    'LungCancer_and_Smoked',
    'LungCancer_SmokerNum',
    'FollowUpMonths',    
    'TumorSize'
]

X = df_cirurgia[features].copy()

# Converter variáveis categóricas em dummies
X = pd.get_dummies(X, columns=['Gender', 'TumorType', 'CancerStage', 'Metastasis', 'TreatmentType', 'SmokingStatus'], drop_first=True)

y = df_cirurgia['SurvivalStatus']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ======================
# 3. ANÁLISE EXPLORATÓRIA DETALHADA
# ======================
print("\nRealizando análise exploratória detalhada...")

plt.figure(figsize=(20, 15))

# Gráfico 1: Distribuição de tratamentos
plt.subplot(3, 3, 1)
treatment_counts = df_cirurgia['TreatmentType'].value_counts()
plt.pie(treatment_counts, labels=treatment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribuição de Tipos de Tratamento')

# Gráfico 2: Sobrevivência por tipo de tratamento
plt.subplot(3, 3, 2)
sns.barplot(x='TreatmentType', y='SurvivalStatus', data=df_cirurgia, estimator=np.mean)
plt.title('Taxa de Óbito por Tipo de Tratamento')
plt.ylabel('Taxa de Mortalidade')
plt.xticks(rotation=45)

# Gráfico 3: Combinação de tratamentos
plt.subplot(3, 3, 3)
treatment_comb = df_cirurgia.groupby(['Received_Radiation', 'Received_Chemotherapy', 
                             'Received_Targeted Therapy', 'Received_Surgery']).size().reset_index(name='Count')
treatment_comb['Combination'] = treatment_comb.apply(lambda row: 
    f"R:{row['Received_Radiation']} C:{row['Received_Chemotherapy']} T:{row['Received_Targeted Therapy']} S:{row['Received_Surgery']}", 
    axis=1)
sns.barplot(x='Combination', y='Count', data=treatment_comb)
plt.title('Combinações de Tratamentos')
plt.xlabel('Combinação (R:Radiação, C:Quimio, T:Terapia Alvo, S:Cirurgia)')
plt.xticks(rotation=45)

# Gráfico 4: Intensidade de tratamento vs. Sobrevivência
plt.subplot(3, 3, 4)
sns.boxplot(x='SurvivalStatus', y='Chemo_Intensity', data=df_cirurgia)
plt.title('Intensidade de Quimioterapia por Sobrevivência')
plt.xlabel('Status de Sobrevivência')
plt.xticks([0, 1], ['Vivo', 'Óbito'])

# Gráfico 5: Intensidade de radiação vs. Sobrevivência
plt.subplot(3, 3, 5)
sns.boxplot(x='SurvivalStatus', y='Radiation_Intensity', data=df_cirurgia)
plt.title('Intensidade de Radiação por Sobrevivência')
plt.xlabel('Status de Sobrevivência')
plt.xticks([0, 1], ['Vivo', 'Óbito'])

# Gráfico 6: Tratamentos por estágio do câncer
plt.subplot(3, 3, 6)
treatment_stage = pd.melt(df_cirurgia, id_vars=['CancerStage'], 
                         value_vars=['Received_Radiation', 'Received_Chemotherapy', 
                                    'Received_Targeted Therapy', 'Received_Surgery'],
                         var_name='Treatment', value_name='Received')
treatment_stage['Treatment'] = treatment_stage['Treatment'].str.replace('Received_', '')
sns.barplot(x='CancerStage', y='Received', hue='Treatment', data=treatment_stage, estimator=np.mean)
plt.title('Proporção de Tratamentos por Estágio do Câncer')
plt.ylabel('Proporção de Pacientes')

# Gráfico 7: Eficácia de combinações de tratamento
plt.subplot(3, 3, 7)
df_cirurgia['Treatment_Combo'] = df_cirurgia.apply(lambda row: 
    f"R:{row['Received_Radiation']} C:{row['Received_Chemotherapy']} T:{row['Received_Targeted Therapy']} S:{row['Received_Surgery']}", 
    axis=1)
combo_survival = df_cirurgia.groupby('Treatment_Combo')['SurvivalStatus'].mean().reset_index()
sns.barplot(x='Treatment_Combo', y='SurvivalStatus', data=combo_survival)
plt.title('Eficácia de Combinações de Tratamento')
plt.ylabel('Taxa de Mortalidade')
plt.xlabel('Combinação de Tratamentos')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/advanced_eda_results.png', dpi=300)
plt.close()

# ======================
# 4. MODELAGEM COM TRATAMENTOS
# ======================
print("\nIniciando modelagem com análise de tratamentos...")
model_results = []

def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, pred)
    auc_score = roc_auc_score(y_test, prob) if prob is not None else None
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    
    return {
        'Modelo': model_name,
        'Acurácia': acc,
        'AUC': auc_score,
        'Tempo de Treino': train_time,
        'Matriz Confusão': cm,
        'Classification Report': report
    }, model

# 4.1 Árvore de Decisão (com balanceamento de classes)
print("\nTreinando Árvore de Decisão...")
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid_dt = GridSearchCV(dt, DT_PARAMS, cv=5, scoring='roc_auc')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

dt_results, dt_model = train_evaluate_model(best_dt, "Árvore de Decisão", X_train, y_train, X_test, y_test)
model_results.append(dt_results)

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, classes, title, filename, y_true=None, y_pred=None):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.title(title)
    
    # Calcular métricas se y_true e y_pred forem fornecidos
    if y_true is not None and y_pred is not None:
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        plt.figtext(0.99, 0.01, f"Acurácia: {acc:.2f}\nEspecificidade: {specificity:.2f}\nPrecisão: {precision:.2f}",
                    horizontalalignment='right', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Árvore de Decisão
plot_confusion_matrix(
    dt_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - Árvore de Decisão (Estágio III e IV)',
    filename=f'{OUTPUT_DIR}/confusion_matrix_decision_tree.png',
    y_true=y_test,
    y_pred=best_dt.predict(X_test)
)

# 4.2 Random Forest (com balanceamento de classes)
print("\nTreinando Random Forest...")
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_rf = GridSearchCV(rf, RF_PARAMS, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

rf_results, rf_model = train_evaluate_model(best_rf, "Random Forest", X_train, y_train, X_test, y_test)
model_results.append(rf_results)

# Random Forest
plot_confusion_matrix(
    rf_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - Random Forest (Estágio III e IV)',
    filename=f'{OUTPUT_DIR}/confusion_matrix_random_forest.png',
    y_true=y_test,
    y_pred=best_rf.predict(X_test)
)

# 4.3 XGBoost
print("\nTreinando XGBoost...")
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_xgb = GridSearchCV(xgb, XGB_PARAMS, cv=5, scoring='roc_auc', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

xgb_results, xgb_model = train_evaluate_model(best_xgb, "XGBoost", X_train, y_train, X_test, y_test)
model_results.append(xgb_results)

# XGBoost
plot_confusion_matrix(
    xgb_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - XGBoost (Estágio III e IV)',
    filename=f'{OUTPUT_DIR}/confusion_matrix_xgboost.png',
    y_true=y_test,
    y_pred=best_xgb.predict(X_test)
)

# Converter resultados para DataFrame
results_df = pd.DataFrame(model_results)

# ======================
# 5. ANÁLISE DE EFICÁCIA DE TRATAMENTOS
# ======================
print("\nAnalisando eficácia de tratamentos...")

# 5.1 Análise de eficácia por estágio
treatment_efficacy = []
for stage in [3, 4]:  # Apenas estágios III e IV
    stage_data = df_cirurgia[df_cirurgia['CancerStageNumeric'] == stage]
    for treatment in treatment_types:
        col_name = f'Received_{treatment}'
        if col_name in stage_data.columns:
            treatment_group = stage_data[stage_data[col_name] == 1]
            control_group = stage_data[stage_data[col_name] == 0]
            
            if len(treatment_group) > 10 and len(control_group) > 10:
                mortality_treatment = treatment_group['SurvivalStatus'].mean()
                mortality_control = control_group['SurvivalStatus'].mean()
                efficacy = mortality_control - mortality_treatment  # Redução na mortalidade
                
                treatment_efficacy.append({
                    'Stage': stage,
                    'Treatment': treatment,
                    'Efficacy': efficacy,
                    'Mortality_Treatment': mortality_treatment,
                    'Mortality_Control': mortality_control,
                    'Sample_Size': len(treatment_group)
                })

efficacy_df = pd.DataFrame(treatment_efficacy)

# Plot de eficácia
plt.figure(figsize=(12, 8))
sns.barplot(x='Stage', y='Efficacy', hue='Treatment', data=efficacy_df)
plt.title('Eficácia de Tratamentos por Estágio do Câncer')
plt.ylabel('Redução na Mortalidade')
plt.xlabel('Estágio do Câncer')
plt.legend(title='Tratamento')
plt.savefig(f'{OUTPUT_DIR}/treatment_efficacy_by_stage.png', dpi=300)
plt.close()

# ======================
# 6. RELATÓRIO FINAL
# ======================
print("\nGerando relatório final...")

# Salvar resultados
results_df.to_csv(f'{OUTPUT_DIR}/treatment_analysis_results.csv', index=False)
efficacy_df.to_csv(f'{OUTPUT_DIR}/treatment_efficacy_analysis.csv', index=False)

# Imprimir insights
print("\n" + "="*50)
print("PRINCIPAIS DESCOBERTAS SOBRE TRATAMENTOS")
print("="*50)

# 1. Distribuição de tratamentos
print(f"\n[1. DISTRIBUIÇÃO DE TRATAMENTOS]")
for t in treatment_types:
    perc = df_cirurgia[f'Received_{t}'].mean() * 100
    print(f"- {t}: {perc:.1f}% dos pacientes")

# 2. Combinações mais comuns
top_combos = df_cirurgia['Treatment_Combo'].value_counts().head(3)
print("\n[2. COMBINAÇÕES MAIS COMUNS]")
for combo, count in top_combos.items():
    print(f"- {combo}: {count} pacientes ({count/len(df_cirurgia)*100:.1f}%)")

# 3. Eficácia dos tratamentos
print("\n[3. EFICÁCIA DOS TRATAMENTOS]")
for _, row in efficacy_df.iterrows():
    if row['Efficacy'] > 0:
        print(f"- Estágio {row['Stage']} | {row['Treatment']}: "
              f"Redução de {row['Efficacy']*100:.1f}% na mortalidade "
              f"(Controle: {row['Mortality_Control']*100:.1f}% → Tratamento: {row['Mortality_Treatment']*100:.1f}%)")

# 4. Insights do modelo
best_model = results_df.loc[results_df['AUC'].idxmax()]
print(f"\n[4. MELHOR MODELO: {best_model['Modelo']} (AUC = {best_model['AUC']:.3f})]")
print("Variáveis mais importantes:")

# Seleciona o modelo treinado correspondente ao melhor
if best_model['Modelo'] == "XGBoost":
    model = best_xgb
elif best_model['Modelo'] == "Random Forest":
    model = best_rf
elif best_model['Modelo'] == "Árvore de Decisão":
    model = best_dt
else:
    model = None

if model is not None and hasattr(model, 'feature_importances_'):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(5)
    for feat, imp in top_features.items():
        print(f"- {feat}: {imp:.3f}")

print("\n" + "="*50)
print("ANÁLISE CONCLUÍDA! RESULTADOS SALVOS EM ARQUIVOS.")

# Mostrar tempo total de processamento
total_time = time.time() - start_time
print(f"\nTempo total de processamento: {total_time:.2f} segundos")

def plot_regression_metrics(y_true, y_pred, filename):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - sse / sst if sst != 0 else 0

    plt.figure(figsize=(6, 4))
    plt.axis('off')
    plt.title("Métricas de Regressão")
    text = f"SSE: {sse:.2f}\nSST: {sst:.2f}\nR²: {r2:.3f}"
    plt.text(0.5, 0.5, text, fontsize=14, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(filename, dpi=300)
    plt.close()

# Exemplo de uso (substitua pelos seus dados de regressão):
# plot_regression_metrics(y_reg, y_pred_reg, 'output/regression_metrics.png')

# Salvar tempo de processamento e hiperparâmetros em um arquivo .txt
with open(f'{OUTPUT_DIR}/relatorio_execucao.txt', 'w', encoding='utf-8') as f:
    f.write("Tempo total de processamento: {:.2f} segundos\n\n".format(total_time))
    f.write("Hiperparâmetros utilizados:\n")
    f.write("Árvore de Decisão (DT_PARAMS):\n")
    for k, v in DT_PARAMS.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nRandom Forest (RF_PARAMS):\n")
    for k, v in RF_PARAMS.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nXGBoost (XGB_PARAMS):\n")
    for k, v in XGB_PARAMS.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n" + "="*50 + "\n")
    f.write("PRINCIPAIS DESCOBERTAS SOBRE TRATAMENTOS\n")
    f.write("="*50 + "\n\n")
    # Inclua as descobertas impressas no terminal:
    f.write("[1. DISTRIBUIÇÃO DE TRATAMENTOS]\n")
    for t in treatment_types:
        perc = df_cirurgia[f'Received_{t}'].mean() * 100
        f.write(f"- {t}: {perc:.1f}% dos pacientes\n")
    f.write("\n[2. COMBINAÇÕES MAIS COMUNS]\n")
    top_combos = df_cirurgia['Treatment_Combo'].value_counts().head(3)
    for combo, count in top_combos.items():
        f.write(f"- {combo}: {count} pacientes ({count/len(df_cirurgia)*100:.1f}%)\n")
    f.write("\n[3. EFICÁCIA DOS TRATAMENTOS]\n")
    for _, row in efficacy_df.iterrows():
        if row['Efficacy'] > 0:
            f.write(f"- Estágio {int(row['Stage'])} | {row['Treatment']}: "
                    f"Redução de {row['Efficacy']*100:.1f}% na mortalidade "
                    f"(Controle: {row['Mortality_Control']*100:.1f}% → Tratamento: {row['Mortality_Treatment']*100:.1f}%)\n")
    f.write("\n[4. MELHOR MODELO]\n")
    f.write(f"{best_model['Modelo']} (AUC = {best_model['AUC']:.3f})\n")
    f.write("Variáveis mais importantes:\n")
    if model is not None and hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importance.sort_values(ascending=False).head(5)
        for feat, imp in top_features.items():
            f.write(f"- {feat}: {imp:.3f}\n")

plt.figure(figsize=(10,6))
sns.boxplot(x='SurvivalStatus', y='DiasAteACirurgia', data=df_cirurgia)
plt.title('Dias até a Cirurgia por Status de Sobrevivência')
plt.xlabel('Status de Sobrevivência')
plt.ylabel('Dias até a Cirurgia')
#plt.show()

print(df_cirurgia.groupby('SurvivalStatus')['DiasAteACirurgia'].mean())

# Supondo que seu modelo se chama 'dt' (DecisionTreeClassifier ou DecisionTreeRegressor)
plt.figure(figsize=(20, 8))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=[str(c) for c in best_dt.classes_] if hasattr(best_dt, "classes_") else None,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Árvore de Decisão - Previsão de Sobrevivência")

# Adiciona legenda para as classes
plt.figtext(
    0.99, 0.01,
    "Legenda: class = 0 (Vivo), class = 1 (Óbito)\n samples = total de pacientes",
    horizontalalignment='right',
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/arvore_decisao.png", dpi=200)
plt.close()

from sklearn.metrics import accuracy_score

def plot_overfit_curve(X_train, X_test, y_train, y_test, output_path):
    depths = range(1, 21)
    train_scores = []
    test_scores = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_scores, label='Acurácia Treino', marker='o')
    plt.plot(depths, test_scores, label='Acurácia Teste', marker='s')
    plt.xlabel('Profundidade da Árvore (max_depth)')
    plt.ylabel('Acurácia')
    plt.title('Curva de Overfitting - Decision Tree')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

# Exemplo de uso:
plot_overfit_curve(X_train, X_test, y_train, y_test, f"{OUTPUT_DIR}/overfit_curve.png")

print(df_cirurgia.columns)

