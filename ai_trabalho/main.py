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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, auc)
from xgboost import XGBClassifier
import time
import os

os.makedirs('output', exist_ok=True)

# ======================
# 1. CARREGAR OS DADOS
# ======================
print("Carregando dados...")
df = pd.read_excel('archive/output.xlsx', sheet_name='Sheet1')

# Filtrar apenas estágios III e IV
df = df[df['CancerStage'].isin(['III', 'IV'])]

# ======================
# 2. PRÉ-PROCESSAMENTO AVANÇADO
# ======================
print("\nRealizando pré-processamento avançado...")

# Mapeamento de estágios cancerígenos
stage_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
df['CancerStageNumeric'] = df['CancerStage'].map(stage_mapping)

# Criar variável binária para estágios avançados
df['AdvancedStage'] = df['CancerStage'].apply(lambda x: 1 if x in ['III', 'IV'] else 0)

# Tratamento de TreatmentType
treatment_types = ['Radiation', 'Chemotherapy', 'Targeted Therapy', 'Surgery']
for t in treatment_types:
    df[f'Received_{t}'] = df['TreatmentType'].apply(lambda x: 1 if t in str(x) else 0)

# Combinar com sessões de tratamento
df['Chemo_Intensity'] = df.apply(lambda row: row['ChemotherapySessions'] if row['Received_Chemotherapy'] else 0, axis=1)
df['Radiation_Intensity'] = df.apply(lambda row: row['RadiationSessions'] if row['Received_Radiation'] else 0, axis=1)

# Variável alvo
df['SurvivalStatus'] = df['SurvivalStatus'].apply(lambda x: 1 if x == 'Deceased' else 0)

# Tratar valores missing
df.fillna({
    'Comorbidities': 'None',
    'GeneticMutation': 'Unknown',
    'AlcoholUse': 'Unknown',
    'SmokingStatus': 'Unknown',
    'TreatmentType': 'None'
}, inplace=True)

# Engenharia de features: Comorbidades
df['Comorbidity_Count'] = df['Comorbidities'].apply(lambda x: len(str(x).split(',')) if x != 'None' else 0)

# Converter variáveis categóricas para numéricas
if df['Metastasis'].dtype == object:
    df['Metastasis'] = df['Metastasis'].map({'No': 0, 'Yes': 1})

# Selecionar features relevantes
features = [
    'Age', 'TumorSize', 'Metastasis', 'CancerStageNumeric', 'AdvancedStage',
    'Received_Radiation', 'Received_Chemotherapy', 'Received_Targeted Therapy', 'Received_Surgery',
    'Chemo_Intensity', 'Radiation_Intensity', 'Comorbidity_Count'
]
X = df[features]
y = df['SurvivalStatus']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ======================
# 3. ANÁLISE EXPLORATÓRIA DETALHADA
# ======================
print("\nRealizando análise exploratória detalhada...")

plt.figure(figsize=(20, 15))

# Gráfico 1: Distribuição de tratamentos
plt.subplot(3, 3, 1)
treatment_counts = df['TreatmentType'].value_counts()
plt.pie(treatment_counts, labels=treatment_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribuição de Tipos de Tratamento')

# Gráfico 2: Sobrevivência por tipo de tratamento
plt.subplot(3, 3, 2)
sns.barplot(x='TreatmentType', y='SurvivalStatus', data=df, estimator=np.mean)
plt.title('Taxa de Óbito por Tipo de Tratamento')
plt.ylabel('Taxa de Mortalidade')
plt.xticks(rotation=45)

# Gráfico 3: Combinação de tratamentos
plt.subplot(3, 3, 3)
treatment_comb = df.groupby(['Received_Radiation', 'Received_Chemotherapy', 
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
sns.boxplot(x='SurvivalStatus', y='Chemo_Intensity', data=df)
plt.title('Intensidade de Quimioterapia por Sobrevivência')
plt.xlabel('Status de Sobrevivência')
plt.xticks([0, 1], ['Vivo', 'Óbito'])

# Gráfico 5: Intensidade de radiação vs. Sobrevivência
plt.subplot(3, 3, 5)
sns.boxplot(x='SurvivalStatus', y='Radiation_Intensity', data=df)
plt.title('Intensidade de Radiação por Sobrevivência')
plt.xlabel('Status de Sobrevivência')
plt.xticks([0, 1], ['Vivo', 'Óbito'])

# Gráfico 6: Tratamentos por estágio do câncer
plt.subplot(3, 3, 6)
treatment_stage = pd.melt(df, id_vars=['CancerStage'], 
                         value_vars=['Received_Radiation', 'Received_Chemotherapy', 
                                    'Received_Targeted Therapy', 'Received_Surgery'],
                         var_name='Treatment', value_name='Received')
treatment_stage['Treatment'] = treatment_stage['Treatment'].str.replace('Received_', '')
sns.barplot(x='CancerStage', y='Received', hue='Treatment', data=treatment_stage, estimator=np.mean)
plt.title('Proporção de Tratamentos por Estágio do Câncer')
plt.ylabel('Proporção de Pacientes')

# Gráfico 7: Eficácia de combinações de tratamento
plt.subplot(3, 3, 7)
df['Treatment_Combo'] = df.apply(lambda row: 
    f"R:{row['Received_Radiation']} C:{row['Received_Chemotherapy']} T:{row['Received_Targeted Therapy']} S:{row['Received_Surgery']}", 
    axis=1)
combo_survival = df.groupby('Treatment_Combo')['SurvivalStatus'].mean().reset_index()
sns.barplot(x='Treatment_Combo', y='SurvivalStatus', data=combo_survival)
plt.title('Eficácia de Combinações de Tratamento')
plt.ylabel('Taxa de Mortalidade')
plt.xlabel('Combinação de Tratamentos')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('output/advanced_eda_results.png', dpi=300)
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

# 4.1 Árvore de Decisão
print("\nTreinando Árvore de Decisão...")
dt_params = {
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20, 30]
}
dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, dt_params, cv=5, scoring='roc_auc')
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_

dt_results, dt_model = train_evaluate_model(best_dt, "Árvore de Decisão", X_train, y_train, X_test, y_test)
model_results.append(dt_results)

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Árvore de Decisão
plot_confusion_matrix(
    dt_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - Árvore de Decisão',
    filename='output/confusion_matrix_decision_tree.png'
)

# 4.2 Random Forest
print("\nTreinando Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

rf_results, rf_model = train_evaluate_model(best_rf, "Random Forest", X_train, y_train, X_test, y_test)
model_results.append(rf_results)

# Random Forest
plot_confusion_matrix(
    rf_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - Random Forest',
    filename='output/confusion_matrix_random_forest.png'
)

# 4.3 XGBoost
print("\nTreinando XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
grid_xgb = GridSearchCV(xgb, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

xgb_results, xgb_model = train_evaluate_model(best_xgb, "XGBoost", X_train, y_train, X_test, y_test)
model_results.append(xgb_results)

# XGBoost
plot_confusion_matrix(
    xgb_results['Matriz Confusão'],
    classes=['Vivo', 'Óbito'],
    title='Matriz de Confusão - XGBoost',
    filename='output/confusion_matrix_xgboost.png'
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
    stage_data = df[df['CancerStageNumeric'] == stage]
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
plt.savefig('output/treatment_efficacy_by_stage.png', dpi=300)
plt.close()

# ======================
# 6. RELATÓRIO FINAL
# ======================
print("\nGerando relatório final...")

# Salvar resultados
results_df.to_csv('output/treatment_analysis_results.csv', index=False)
efficacy_df.to_csv('output/treatment_efficacy_analysis.csv', index=False)

# Imprimir insights
print("\n" + "="*50)
print("PRINCIPAIS DESCOBERTAS SOBRE TRATAMENTOS")
print("="*50)

# 1. Distribuição de tratamentos
print(f"\n[1. DISTRIBUIÇÃO DE TRATAMENTOS]")
for t in treatment_types:
    perc = df[f'Received_{t}'].mean() * 100
    print(f"- {t}: {perc:.1f}% dos pacientes")

# 2. Combinações mais comuns
top_combos = df['Treatment_Combo'].value_counts().head(3)
print("\n[2. COMBINAÇÕES MAIS COMUNS]")
for combo, count in top_combos.items():
    print(f"- {combo}: {count} pacientes ({count/len(df)*100:.1f}%)")

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
if best_model['Modelo'] == "XGBoost" and 'feature_importances_' in dir(xgb_model):
    importance = pd.Series(xgb_model.feature_importances_, index=features)
    top_features = importance.sort_values(ascending=False).head(5)
    for feat, imp in top_features.items():
        print(f"- {feat}: {imp:.3f}")
elif best_model['Modelo'] == "Random Forest" and 'feature_importances_' in dir(rf_model):
    importance = pd.Series(rf_model.feature_importances_, index=features)
    top_features = importance.sort_values(ascending=False).head(5)
    for feat, imp in top_features.items():
        print(f"- {feat}: {imp:.3f}")
elif best_model['Modelo'] == "Árvore de Decisão" and 'feature_importances_' in dir(dt_model):
    importance = pd.Series(dt_model.feature_importances_, index=features)
    top_features = importance.sort_values(ascending=False).head(5)
    for feat, imp in top_features.items():
        print(f"- {feat}: {imp:.3f}")

print("\n" + "="*50)
print("ANÁLISE CONCLUÍDA! RESULTADOS SALVOS EM ARQUIVOS.")