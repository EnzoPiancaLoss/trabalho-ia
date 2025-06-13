#pip install pandas
#pip install pandas lifelines matplotlib

import pandas as pd
import os

from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt


os.system('cls')
df = pd.read_csv('china_cancer_patients_synthetic.csv')



df['Evento'] = df['SurvivalStatus'].apply(lambda x: 1 if x == 'Deceased' else 0)


print(df[['SurvivalStatus', 'Evento', 'FollowUpMonths']].head())

kmf = KaplanMeierFitter()


kmf.fit(durations=df['FollowUpMonths'], event_observed=df['Evento'])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Curva de Sobrevivência (Kaplan-Meier)')
plt.xlabel('Tempo (meses)')
plt.ylabel('Taxa de Sobrevivência')
plt.grid(True)
plt.show()




plt.figure(figsize=(10, 6))
for tumor_type in df['TumorType'].unique():
    mask = df['TumorType'] == tumor_type
    kmf.fit(durations=df[mask]['FollowUpMonths'], event_observed=df[mask]['Evento'])
    kmf.plot_survival_function(label=tumor_type)

plt.title('Sobrevivência por Tipo de Tumor')
plt.xlabel('Tempo (meses)')
plt.ylabel('Taxa de Sobrevivência')
plt.legend()
plt.grid(True)
plt.show()


# Visualizar as primeiras linhas do DataFrame
#print(df.head())

# Visualizar informações sobre o DataFrame (colunas, tipos de dados, etc.)
#print(df.info())

# Visualizar estatísticas descritivas
#print(df.describe())

