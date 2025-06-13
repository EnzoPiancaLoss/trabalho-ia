import os

import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

os.system("cls")

# Carregar os dados
df = pd.read_csv('china_cancer_patients_synthetic.csv')

# Filtrar apenas estágios III e IV
df_filtered = df[df['CancerStage'].isin(['III', 'IV'])]

# Converter 'SurvivalStatus' para binário (1 = Óbito, 0 = Censurado/Vivo)
df_filtered['Evento'] = df_filtered['SurvivalStatus'].apply(lambda x: 1 if x == 'Deceased' else 0)

# Inicializar o modelo Kaplan-Meier
kmf = KaplanMeierFitter()

# Plotar curva para cada província
plt.figure(figsize=(12, 7))

for province in df_filtered['Province'].unique():
    mask = df_filtered['Province'] == province
    kmf.fit(
        durations=df_filtered[mask]['FollowUpMonths'],
        event_observed=df_filtered[mask]['Evento'],
        label=province
    )
    kmf.plot_survival_function(ci_show=False)  # Remover intervalo de confiança para clareza

plt.title('Taxa de Sobrevivência por Província (Estágios III e IV)')
plt.xlabel('Tempo de Acompanhamento (meses)')
plt.ylabel('Probabilidade de Sobrevivência')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legenda fora do gráfico
plt.tight_layout()  # Ajustar layout para evitar cortes
plt.show()