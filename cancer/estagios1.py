import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import os

os.system('cls')
df = pd.read_csv('china_cancer_patients_synthetic.csv')

df['Evento'] = df['SurvivalStatus'].apply(lambda x: 1 if x == 'Deceased' else 0)

kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for stage in sorted(df['CancerStage'].unique()):
    mask = df['CancerStage'] == stage
    kmf.fit(
        durations=df[mask]['FollowUpMonths'],
        event_observed=df[mask]['Evento'],
        label=f'Estágio {stage}'
    )
    kmf.plot_survival_function(ci_show=False) 

plt.title('Taxa de Sobrevivência por Estágio do Câncer')
plt.xlabel('Tempo de Acompanhamento (meses)')
plt.ylabel('Probabilidade de Sobrevivência')
plt.grid(True)
plt.legend()
plt.show()