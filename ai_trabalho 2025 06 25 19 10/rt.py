import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PATH_DATABASE = "archive/china_cancer_patients_synthetic.csv"

def mortalidade_geral():
# Carregar os dados
    df = pd.read_csv('archive/china_cancer_patients_synthetic.csv')

    # Contar o número de pacientes vivos e mortos
    survival_counts = df['SurvivalStatus'].value_counts()

    # Criar o gráfico de torta
    plt.figure(figsize=(8, 6))
    plt.pie(survival_counts, 
            labels=survival_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999'],
            explode=(0.1, 0))  # Explode a primeira fatia

    # Adicionar título
    plt.title('Distribuição de Pacientes por Status de Sobrevivência', pad=20)

    # Mostrar o gráfico
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.show()



def mortalidade_3_4():
    # Carregar os dados
    df = pd.read_csv('archive/china_cancer_patients_synthetic.csv')

    # Filtrar apenas os estágios III e IV
    filtered_df = df[df['CancerStage'].str.contains('III|IV', na=False)]

    # Contar o número de pacientes vivos e mortos nos estágios filtrados
    survival_counts = filtered_df['SurvivalStatus'].value_counts()

    # Criar o gráfico de torta
    plt.figure(figsize=(8, 6))
    plt.pie(survival_counts, 
            labels=survival_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            colors=['#66b3ff', '#ff9999'],
            explode=(0.1, 0))  # Explode a primeira fatia

    # Adicionar título
    plt.title('Distribuição de Pacientes por Status de Sobrevivência (Estágios III e IV)', pad=20)

    # Mostrar o gráfico
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.show()

def plot_mortalidade(estagios=None, separar_estagios=False, separar_generos=True):
    """
    Gera gráficos de mortalidade com opções de filtro por estágio e separação por gênero.
    
    Parâmetros:
    -----------
    estagios : list, optional
        Lista de estágios a incluir (ex: ["III", "IV"]). Se None, inclui todos.
    separar_estagios : bool, optional
        Se True, cria subplots separados para cada estágio.
    separar_generos : bool, optional
        Se True, separa os dados por gênero (Male, Female, Other).
    """
    # Carregar os dados
    df = pd.read_csv('archive/china_cancer_patients_synthetic.csv')
    
    # Verificar e padronizar os valores de SurvivalStatus
    if 'Deceased' in df['SurvivalStatus'].values:
        df['SurvivalStatus'] = df['SurvivalStatus'].replace({'Deceased': 'Dead'})
    
    # Definir cores
    cores_status = {'Alive': '#66b3ff', 'Dead': '#ff9999'}
    cores_genero = {'Male': '#1f77b4', 'Female': '#ff7f0e', 'Other': '#2ca02c'}
    
    # Filtrar estágios se especificado
    if estagios is not None:
        regex_filter = "|".join(estagios)
        filtered_df = df[df['CancerStage'].str.contains(regex_filter, na=False)]
        titulo_estagios = f" (Estágios {', '.join(estagios)})"
    else:
        filtered_df = df
        titulo_estagios = ""
        estagios = ["I", "II", "III", "IV"] if separar_estagios else []
    
    if separar_estagios:
        # Criar subplots para cada estágio
        fig, axs = plt.subplots(1, len(estagios), figsize=(5*len(estagios), 6))
        if len(estagios) == 1:
            axs = [axs]
        
        for i, estagio in enumerate(estagios):
            df_estagio = filtered_df[filtered_df['CancerStage'].str.contains(estagio, na=False)]
            
            if separar_generos:
                # Gráfico de barras agrupadas por gênero
                cross_tab = pd.crosstab(df_estagio['Gender'], df_estagio['SurvivalStatus'])
                # Garantir que temos todas as categorias de gênero
                cross_tab = cross_tab.reindex(['Male', 'Female', 'Other'], fill_value=0)
                # Garantir que temos ambas as categorias de status
                for status in ['Alive', 'Dead']:
                    if status not in cross_tab.columns:
                        cross_tab[status] = 0
                
                cross_tab.plot(kind='bar', ax=axs[i], color=[cores_status[s] for s in cross_tab.columns],
                              edgecolor='black', width=0.8)
                
                # Adicionar valores nas barras
                for p in axs[i].containers:
                    axs[i].bar_label(p, label_type='edge', padding=3)
                
                axs[i].set_title(f'Estágio {estagio}')
                axs[i].set_ylabel('Número de Pacientes')
                axs[i].legend(title='Status')
            else:
                # Gráfico de pizza simples para o estágio
                counts = df_estagio['SurvivalStatus'].value_counts()
                axs[i].pie(counts, labels=counts.index, autopct='%1.1f%%',
                          colors=[cores_status[s] for s in counts.index],
                          startangle=90, explode=(0.1, 0))
                axs[i].set_title(f'Estágio {estagio}')
                axs[i].axis('equal')
        
        fig.suptitle(f'Distribuição de Sobrevivência por Estágio{titulo_estagios}', y=1.05)
    
    else:
        # Gráfico único
        plt.figure(figsize=(10, 6))
        
        if separar_generos:
            # Gráfico de barras agrupadas
            cross_tab = pd.crosstab(filtered_df['Gender'], filtered_df['SurvivalStatus'])
            # Garantir todas as categorias
            cross_tab = cross_tab.reindex(['Male', 'Female', 'Other'], fill_value=0)
            for status in ['Alive', 'Dead']:
                if status not in cross_tab.columns:
                    cross_tab[status] = 0
            
            ax = cross_tab.plot(kind='bar', color=[cores_status[s] for s in cross_tab.columns],
                              edgecolor='black', width=0.8)
            
            # Adicionar valores nas barras
            for p in ax.containers:
                ax.bar_label(p, label_type='edge', padding=3)
            
            plt.ylabel('Número de Pacientes')
            plt.legend(title='Status')
        else:
            # Gráfico de pizza simples
            counts = filtered_df['SurvivalStatus'].value_counts()
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
                   colors=[cores_status[s] for s in counts.index],
                   startangle=90, explode=(0.1, 0))
            plt.axis('equal')
        
        plt.title(f'Distribuição de Sobrevivência{titulo_estagios}')
    
    plt.tight_layout()
    plt.show()
# Exemplos de uso:



def genero(estagios=None, separar_estagios=False):
    """
    Gera um gráfico de distribuição de pacientes por gênero, com opções de filtro por estágio e separação por colunas.
    
    Parâmetros:
    -----------
    estagios : list, opcional
        Lista de estágios a serem filtrados (ex: ["I", "III"]). 
        Se None, inclui todos os estágios (I, II, III, IV).
    separar_estagios : bool, opcional
        Se True, cria colunas separadas para cada estágio no gráfico.
    """
    # Carregar os dados
    df = pd.read_csv('archive/china_cancer_patients_synthetic.csv')
    
    # Filtrar estágios específicos (se fornecido)
    if estagios is not None:
        regex_filter = "|".join(estagios)
        filtered_df = df[df['CancerStage'].str.contains(regex_filter, na=False)]
        titulo_estagios = f" (Estágios {', '.join(estagios)})"
    else:
        filtered_df = df
        estagios = ["I", "II", "III", "IV"]  # Todos os estágios se nenhum for especificado
        titulo_estagios = ""
    
    # Definir cores para cada gênero
    colors = {'Male': '#66b3ff', 'Female': '#ff9999', 'Other': '#99ff99'}
    
    if separar_estagios:
        # Criar DataFrame com contagem de gêneros por estágio
        grouped = filtered_df.groupby(['CancerStage', 'Gender']).size().unstack(fill_value=0)
        
        # Ordenar os estágios conforme a lista fornecida (ou ordem padrão)
        grouped = grouped.loc[sorted(grouped.index, key=lambda x: estagios.index(x)) if estagios else grouped.index]
        
        # Plotar gráfico de barras agrupadas
        plt.figure(figsize=(10, 6))
        ax = grouped.plot(kind='bar', color=[colors[g] for g in grouped.columns], edgecolor='black', width=0.8)
        
        # Adicionar valores em cima das barras
        for p in ax.patches:
            ax.annotate(str(p.get_height()), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
        
        plt.xlabel('Estágio do Câncer', fontsize=12)
        plt.ylabel('Número de Pacientes', fontsize=12)
        plt.title(f'Distribuição de Gêneros por Estágio{titulo_estagios}', pad=20, fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(title='Gênero')
        
    else:
        # Gráfico simples (todos os estágios juntos)
        gender_counts = filtered_df['Gender'].value_counts()
        
        plt.figure(figsize=(8, 6))
        gender_counts.plot(kind='bar', color=[colors[g] for g in gender_counts.index], edgecolor='black')
        
        plt.xlabel('Gênero', fontsize=12)
        plt.ylabel('Número de Pacientes', fontsize=12)
        plt.title(f'Distribuição de Pacientes por Gênero{titulo_estagios}', pad=20, fontsize=14)
        
        for i, count in enumerate(gender_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def ademar():
    # Carregar os dados
    df = pd.read_csv('archive/china_cancer_patients_synthetic.csv')

    # Verificar e padronizar os valores de SurvivalStatus
    if 'Deceased' in df['SurvivalStatus'].values:
        df['SurvivalStatus'] = df['SurvivalStatus'].replace({'Deceased': 'Dead'})

    # Converter SurvivalStatus para valores numéricos (0 para Dead, 1 para Alive)
    df['SurvivalStatus_num'] = df['SurvivalStatus'].map({'Dead': 0, 'Alive': 1})

    # Adicionar jitter para melhor visualização
    np.random.seed(42)  # Para reprodutibilidade
    jitter = np.random.normal(0, 0.05, size=len(df))
    df['SurvivalStatus_jitter'] = df['SurvivalStatus_num'] + jitter

    # Criar o gráfico de dispersão
    plt.figure(figsize=(10, 6))

    # Plotar pontos para cada status
    for status, color in [('Dead', 'red'), ('Alive', 'green')]:
        subset = df[df['SurvivalStatus'] == status]
        plt.scatter(
            subset['ChemotherapySessions'], 
            subset['SurvivalStatus_jitter'], 
            alpha=0.6,
            color=color,
            label=status
        )

    # Ajustar eixos e labels
    plt.yticks([0, 1], ['Dead', 'Alive'])
    plt.xlabel('Número de Sessões de Quimioterapia', fontsize=12)
    plt.ylabel('Status de Sobrevivência', fontsize=12)
    plt.title('Relação entre Quimioterapia e Sobrevivência', fontsize=14, pad=20)

    # Adicionar linha de tendência (opcional)
    sns.regplot(
        x='ChemotherapySessions', 
        y='SurvivalStatus_num', 
        data=df, 
        logistic=True,
        scatter=False,
        color='blue',
        line_kws={'label': 'Tendência'}
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def quimioterapia1(cancer_stage=None, arquivo_dados=PATH_DATABASE):
    """
    Gera um gráfico de regressão logística a partir de um arquivo de dados.
    - Assume que 'SurvivalStatus' contém strings como "Alive" e "Deceased".
    - Converte automaticamente para 0 (Deceased) e 1 (Alive).
    """
    # Configuração do estilo
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1️⃣ Carregar dados do arquivo
    try:
        if arquivo_dados.endswith('.csv'):
            df = pd.read_csv(arquivo_dados)
        elif arquivo_dados.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(arquivo_dados)
        else:
            raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")
        
        # Verificar colunas necessárias
        required_cols = ['ChemotherapySessions', 'CancerStage', 'SurvivalStatus']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O arquivo deve ter as colunas: {required_cols}.")

        # 2️⃣ Converter SurvivalStatus para numérico (Alive=1, Deceased=0)
        status_map = {'Alive': 1, 'Deceased': 0}
        df['SurvivalStatus'] = df['SurvivalStatus'].map(status_map).astype(int)

        # Extrair dados
        chemotherapy = df['ChemotherapySessions'].values
        cancer_stages = df['CancerStage'].values
        survival_status = df['SurvivalStatus'].values  # Agora é 0 ou 1

    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        return

    # 3️⃣ Filtrar por estágio (se fornecido)
    if cancer_stage is not None:
        if isinstance(cancer_stage, list):
            mask = np.isin(cancer_stages, cancer_stage)
            title_stage = f' (Estágios {", ".join(cancer_stage)})'
        else:
            mask = (cancer_stages == cancer_stage)
            title_stage = f' (Estágio {cancer_stage})'
        
        chemotherapy = chemotherapy[mask]
        survival_status = survival_status[mask]
    else:
        title_stage = ' (Todos os Estágios)'

    # 4️⃣ Criar gráfico
    fig, ax = plt.subplots(figsize=(8, 6))

    # Gráfico de dispersão
    ax.scatter(
        chemotherapy, 
        survival_status, 
        alpha=0.6,
        color='#1f77b4',
        edgecolor='w',
        linewidth=0.5,
        label='Dados Reais'
    )

    # Linha de regressão logística (só funciona se houver variabilidade nos dados)
    if len(np.unique(survival_status)) >= 2:  # Pelo menos 0s e 1s
        sns.regplot(
            x=chemotherapy,
            y=survival_status,
            logistic=True,
            scatter=False,
            color='#d62728',
            line_kws={'linewidth': 2, 'label': 'Regressão Logística'},
            ax=ax
        )
    else:
        print("Aviso: Dados insuficientes para regressão (todos os status são iguais).")

    # Ajustes do gráfico
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Deceased', 'Alive'])  # Rótulos originais
    ax.set_xlabel('Número de Sessões de Quimioterapia', fontsize=12)
    ax.set_ylabel('Status de Sobrevivência', fontsize=12)
    ax.set_title(f'Efeito da Quimioterapia na Sobrevivência{title_stage}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_quimio_barras(cancer_stage=None, arquivo_dados=PATH_DATABASE):
    """
    Gera um gráfico de barras agrupadas mostrando a distribuição de sessões de quimioterapia
    para pacientes vivos e mortos.
    
    Parâmetros:
    -----------
    cancer_stage : str ou list, opcional
        Filtra por estágio do câncer (ex: "III" ou ["III", "IV"]). Se None, usa todos.
    arquivo_dados : str, opcional
        Caminho do arquivo de dados (CSV, Excel, etc.). Padrão: 'dados_quimioterapia.csv'.
    """
    # Configuração do estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Carregar dados
        if arquivo_dados.endswith('.csv'):
            df = pd.read_csv(arquivo_dados)
        elif arquivo_dados.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(arquivo_dados)
        else:
            raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")
        
        # Verificar colunas necessárias
        required_cols = ['ChemotherapySessions', 'CancerStage', 'SurvivalStatus']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O arquivo deve ter as colunas: {required_cols}.")
        
        # Mapear SurvivalStatus para numérico
        status_map = {'Alive': 'Alive', 'Deceased': 'Deceased'}
        df['Status'] = df['SurvivalStatus'].map(status_map)
        
        # Filtrar por estágio
        if cancer_stage is not None:
            if isinstance(cancer_stage, list):
                df = df[df['CancerStage'].isin(cancer_stage)]
                title_stage = f' (Estágios {", ".join(cancer_stage)})'
            else:
                df = df[df['CancerStage'] == cancer_stage]
                title_stage = f' (Estágio {cancer_stage})'
        else:
            title_stage = ' (Todos os Estágios)'
        
        # Verificar se há dados suficientes
        if df.empty:
            print("Nenhum dado disponível após o filtro.")
            return
        
        # Criar grupos de sessões (caso haja muitos valores únicos)
        if df['ChemotherapySessions'].nunique() > 10:
            df['SessionGroup'] = pd.cut(df['ChemotherapySessions'], 
                                       bins=5, 
                                       precision=0,
                                       include_lowest=True)
            group_col = 'SessionGroup'
        else:
            group_col = 'ChemotherapySessions'
        
        # Contar ocorrências
        contagem = df.groupby([group_col, 'Status']).size().unstack().fillna(0)
        
        # Ordenar por número de sessões
        if group_col == 'ChemotherapySessions':
            contagem = contagem.sort_index()
        
        # Plotar gráfico de barras
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Posições das barras
        bar_width = 0.35
        indices = np.arange(len(contagem))
        
        # Barras para Alive
        alive_bars = ax.bar(indices - bar_width/2, contagem['Alive'], 
                           width=bar_width, color='#2ca02c', label='Alive')
        
        # Barras para Deceased
        deceased_bars = ax.bar(indices + bar_width/2, contagem['Deceased'], 
                              width=bar_width, color='#d62728', label='Deceased')
        
        # Configurações do gráfico
        ax.set_title(f'Distribuição de Sessões de Quimioterapia por Sobrevivência{title_stage}', fontsize=16)
        ax.set_xlabel('Sessões de Quimioterapia', fontsize=12)
        ax.set_ylabel('Número de Pacientes', fontsize=12)
        
        # Rótulos do eixo X
        ax.set_xticks(indices)
        ax.set_xticklabels([str(x) for x in contagem.index], rotation=45)
        
        # Adicionar valores nas barras
        for bars in [alive_bars, deceased_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # Deslocamento vertical
                                textcoords="offset points",
                                ha='center', va='bottom')
        
        # Grade e legenda
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Status')
        
        # Melhorar layout
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erro ao processar os dados: {e}")



def plot_quimio_barras_detalhado(cancer_stage=None, arquivo_dados=PATH_DATABASE):
    """
    Gera um gráfico de barras agrupadas mostrando a distribuição detalhada de sessões de quimioterapia
    para pacientes vivos e mortos, com barras separadas para cada número específico de sessões.
    
    Parâmetros:
    -----------
    cancer_stage : str ou list, opcional
        Filtra por estágio do câncer (ex: "III" ou ["III", "IV"]). Se None, usa todos.
    arquivo_dados : str, opcional
        Caminho do arquivo de dados (CSV, Excel, etc.). Padrão: 'dados_quimioterapia.csv'.
    """
    # Configuração do estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Carregar dados
        if arquivo_dados.endswith('.csv'):
            df = pd.read_csv(arquivo_dados)
        elif arquivo_dados.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(arquivo_dados)
        else:
            raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")
        
        # Verificar colunas necessárias
        required_cols = ['ChemotherapySessions', 'CancerStage', 'SurvivalStatus']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O arquivo deve ter as colunas: {required_cols}.")
        
        # Mapear SurvivalStatus para texto consistente
        status_map = {'Alive': 'Alive', 'Deceased': 'Deceased'}
        df['Status'] = df['SurvivalStatus'].map(status_map)
        
        # Filtrar por estágio
        if cancer_stage is not None:
            if isinstance(cancer_stage, list):
                df = df[df['CancerStage'].isin(cancer_stage)]
                title_stage = f' (Estágios {", ".join(cancer_stage)})'
            else:
                df = df[df['CancerStage'] == cancer_stage]
                title_stage = f' (Estágio {cancer_stage})'
        else:
            title_stage = ' (Todos os Estágios)'
        
        # Verificar se há dados suficientes
        if df.empty:
            print("Nenhum dado disponível após o filtro.")
            return
        
        # Contar ocorrências para cada combinação de sessões e status
        contagem = df.groupby(['ChemotherapySessions', 'Status']).size().unstack().fillna(0)
        
        # Preencher todos os status possíveis (caso falte algum)
        for status in ['Alive', 'Deceased']:
            if status not in contagem.columns:
                contagem[status] = 0
        
        # Ordenar por número de sessões
        contagem = contagem.sort_index()
        
        # Plotar gráfico de barras
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Configurar posições e largura das barras
        bar_width = 0.35
        indices = np.arange(len(contagem))
        
        # Barras para Alive (Vivo)
        alive_bars = ax.bar(indices - bar_width/2, contagem['Alive'], 
                           width=bar_width, color='#2ca02c', label='Alive',
                           edgecolor='grey', alpha=0.9)
        
        # Barras para Deceased (Morto)
        deceased_bars = ax.bar(indices + bar_width/2, contagem['Deceased'], 
                              width=bar_width, color='#d62728', label='Deceased',
                              edgecolor='grey', alpha=0.9)
        
        # Configurações do gráfico
        ax.set_title(f'Distribuição Detalhada de Sessões de Quimioterapia por Sobrevivência{title_stage}', 
                    fontsize=16, pad=20)
        ax.set_xlabel('Número de Sessões de Quimioterapia', fontsize=12)
        ax.set_ylabel('Número de Pacientes', fontsize=12)
        
        # Rótulos do eixo X (cada número de sessão)
        ax.set_xticks(indices)
        ax.set_xticklabels([str(int(x)) for x in contagem.index], fontsize=10)
        
        # Adicionar valores nas barras
        for bars in [alive_bars, deceased_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9)
        
        # Adicionar linha de grade e legenda
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Status de Sobrevivência', loc='upper right')
        
        # Melhorar layout
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erro ao processar os dados: {e}")
        import traceback
        traceback.print_exc()






#plot_sessoes_sobrevivencia()
# plot_quimio_barras(cancer_stage=["III", "IV"])
# plot_quimio_barras_detalhado(cancer_stage=["I", "II"])

# quimioterapia1(["III", "IV"])
# quimioterapia1()
# ademar()
#plot_mortalidade(estagios=["III", "IV"])
# Exemplos de uso:
# genero(estagios=["I", "III", "IV"], separar_estagios=True)  # Barras agrupadas por estágio
# genero(separar_estagios=True)  # Todos os estágios separados
# genero()  # Gráfico padrão (todos juntos)

# plot_mortalidade()  # Equivalente a mortalidade_geral()
# plot_mortalidade(estagios=["III", "IV"])  # Equivalente a mortalidade_3_4()
# plot_mortalidade(estagios=["I", "II", "IV"], separar_estagios=True)  # Gráficos separados

def plot_radiation_barras_detalhado(cancer_stage=None, arquivo_dados="archive/china_cancer_patients_synthetic.csv"):
    """
    Gera um gráfico de barras agrupadas mostrando a distribuição detalhada de sessões de radioterapia
    para pacientes vivos e mortos, com barras separadas para cada número específico de sessões.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        # Carregar dados
        if arquivo_dados.endswith('.csv'):
            df = pd.read_csv(arquivo_dados)
        elif arquivo_dados.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(arquivo_dados)
        else:
            raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")

        # Verificar colunas necessárias
        required_cols = ['RadiationSessions', 'CancerStage', 'SurvivalStatus']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O arquivo deve ter as colunas: {required_cols}.")

        # Mapear SurvivalStatus para texto consistente
        status_map = {'Alive': 'Alive', 'Deceased': 'Deceased'}
        df['Status'] = df['SurvivalStatus'].map(status_map)

        # Filtrar por estágio
        if cancer_stage is not None:
            if isinstance(cancer_stage, list):
                df = df[df['CancerStage'].isin(cancer_stage)]
                title_stage = f' (Estágios {", ".join(cancer_stage)})'
            else:
                df = df[df['CancerStage'] == cancer_stage]
                title_stage = f' (Estágio {cancer_stage})'
        else:
            title_stage = ' (Todos os Estágios)'

        # Verificar se há dados suficientes
        if df.empty:
            print("Nenhum dado disponível após o filtro.")
            return

        # Contar ocorrências para cada combinação de sessões e status
        contagem = df.groupby(['RadiationSessions', 'Status']).size().unstack().fillna(0)

        # Preencher todos os status possíveis (caso falte algum)
        for status in ['Alive', 'Deceased']:
            if status not in contagem.columns:
                contagem[status] = 0

        # Ordenar por número de sessões
        contagem = contagem.sort_index()

        # Plotar gráfico de barras
        fig, ax = plt.subplots(figsize=(14, 8))

        bar_width = 0.35
        indices = np.arange(len(contagem))

        alive_bars = ax.bar(indices - bar_width/2, contagem['Alive'],
                            width=bar_width, color='#2ca02c', label='Alive',
                            edgecolor='grey', alpha=0.9)
        deceased_bars = ax.bar(indices + bar_width/2, contagem['Deceased'],
                               width=bar_width, color='#d62728', label='Deceased',
                               edgecolor='grey', alpha=0.9)

        ax.set_title(f'Distribuição Detalhada de Sessões de Radioterapia por Sobrevivência{title_stage}',
                     fontsize=16, pad=20)
        ax.set_xlabel('Número de Sessões de Radioterapia', fontsize=12)
        ax.set_ylabel('Número de Pacientes', fontsize=12)
        ax.set_xticks(indices)
        ax.set_xticklabels([str(int(x)) for x in contagem.index], fontsize=10)

        for bars in [alive_bars, deceased_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=9)

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Status de Sobrevivência', loc='upper right')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro ao processar os dados: {e}")


# Exemplo de uso:

def sobrevivencia_fumantes_por_idade_individual(arquivo_dados=PATH_DATABASE):
    df = pd.read_csv(arquivo_dados)

    # Agrupar Former e Current em "Smoker"
    df['SmokingGroup'] = df['SmokingStatus'].replace({'Current': 'Smoker', 'Former': 'Smoker', 'Never': 'Never'})

    # Contar vivos e mortos por idade e grupo de fumante
    contagem = (
        df.groupby(['Age', 'SmokingGroup', 'SurvivalStatus'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Plotar gráfico de barras agrupadas
    fig, ax = plt.subplots(figsize=(18, 7))

    statuses = ['Alive', 'Deceased']
    smoking_groups = ['Smoker', 'Never']
    bar_width = 0.35
    ages = sorted(contagem['Age'].unique())
    x = np.arange(len(ages))

    for i, smoke in enumerate(smoking_groups):
        data = contagem[contagem['SmokingGroup'] == smoke]
        data = data.set_index('Age').reindex(ages, fill_value=0)
        for j, status in enumerate(statuses):
            offset = (i - 0.5) * bar_width + (j - 0.5) * (bar_width/2)
            bars = ax.bar(
                x + offset,
                data[status],
                width=bar_width/2,
                label=f'{smoke} - {status}' if i == 0 else "",
                color=sns.color_palette('Set2')[i],
                alpha=0.7 if status == 'Alive' else 1.0,
                edgecolor='black' if status == 'Deceased' else None,
                hatch='//' if status == 'Deceased' else ''
            )
            # Adiciona o número em cima de cada barra
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(ages, rotation=90)
    ax.set_xlabel('Idade')
    ax.set_ylabel('Número de Pacientes')
    ax.set_title('Contagem de Sobreviventes e Óbitos por Idade e Grupo de Fumante')
    # Legenda customizada
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=sns.color_palette('Set2')[0], label='Smoker'),
        Patch(facecolor=sns.color_palette('Set2')[1], label='Never'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Deceased'),
        Patch(facecolor='grey', alpha=0.7, label='Alive')
    ]
    ax.legend(handles=legend_elements, title='Grupo de Fumante / Sobrevivência', loc='upper right')
    plt.tight_layout()
    plt.show()
    
# sobrevivencia_fumantes_por_idade_individual()

def grafico_tumor_mortalidade(arquivo_dados=PATH_DATABASE):
    df = pd.read_csv(arquivo_dados)
    # Corrigir separador decimal se necessário
    df['TumorSize'] = df['TumorSize'].astype(str).str.replace(',', '.')
    df['TumorSize'] = pd.to_numeric(df['TumorSize'], errors='coerce')

    # Padronizar status
    df['SurvivalStatus'] = df['SurvivalStatus'].replace({'Deceased': 'Dead'})
    df = df[df['TumorSize'].notna()]

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='SurvivalStatus', y='TumorSize', data=df, inner=None, palette='Set2')
    sns.boxplot(x='SurvivalStatus', y='TumorSize', data=df, width=0.2, showcaps=True, boxprops={'facecolor':'None'}, showfliers=False, whiskerprops={'linewidth':2})
    plt.title('Distribuição do Tamanho do Tumor por Status de Sobrevivência')
    plt.xlabel('Status de Sobrevivência')
    plt.ylabel('Tamanho do Tumor')
    plt.tight_layout()
    plt.show()

def grafico_mortalidade_metastasis(arquivo_dados=PATH_DATABASE):


    df = pd.read_csv(arquivo_dados)
    # Filtrar apenas estágios III e IV
    df = df[df['CancerStage'].str.contains('III|IV', na=False)]

    # Padronizar status
    df['SurvivalStatus'] = df['SurvivalStatus'].replace({'Deceased': 'Dead'})
    # Padronizar metástase
    df['Metastasis'] = df['Metastasis'].replace({'Yes': 'Metastasis', 'No': 'No Metastasis'})

    # Contar vivos e mortos por metástase
    contagem = df.groupby(['Metastasis', 'SurvivalStatus']).size().unstack(fill_value=0)
    contagem = contagem[['Alive', 'Dead']] if 'Dead' in contagem.columns else contagem

    contagem.plot(kind='bar', color=['#66b3ff', '#ff9999'], edgecolor='black')
    plt.title('Mortalidade por Presença de Metástase (Estágios III e IV)')
    plt.xlabel('Metástase')
    plt.ylabel('Número de Pacientes')
    plt.legend(title='Status de Sobrevivência')
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
grafico_mortalidade_metastasis()



