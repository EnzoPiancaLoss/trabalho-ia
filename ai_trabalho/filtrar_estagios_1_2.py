import pandas as pd

def filter_cancer_stages(input_file, output_file):
    """
    Filtra o arquivo Excel removendo pacientes com estágio de câncer 1 ou 2.
    
    Args:
        input_file (str): Caminho do arquivo Excel de entrada
        output_file (str): Caminho do arquivo Excel de saída
    """
    # Ler o arquivo Excel
    df = pd.read_excel(input_file, sheet_name='china_cancer_patients_synthetic')
    
    # Filtrar os pacientes - manter apenas estágios diferentes de I e II
    # Assumindo que a coluna de estágio é 'CancerStage' e contém valores como 'I', 'II', 'III', 'IV'
    filtered_df = df[~df['CancerStage'].isin(['I', 'II'])]
    
    # Salvar o novo arquivo Excel
    filtered_df.to_excel(output_file, sheet_name='china_cancer_patients_filtered', index=False)
    
    print(f"Arquivo filtrado salvo como {output_file}. Removidos {len(df) - len(filtered_df)} pacientes.")

# Exemplo de uso
if __name__ == "__main__":
    input_filename = "inteligenciaartifical.xlsx"
    output_filename = "inteligenciaartifical_filtered.xlsx"
    filter_cancer_stages(input_filename, output_filename)