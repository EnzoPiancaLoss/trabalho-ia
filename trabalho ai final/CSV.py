import pandas as pd

PATH_CSV = "china_cancer_patients_synthetic.csv"

PATH_EX = "china_cancer_patients_synthetic.xlsx"

def excel_to_csv(excel_path, csv_path, sheet_name=0):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.to_csv(csv_path, index=False)
    print(f"Arquivo Excel '{excel_path}' convertido para CSV '{csv_path}'.")

def csv_to_excel(csv_path, excel_path):
    df = pd.read_csv(csv_path)
    df.to_excel(excel_path, index=False)
    print(f"Arquivo CSV '{csv_path}' convertido para Excel '{excel_path}'.")

# Exemplos de uso:
# excel_to_csv('arquivo.xlsx', 'arquivo.csv')
# csv_to_excel('arquivo.csv', 'arquivo.xlsx')

# csv_to_excel("archive/china_cancer_patients_synthetic.csv", PATH_EX)
excel_to_csv("china_cancer_patients_synthetic.xlsx",PATH_CSV)