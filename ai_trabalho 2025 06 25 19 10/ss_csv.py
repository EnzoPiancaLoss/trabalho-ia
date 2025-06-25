import pandas as pd

def csv_para_excel(caminho_csv, caminho_excel):
    df = pd.read_csv(caminho_csv)
    df.to_excel(caminho_excel, index=False)

if __name__ == "__main__":
    caminho_csv = "archive/china_cancer_patients_synthetic.csv"
    caminho_excel = "output.xlsx"
    csv_para_excel(caminho_csv, caminho_excel)
    print("Conversão concluída!")