import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Cargar el archivo CSV con separador ';'
#df = pd.read_csv('Data.csv', sep=';') #Cambiar la ruta según sea necesario
df = pd.read_csv("PE_datos.csv", sep=";")

# Convertir columnas numéricas que están en formato texto
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float, errors='ignore')

# Imputar valores faltantes
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        # Imputar valores faltantes en variables numéricas con la media
        df[column].fillna(df[column].mean(), inplace=True)
    else:
        # Imputar valores faltantes en variables categóricas con la moda
        df[column].fillna(df[column].mode()[0], inplace=True)

# Confirmar que no hay valores faltantes
print("Valores faltantes después de la imputación:")
print(df.isnull().sum())

# Mostrar las primeras filas del DataFrame para verificar la conversión
print(df.head())

# Análisis descriptivo
print(df.describe())

# Crear un archivo PDF para guardar las gráficas
output_pdf = 'Graficas_EDA.pdf' #Cambiar la ruta según sea necesario
with PdfPages(output_pdf) as pdf:
    # Visualización de la distribución de la edad
    plt.figure(figsize=(10, 6))
    sns.histplot(df['EDAD'], bins=30, kde=True)
    plt.title('Distribución de la Edad')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Relación entre edad y valor de factura
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='EDAD', y='REALVTAMNFACTURA', data=df)
    plt.title('Relación entre Edad y Valor de Factura')
    plt.xlabel('Edad')
    plt.ylabel('Valor de Factura')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Matriz de correlación
    plt.figure(figsize=(14, 10))  # Aumentar el tamaño de la figura
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        annot_kws={"size": 10},  # Tamaño del texto de las anotaciones
        cbar_kws={"shrink": 0.8}  # Ajustar el tamaño de la barra de color
    )
    plt.title('Matriz de Correlación', fontsize=16)  # Título más grande
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotar etiquetas del eje X
    plt.yticks(fontsize=10)  # Ajustar tamaño de etiquetas del eje Y
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Análisis de distribución por variable
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Si la columna es numérica, mostrar un histograma
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(df[column], bins=30, kde=True)
            plt.title(f'Distribución de la variable numérica: {column}')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
        else:
            # Si la columna es categórica, mostrar un gráfico de barras
            sns.countplot(x=column, data=df, order=df[column].value_counts().index)
            plt.title(f'Distribución de la variable categórica: {column}')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45, ha='right')  # Rotar etiquetas si son largas
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()

# Guardar el DataFrame limpio para futuros análisis predictivos
output_csv = 'data_input.parquet'#Cambiar la ruta según sea necesario
#df.to_csv(output_csv, index=False)
df.to_parquet(output_csv)

print(f"Análisis exploratorio de datos (EDA) completado. Las gráficas se han guardado en '{output_pdf}' y el archivo limpio en '{output_csv}'.")
