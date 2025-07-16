# tfm-modelo-riesgo-vd
Este repositorio contiene el código elaborado del modelo de riesgos para una empresa de venta directa, el cual será empleado para análisis y desarrollo de un modelo de Machine Learning. 

Su estructura se organiza de la siguiente manera:

app/: Contiene la aplicación web desarrollada con Streamlit, que consume el modelo entrenado (.pickle) y la data de prueba para predicciones.

exploraciondedatos/: Incluye el script EDA_TFM.py con el Análisis Exploratorio de Datos (EDA) de la data.

train/ y predict/: Subcarpetas que alojan la data y el código para las fases de entrenamiento y predicción del modelo, respectivamente.

Todo el código presente en este repositorio es de autoría exclusiva del propietario y no contiene contribuciones externas.

## Instalar el ambiente de conda

### Configuración del Entorno de Desarrollo
Para replicar el entorno de desarrollo y ejecutar este proyecto, sigue estos pasos:

#### Crea un nuevo entorno Conda:

Abre tu terminal (o Anaconda Prompt) y ejecuta:


conda create --name nombre_de_tu_env python=3.9
(Puedes cambiar nombre_de_tu_env por el nombre que prefieras para tu entorno y 3.9 por la versión de Python que uses).

#### Activa el entorno:

conda activate nombre_de_tu_env

Verás el nombre de tu entorno entre paréntesis en tu terminal.


#### Instala las dependencias:

Con el entorno activado y ubicado en la raíz de este repositorio, instala todas las librerías necesarias usando el archivo requirements.txt:

pip install -r requirements.txt

¡Listo! Tu entorno estará configurado con todas las dependencias requeridas para ejecutar el código del proyecto.