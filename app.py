# CASO 01 | Estimación de pedidos de kits de robótica - SENATI
# Flask + HTML + CSS (sin JavaScript). Usa el modelo guardado en modelo_kits.pkl.

from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
carpeta = Path(__file__).resolve().parent

# Cargamos el modelo entrenado con entrenar_modelo.py (una sola vez).
paquete = joblib.load(carpeta / 'modelo_kits.pkl')
modelo = paquete['modelo']
columnas = paquete['columnas']
rangos = paquete['rangos']
mae = paquete['mae']
r2 = paquete['r2']


def leer_numero(texto):
    """Convierte el texto del formulario en número (acepta coma decimal)."""
    return float(texto.replace(',', '.'))


@app.route('/', methods=['GET', 'POST'])
def inicio():
    resultado = None
    error = ''
    # Valores que el usuario escribió, para no perderlos al recargar.
    escritos = {'consultas_web': '', 'inversion_publicidad_soles': '',
                'campana_escolar': '0'}

    if request.method == 'POST':
        for campo in escritos:
            escritos[campo] = request.form.get(campo, '').strip()

        try:
            consultas = leer_numero(escritos['consultas_web'])
            inversion = leer_numero(escritos['inversion_publicidad_soles'])
            campana = int(escritos['campana_escolar'])

            # Validación de rangos pedida en el caso.
            c_min, c_max = rangos['consultas_web']
            i_min, i_max = rangos['inversion_publicidad_soles']
            if not c_min <= consultas <= c_max:
                error = f'Las consultas web deben estar entre {c_min} y {c_max}.'
            elif not i_min <= inversion <= i_max:
                error = f'La inversión debe estar entre S/ {i_min} y S/ {i_max}.'
            elif campana not in (0, 1):
                error = 'La campaña escolar solo puede ser 0 (No) o 1 (Sí).'
            else:
                # Usamos un DataFrame con los mismos nombres de columna del entrenamiento.
                fila = pd.DataFrame([[consultas, inversion, campana]],
                                    columns=columnas)
                estimacion = modelo.predict(fila)[0]
                estimacion = max(estimacion, 0)  # no existen pedidos negativos
                resultado = {
                    'pedidos': round(estimacion),
                    'exacto': f'{estimacion:.1f}',
                    'minimo': max(round(estimacion - mae), 0),
                    'maximo': round(estimacion + mae),
                }
        except ValueError:
            error = 'Escribe valores numéricos válidos en todos los campos.'

    return render_template('index.html', resultado=resultado, error=error,
                           escritos=escritos, rangos=rangos, mae=mae, r2=r2)


if __name__ == '__main__':
    # debug=True solo para practicar en la computadora.
    app.run(debug=True)
