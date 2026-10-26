# CASO 01 | Pedidos de kits de robótica - SENATI
# Entrena una regresión lineal múltiple y guarda el modelo en .pkl y .json.
# Ejecutar una vez:  python entrenar_modelo.py

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

carpeta = Path(__file__).resolve().parent

# 1. Leer datos (utf-8-sig quita el carácter invisible del inicio del CSV)
datos = pd.read_csv(carpeta / 'datos.csv', encoding='utf-8-sig')
print('Filas y columnas:', datos.shape)
print('Valores ausentes:\n', datos.isna().sum(), '\n')

# 2. Variables de entrada (X) y respuesta (y). 'semana' es solo un código.
columnas = ['consultas_web', 'inversion_publicidad_soles', 'campana_escolar']
objetivo = 'pedidos_kits'
X = datos[columnas]
y = datos[objetivo]

# 3. Separar 80% entrenamiento / 20% prueba
X_ent, X_pru, y_ent, y_pru = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f'Entrenamiento: {len(X_ent)} filas | Prueba: {len(X_pru)} filas\n')

# 4. Entrenar
modelo = LinearRegression()
modelo.fit(X_ent, y_ent)

print('Intercepto:', round(modelo.intercept_, 4))
for nombre, coef in zip(columnas, modelo.coef_):
    print(f'  {nombre}: {coef:.4f}')

# 5. Evaluar solo con datos de prueba
pred = modelo.predict(X_pru)
mae = mean_absolute_error(y_pru, pred)
r2 = r2_score(y_pru, pred)
print(f'\nMAE (prueba): {mae:.2f} pedidos')
print(f'R²  (prueba): {r2:.3f}')

# 6. Pruebas obligatorias del caso
pruebas = pd.DataFrame(
    [[20, 200, 0], [55, 800, 1]], columns=columnas)
print('\nPruebas obligatorias:')
for fila, p in zip(pruebas.values, modelo.predict(pruebas)):
    print(f'  {fila.tolist()} -> {p:.2f} pedidos')

# 7. Guardar el modelo (.pkl) con sus datos de apoyo para la web
paquete = {
    'modelo': modelo,
    'columnas': columnas,
    'mae': float(mae),
    'r2': float(r2),
    'rangos': {'consultas_web': [5, 70],
               'inversion_publicidad_soles': [0, 1100],
               'campana_escolar': [0, 1]},
}
joblib.dump(paquete, carpeta / 'modelo_kits.pkl')

# 8. Guardar coeficientes en modelo.json (lo pide la rúbrica)
with open(carpeta / 'modelo.json', 'w', encoding='utf-8') as f:
    json.dump({
        'intercepto': float(modelo.intercept_),
        'coeficientes': dict(zip(columnas, map(float, modelo.coef_))),
        'mae_prueba': float(mae),
        'r2_prueba': float(r2),
    }, f, indent=2, ensure_ascii=False)

print('\nGuardados: modelo_kits.pkl y modelo.json')
