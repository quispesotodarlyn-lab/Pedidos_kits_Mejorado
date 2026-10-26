# Caso 01 — Pedidos de kits de robótica (SENATI)

## Archivos
- `datos.csv`: 110 semanas simuladas.
- `entrenar_modelo.py`: entrena `LinearRegression` (80/20) y genera `modelo_kits.pkl` y `modelo.json`.
- `app.py`, `templates/index.html`, `static/estilos.css`: web en Flask + HTML + CSS (sin JavaScript).

## Pasos
```powershell
pip install -r requirements.txt
python entrenar_modelo.py   # vuelve a crear el .pkl con TU versión de scikit-learn
python app.py               # abre http://127.0.0.1:5000
```

## Resultados (prueba, 22 filas)
- Ecuación: pedidos = 1.54 + 0.346·consultas + 0.0067·soles_publicidad + 5.02·campaña
- MAE = 2.95 pedidos | R² = 0.79
- Prueba 1 (20 consultas, S/ 200, sin campaña): 9.8 ≈ 10 kits
- Prueba 2 (55 consultas, S/ 800, con campaña): 31.0 ≈ 31 kits
