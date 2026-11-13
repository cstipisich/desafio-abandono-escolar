# Desafío Ministerio — US$1.000.000 contra el abandono escolar

**Equipo:** [Lavia, Ram, Salazar, Stipisich]
**Curso:** Aplicación de Ciencia de Datos — Clase 6/7
**Fecha:** noviembre 2026

## Qué hicimos, en una frase

Entrenamos un modelo de riesgo de abandono (LightGBM, AUC 0,842 en validación 2020),
estimamos la probabilidad de abandonar de los 298.173 alumnos de la foto 2021 y
repartimos el millón resolviendo el problema de optimización con condiciones KKT:
terminamos dando plata a **4.306 alumnos** (1,4 % de la foto) con una ganancia
esperada de **588,4 abandonos evitados**, 18× más que repartir el millón entre todos.

## 1. Modelo de riesgo (`01_entrenar_modelo.py`)

- **Datos**: `chile_2014_2020.csv`, filtrado a 2019–2020 → 587.598 filas.
- **Etiqueta**: `abandono_t1` (tasa base 1,48 %).
- **Split temporal**: entrenamos con 2019 (293.716) y validamos con 2020 (293.882).
  Lo hicimos así para no filtrar información del futuro: el modelo que predice 2021
  tiene que haberse entrenado con años anteriores.
- **Features** (del recorte de la clase 6, sin datos del futuro): promedio, asistencia,
  edad, repetición, cambios de escuela, agregados de escuela (`esc_n`, `esc_prom`,
  `esc_asist`, `esc_rep`, `esc_ab_lag`), dependencia, ruralidad, región, enseñanza,
  grado, jornada, rama, especialidad, género y situación final (`sit_fin`).
- **Modelo**: LightGBM binario con `learning_rate=0.05`, `num_leaves=63`,
  `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`
  y `early_stopping=100`. Probamos con `is_unbalance=True` y el modelo se estancaba en
  la iteración 1; lo sacamos y el AUC subió a 0,842.
- **Resultados**: AUC val 2020 = **0,8424**; mejor iteración 69; Brier tras calibración
  isotónica = **0,00965**. La calibración es clave: sin ella las probabilidades de
  LightGBM no son comparables entre sí y el KKT repartiría mal.
- **Predicción 2021**: 298.173 alumnos; `p` media 1,10 %, mediana 0,46 %, máximo 1,0.
  La cola de riesgo (alumnos con `p` alto) es chica pero existe, y es donde el KKT
  concentra la plata.

## 2. Reparto óptimo (`02_optimizar_montos.py`)

Con `ganancia_i(m) = p_i · 0,6 · (1 − e^{−m/200})`, maximizamos `Σ ganancia_i` sujeto
a `Σ m_i ≤ 1.000.000` y `m_i ≥ 0`. Por KKT, todos los alumnos con `m_i > 0` tienen el
mismo retorno marginal λ:
m_i = 200 · ln( p_i · 0,6 / (200·λ) ) si p_i · 0,6 / (200·λ) > 1
m_i = 0 en caso contrario


Calibramos λ por bisección (`brentq`) hasta que `Σ m_i = 1.000.000`. Resultado:
**λ = 2,308·10⁻⁴**, 4.306 alumnos con monto positivo, mínimo US$ 38,07,
mediana US$ 258,55, máximo US$ 512,96. La suma da exactamente US$ 1.000.000.

## 3. Comparación de estrategias (`06_tabla_estrategias.py`)

           estrategia  alumnos_con_plata  monto_por_alumno_mediana  ganancia  ratio_vs_kkt
 KKT óptimo (entrega)               4306                    258.55    588.37         1.000
  Top-1 % concentrado               2982                    335.35    566.49         0.963
  Top-2 % concentrado               5963                    167.70    475.48         0.808
  Top-0.5 % concentrado             1491                    670.69    459.52         0.781
  Top-5 % concentrado              14909                     67.07    296.06         0.503
  Top-10 % concentrado              29817                     33.54    190.90         0.324
  Top-20 % concentrado              59635                     16.77    117.46         0.200
 Uniforme entre todos             298173                      3.35     32.68         0.056

Lo que muestra la tabla: el KKT le gana al uniforme por ~18× (repartir ~3 US$ por
alumno no mueve el riesgo) y le gana al top-1 % por ~4 % (concentrar todo satura el
efecto `1 − e^{−m/200}`). El óptimo está en un punto **intermedio**: pocos alumnos,
montos moderados, todos con el mismo retorno marginal.

## 4. Las tres preguntas del enunciado

**¿Alcanza con ordenar a los alumnos?** No. El orden importa (prioriza a los de mayor
riesgo), pero también importa *cuánto vale* esa probabilidad: entra multiplicando en
la ganancia. Y como el efecto satura, no conviene concentrar todo en los primeros.

**¿Poco a muchos o mucho a pocos?** Ni uno ni otro: **intermedio**. Pocos alumnos
(4.306 de 298.173, el 1,4 %), con montos moderados (mediana US$ 258, máximo US$ 513).
El umbral de KKT deja afuera a los de riesgo bajo; los de riesgo alto reciben más,
pero no hasta 1.000 porque el marginal se iguala antes.

**¿Es razonable la hipótesis del ministerio?** Es una hipótesis causal fuerte: el
efecto solo existe en quien habría abandonado. El ejercicio la asume; nuestro modelo
solo estima `p_i`. Para validarla haría falta un diseño causal (asignación aleatoria,
regresión discontinua, diff-in-diff). Con datos observacionales, el efecto estimado se
confunde con selección.

## 5. Entrega

`entrega.csv` con `id_alumno, monto`:
- 298.173 filas (una por alumno de 2021);
- montos ≥ 0;
- suma = US$ 1.000.000,00 exactos;
- solo IDs de la foto 2021, sin duplicados.

Validación automática: `04_validar_entrega.py` → `OK validación`.
OK validación
Filas: 298,173
Suma:  1,000,000.00
Con monto > 0: 4,306
Máximo: 512.17

## 6. Reproducir

```powershell
cd D:\Ciencia_Datos_UNO\Aplicacion_Ciencia_Datos\Clase_6\solucion
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" -m pip install pandas numpy scikit-learn lightgbm pyarrow scipy joblib matplotlib
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 01_entrenar_modelo.py
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 02_optimizar_montos.py
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 03_generar_entrega.py
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 04_validar_entrega.py
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 05_graficos.py
& "C:\Users\Usuario\AppData\Local\Programs\Python\Python313\python.exe" 06_tabla_estrategias.py

