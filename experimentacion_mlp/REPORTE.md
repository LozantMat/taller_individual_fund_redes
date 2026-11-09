# Reporte: comportamiento del pipeline final

## 1. Protocolo

- **Baseline:** MLP 64x64 con ReLU, momentum (β = 0.9), lr = 0.01, lotes de 32, sin regularización y 100 épocas. Es la mejor configuración de `Optimizacion_modelo_neuronal.ipynb`.
- **Datos:** el conjunto de prueba es el mismo del notebook (14.000 imágenes). Del conjunto de entrenamiento se separó un 10 % para validación, así que el modelo se entrena con 50.400 imágenes y se valida con 5.600.
- **Selección:** early stopping y la elección de configuraciones usan solo la validación. La prueba se reporta, pero no influye en ninguna decisión.
- **Comparación justa:** el baseline se volvió a entrenar con este mismo protocolo y obtuvo **97,62 %** de accuracy de prueba. En el notebook había obtenido 97,87 %, pero con 5.600 imágenes más de entrenamiento.
- **Ruido de la medición:** con 14.000 imágenes de prueba y una accuracy cercana al 98 %, el error estándar es de unos ±0,12 pp. Por eso, diferencias menores a ~0,2 pp no se consideran significativas.

## 2. Resultados por etapa

| Etapa | Configuración | Acc. val. | Acc. prueba | Δ vs baseline | Épocas (restaurada / corridas) |
|---|---|---|---|---|---|
| Baseline | 64x64, lr 0.01 | 0,9761 | 0,9762 | — | 100 / 100 |
| Dropout | 0.1 | 0,9780 | 0,9769 | +0,07 pp | 47 / 77 |
| Dropout | 0.2 | 0,9748 | 0,9745 | −0,17 pp | 41 / 71 |
| Dropout | 0.4 | 0,9671 | 0,9679 | −0,84 pp | 70 / 100 |
| Learning rate (dropout 0.1) | 0.005 | 0,9762 | 0,9750 | −0,12 pp | 21 / 51 |
| Learning rate (dropout 0.1) | 0.0075 | 0,9752 | 0,9763 | +0,01 pp | 16 / 46 |
| Learning rate (dropout 0.1) | 0.015 | 0,9725 | 0,9763 | +0,01 pp | 19 / 49 |
| Learning rate (dropout 0.1) | 0.02 | 0,9718 | 0,9756 | −0,06 pp | 11 / 41 |
| Aumento de datos | dropout 0.0 | **0,9838** | **0,9856** | **+0,94 pp** | 133 / 150 |
| Aumento de datos | dropout 0.1 | 0,9834 | 0,9836 | +0,74 pp | 136 / 150 |
| Aumento de datos | dropout 0.2 | 0,9775 | 0,9805 | +0,43 pp | 136 / 150 |
| Aumento de datos | dropout 0.4 | 0,9625 | 0,9652 | −1,10 pp | 140 / 150 |
| **Final** | aumento, sin dropout, hasta 300 épocas | 0,9834 | **0,9844** | **+0,81 pp** | 164 / 214 |

### Dropout

Por sí solo, el dropout casi no mejora la accuracy. Con una tasa de 0.1 la ganancia es de +0,07 pp, que está dentro del ruido; con 0.2 y 0.4 la accuracy baja. La red de 64x64 tiene poca capacidad, y apagar neuronas la reduce todavía más.

Con 0.4 el modelo ya no alcanza a ajustar ni siquiera el conjunto de entrenamiento (95,4 % de accuracy de entrenamiento), es decir, hay subajuste.

Donde sí ayuda el dropout es en la **pérdida**. Combinado con early stopping, la pérdida de prueba pasa de 0,148 en el baseline a 0,090 con dropout 0.1. Las predicciones correctas siguen siendo casi las mismas, pero el modelo deja de dar probabilidades extremas y de estar tan sobreconfiado cuando se equivoca.

### Learning rate

Entre 0.005 y 0.02 todas las variantes quedan a menos de ±0,12 pp del baseline, así que **0.01 sigue siendo una buena elección** y se mantuvo. Lo que sí cambia es la velocidad: con un learning rate más alto, la pérdida de validación llega antes a su mínimo (época 11 con 0.02, época 47 con 0.01).

### Aumento de datos

Es la mejora más importante. Las transformaciones son traslaciones de ±2 píxeles y rotaciones de hasta ±10° (ver `resultados/ejemplos_aumento.png`). Hay dos efectos claros:

1. **Ya no hay sobreajuste.** La pérdida de validación baja de forma continua durante cientos de épocas, en lugar de subir desde la época 12 como pasaba en el baseline.
2. **El aumento de datos y el dropout se estorban.** Ambos son formas de regularización, y aplicarlos juntos a una red pequeña regulariza de más. Por eso, con aumento de datos, la accuracy baja a medida que sube el dropout: 98,56 % con 0.0, 98,36 % con 0.1, 98,05 % con 0.2 y 96,52 % con 0.4. La mejor combinación resultó ser **aumento de datos sin dropout**.

## 3. Comportamiento del pipeline final

La configuración final es la del baseline más el aumento de datos y early stopping (paciencia de 50 épocas sobre la pérdida de validación), con un máximo de 300 épocas.

| Métrica (prueba) | Baseline | Final |
|---|---|---|
| Accuracy | 97,62 % | **98,44 %** |
| F1 ponderado | 0,9762 | **0,9844** |
| Pérdida | 0,148 | **0,051** |
| Errores (de 14.000) | 333 | **219 (−34 %)** |
| Accuracy de entrenamiento | 100 % | 98,0 % (sobre datos aumentados) |
| Tiempo de entrenamiento | 1,0 min | 14,8 min |

**Dinámica del entrenamiento** (`resultados/final_curvas.png`)

- La pérdida de entrenamiento queda **por encima** de la de validación y la de prueba durante todo el entrenamiento: 0,062 frente a 0,046 y 0,051. Esto es lo esperado, porque el modelo se entrena con imágenes transformadas, que son más difíciles que las originales con las que se evalúa.
- Esa relación es la contraria a la del baseline, cuya pérdida de entrenamiento era de 10⁻⁴ mientras la de prueba subía. Es la señal más clara de que se eliminó el sobreajuste.
- La accuracy de validación sube rápido hasta la época ~50 (98,1 %), luego avanza despacio hasta la ~100 (98,3 %) y después queda en una meseta entre 98,3 % y 98,6 %.
- El entrenamiento se detuvo en la época 214 y se restauraron los pesos de la época 164.

**Efecto de las épocas adicionales**

- Ampliar el máximo de 150 a 300 épocas casi no cambió el resultado. El modelo final (98,44 %) y la corrida de 150 épocas (98,56 %) difieren en 17 imágenes, lo que está dentro del ruido.
- Las épocas extra sirvieron sobre todo para confirmar que el modelo había llegado a su meseta. El límite útil lo pone la capacidad de la red, no la cantidad de épocas.
- La época restaurada (164) corresponde a un mínimo aislado de una pérdida de validación que oscila bastante (entre 0,046 y 0,055 en esa zona). Un early stopping con paciencia amplia no evita esa oscilación: termina eligiendo el punto más bajo de una serie con ruido.

**Errores por clase** (`resultados/final_matriz_confusion.png`)

- El F1 mejora en las 10 clases. Las mayores ganancias están en los dígitos 5 (0,969 → 0,983), 3 (0,974 → 0,986), 8 (0,971 → 0,983) y 2 (0,973 → 0,985), que son los que más varían en inclinación y posición.
- Hay una confusión que no mejora: los **4 clasificados como 9** pasan de 23 a 28 casos, y hoy son la principal fuente de error.
- Probablemente las rotaciones de ±10° hacen que un 4 cerrado se parezca todavía más a un 9. Si se quisiera atacar ese error, la rotación es el parámetro que habría que ajustar.

## 4. Conclusiones

1. **El aumento de datos es el cambio que más aporta**: +0,8 a +0,9 pp de accuracy, un tercio menos de errores y una pérdida de prueba casi tres veces menor.
2. **El dropout no conviene en esta red de 64x64** cuando ya hay aumento de datos. Con una tasa de 0.1 o 0.2 solo resta, y con 0.4 la red queda subajustada.
3. **El learning rate de 0.01 ya estaba bien elegido.** Los valores cercanos dan resultados equivalentes.
4. **Early stopping con paciencia amplia cumplió su función.** El modelo pudo atravesar las mesetas largas que produce el aumento de datos y se detuvo solo cuando dejó de mejorar.
5. **Más épocas solo valen la pena junto con un regularizador.** Sin aumento de datos, entrenar más solo agrava el sobreajuste. Con aumento de datos, el modelo mejora hasta la época ~100 y después se estanca.

**Siguientes pasos sugeridos**

- Probar una red más ancha (128x128) con aumento de datos. Con más capacidad, el dropout podría volver a ser útil.
- Agregar decaimiento del learning rate (el parámetro `lr_decay` ya está disponible) para reducir la oscilación de la meseta.
- Reentrenar el modelo final con entrenamiento y validación juntos, usando el número de épocas que se encontró.
