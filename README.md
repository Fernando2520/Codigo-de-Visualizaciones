# Codigo-de-Visualizaciones Table with Embedded Chart

https://observablehq.com/@d3/box-plot Link de la fuente. 

La visualización es cualquier tipo de representación gráfica que nos permita:

Exploración, nos ayuda a obtener una visión más profunda a través de una gran cantidad de datos

Descubrimiento, También da visión extra

Análisis, se puede utilizar para encontrar patrón

Comunicación,visualización nos ayudan a comunicarnos mejor

Un diagrama de caja y bigotes muestra estadísticas resumidas de una distribución 
cuantitativa. Aquí, la distribución de precios (eje y) 
de un conjunto de diamantes se traza para un rango dado de valores en quilates (eje x).

chart = {
  const svg = d3.select(DOM.svg(width, height));

  const g = svg.append("g")
    .selectAll("g")
    .data(bins)
    .join("g");

  g.append("path")
      .attr("stroke", "currentColor")
      .attr("d", d => `
        M${x((d.x0 + d.x1) / 2)},${y(d.range[1])}
        V${y(d.range[0])}
      `);

  g.append("path")
      .attr("fill", "#ddd")
      .attr("d", d => `
        M${x(d.x0) + 1},${y(d.quartiles[2])}
        H${x(d.x1)}
        V${y(d.quartiles[0])}
        H${x(d.x0) + 1}
        Z
      `);

  g.append("path")
      .attr("stroke", "currentColor")
      .attr("stroke-width", 2)
      .attr("d", d => `
        M${x(d.x0) + 1},${y(d.quartiles[1])}
        H${x(d.x1)}
      `);

  g.append("g")
      .attr("fill", "currentColor")
      .attr("fill-opacity", 0.2)
      .attr("stroke", "none")
      .attr("transform", d => `translate(${x((d.x0 + d.x1) / 2)},0)`)
    .selectAll("circle")
    .data(d => d.outliers)
    .join("circle")
      .attr("r", 2)
      .attr("cx", () => (Math.random() - 0.5) * 4)
      .attr("cy", d => y(d.y));

  svg.append("g")
      .call(xAxis);

  svg.append("g")
      .call(yAxis);

  return svg.node();
}

bins = d3.histogram()
    .thresholds(n)
    .value(d => d.x)
  (data)
    .map(bin => {
      bin.sort((a, b) => a.y - b.y);
      const values = bin.map(d => d.y);
      const min = values[0];
      const max = values[values.length - 1];
      const q1 = d3.quantile(values, 0.25);
      const q2 = d3.quantile(values, 0.50);
      const q3 = d3.quantile(values, 0.75);
      const iqr = q3 - q1; // interquartile range
      const r0 = Math.max(min, q1 - iqr * 1.5);
      const r1 = Math.min(max, q3 + iqr * 1.5);
      bin.quartiles = [q1, q2, q3];
      bin.range = [r0, r1];
      bin.outliers = bin.filter(v => v.y < r0 || v.y > r1); // TODO
      return bin;
    })
    
    data = d3.csvParse(await FileAttachment("diamonds.csv").text(), ({carat, price}) => ({x: +carat, y: +price}))
    
    x = ƒ(n)
    x = d3.scaleLinear()
    .domain([d3.min(bins, d => d.x0), d3.max(bins, d => d.x1)])
    .rangeRound([margin.left, width - margin.right])
    
    y = ƒ(n)
    y = d3.scaleLinear()
    .domain([d3.min(bins, d => d.range[0]), d3.max(bins, d => d.range[1])]).nice()
    .range([height - margin.bottom, margin.top])
    
    xAxis = ƒ(g)
    xAxis = g => g
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(n).tickSizeOuter(0))
    
    yAxis = ƒ(g)
    yAxis = g => g
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(null, "s"))
    .call(g => g.select(".domain").remove())
    
    n = 23.85
    n = width / 40
    
    height = 600
    height = 600
    
    margin = Object {top: 20, right: 20, bottom: 30, left: 40}
    margin = ({top: 20, right: 20, bottom: 30, left: 40})
    
    d3 = Object {event: null, format: ƒ(t), formatPrefix: ƒ(t, n), timeFormat: ƒ(t), timeParse: ƒ(t), utcFormat: ƒ(t), utcParse: ƒ(t), FormatSpecifier: ƒ(t), active: ƒ(t, n), arc: ƒ(), area: ƒ(), areaRadial: ƒ(), ascending: ƒ(t, n), autoType: ƒ(t), axisBottom: ƒ(t), axisLeft: ƒ(t), axisRight: ƒ(t), axisTop: ƒ(t), bisect: ƒ(n, e, r, i), bisectLeft: ƒ(n, e, r, i), …}
    d3 = require("d3@5")
    

# Codigo-de-Visualizaciones Components of Components
https://unipython.com/multiclass-and-multilabel-algorithms-algoritmos-multiclase-multietiqueta/ Link de la fuente 

Esta visualización de componentes , es para determinar una visualización en la cual nos permite hacer una comparacion de datos 
enre tablas, de como va creciendo los porcentajes o de como va aumentando las posibles causas  de alguna información. 

    import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
  # separacion del hiperplano
  w = clf.coef_[0]
  a = -w[0] / w[1]
  xx = np.linspace(min_x - 5, max_x + 5) 
  yy = a * xx - (clf.intercept_[0]) / w[1]
  plt.plot(xx, yy, linestyle, label=label)

def plot_subfigure(X, Y, subplot, title, transform):
  if transform == "pca":
    X = PCA(n_components=2).fit_transform(X)
  elif transform == "cca":
    X = CCA(n_components=2).fit(X, Y).transform(X)
  else:
    raise ValueError

  min_x = np.min(X[:, 0])
  max_x = np.max(X[:, 0])

  min_y = np.min(X[:, 1])
  max_y = np.max(X[:, 1])

  classif = OneVsRestClassifier(SVC(kernel='linear'))
  classif.fit(X, Y)

  plt.subplot(2, 2, subplot)
  plt.title(title)

  zero_class = np.where(Y[:, 0])
  one_class = np.where(Y[:, 1])
  plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
  plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='g', facecolors='none', linewidths=2, label='Class 1')
  plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='red', facecolors='none', linewidths=2, label='Class 2')

  plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--', 'Boundary\nfor class 1')
  plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.', 'Boundary\nfor class 2')
  plt.xticks(())
  plt.yticks(())

  plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
  plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
  if subplot == 2:
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend(loc="upper left")

plt.figure(figsize=(8, 6))

X, Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=False, random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()


    
    
    
