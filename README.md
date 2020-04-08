# Codigo-de-Visualizaciones

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
    
    
    
    
    
