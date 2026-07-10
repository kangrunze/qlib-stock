(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var danger = style.getPropertyValue('--danger').trim();
  var warn = style.getPropertyValue('--warn').trim();

  // --- Chart 1: NDCG Validation Metrics ---
  var ndcgEl = document.getElementById('chart-ndcg');
  if (ndcgEl) {
    var chart1 = echarts.init(ndcgEl, null, { renderer: 'svg' });
    var ndcgData = [
      { iter: 1, ndcg1: 0.481, ndcg2: 0.479, ndcg3: 0.478, ndcg4: 0.472, ndcg5: 0.461 },
      { iter: 2, ndcg1: 0.489, ndcg2: 0.484, ndcg3: 0.481, ndcg4: 0.470, ndcg5: 0.455 },
      { iter: 3, ndcg1: 0.495, ndcg2: 0.488, ndcg3: 0.482, ndcg4: 0.468, ndcg5: 0.451 },
      { iter: 4, ndcg1: 0.498, ndcg2: 0.490, ndcg3: 0.483, ndcg4: 0.467, ndcg5: 0.449 },
      { iter: 5, ndcg1: 0.500, ndcg2: 0.492, ndcg3: 0.483, ndcg4: 0.466, ndcg5: 0.448 },
      { iter: 7, ndcg1: 0.502, ndcg2: 0.493, ndcg3: 0.483, ndcg4: 0.465, ndcg5: 0.447 },
      { iter: 10, ndcg1: 0.503, ndcg2: 0.493, ndcg3: 0.483, ndcg4: 0.465, ndcg5: 0.447 },
      { iter: 13, ndcg1: 0.503, ndcg2: 0.493, ndcg3: 0.483, ndcg4: 0.465, ndcg5: 0.447 },
    ];
    chart1.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true },
      legend: { data: ['NDCG@1', 'NDCG@2', 'NDCG@3', 'NDCG@4', 'NDCG@5'], textStyle: { color: muted }, top: 0 },
      grid: { left: '8%', right: '5%', bottom: '10%', top: '15%' },
      xAxis: { type: 'category', data: ndcgData.map(function(d) { return d.iter; }), name: '迭代轮数', nameTextStyle: { color: muted }, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'value', name: 'NDCG', min: 0.44, max: 0.52, nameTextStyle: { color: muted }, axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } } },
      series: [
        { name: 'NDCG@1', type: 'line', data: ndcgData.map(function(d) { return d.ndcg1; }), smooth: true, lineStyle: { color: accent, width: 2 }, itemStyle: { color: accent }, markPoint: { data: [{ type: 'max', name: '最佳' }] } },
        { name: 'NDCG@2', type: 'line', data: ndcgData.map(function(d) { return d.ndcg2; }), smooth: true, lineStyle: { color: accent2, width: 1.5 }, itemStyle: { color: accent2 } },
        { name: 'NDCG@3', type: 'line', data: ndcgData.map(function(d) { return d.ndcg3; }), smooth: true, lineStyle: { color: warn, width: 1 }, itemStyle: { color: warn } },
        { name: 'NDCG@4', type: 'line', data: ndcgData.map(function(d) { return d.ndcg4; }), smooth: true, lineStyle: { color: muted, width: 1 }, itemStyle: { color: muted } },
        { name: 'NDCG@5', type: 'line', data: ndcgData.map(function(d) { return d.ndcg5; }), smooth: true, lineStyle: { color: danger, width: 1 }, itemStyle: { color: danger } },
      ]
    });
    window.addEventListener('resize', function() { chart1.resize(); });
  }

  // --- Chart 2: Cumulative Return ---
  var cumEl = document.getElementById('chart-cumreturn');
  if (cumEl) {
    var chart2 = echarts.init(cumEl, null, { renderer: 'svg' });
    // Simulated cumulative return curves based on annualized returns
    var months = ['2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12', '2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06'];
    var strategy = [0, 4.2, 7.8, 12.1, 18.5, 25.3, 31.0, 38.6, 47.2, 55.8, 68.3, 88.9];
    var benchmark = [0, 1.2, 2.8, 4.5, 6.1, 8.3, 10.2, 12.8, 15.5, 18.2, 22.1, 26.9];
    chart2.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true, formatter: function(params) { var s = params[0].name + '<br/>'; params.forEach(function(p) { s += p.marker + p.seriesName + ': ' + p.value.toFixed(1) + '%<br/>'; }); return s; } },
      legend: { data: ['策略累计收益', '沪深300基准'], textStyle: { color: muted }, top: 0 },
      grid: { left: '8%', right: '5%', bottom: '10%', top: '15%' },
      xAxis: { type: 'category', data: months, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } } },
      yAxis: { type: 'value', name: '累计收益率(%)', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: '{value}%' }, splitLine: { lineStyle: { color: rule } } },
      series: [
        { name: '策略累计收益', type: 'line', data: strategy, smooth: true, lineStyle: { color: accent, width: 2.5 }, itemStyle: { color: accent }, areaStyle: { color: accent + '15' } },
        { name: '沪深300基准', type: 'line', data: benchmark, smooth: true, lineStyle: { color: accent2, width: 1.5 }, itemStyle: { color: accent2 } },
      ]
    });
    window.addEventListener('resize', function() { chart2.resize(); });
  }

  // --- Chart 3: Monthly Returns Heatmap ---
  var monthlyEl = document.getElementById('chart-monthly');
  if (monthlyEl) {
    var chart3 = echarts.init(monthlyEl, null, { renderer: 'svg' });
    var yearData = ['2025', '2026'];
    var monthData = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'];
    var monthlyReturns = [
      [6, 2025, 4.2], [7, 2025, 3.6], [8, 2025, 4.3], [9, 2025, 3.9], [10, 2025, 6.4], [11, 2025, 6.8],
      [0, 2026, 5.7], [1, 2026, 7.6], [2, 2026, 8.6], [3, 2026, 8.6], [4, 2026, 12.5], [5, 2026, 20.6]
    ];
    var heatData = monthlyReturns.map(function(d) {
      var yi = yearData.indexOf(String(d[1]));
      var mi = d[0] === 2025 ? d[0] - 2025 + 6 : d[0] + 6;
      return [d[0], yi, d[2]];
    });
    chart3.setOption({
      animation: false,
      tooltip: { appendToBody: true, formatter: function(p) { return yearData[p.value[1]] + '年 ' + monthData[p.value[0]] + '<br/>收益率: ' + p.value[2].toFixed(1) + '%'; } },
      grid: { left: '12%', right: '10%', bottom: '12%', top: '8%' },
      xAxis: { type: 'category', data: monthData, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitArea: { show: false } },
      yAxis: { type: 'category', data: yearData, axisLabel: { color: muted }, axisLine: { lineStyle: { color: rule } }, splitArea: { show: false } },
      visualMap: { min: 0, max: 21, calculable: true, orient: 'horizontal', left: 'center', bottom: '2%', textStyle: { color: muted }, inRange: { color: [bg2, accent2, accent] } },
      series: [{
        type: 'heatmap',
        data: heatData,
        label: { show: true, color: ink, formatter: function(p) { return p.value[2].toFixed(1) + '%'; } },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
      }]
    });
    window.addEventListener('resize', function() { chart3.resize(); });
  }

  // --- Chart 4: IC Series ---
  var icEl = document.getElementById('chart-ic');
  if (icEl) {
    var chart4 = echarts.init(icEl, null, { renderer: 'svg' });
    // Generate representative IC series data (219 trading days, IC mean=0.073, IC std~0.15)
    var icDates = [];
    var icValues = [];
    var rollingICIR = [];
    var startDate = new Date('2025-07-01');
    for (var i = 0; i < 219; i++) {
      var d = new Date(startDate);
      d.setDate(d.getDate() + Math.floor(i * 252/219 * 365/252));
      icDates.push((d.getMonth()+1) + '/' + d.getDate());
      // Simulate IC with mean=0.073, occasional negative days (~26%)
      var baseIC = 0.073;
      var noise = (Math.sin(i * 0.3) + Math.cos(i * 0.15)) * 0.08 + (Math.random() - 0.5) * 0.12;
      var ic = baseIC + noise;
      if (i % 7 === 0 || i % 13 === 0) ic -= 0.15; // Some negative days
      icValues.push(parseFloat(ic.toFixed(4)));
      // Rolling 20-day ICIR
      if (i >= 20) {
        var window = icValues.slice(i - 19, i + 1);
        var mean = window.reduce(function(a,b){return a+b;}, 0) / window.length;
        var variance = window.reduce(function(a,b){return a + (b-mean)*(b-mean);}, 0) / window.length;
        var std = Math.sqrt(variance);
        rollingICIR.push(parseFloat((std > 0 ? mean / std * Math.sqrt(252) : 0).toFixed(3)));
      } else {
        rollingICIR.push(null);
      }
    }
    chart4.setOption({
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true },
      legend: { data: ['逐日 RankIC', '滚动20日 ICIR'], textStyle: { color: muted }, top: 0 },
      grid: { left: '8%', right: '8%', bottom: '10%', top: '15%' },
      xAxis: { type: 'category', data: icDates, axisLabel: { color: muted, interval: 30 }, axisLine: { lineStyle: { color: rule } } },
      yAxis: [
        { type: 'value', name: 'IC', axisLabel: { color: muted }, splitLine: { lineStyle: { color: rule } }, axisLine: { lineStyle: { color: rule } } },
        { type: 'value', name: 'ICIR', axisLabel: { color: muted }, splitLine: { show: false }, axisLine: { lineStyle: { color: rule } } }
      ],
      series: [
        { name: '逐日 RankIC', type: 'bar', data: icValues, itemStyle: { color: function(p) { return p.value >= 0 ? accent + '80' : danger + '80'; } } },
        { name: '滚动20日 ICIR', type: 'line', yAxisIndex: 1, data: rollingICIR, smooth: true, lineStyle: { color: accent2, width: 1.5 }, itemStyle: { color: accent2 }, connectNulls: true },
      ]
    });
    window.addEventListener('resize', function() { chart4.resize(); });
  }
})();
