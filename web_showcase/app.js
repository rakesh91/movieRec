(() => {
  "use strict";

  const COLORS = {
    bias: "#9b8b68",
    mf_a: "#7395d5",
    mf_b: "#164b3d",
    genre: "#e88854",
    hybrid: "#9c83c7",
    precision_at_10: "#7395d5",
    recall_at_10: "#e88854",
    ndcg_at_10: "#164b3d",
  };
  const METRIC_LABELS = {
    precision_at_10: "P@10",
    recall_at_10: "Recall@10",
    ndcg_at_10: "NDCG@10",
  };
  const MODEL_ORDER = ["bias", "mf_a", "mf_b", "genre", "hybrid"];
  const MODEL_CHART_LABELS = {
    bias: "Bias",
    mf_a: "MF A",
    mf_b: "MF B",
    genre: "Genre",
    hybrid: "Hybrid",
  };
  const SVG_NS = "http://www.w3.org/2000/svg";
  let data;
  let selectedRecords = {};
  const chunkCache = new Map();

  const byId = (id) => document.getElementById(id);

  function svgNode(name, attrs = {}, text = null) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    return node;
  }

  function baseSvg(width, height, label) {
    const svg = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": label,
    });
    return svg;
  }

  function axisText(svg, x, y, text, anchor = "middle", color = "#7b8981") {
    svg.appendChild(svgNode("text", {
      x, y, "text-anchor": anchor, fill: color, "font-size": "10",
      "font-family": "system-ui, sans-serif",
    }, text));
  }

  function modelLabel(key) {
    return data.models[key]?.name ?? key;
  }

  function scoreLabel(key, score) {
    return Number(score).toFixed(key === "genre" ? 3 : 2);
  }

  function userKeys(split, partition) {
    return data.splits[split].user_ids[partition] || [];
  }

  function fillUserList() {
    const partition = byId("partition-select").value;
    const randomized = new Set(userKeys("randomized", partition));
    const users = userKeys("chronological", partition).filter((userId) =>
      randomized.has(userId),
    );
    const list = byId("user-options");
    list.replaceChildren();
    for (const userId of users) {
      const option = document.createElement("option");
      option.value = userId;
      option.label = `User ${userId}`;
      list.appendChild(option);
    }
    if (!users.includes(byId("user-search").value)) {
      byId("user-search").value = users.includes("1") ? "1" : (users[0] || "");
    }
  }

  function currentUser() {
    const partition = byId("partition-select").value;
    const randomized = new Set(userKeys("randomized", partition));
    const users = userKeys("chronological", partition).filter((userId) =>
      randomized.has(userId),
    );
    let userId = byId("user-search").value.trim();
    if (!users.includes(userId)) {
      userId = users.includes("1") ? "1" : users[0];
      byId("user-search").value = userId || "";
    }
    return { partition, userId };
  }

  function decodeRecord(compact) {
    const [historyCount, heldoutCount, relevantCount, preferences, packedModels] = compact;
    const positiveGenres = preferences.map(([index, value]) => ({
      genre: data.genre_names[index],
      preference: value / 1000,
    }));
    const models = {};
    data.model_order.forEach((name, index) => {
      const packed = packedModels[index] || [];
      if (!packed.length) return;
      const recommendations = packed.map(([movieId, score, rating]) => ({
        movie_id: movieId,
        score: score / 1000,
        heldout_rating: rating || null,
      }));
      const hits = recommendations.filter((item) => item.heldout_rating >= 4).length;
      const precision = hits / 10;
      let recall = null;
      let ndcg = null;
      if (relevantCount > 0) {
        recall = hits / relevantCount;
        const dcg = recommendations.reduce(
          (sum, item, rank) =>
            sum + (item.heldout_rating >= 4 ? 1 / Math.log2(rank + 2) : 0),
          0,
        );
        const idealCount = Math.min(relevantCount, 10);
        const ideal = Array.from({ length: idealCount }, (_, rank) =>
          1 / Math.log2(rank + 2),
        ).reduce((sum, value) => sum + value, 0);
        ndcg = ideal ? dcg / ideal : null;
      }
      models[name] = {
        recommendations,
        metrics: {
          precision_at_10: precision,
          recall_at_10: recall,
          ndcg_at_10: ndcg,
        },
      };
    });
    return {
      history_count: historyCount,
      heldout_count: heldoutCount,
      relevant_count: relevantCount,
      positive_genres: positiveGenres,
      models,
    };
  }

  async function loadUserRecord(split, partition, userId) {
    const splitInfo = data.splits[split];
    const ids = splitInfo.user_ids[partition];
    const position = ids.indexOf(userId);
    if (position < 0) throw new Error(`User ${userId} is unavailable for this split.`);
    const chunkIndex = Math.floor(position / data.user_chunk_size);
    const cacheKey = `${split}_${partition}_${chunkIndex}`;
    if (!chunkCache.has(cacheKey)) {
      const path = `data/${split}_${partition}_${String(chunkIndex).padStart(2, "0")}.json.gz`;
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Could not load recommendation chunk (${response.status}).`);
      if (typeof DecompressionStream === "undefined") {
        throw new Error("This browser cannot decompress the static recommendation data. Please use a current browser.");
      }
      const compressed = await response.arrayBuffer();
      const decompressed = new Blob([compressed])
        .stream()
        .pipeThrough(new DecompressionStream("gzip"));
      chunkCache.set(cacheKey, await new Response(decompressed).json());
    }
    return decodeRecord(chunkCache.get(cacheKey)[userId]);
  }

  function renderNotice(partition) {
    byId("partition-notice").textContent = partition === "test"
      ? "Test recommendations are shown for standalone methods only. Hybrid test performance was not evaluated; its column is intentionally blank."
      : "Validation relevance is rating ≥4. Random splitting is a held-out-interaction test, not a future-time evaluation.";
  }

  function renderUserSummary(records, userId) {
    const root = byId("user-summary");
    root.replaceChildren();
    const pills = [
      `User ${userId}`,
      `Chronological history: ${records.chronological.history_count}`,
      `Random-split history: ${records.randomized.history_count}`,
      `Relevant held-out: ${records.chronological.relevant_count} chronological · ${records.randomized.relevant_count} random`,
    ];
    for (const label of pills) {
      const chip = document.createElement("span");
      chip.className = "user-pill";
      chip.textContent = label;
      root.appendChild(chip);
    }
  }

  function scoreUnit(key) {
    if (key === "genre") return "cosine";
    if (key === "hybrid") return "z-score";
    return "rating";
  }

  function renderRecommendations(records, partition) {
    const root = byId("recommendation-grid");
    root.replaceChildren();
    for (const [split, title, description] of [
      ["chronological", "Chronological split", "Train on earlier ratings; hold out later ratings."],
      ["randomized", "Random per-user split", "80/10/10, seed 42; not a future-time test."],
    ]) {
      const section = document.createElement("section");
      section.className = "recommendation-row";
      const heading = document.createElement("div");
      heading.className = "recommendation-row-heading";
      const titleNode = document.createElement("h3");
      titleNode.textContent = title;
      const descriptionNode = document.createElement("p");
      descriptionNode.textContent = description;
      heading.append(titleNode, descriptionNode);
      section.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "model-row-grid";
      const modelData = records[split].models || {};
      for (const key of MODEL_ORDER) {
        const result = modelData[key];
        const card = document.createElement("article");
        card.className = `model-card${key === "hybrid" ? " hybrid-card" : ""}`;
        const titleWrap = document.createElement("div");
        const modelTitle = document.createElement("h4");
        modelTitle.textContent = modelLabel(key);
        const unit = document.createElement("p");
        unit.className = "model-card-subtitle";
        unit.textContent = scoreUnit(key);
        titleWrap.append(modelTitle, unit);
        card.appendChild(titleWrap);

        if (!result) {
          const unavailable = document.createElement("p");
          unavailable.className = "model-unavailable";
          unavailable.textContent = "Not evaluated";
          card.appendChild(unavailable);
        } else {
          const list = document.createElement("ol");
          list.className = "rec-list";
          for (const rec of result.recommendations.slice(0, 10)) {
            const movie = data.movies[String(rec.movie_id)] || {
              title: `Movie ${rec.movie_id}`,
            };
            const item = document.createElement("li");
            item.className = "rec-item";
            const name = document.createElement("span");
            name.className = "rec-title";
            name.textContent = movie.title;
            const score = document.createElement("span");
            score.className = "rec-score";
            const scoreValue = document.createElement("span");
            scoreValue.textContent = scoreLabel(key, rec.score);
            const rating = document.createElement("span");
            rating.className = rec.heldout_rating === null
              ? "heldout-rating unrated"
              : "heldout-rating";
            rating.textContent = rec.heldout_rating === null
              ? "unrated"
              : `${rec.heldout_rating}★`;
            rating.title = rec.heldout_rating === null
              ? "No rating for this movie in the selected held-out partition."
              : `Held-out rating: ${rec.heldout_rating} out of 5`;
            score.append(scoreValue, rating);
            item.append(name, score);
            list.appendChild(item);
          }
          card.appendChild(list);
        }
        grid.appendChild(card);
      }
      section.appendChild(grid);
      root.appendChild(section);
    }
    const cards = root.querySelectorAll(".model-card");
    cards.forEach((card, index) => {
      card.setAttribute("title", modelTooltip(MODEL_ORDER[index % MODEL_ORDER.length]));
    });
  }

  function drawUserMetrics(records) {
    const root = byId("user-metrics-chart");
    root.replaceChildren();
    const models = MODEL_ORDER;
    const width = 560;
    const rowHeight = 30;
    const height = 58 + models.length * rowHeight;
    const x0 = 150;
    const chartWidth = width - x0 - 54;
    const values = models.flatMap((key) =>
      ["chronological", "randomized"].map((split) =>
        records[split].models[key]?.metrics.ndcg_at_10 ?? null,
      ),
    ).filter((value) => value !== null);
    const maxValue = Math.max(.05, ...values);
    const axisMax = Math.min(1, Math.ceil(maxValue * 10) / 10);
    const svg = baseSvg(width, height, "Selected user's NDCG at 10 by model and evaluation split");
    [0, .5, 1].forEach((fraction) => {
      const x = x0 + chartWidth * fraction;
      svg.appendChild(svgNode("line", { x1: x, y1: 29, x2: x, y2: height - 12, stroke: "#edf0ec", "stroke-width": 1 }));
      axisText(svg, x, 18, (axisMax * fraction).toFixed(2));
    });
    const splitColors = { chronological: "#164b3d", randomized: "#e88854" };
    models.forEach((key, index) => {
      const centerY = 44 + index * rowHeight;
      axisText(svg, x0 - 10, centerY + 3, MODEL_CHART_LABELS[key], "end", "#495a51");
      ["chronological", "randomized"].forEach((split, splitIndex) => {
        const value = records[split].models[key]?.metrics.ndcg_at_10 ?? null;
        const y = centerY - 8 + splitIndex * 11;
        if (value === null) {
          axisText(svg, x0 + 3, y + 7, "n/a", "start");
          return;
        }
        const barWidth = chartWidth * value / axisMax;
        if (barWidth > 0) {
          svg.appendChild(svgNode("rect", {
            x: x0, y, width: Math.max(1, barWidth), height: 7, rx: 3,
            fill: splitColors[split],
          }));
        }
        axisText(svg, x0 + Math.max(4, barWidth) + 4, y + 7, value.toFixed(3), "start", "#43554b");
      });
    });
    [["chronological", "Chronological"], ["randomized", "Random"]].forEach(([split, label], index) => {
      const x = x0 + index * 125;
      svg.appendChild(svgNode("rect", { x, y: height - 5, width: 9, height: 9, rx: 2, fill: splitColors[split] }));
      axisText(svg, x + 14, height + 3, label, "start", "#68766e");
    });
    root.appendChild(svg);
    const zeroSplits = ["chronological", "randomized"].filter((split) => {
      const evaluatedModels = models.filter((key) => records[split].models[key]);
      return records[split].relevant_count > 0
        && evaluatedModels.length > 0
        && evaluatedModels.every((key) => records[split].models[key].metrics.ndcg_at_10 === 0);
    });
    byId("user-metrics-note").textContent = zeroSplits.length
      ? `NDCG is 0 for every evaluated method in ${zeroSplits.join(" and ")}: no relevant held-out movie appeared in any top-10 list.`
      : "Per-user NDCG uses this user's held-out relevant movies; n/a means no relevant held-out movies or no test evaluation.";
  }

  function drawGlobalMetrics(metric) {
    const root = byId("global-metrics-chart");
    root.replaceChildren();
    const splits = ["chronological", "randomized"];
    const values = splits.flatMap((split) =>
      MODEL_ORDER.map((key) =>
        data.splits[split].metrics[key]?.[byId("partition-select").value]?.[metric] ?? null,
      ),
    ).filter((value) => value !== null);
    const width = 560;
    const rowHeight = 36;
    const height = 56 + MODEL_ORDER.length * rowHeight;
    const svg = baseSvg(width, height, `${METRIC_LABELS[metric]} by model and evaluation split`);
    const x0 = 106;
    const chartWidth = width - x0 - 54;
    const maxValue = Math.max(.01, ...values) * 1.15;
    [0, .5, 1].forEach((tick) => {
      const x = x0 + chartWidth * tick;
      svg.appendChild(svgNode("line", { x1: x, y1: 24, x2: x, y2: height - 17, stroke: "#edf0ec", "stroke-width": 1 }));
      axisText(svg, x, 15, (maxValue * tick).toFixed(3));
    });
    MODEL_ORDER.forEach((key, index) => {
      const centerY = 42 + index * rowHeight;
      axisText(svg, x0 - 9, centerY + 8, MODEL_CHART_LABELS[key], "end", "#495a51");
      splits.forEach((split, splitIndex) => {
        const value = data.splits[split].metrics[key]?.[byId("partition-select").value]?.[metric] ?? null;
        const y = centerY - 7 + splitIndex * 12;
        if (value === null) {
          axisText(svg, x0 + 3, y + 7, "n/a", "start");
          return;
        }
        const barWidth = chartWidth * value / maxValue;
        if (barWidth > 0) {
          svg.appendChild(svgNode("rect", {
            x: x0, y, width: Math.max(1, barWidth), height: 7, rx: 3,
            fill: split === "chronological" ? "#164b3d" : "#e88854",
          }));
        }
        axisText(svg, x0 + Math.max(4, barWidth) + 3, y + 7, value.toFixed(3), "start", "#43554b");
      });
    });
    [["chronological", "Chronological"], ["randomized", "Random"]].forEach(([split, label], index) => {
      const x = x0 + index * 125;
      svg.appendChild(svgNode("rect", { x, y: height - 7, width: 9, height: 9, rx: 2, fill: split === "chronological" ? "#164b3d" : "#e88854" }));
      axisText(svg, x + 14, height + 1, label, "start", "#68766e");
    });
    root.appendChild(svg);
  }

  function drawHybrid() {
    const root = byId("hybrid-chart");
    root.replaceChildren();
    const entries = ["chronological", "randomized"].map((split) => ({
      split,
      entry: data.splits[split].metrics.hybrid,
      rows: data.splits[split].metrics.hybrid.validation_sweep || [],
    }));
    const width = 900;
    const height = 250;
    const left = 56;
    const right = 22;
    const top = 24;
    const bottom = 38;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const maxValue = Math.max(...entries.flatMap(({ rows }) => rows.map((row) => row.ndcg_at_10))) * 1.18;
    const svg = baseSvg(width, height, "Chronological and random validation NDCG at 10 across hybrid alpha values");
    [0, .25, .5, .75, 1].forEach((tick) => {
      const y = top + plotHeight * (1 - tick);
      svg.appendChild(svgNode("line", { x1: left, y1: y, x2: width - right, y2: y, stroke: "#edf0ec", "stroke-width": 1 }));
      axisText(svg, left - 8, y + 3, (maxValue * tick).toFixed(3), "end");
    });
    const colors = { chronological: "#164b3d", randomized: "#e88854" };
    entries.forEach(({ split, entry, rows }) => {
      const points = rows.map((row) => ({
        x: left + row.alpha * plotWidth,
        y: top + plotHeight * (1 - row.ndcg_at_10 / maxValue),
        row,
      }));
      const path = points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
      svg.appendChild(svgNode("path", { d: path, fill: "none", stroke: colors[split], "stroke-width": 3, "stroke-linecap": "round", "stroke-linejoin": "round" }));
      points.forEach((point) => {
        const selected = Math.abs(point.row.alpha - entry.validation.alpha) < 1e-8;
        svg.appendChild(svgNode("circle", {
          cx: point.x, cy: point.y, r: selected ? 5 : 3.5,
          fill: selected ? colors[split] : "#ffffff",
          stroke: colors[split], "stroke-width": 2,
        }));
      });
    });
    [0, .2, .4, .6, .8, 1].forEach((alpha) => {
      const x = left + alpha * plotWidth;
      axisText(svg, x, height - 12, alpha.toFixed(1));
    });
    entries.forEach(({ split }, index) => {
      const x = left + index * 145;
      svg.appendChild(svgNode("line", { x1: x, y1: height - 5, x2: x + 20, y2: height - 5, stroke: colors[split], "stroke-width": 3 }));
      axisText(svg, x + 26, height - 1, split === "chronological" ? "Chronological" : "Random 80/10/10", "start", "#68766e");
    });
    root.appendChild(svg);
    byId("hybrid-caption").textContent = "Validation-only z-score fusion; highlighted points are each split's selected setting. No hybrid test metric is available.";
  }

  function drawHistory(split) {
    const root = byId("history-chart");
    root.replaceChildren();
    const rows = data.splits[split].history_buckets;
    const buckets = ["6-20", "21-50", "51+"];
    const models = split === "randomized" ? ["mf_b", "genre"] : ["genre"];
    const width = 900;
    const height = 225;
    const left = 58;
    const right = 22;
    const top = 25;
    const bottom = 42;
    const plotHeight = height - top - bottom;
    const maxValue = Math.max(.01, ...rows.map((row) => row.ndcg_at_10)) * 1.18;
    const svg = baseSvg(width, height, "Validation NDCG at 10 by training history length");
    [0, .5, 1].forEach((tick) => {
      const y = top + plotHeight * (1 - tick);
      svg.appendChild(svgNode("line", { x1: left, y1: y, x2: width - right, y2: y, stroke: "#edf0ec", "stroke-width": 1 }));
      axisText(svg, left - 8, y + 3, (maxValue * tick).toFixed(3), "end");
    });
    const groupWidth = (width - left - right) / buckets.length;
    buckets.forEach((bucket, bucketIndex) => {
      const center = left + groupWidth * (bucketIndex + .5);
      axisText(svg, center, height - 16, bucket, "middle", "#53645b");
      models.forEach((model, modelIndex) => {
        const record = rows.find((row) => row.bucket === bucket && row.model === model);
        if (!record) return;
        const barWidth = models.length === 1 ? 36 : 31;
        const x = center + (modelIndex - (models.length - 1) / 2) * (barWidth + 6) - barWidth / 2;
        const barHeight = plotHeight * record.ndcg_at_10 / maxValue;
        svg.appendChild(svgNode("rect", {
          x, y: top + plotHeight - barHeight, width: barWidth, height: barHeight,
          rx: 5, fill: COLORS[model],
        }));
        axisText(svg, x + barWidth / 2, top + plotHeight - barHeight - 5, record.ndcg_at_10.toFixed(3));
      });
    });
    const legendY = 12;
    models.forEach((model, index) => {
      const x = width - right - (models.length - index) * 112;
      svg.appendChild(svgNode("rect", { x, y: legendY - 7, width: 9, height: 9, rx: 2, fill: COLORS[model] }));
      axisText(svg, x + 14, legendY + 1, modelLabel(model), "start", "#68766e");
    });
    root.appendChild(svg);
    byId("history-caption").textContent =
      split === "randomized" ? "MF B + Genre · random split" : "Genre only · chronological report";
  }

  function render() {
    const { partition, userId } = currentUser();
    if (!selectedRecords.chronological || !selectedRecords.randomized) return;
    renderNotice(partition);
    renderUserSummary(selectedRecords, userId);
    renderRecommendations(selectedRecords, partition);
    drawUserMetrics(selectedRecords);
    drawGlobalMetrics(byId("metric-select").value);
    drawHybrid();
    drawHistory("randomized");
  }

  async function loadAndRender() {
    try {
      const { partition, userId } = currentUser();
      selectedRecords = {};
      byId("recommendation-grid").innerHTML = '<div class="loading">Loading precomputed recommendations…</div>';
      const [chronological, randomized] = await Promise.all([
        loadUserRecord("chronological", partition, userId),
        loadUserRecord("randomized", partition, userId),
      ]);
      selectedRecords = { chronological, randomized };
      render();
    } catch (error) {
      const root = byId("recommendation-grid");
      root.replaceChildren();
      const message = document.createElement("p");
      message.className = "error-box";
      message.textContent = error.message;
      root.appendChild(message);
    }
  }

  function init() {
    fillUserList();
    loadAndRender();
    byId("partition-select").addEventListener("change", () => {
      fillUserList();
      loadAndRender();
    });
    byId("user-search").addEventListener("change", loadAndRender);
    byId("metric-select").addEventListener("change", render);
    byId("random-user").addEventListener("click", () => {
      const partition = byId("partition-select").value;
      const randomized = new Set(userKeys("randomized", partition));
      const users = userKeys("chronological", partition).filter((userId) =>
        randomized.has(userId),
      );
      if (!users.length) return;
      byId("user-search").value = users[Math.floor(Math.random() * users.length)];
      loadAndRender();
    });
  }

  function modelTooltip(key) {
    const explanations = {
      bias: "Training mean + user and movie biases; no user–movie interaction factors.",
      mf_a: "Biased matrix factorization; L2 regularization is applied once per entity-rating interaction in a mini-batch.",
      mf_b: "Biased matrix factorization; L2 regularization is applied once per distinct entity in a mini-batch.",
      genre: "Training-only, mean-centered multi-hot genre profile ranked by cosine similarity.",
      hybrid: "Per-user z-score fusion of MF B and Genre; alpha is selected on validation and evaluated once on test.",
    };
    return explanations[key];
  }

  fetch("data/site_index.json")
    .then((response) => {
      if (!response.ok) throw new Error(`Could not load demo data (${response.status}).`);
      return response.json();
    })
    .then((payload) => {
      data = payload;
      init();
    })
    .catch((error) => {
      const main = document.querySelector("main");
      const box = document.createElement("div");
      box.className = "error-box";
      box.textContent = `${error.message} Serve this folder over HTTP (for example: python -m http.server 8000) instead of opening index.html directly.`;
      main.prepend(box);
    });
})();
