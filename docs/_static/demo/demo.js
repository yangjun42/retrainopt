"use strict";

const policies = [
  {id: "optimized", name: "Optimized", color: "#08775d"},
  {id: "periodic", name: "Every 6 steps", color: "#986007"},
  {id: "always", name: "Always", color: "#b54b43"},
  {id: "never", name: "Never", color: "#6a7685"},
];
const byId = id => document.getElementById(id);
const number = (value, digits = 2) => value.toLocaleString("en", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const control = {scenario: byId("scenario"), quality: byId("quality"), training: byId("training")};
let data;

function showError(error) {
  const status = byId("status");
  status.classList.add("error");
  status.textContent = "The example data could not be loaded. Reload the page or open the demo guide above.";
  console.error("RetrainOpt demo:", error);
}

function renderTimeline(result) {
  const left = 130, width = 620, rowHeight = 51;
  const x = step => left + step * width / (data.steps - 1);
  let svg = '<title>When and how much to retrain</title><desc>Circles at step zero show the existing model. Bars show retraining; their height and label give the percentage of available data used.</desc>';
  for (const step of [0, 4, 8, 12, 16, 20, 23]) {
    svg += `<line class="axis" x1="${x(step)}" x2="${x(step)}" y1="10" y2="225" stroke-dasharray="2 5"/><text x="${x(step)}" y="245" text-anchor="middle">${step}</text>`;
  }
  policies.forEach((policy, row) => {
    const baseline = 52 + row * rowHeight;
    const plan = result.policies[policy.id];
    svg += `<text class="row-label" x="0" y="${baseline - 12}" style="fill:${policy.color}">${policy.name}</text><line class="axis" x1="${left}" x2="${left + width}" y1="${baseline}" y2="${baseline}"/>`;
    svg += `<circle cx="${left}" cy="${baseline - 3}" r="4" fill="white" stroke="${policy.color}" stroke-width="2"><title>Existing model; no training charge</title></circle>`;
    plan.actions.forEach((samples, step) => {
      if (!samples) return;
      const percent = 100 * samples / data.dataset_size;
      const height = 38 * samples / data.dataset_size;
      svg += `<rect x="${x(step) - 6}" y="${baseline - height}" width="12" height="${height}" rx="2" fill="${policy.color}"><title>Step ${step}: retrain with ${percent}% of data (${samples} samples)</title></rect>`;
      if (policy.id === "optimized") svg += `<text class="step-value" x="${x(step)}" y="${baseline - height - 5}" text-anchor="middle">${percent}%</text>`;
    });
  });
  svg += '<text x="130" y="267">Time step</text><text x="750" y="267" text-anchor="end">○ Existing model at step 0</text>';
  byId("timeline").innerHTML = svg;
}

function renderStream(scenario) {
  const left = 36, top = 18, width = 372, height = 155;
  const x = step => left + width * step / (data.steps - 1);
  const y = value => top + height * (1 - value);
  let svg = `<title>${scenario.name}: synthetic regime over time</title>`;
  for (const value of [0, 0.5, 1]) {
    svg += `<line class="axis" x1="${left}" x2="${left + width}" y1="${y(value)}" y2="${y(value)}"/><text x="${left - 8}" y="${y(value) + 4}" text-anchor="end">${value}</text>`;
  }
  const line = scenario.regime.map((value, step) => `${step ? "L" : "M"}${x(step)},${y(value)}`).join(" ");
  svg += `<path d="${line} L${left + width},${top + height} L${left},${top + height} Z" fill="#e4f3eb"/><path d="${line}" fill="none" stroke="#08775d" stroke-width="3" stroke-linejoin="round"/>`;
  scenario.regime.forEach((value, step) => {
    svg += `<circle cx="${x(step)}" cy="${y(value)}" r="2.5" fill="#08775d"><title>Step ${step}: ${number(value)}</title></circle>`;
  });
  for (const step of [0, 8, 16, 23]) svg += `<text x="${x(step)}" y="197" text-anchor="middle">${step}</text>`;
  svg += '<text x="36" y="218">Time step</text>';
  byId("stream").innerHTML = svg;
}

function renderBreakdown(result) {
  const left = 5, width = 354;
  const max = Math.max(...policies.map(policy => result.policies[policy.id].total_cost));
  const scale = cost => cost / max * width;
  let svg = '<title>Cost breakdown in demo objective units</title><desc>Dark green is training cost; light green is total raw loss multiplied by the selected quality weight.</desc>';
  policies.forEach((policy, row) => {
    const value = result.policies[policy.id], top = row * 48 + 7;
    const trainingWidth = scale(value.training_cost), lossWidth = scale(value.weighted_loss_cost);
    svg += `<text x="${left}" y="${top + 5}">${policy.name}</text><rect x="${left}" y="${top + 13}" width="${trainingWidth}" height="14" rx="2" fill="#08775d"><title>Training: ${number(value.training_cost)}</title></rect><rect x="${left + trainingWidth}" y="${top + 13}" width="${lossWidth}" height="14" rx="2" fill="#b9d9cc"><title>Weighted loss: ${number(value.weighted_loss_cost)}</title></rect><text class="total-label" x="${left + trainingWidth + lossWidth + 8}" y="${top + 25}">${number(value.total_cost)}</text>`;
  });
  svg += '<text x="5" y="215">Total cost = both parts together</text>';
  byId("breakdown").innerHTML = svg;
}

function render() {
  const scenario = data.scenarios.find(item => item.id === control.scenario.value);
  const quality = data.quality_weights[Number(control.quality.value)];
  const training = data.training_multipliers[Number(control.training.value)];
  const result = data.cases.find(item => item.scenario === scenario.id && item.quality_weight === quality && item.training_multiplier === training);
  if (!result) throw new Error("Missing precomputed scenario.");
  byId("quality-value").textContent = `${number(quality, 1)}×`;
  byId("training-value").textContent = `${number(training, 1)}×`;
  control.quality.setAttribute("aria-valuetext", `${quality} times raw prediction loss`);
  control.training.setAttribute("aria-valuetext", `${training} times base training price`);
  byId("scenario-description").textContent = scenario.description;
  const optimized = result.policies.optimized;
  const averageVolume = optimized.retrain_count ? optimized.total_samples / optimized.retrain_count / data.dataset_size * 100 : 0;
  const summary = optimized.retrain_count
    ? `The optimized plan retrains ${optimized.retrain_count} ${optimized.retrain_count === 1 ? "time" : "times"}, using ${number(averageVolume, 0)}% of available data per fit on average.`
    : "The optimized plan keeps the initial model for all 24 steps. Here, retraining costs more than the loss it avoids.";
  byId("status").textContent = `${scenario.name} · Quality weight ${quality}× · Training price ${training}×`;
  byId("schedule-summary").textContent = summary;
  byId("cards").innerHTML = policies.map(policy => {
    const plan = result.policies[policy.id];
    return `<article class="card ${policy.id}"><h3 class="card-label" style="color:${policy.color}">${policy.name}</h3><p class="card-value">${number(plan.total_cost)}<span class="card-unit">cost</span></p><p class="card-note">${plan.retrain_count} retrains<br>${number(plan.raw_loss / data.steps, 3)} average raw loss</p></article>`;
  }).join("");
  byId("numbers").innerHTML = policies.map(policy => {
    const plan = result.policies[policy.id];
    return `<tr><th scope="row">${policy.name}</th><td>${number(plan.training_cost)}</td><td>${number(plan.raw_loss)}</td><td>${number(plan.weighted_loss_cost)}</td><td>${number(plan.total_cost)}</td><td>${plan.retrain_count}</td><td>${plan.total_samples}</td></tr>`;
  }).join("");
  document.querySelectorAll(".presets button").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.quality === control.quality.value && button.dataset.training === control.training.value));
  });
  renderTimeline(result);
  renderStream(scenario);
  renderBreakdown(result);
}

async function initialize() {
  const response = await fetch("data.json");
  if (!response.ok) throw new Error(`Data request failed: ${response.status}`);
  data = await response.json();
  if (data.schema_version !== 1 || data.cases.length !== 60) throw new Error("Unsupported demo data.");
  byId("version").textContent = `RetrainOpt ${data.package_version}`;
  control.scenario.innerHTML = data.scenarios.map(item => `<option value="${item.id}">${item.name}</option>`).join("");
  Object.values(control).forEach(input => {
    input.disabled = false;
    input.addEventListener("input", () => { try { render(); } catch (error) { showError(error); } });
  });
  document.querySelectorAll(".presets button").forEach(button => {
    button.disabled = false;
    button.addEventListener("click", () => {
      control.quality.value = button.dataset.quality;
      control.training.value = button.dataset.training;
      render();
    });
  });
  render();
}

initialize().catch(showError);
