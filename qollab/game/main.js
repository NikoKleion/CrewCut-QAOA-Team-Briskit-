import * as game from 'dispatch_game';

const BANDS = [[0, 0, 'best plan'], [0, 0.01, 'up to 1%'], [0.01, 0.05, '1% to 5%'], [0.05, 0.1, '5% to 10%'],
  [0.1, Infinity, 'more than 10%']];
const TIE = 1e-9;

const element = (id) => document.getElementById(id);
const percent = (value, digits = 2) => `${(100 * value).toFixed(digits)}%`;

function make(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

function cell(row, text, header = false) {
  const item = make(header ? 'th' : 'td', text);
  if (header) item.scope = 'row';
  row.appendChild(item);
}

// https://qollab.xyz/learn/docs/js-qiskit-projects: call Python through .callPromising
// https://pyodide.org/en/stable/usage/api/js-api.html: toJs turns a dict into a Map by default; destroy frees the proxy
async function python(call, ...values) {
  const proxy = await call.callPromising(...values);
  const result = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return result;
}

const data = await python(game.setup);
const n = data.repairs;
const { hours, urgency, order } = data;
const plans = 2 ** n;

// the page's own scoring: run each crew through its repairs in order
function score(plan) {
  const clock = [0, 0];
  let total = 0;
  for (const j of order) {
    const crew = (plan >> j) & 1;
    clock[crew] += hours[j];
    total += urgency[j] * clock[crew];
  }
  return total;
}

function everyScore() {
  return Array.from({ length: plans }, (_, plan) => score(plan));
}

const costs = everyScore();
const best = Math.min(...costs);
const gap = (cost) => cost / best - 1;
const better = (cost) => costs.filter((other) => other < cost - TIE).length;
const letters = (plan) => Array.from({ length: n }, (_, j) => ((plan >> j) & 1 ? 'B' : 'A')).join('');

const disagreement = Math.max(...costs.map((cost, plan) => Math.abs(cost - data.costs[plan])));
element('selfcheck').textContent = disagreement < 1e-9
  ? `Check passed: this page scored all ${plans} plans by running each crew's schedule, and every score equals the Python model's (largest difference ${disagreement.toExponential(1)}).`
  : `Check FAILED: the page's scores differ from the Python model's by up to ${disagreement.toExponential(2)}. Do not trust the numbers below.`;

function rulePlan() {
  const load = [0, 0];
  let plan = 0;
  for (const j of order) {
    const crew = load[0] <= load[1] ? 0 : 1;
    load[crew] += hours[j];
    plan |= crew << j;
  }
  return plan;
}

function drawLanes(target, crews) {
  target.replaceChildren();
  const longest = hours.reduce((sum, h) => sum + h, 0);
  for (const crew of [0, 1]) {
    const name = crew ? 'B' : 'A';
    const jobs = order.filter((j) => crews[j] === crew);
    const total = jobs.reduce((sum, j) => sum + hours[j], 0);
    target.appendChild(make('span', `Crew ${name}`, 'lane-name'));
    const lane = make('div', undefined, 'lane');
    for (const j of jobs) {
      const block = make('div', String(j + 1), `job crew-${name.toLowerCase()}`);
      block.style.width = `${100 * hours[j] / longest}%`;
      block.title = `repair ${j + 1}: ${hours[j]} h, urgency ${urgency[j].toFixed(2)}`;
      lane.appendChild(block);
    }
    target.appendChild(lane);
    target.appendChild(make('span', jobs.length ? `repairs ${jobs.map((j) => j + 1).join(', ')}; ${total} h` : 'no repairs', 'lane-text'));
  }
}

const crewsOf = (plan) => Array.from({ length: n }, (_, j) => (plan >> j) & 1);
const choice = Array(n).fill(null);
const mostUrgent = urgency.indexOf(Math.max(...urgency));

const form = element('repairs');
for (let j = 0; j < n; j += 1) {
  const group = make('fieldset');
  const legend = make('legend', `Repair ${j + 1}`);
  group.appendChild(legend);
  group.appendChild(make('span', `${hours[j]} h, urgency ${urgency[j].toFixed(2)}${j === mostUrgent ? ' (critical load)' : ''}`, 'facts'));
  for (const crew of [0, 1]) {
    const label = make('label');
    const input = make('input');
    input.type = 'radio';
    input.name = `repair-${j}`;
    input.value = String(crew);
    input.addEventListener('change', () => { choice[j] = crew; redraw(); });
    label.append(input, ` Crew ${crew ? 'B' : 'A'}`);
    group.appendChild(label);
  }
  form.appendChild(group);
}

function playerPlan() {
  return choice.reduce((plan, crew, j) => plan | (crew << j), 0);
}

function redraw() {
  drawLanes(element('timeline'), choice);
  const complete = choice.every((crew) => crew !== null);
  element('score').textContent = complete ? score(playerPlan()).toFixed(2) : `${choice.filter((crew) => crew !== null).length} of ${n} repairs assigned`;
  element('lock').disabled = !complete;
}
redraw();

let locked = null;

function reveal() {
  locked = playerPlan();
  for (const input of form.querySelectorAll('input')) input.disabled = true;
  element('lock').disabled = true;

  const started = performance.now();
  const table = everyScore();
  const elapsed = performance.now() - started;
  element('search-time').textContent = elapsed < 1 ? 'under 1' : elapsed.toFixed(1);

  const bestPlan = table.indexOf(Math.min(...table));
  const mine = costs[locked];
  const ahead = better(mine);
  element('verdict').textContent = ahead === 0
    ? `You found a best plan: ${mine.toFixed(2)}. No plan scores better.`
    : `Your plan scores ${mine.toFixed(2)}, ${percent(gap(mine))} above the best plan (${best.toFixed(2)}). ${ahead} of the ${plans} plans score better.`;

  const body = element('standings').tBodies[0];
  body.replaceChildren();
  for (const [name, plan] of [['best plan', bestPlan], ['your plan', locked], ['simple rule: next repair to the crew that is free first', rulePlan()]]) {
    const row = body.insertRow();
    cell(row, name, true);
    cell(row, letters(plan));
    cell(row, costs[plan].toFixed(2));
    cell(row, gap(costs[plan]) < TIE ? 'best plan' : `+${percent(gap(costs[plan]))}`);
    cell(row, String(better(costs[plan])));
  }
  drawLanes(element('best-timeline'), crewsOf(bestPlan));
  element('rule-note').textContent = `On this instance the simple rule is ${percent(gap(costs[rulePlan()]))} above the best plan. On ${data.rule_best_on} of the ${data.instances} test instances of this size it is already a best plan; this is the first of the ${data.instances}.`;
  element('search').hidden = false;
  element('circuits').hidden = false;
  element('search-title').focus?.();
}
element('lock').addEventListener('click', reveal);

function measures(run) {
  const shots = run.plans.reduce((sum, [, count]) => sum + count, 0);
  const valid = run.plans.filter(([plan]) => plan >= 0);
  const count = (keep) => valid.filter(([plan]) => keep(costs[plan])).reduce((sum, [, c]) => sum + c, 0);
  return { shots, valid: count(() => true) / shots, best: count((cost) => gap(cost) < TIE) / shots,
    beat: count((cost) => cost < costs[locked] - TIE) / shots,
    band: BANDS.map(([low, high]) => count((cost) => (high === 0 ? gap(cost) < TIE : gap(cost) >= Math.max(low, TIE) && gap(cost) < high)) / shots) };
}

function show(result) {
  const crewcut = measures(result.crewcut);
  const standard = result.standard ? measures(result.standard) : null;
  const skipped = 'not run here';
  const chance = { best: costs.filter((cost) => gap(cost) < TIE).length / plans, beat: better(costs[locked]) / plans };
  const shotsFor = (share) => String(Math.ceil(Math.log(0.05) / Math.log(1 - share)));
  const rows = [
    ['qubits', String(result.crewcut.qubits), String(2 * n), '-'],
    ['pair gates in one layer', String(data.crewcut_gates), String(data.standard_gates), '-'],
    ['two-qubit gates after compiling for this backend', String(result.crewcut.compiled), standard ? String(result.standard.compiled) : skipped, '-'],
    ['shots that are a valid plan', percent(crewcut.valid), standard ? percent(standard.valid) : skipped, '100.00%'],
    ['shots that are a best plan, this run', percent(crewcut.best), standard ? percent(standard.best) : skipped, percent(chance.best)],
    ['the same without sampling error (trained circuit, exact)', percent(data.exact.crewcut.best), percent(data.exact.standard.best), percent(chance.best)],
    ['shots needed for a 95% chance of a best plan (from the exact share)', shotsFor(data.exact.crewcut.best), shotsFor(data.exact.standard.best), shotsFor(chance.best)],
    ['shots that beat your plan, this run', percent(crewcut.beat), standard ? percent(standard.beat) : skipped, percent(chance.beat)],
  ];
  const compare = element('compare').tBodies[0];
  compare.replaceChildren();
  for (const [name, ...values] of rows) {
    const row = compare.insertRow();
    cell(row, name, true);
    for (const value of values) cell(row, value);
  }

  const bands = element('bands').tBodies[0];
  bands.replaceChildren();
  BANDS.forEach(([low, high, label], i) => {
    const row = bands.insertRow();
    cell(row, label, true);
    cell(row, percent(crewcut.band[i]));
    cell(row, standard ? percent(standard.band[i]) : skipped);
    cell(row, percent(costs.filter((cost) => (high === 0 ? gap(cost) < TIE : gap(cost) >= Math.max(low, TIE) && gap(cost) < high)).length / plans));
  });
  const invalid = bands.insertRow();
  cell(invalid, 'not a valid plan', true);
  cell(invalid, percent(1 - crewcut.valid));
  cell(invalid, standard ? percent(1 - standard.valid) : skipped);
  cell(invalid, '0.00%');

  const notes = [`Backend: ${result.backend}. ${result.shots} shots for each circuit.`,
    `Crew-Cut job ${result.crewcut.seconds.toFixed(1)} s${standard ? `, standard job ${result.standard.seconds.toFixed(1)} s` : ''}.`,
    `Standard QAOA's penalty: ${data.penalty_name}.`];
  if (!standard) notes.push('Standard QAOA was not run: Qollab allows one job in a run on IonQ backends. Its exact figures are shown instead. Pick the built-in simulator to run both.');
  for (const run of [result.crewcut, result.standard]) if (run && run.note) notes.push(run.note);
  element('run-note').textContent = notes.join(' ');
  element('result').hidden = false;
}

const button = element('run');
const status = element('status');

function clicked() {
  return new Promise((resolve) => { button.onclick = resolve; });
}

let oneJob = false;
while (!oneJob) {
  await clicked();
  const shots = Number(element('shots').value);
  button.disabled = true;
  status.textContent = `Running ${shots} shots of each circuit...`;
  try {
    const result = await python(game.run_circuits, backend, shots);
    show(result);
    oneJob = result.one_job;
    status.textContent = oneJob
      ? 'Done. IonQ backends take one job per run: press Stop, then Run, for another.'
      : 'Done. Press Run the circuits again for a fresh sample.';
  } catch (error) {
    status.textContent = `The run failed: ${error.message ?? error}`;
    console.log(String(error));
  }
  button.disabled = oneJob;
}
