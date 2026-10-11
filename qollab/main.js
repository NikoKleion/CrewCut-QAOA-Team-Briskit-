import * as game from 'dispatch_game';
import { startReport } from './report.js';

const TIE = 1e-9;
const SVG = 'http://www.w3.org/2000/svg';
// place of each site on the map, in percent of its width and height; the depot is where the crews wait.
// No line between two sites passes under a third.
const SITES = [[27, 24], [29, 76], [41, 50], [46, 15], [44, 86], [72, 26], [71, 74], [90, 49]];
const DEPOT = [7, 50];
const HOUR = 150;

const element = (id) => document.getElementById(id);
const percent = (value, digits = 2) => `${(100 * value).toFixed(digits)}%`;

function make(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}

function svg(tag, attributes = {}, text) {
  const item = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) item.setAttribute(name, value);
  if (text !== undefined) item.textContent = text;
  return item;
}

// Waiting is done on a timer, because browsers stop animation frames in a tab that is not in front.
// https://qollab.xyz/learn/docs/js-qiskit-projects: Qollab's setTimeout returns a promise. If that promise settles
// before the time has passed (a setTimeout that takes no callback), the rest of the wait is done on animation frames.
function sleep(ms) {
  return new Promise((resolve) => {
    let done = false;
    const end = performance.now() + ms;
    const finish = () => { if (!done) { done = true; resolve(); } };
    const onFrames = () => (performance.now() >= end ? finish() : requestAnimationFrame(onFrames));
    let returned;
    try { returned = setTimeout(finish, ms); } catch (error) { returned = null; }
    if (returned && typeof returned.then === 'function') {
      returned.then(() => (performance.now() >= end - 5 ? finish() : requestAnimationFrame(onFrames)), onFrames);
    }
  });
}

// seeded numbers for the order in which shots are drawn on the page: mulberry32
function seeded(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// https://qollab.xyz/learn/docs/js-qiskit-projects: call Python through .callPromising
// https://pyodide.org/en/stable/usage/api/js-api.html: toJs turns a dict into a Map by default; destroy frees the proxy
async function python(call, ...values) {
  const proxy = await call.callPromising(...values);
  const result = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return result;
}

const jobs = { used: false };
startReport(jobs);

const data = await python(game.setup);
const n = data.repairs;
const { hours, urgency, order } = data;
const plans = 2 ** n;

// the page's own scoring: run each crew through its sites in order
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

const everyScore = () => Array.from({ length: plans }, (_, plan) => score(plan));
const costs = everyScore();
const best = Math.min(...costs);
const worst = Math.max(...costs);
const bestPlan = costs.indexOf(best);
const gap = (cost) => cost / best - 1;
const better = (cost) => costs.filter((other) => other < cost - TIE).length;
const crewsOf = (plan) => Array.from({ length: n }, (_, j) => (plan >> j) & 1);
const ranked = Array.from({ length: plans }, (_, plan) => plan).sort((a, b) => costs[a] - costs[b] || a - b);
const rank = Array(plans);
ranked.forEach((plan, place) => { rank[plan] = place; });

// the same score as links: two sites on one crew cost the later one's urgency times the earlier one's hours
const links = [];
order.forEach((k, a) => order.slice(a + 1).forEach((j) => links.push([k, j, urgency[j] * hours[k]])));
const fixed = hours.reduce((sum, h, j) => sum + urgency[j] * h, 0);
const weight = links.reduce((sum, [, , c]) => sum + c, 0);
const together = (plan, k, j) => ((plan >> k) & 1) === ((plan >> j) & 1);
const paid = (plan) => links.reduce((sum, [k, j, c]) => sum + (together(plan, k, j) ? c : 0), 0);

const disagreement = Math.max(...costs.map((cost, plan) => Math.abs(cost - data.costs[plan])));
const linkGap = Math.max(...costs.map((cost, plan) => Math.abs(fixed + paid(plan) - cost)));
const counted28 = links.length === data.crewcut_gates && 2 * links.length + n === data.standard_gates;
element('selfcheck').textContent = disagreement < 1e-9 && linkGap < 1e-9 && counted28
  ? `Check passed: this page scored all ${plans} plans twice, by running each crew's schedule and by adding up links, and both equal the Python model's scores. Gates in a layer: ${data.crewcut_gates} and ${data.standard_gates}, as the circuits have.`
  : `Check FAILED: schedule against the Python model ${disagreement.toExponential(2)}, links against schedule ${linkGap.toExponential(2)}, gate counts ${counted28 ? 'agree' : 'differ'}. Do not trust the numbers below.`;

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

// ---------------------------------------------------------------- the map
const map = element('map');
const sky = svg('svg', { viewBox: '0 0 160 70', preserveAspectRatio: 'none', class: 'sky', 'aria-hidden': 'true' });
for (const d of ['M0 35 H160', 'M0 12 H160', 'M0 58 H160', 'M44 0 V70', 'M81 0 V70', 'M116 0 V70']) sky.appendChild(svg('path', { d, class: 'road' }));
for (let i = 0; i < 46; i += 1) {
  const x = (i * 37) % 164;
  const y = (i * 53) % 70;
  sky.appendChild(svg('line', { x1: x, y1: y, x2: x - 1.6, y2: y + 4.5, class: 'rain', style: `animation-delay:${(i % 9) * 0.11}s` }));
}
const routes = svg('g');
sky.appendChild(routes);
map.appendChild(sky);
const flash = make('div', undefined, 'flash');
map.appendChild(flash);
const depot = make('div', 'depot', 'depot');
depot.style.left = `${DEPOT[0]}%`;
depot.style.top = `${DEPOT[1]}%`;
map.appendChild(depot);

const choice = Array(n).fill(null);
const mostUrgent = urgency.indexOf(Math.max(...urgency));
let busy = true;
let locked = null;

const sites = hours.map((h, j) => {
  const button = make('button', undefined, 'site ok');
  button.type = 'button';
  button.style.left = `${SITES[j][0]}%`;
  button.style.top = `${SITES[j][1]}%`;
  const dot = 0.55 + 0.5 * Math.sqrt(urgency[j] / urgency[mostUrgent]);
  button.style.setProperty('--size', `${dot.toFixed(2)}`);
  button.append(make('span', String(j + 1), 'num'), make('span', `${h} h`, 'hrs'), make('span', '', 'tag'));
  button.addEventListener('click', () => {
    if (busy || locked !== null) return;
    choice[j] = choice[j] === 0 ? 1 : 0;
    redraw();
  });
  map.appendChild(button);
  return button;
});

const markers = [0, 1].map((crew) => {
  const marker = make('div', crew ? 'B' : 'A', `crew crew-${crew ? 'b' : 'a'}`);
  marker.setAttribute('aria-hidden', 'true');
  map.appendChild(marker);
  return marker;
});

// a crew stands left of the site it works on, clear of the site's crew letter; at the depot, one above and one below
function place(marker, crew, spot) {
  const home = spot === DEPOT;
  marker.style.left = home ? `${spot[0]}%` : `calc(${spot[0]}% - 2.5rem)`;
  marker.style.top = home ? `calc(${spot[1]}% ${crew ? '+' : '-'} 1.8rem)` : `calc(${spot[1]}% + 0.3rem)`;
}

function describe(j, state) {
  const crew = choice[j] === null ? 'no crew yet' : `crew ${choice[j] ? 'B' : 'A'}`;
  const what = { ok: 'power on', dark: 'dark', lit: 'power back' }[state];
  return `Site ${j + 1}: ${hours[j]} hours of work, urgency ${urgency[j].toFixed(2)} an hour${j === mostUrgent ? ', critical load' : ''}; ${what}; ${crew}`;
}

function paint(j, state, crews = choice) {
  const crew = crews[j];
  sites[j].className = `site ${state}${crew === null ? '' : crew ? ' of-b' : ' of-a'}${j === mostUrgent ? ' critical' : ''}`;
  sites[j].querySelector('.tag').textContent = crew === null ? '' : crew ? 'B' : 'A';
  sites[j].setAttribute('aria-label', describe(j, state));
  sites[j].title = `urgency ${urgency[j].toFixed(2)} an hour`;
}

async function storm() {
  for (const j of [2, 5, 0, 7, 3, 6, 1, 4]) {
    flash.classList.add('on');
    paint(j, 'dark');
    await sleep(90);
    flash.classList.remove('on');
    await sleep(170);
  }
}

// ---------------------------------------------------------------- the crews' lanes
function schedule(crews) {
  const clock = [0, 0];
  const start = Array(n).fill(null);
  const end = Array(n).fill(null);
  for (const j of order) {
    if (crews[j] === null) continue;
    start[j] = clock[crews[j]];
    clock[crews[j]] += hours[j];
    end[j] = clock[crews[j]];
  }
  return { start, end, last: Math.max(...clock) };
}

const total = hours.reduce((sum, h) => sum + h, 0);

function drawLanes(crews) {
  const target = element('lanes');
  target.replaceChildren();
  for (const crew of [0, 1]) {
    const name = crew ? 'B' : 'A';
    const jobsOf = order.filter((j) => crews[j] === crew);
    const sum = jobsOf.reduce((s, j) => s + hours[j], 0);
    target.appendChild(make('span', `Crew ${name}`, 'lane-name'));
    const lane = make('div', undefined, 'lane');
    for (const j of jobsOf) {
      const block = make('div', String(j + 1), `job crew-${name.toLowerCase()}`);
      block.style.width = `${100 * hours[j] / total}%`;
      lane.appendChild(block);
    }
    lane.appendChild(make('i', undefined, 'playhead'));
    target.appendChild(lane);
    target.appendChild(make('span', jobsOf.length ? `sites ${jobsOf.map((j) => j + 1).join(', ')} · ${sum} h` : 'nothing yet', 'lane-text'));
  }
}

const playerPlan = () => choice.reduce((plan, crew, j) => plan | (crew << j), 0);

function redraw() {
  for (let j = 0; j < n; j += 1) paint(j, 'dark');
  drawLanes(choice);
  const complete = choice.every((crew) => crew !== null);
  element('send').disabled = !complete;
  element('mine').textContent = complete ? score(playerPlan()).toFixed(2) : `${choice.filter((crew) => crew !== null).length} of ${n} sites assigned`;
}

// the clock: each hour adds the urgency of every site still dark, so the meter ends at the plan's score
async function play(crews, meter) {
  busy = true;
  const { start, end, last } = schedule(crews);
  drawLanes(crews);
  routes.replaceChildren();
  const at = [DEPOT, DEPOT];
  let dark = 0;
  for (let hour = 0; hour <= last; hour += 1) {
    for (let j = 0; j < n; j += 1) paint(j, end[j] <= hour ? 'lit' : 'dark', crews);
    for (const crew of [0, 1]) {
      const now = order.find((j) => crews[j] === crew && start[j] <= hour && hour < end[j]);
      const done = order.filter((j) => crews[j] === crew && end[j] <= hour).pop();
      place(markers[crew], crew, now !== undefined ? SITES[now] : done !== undefined ? SITES[done] : DEPOT);
      // the order of work as a line; the sky is 160 by 70 and the places are in percent
      if (now !== undefined && SITES[now] !== at[crew]) {
        routes.appendChild(svg('line', { x1: 1.6 * at[crew][0], y1: 0.7 * at[crew][1], x2: 1.6 * SITES[now][0], y2: 0.7 * SITES[now][1], class: `route route-${crew ? 'b' : 'a'}` }));
        at[crew] = SITES[now];
      }
    }
    for (const head of document.querySelectorAll('#lanes .playhead')) head.style.left = `${100 * hour / total}%`;
    element('clock').textContent = String(hour);
    meter.textContent = dark.toFixed(2);
    if (hour === last) break;
    dark += urgency.reduce((sum, u, j) => sum + (end[j] > hour ? u : 0), 0);
    await sleep(HOUR);
  }
  busy = false;
  return dark;
}

element('lucky').addEventListener('click', () => {
  if (busy || locked !== null) return;
  const random = seeded(Math.floor(performance.now()));
  for (let j = 0; j < n; j += 1) choice[j] = random() < 0.5 ? 0 : 1;
  redraw();
});

element('send').addEventListener('click', async () => {
  if (busy || locked !== null) return;
  locked = playerPlan();
  element('send').disabled = true;
  element('lucky').disabled = true;
  const reading = await play(choice, element('mine'));
  const off = Math.abs(reading - costs[locked]);
  element('hint').textContent = off < 1e-9
    ? `The meter stopped at ${reading.toFixed(2)}: exactly this plan's score. Lower is better. Is it the best you could have done?`
    : `Check FAILED: the meter stopped at ${reading} but the plan's score is ${costs[locked]}.`;
  element('scan').disabled = false;
  element('scan-count').textContent = '';
});

// ---------------------------------------------------------------- brute force
const WIDTH = 800;
const STEP = 3;
const barHeight = (cost) => 6 + 118 * (cost - best) / (worst - best);

function board(target, height) {
  const picture = svg('svg', { viewBox: `0 0 ${WIDTH} ${height}`, class: 'bars' });
  target.replaceChildren(picture);
  return picture;
}

function flag(picture, place, text, level, className) {
  const x = place * STEP + 1.2;
  picture.appendChild(svg('line', { x1: x, y1: 14 + 13 * level, x2: x, y2: 140, class: `flagline ${className}` }));
  picture.appendChild(svg('text', { x: x + 3, y: 12 + 13 * level, class: `flag ${className}` }, text));
}

element('scan').addEventListener('click', async () => {
  if (locked === null || element('scan').disabled) return;
  element('scan').disabled = true;
  const picture = board(element('landscape'), 150);
  const started = performance.now();
  everyScore();
  const elapsed = performance.now() - started;
  const bars = [];
  let lead = 0;
  for (let plan = 0; plan < plans; plan += 1) {
    const h = barHeight(costs[plan]);
    const bar = svg('rect', { x: plan * STEP, y: 140 - h, width: STEP - 0.6, height: h, class: 'bar' });
    picture.appendChild(bar);
    bars.push(bar);
    if (costs[plan] < costs[lead] - TIE) {
      bars[lead].classList.remove('lead');
      lead = plan;
    }
    bars[lead].classList.add('lead');
    if (plan % 3 === 2 || plan === plans - 1) {
      element('scan-count').textContent = `scored ${plan + 1} of ${plans} plans · best so far ${costs[lead].toFixed(2)}`;
      await sleep(16);
    }
  }
  await sleep(500);
  element('scan-count').textContent = `all ${plans} scored · sorting them from best to worst`;
  bars.forEach((bar, plan) => { bar.style.transform = `translateX(${(rank[plan] - plan) * STEP}px)`; });
  await sleep(1500);
  for (const plan of ranked.filter((p) => gap(costs[p]) < TIE)) bars[plan].classList.add('lead');
  bars[locked].classList.add('you');
  const rule = rulePlan();
  flag(picture, 0, `best ${best.toFixed(2)}`, 0, 'gold');
  flag(picture, rank[rule], `simple rule +${percent(gap(costs[rule]))}`, 1, 'grey');
  flag(picture, rank[locked], gap(costs[locked]) < TIE ? 'you: a best plan' : `you +${percent(gap(costs[locked]))}`, 2, 'white');
  element('scan-count').textContent = `all ${plans} scored · best plan on the left`;
  const mine = costs[locked];
  const ahead = better(mine);
  const verdict = element('verdict');
  verdict.hidden = false;
  verdict.textContent = ahead === 0
    ? `You found a best plan: ${mine.toFixed(2)}. No plan scores better.`
    : `Your night cost ${mine.toFixed(2)}, ${percent(gap(mine))} more than the best possible (${best.toFixed(2)}). ${ahead} of the ${plans} plans beat yours.`;
  element('search-time').textContent = elapsed < 1 ? 'under a' : `${elapsed.toFixed(1)}`;
  element('rule-note').textContent = `On this night the simple dispatch rule ends ${percent(gap(costs[rule]))} above the best plan. On ${data.rule_best_on} of the ${data.instances} test nights of this size it is already a best plan; this is the first of the ${data.instances}.`;
  element('replay').hidden = false;
  for (const id of ['links-mine', 'links-best', 'layer', 'game-run']) element(id).disabled = false;
  element('game-status').textContent = 'Simulators and noise models are free. This page needs no quantum computer.';
  buildLinks();
  drawLinks(locked);
  registers();
});

element('replay').addEventListener('click', async () => {
  if (busy) return;
  element('replay').disabled = true;
  element('best-meter').hidden = false;
  map.scrollIntoView({ behavior: 'smooth', block: 'center' });
  await sleep(600);
  await play(crewsOf(bestPlan), element('best'));
  element('replay').disabled = false;
});

// ---------------------------------------------------------------- the links
const GRAPH = [800, 350];
const spot = (j) => [SITES[j][0] * GRAPH[0] / 100, SITES[j][1] * GRAPH[1] / 100];
const heaviest = Math.max(...links.map(([, , c]) => c));
let lines = [];
let nodes = [];
let shown = 0;

function buildLinks() {
  const picture = svg('svg', { viewBox: `0 0 ${GRAPH[0]} ${GRAPH[1]}`, class: 'graph' });
  lines = links.map(([k, j, c]) => {
    const [x1, y1] = spot(k);
    const [x2, y2] = spot(j);
    const line = svg('line', { x1, y1, x2, y2 });
    line.style.setProperty('--w', `${(1.5 + 10 * c / heaviest).toFixed(2)}px`);
    picture.appendChild(line);
    return line;
  });
  nodes = hours.map((_, j) => {
    const [x, y] = spot(j);
    const node = svg('g', { transform: `translate(${x} ${y})` });
    node.append(svg('circle', { r: 18 }), svg('text', { y: 6, class: 'num' }, String(j + 1)), svg('text', { x: 24, y: -14, class: 'who' }));
    picture.appendChild(node);
    return node;
  });
  element('graph').replaceChildren(picture);
}

function drawLinks(plan) {
  shown = plan;
  const mine = plan === locked;
  element('links-mine').setAttribute('aria-pressed', String(mine));
  element('links-best').setAttribute('aria-pressed', String(plan === bestPlan));
  links.forEach(([k, j], i) => {
    lines[i].setAttribute('class', together(plan, k, j) ? `link paid of-${(plan >> k) & 1 ? 'b' : 'a'}` : 'link cut');
  });
  nodes.forEach((node, j) => {
    const crew = (plan >> j) & 1;
    node.setAttribute('class', `node of-${crew ? 'b' : 'a'}${j === mostUrgent ? ' critical' : ''}`);
    node.querySelector('.who').textContent = crew ? 'B' : 'A';
  });
  const cost = paid(plan);
  const kept = links.filter(([k, j]) => together(plan, k, j)).length;
  element('l-fixed').textContent = fixed.toFixed(2);
  element('l-paid').textContent = cost.toFixed(2);
  element('l-total').textContent = (fixed + cost).toFixed(2);
  element('l-cut').textContent = (weight - cost).toFixed(2);
  const note = `${mine ? 'Your plan' : 'The best plan'} keeps ${kept} of the ${links.length} links and cuts ${links.length - kept}. Solid lines are links kept, by crew; dashed lines are cut. Site ${mostUrgent + 1} is the critical load: its links are the heavy ones.`;
  element('links-note').textContent = note;
  element('graph').setAttribute('aria-label', `The eight sites and the ${links.length} links between them. ${note}`);
}

element('links-mine').addEventListener('click', () => lines.length && !element('links-mine').disabled && drawLinks(locked));
element('links-best').addEventListener('click', () => lines.length && !element('links-best').disabled && drawLinks(bestPlan));

element('layer').addEventListener('click', async () => {
  if (!lines.length || element('layer').disabled) return;
  const buttons = [element('layer'), element('links-mine'), element('links-best')];
  for (const button of buttons) button.disabled = true;
  for (const line of lines) line.setAttribute('class', 'link all');
  nodes.forEach((node, j) => {
    node.setAttribute('class', `node${j === mostUrgent ? ' critical' : ''}`);
    node.querySelector('.who').textContent = '';
  });
  for (let i = 0; i < links.length; i += 1) {
    const [k, j, c] = links[i];
    lines[i].setAttribute('class', 'link all fire');
    element('layer-note').textContent = `gate ${i + 1} of ${links.length}: sites ${k + 1} and ${j + 1}, link weight ${c.toFixed(2)}`;
    await sleep(120);
    lines[i].setAttribute('class', 'link all fired');
  }
  element('layer-note').textContent = `then one rotation on each of the ${n} qubits`;
  for (const node of nodes) node.classList.add('pulse');
  await sleep(1100);
  element('layer-note').textContent = `One layer: ${links.length} two qubit gates and ${n} rotations on ${n} qubits. The standard form needs ${data.standard_gates} two qubit gates on ${2 * n} qubits for the same layer.`;
  await sleep(900);
  drawLinks(shown);
  for (const button of buttons) button.disabled = false;
});

// ---------------------------------------------------------------- the two registers
function qubit(text, className) {
  return make('span', text, `q ${className}`);
}

function drawStandard(bits) {
  const target = element('reg-standard');
  target.replaceChildren();
  let valid = true;
  for (let j = 0; j < n; j += 1) {
    const a = (bits >> (2 * j)) & 1;
    const b = (bits >> (2 * j + 1)) & 1;
    const fine = a + b === 1;
    valid = valid && fine;
    const pair = make('span', undefined, `pair${fine ? '' : ' bad'}`);
    pair.append(qubit('A', a ? 'on of-a' : 'off'), qubit('B', b ? 'on of-b' : 'off'), make('small', String(j + 1)));
    target.appendChild(pair);
  }
  return valid;
}

async function registers() {
  const target = element('reg-crewcut');
  target.replaceChildren();
  for (let j = 0; j < n; j += 1) {
    const crew = (locked >> j) & 1;
    const one = make('span', undefined, 'pair');
    one.append(qubit(crew ? 'B' : 'A', crew ? 'on of-b' : 'on of-a'), make('small', String(j + 1)));
    target.appendChild(one);
  }
  const random = seeded(20261011);
  let found = 0;
  for (let k = 1; k <= 36; k += 1) {
    found += drawStandard(Math.floor(random() * 65536)) ? 1 : 0;
    element('flicker').textContent = `random patterns tried: ${k} · were a plan: ${found} · of 65,536 patterns only 256 are plans, 1 in 256`;
    await sleep(140);
  }
  let own = 0;
  for (let j = 0; j < n; j += 1) own |= ((locked >> j) & 1 ? 2 : 1) << (2 * j);
  drawStandard(own);
  element('flicker').textContent += ' · shown now: your plan';
}

// ---------------------------------------------------------------- the circuits
function counted(run) {
  const count = Array(plans).fill(0);
  let invalid = 0;
  for (const [plan, c] of run.plans) {
    if (plan < 0) invalid += c; else count[plan] += c;
  }
  return { count, invalid, shots: count.reduce((s, c) => s + c, 0) + invalid };
}

function shares(tally) {
  const share = (keep) => tally.count.reduce((s, c, plan) => s + (keep(costs[plan]) ? c : 0), 0) / tally.shots;
  return { valid: 1 - tally.invalid / tally.shots, best: share((cost) => gap(cost) < TIE),
    beat: share((cost) => cost < costs[locked] - TIE), far: share((cost) => gap(cost) >= 0.1) };
}

async function rain(target, tally, peak, className, random) {
  const picture = board(target, 112);
  const bars = ranked.map((plan, place) => {
    const bar = svg('rect', { x: place * STEP, y: 100, width: STEP - 0.6, height: 0, class: `bar ${className}${gap(costs[plan]) < TIE ? ' lead' : ''}` });
    picture.appendChild(bar);
    return bar;
  });
  picture.appendChild(svg('line', { x1: rank[locked] * STEP + 1.2, y1: 4, x2: rank[locked] * STEP + 1.2, y2: 104, class: 'flagline white' }));
  picture.appendChild(svg('text', { x: rank[locked] * STEP + 5, y: 12, class: 'flag white' }, 'you'));
  const bin = svg('rect', { x: 774, y: 100, width: 20, height: 0, class: 'bar miss' });
  picture.appendChild(bin);
  if (tally.invalid) picture.appendChild(svg('text', { x: 742, y: 12, class: 'flag grey' }, 'not a plan'));
  const draws = [];
  tally.count.forEach((c, plan) => { for (let k = 0; k < c; k += 1) draws.push(rank[plan]); });
  for (let k = 0; k < tally.invalid; k += 1) draws.push(-1);
  for (let k = draws.length - 1; k > 0; k -= 1) {
    const other = Math.floor(random() * (k + 1));
    [draws[k], draws[other]] = [draws[other], draws[k]];
  }
  const seen = Array(plans).fill(0);
  let missed = 0;
  const batch = Math.ceil(draws.length / 36);
  for (let from = 0; from < draws.length; from += batch) {
    for (const place of draws.slice(from, from + batch)) {
      if (place < 0) missed += 1; else seen[place] += 1;
    }
    bars.forEach((bar, place) => {
      const h = 96 * seen[place] / peak;
      bar.setAttribute('y', 100 - h);
      bar.setAttribute('height', h);
    });
    const h = Math.min(96, 96 * missed / peak);
    bin.setAttribute('y', 100 - h);
    bin.setAttribute('height', h);
    await sleep(60);
  }
}

function tile(big, small, className) {
  const item = make('div', undefined, `tile ${className || ''}`);
  item.append(make('b', big), make('span', small));
  return item;
}

function show(result) {
  const random = seeded(result.shots);
  const drawn = { plans: [] };
  const picks = Array(plans).fill(0);
  for (let k = 0; k < result.shots; k += 1) picks[Math.floor(random() * plans)] += 1;
  picks.forEach((c, plan) => { if (c) drawn.plans.push([plan, c]); });
  const tallies = { random: counted(drawn), crewcut: counted(result.crewcut), standard: result.standard ? counted(result.standard) : null };
  const peak = Math.max(...Object.values(tallies).filter(Boolean).map((tally) => Math.max(...tally.count)));
  const measured = { random: shares(tallies.random), crewcut: shares(tallies.crewcut), standard: tallies.standard ? shares(tallies.standard) : null };
  const chance = { best: costs.filter((cost) => gap(cost) < TIE).length / plans, beat: better(costs[locked]) / plans };
  const skipped = 'not run here';
  const shotsFor = (share) => String(Math.ceil(Math.log(0.05) / Math.log(1 - share)));
  element('rain').hidden = false;

  const caption = (m) => `${percent(m.best)} of the shots are a best plan · ${percent(m.beat)} beat your plan · ${percent(m.far)} are more than 10% worse than the best${m.valid < 1 ? ` · ${percent(1 - m.valid)} are not a plan at all` : ''}`;
  element('cap-random').textContent = caption(measured.random);
  element('cap-crewcut').textContent = caption(measured.crewcut);
  element('cap-standard').textContent = measured.standard ? caption(measured.standard)
    : 'Not run on this backend: Qollab allows one job in a run on IonQ backends. Pick the built-in simulator to see both.';
  for (const [id, m, name] of [['row-random', measured.random, 'Plans picked at random'], ['row-standard', measured.standard, 'Standard QAOA'], ['row-crewcut', measured.crewcut, 'CrewCut QAOA']]) {
    element(id).setAttribute('aria-label', m ? `${name}: shots over the 256 plans sorted from best to worst. ${caption(m)}` : `${name}: not run on this backend`);
  }

  const tiles = element('tiles');
  tiles.replaceChildren(
    tile(`${n} vs ${2 * n}`, 'qubits: CrewCut against standard QAOA', 'ours'),
    tile(`${data.crewcut_gates} vs ${data.standard_gates}`, 'two qubit gates in one layer'),
    tile(`${percent(data.exact.crewcut.best)} vs ${percent(data.exact.standard.best)}`, `shots that are a best plan, exact for the trained circuits; picking at random gives ${percent(chance.best)}`, 'ours'),
    tile(`${shotsFor(data.exact.crewcut.best)} vs ${shotsFor(data.exact.standard.best)}`, `shots for a 95% chance of a best plan; picking at random needs ${shotsFor(chance.best)}`),
  );

  const rows = [
    ['two qubit gates after compiling for this backend', String(result.crewcut.compiled), result.standard ? String(result.standard.compiled) : skipped, '-'],
    ['shots that are a valid plan, this run', percent(measured.crewcut.valid), measured.standard ? percent(measured.standard.valid) : skipped, '100.00%'],
    ['shots that are a best plan, this run', percent(measured.crewcut.best), measured.standard ? percent(measured.standard.best) : skipped, percent(measured.random.best)],
    ['the same without sampling error', percent(data.exact.crewcut.best), percent(data.exact.standard.best), percent(chance.best)],
    ['shots that beat your plan, this run', percent(measured.crewcut.beat), measured.standard ? percent(measured.standard.beat) : skipped, percent(measured.random.beat)],
    ['shots more than 10% worse than the best, this run', percent(measured.crewcut.far), measured.standard ? percent(measured.standard.far) : skipped, percent(measured.random.far)],
  ];
  const compare = element('compare').tBodies[0];
  compare.replaceChildren();
  for (const [name, ...values] of rows) {
    const row = compare.insertRow();
    const head = make('th', name);
    head.scope = 'row';
    row.appendChild(head);
    for (const value of values) row.appendChild(make('td', value));
  }
  const notes = [`Backend: ${result.backend}. ${result.shots} shots for each circuit; CrewCut job ${result.crewcut.seconds.toFixed(1)} s${result.standard ? `, standard job ${result.standard.seconds.toFixed(1)} s` : ''}.`,
    `Standard QAOA's penalty: ${data.penalty_name}. The random row is drawn by this page.`];
  for (const run of [result.crewcut, result.standard]) if (run && run.note) notes.push(run.note);
  element('run-note').textContent = notes.join(' ');

  return Promise.all([
    rain(element('row-random'), tallies.random, peak, 'random', seeded(1)),
    tallies.standard ? rain(element('row-standard'), tallies.standard, peak, 'standard', seeded(2)) : null,
    rain(element('row-crewcut'), tallies.crewcut, peak, 'crewcut', seeded(3)),
  ]);
}

const runButton = element('game-run');
const runStatus = element('game-status');
runButton.addEventListener('click', async () => {
  if (locked === null || runButton.disabled) return;
  if (jobs.used) {
    runStatus.textContent = 'IonQ backends take one job per run: press Stop, then Run, for another.';
    return;
  }
  const shots = Number(element('game-shots').value);
  runButton.disabled = true;
  runStatus.textContent = `Running ${shots} shots of each circuit...`;
  try {
    const result = await python(game.run_circuits, backend, shots);
    jobs.used = result.one_job;
    await show(result);
    runStatus.textContent = jobs.used
      ? 'Done. IonQ backends take one job per run: press Stop, then Run, for another.'
      : 'Done. Run again for a fresh sample.';
  } catch (error) {
    runStatus.textContent = `The run failed: ${error.message ?? error}`;
    console.log(String(error));
  }
  runButton.disabled = jobs.used;
});

// ---------------------------------------------------------------- the demo
// seconds from the start, the section to show, what to press there; the times are those of the video script
const SCHEDULE = [
  [0, 's-top', null],
  [10, 's-night', async () => { element('lucky').click(); await sleep(600); element('send').click(); }],
  [27, 's-worth', null],
  [37, 's-search', () => element('scan').click()],
  [52, 's-links', async () => { element('links-best').click(); await sleep(2500); element('layer').click(); }],
  [71, 's-qubits', () => registers()],
  [93, 's-helps', null],
  [108, 'tile-best', null],
  [118, 's-circuits', () => element('game-run').click()],
  [128, 's-noise', () => element('run').click()],
  [138, 's-beyond', null],
  [153, 's-advantage', null],
  [166, 's-method', null],
  [176, 's-top', null],
];
const pace = Number(new URLSearchParams(location.search).get('pace')) || 1;

async function until(test) {
  while (!test()) await sleep(100);
}

element('demo').addEventListener('click', async () => {
  const button = element('demo');
  if (button.disabled || locked !== null) return;
  button.disabled = true;
  element('demo-note').textContent = 'Playing: 2 minutes 56 seconds, in the order of the pamphlet.';
  await until(() => !busy);
  const started = performance.now();
  for (const [second, id, action] of SCHEDULE) {
    await sleep(Math.max(0, started + 1000 * second / pace - performance.now()));
    element(id).scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (action) {
      if (id === 's-search') await until(() => !element('scan').disabled);
      if (id === 's-links') await until(() => !element('layer').disabled);
      await action();
    }
  }
  element('demo-note').textContent = 'The demo has played. Every button above still works.';
});

// ---------------------------------------------------------------- start
for (let j = 0; j < n; j += 1) paint(j, 'ok');
for (const crew of [0, 1]) place(markers[crew], crew, DEPOT);
drawLanes(choice);
await sleep(700);
await storm();
busy = false;
redraw();
