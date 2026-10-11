import * as page from 'page_run';

const BANDS = [[0, 0, 'best plan'], [0, 0.01, 'up to 1%'], [0.01, 0.05, '1% to 5%'], [0.05, 0.1, '5% to 10%'],
  [0.1, Infinity, 'more than 10%']];

const element = (id) => document.getElementById(id);
const status = element('status');
const button = element('run');

function cell(row, text, header = false) {
  const item = document.createElement(header ? 'th' : 'td');
  if (header) item.scope = 'row';
  item.textContent = text;
  row.appendChild(item);
}

function share(count, total) {
  return `${(100 * count / total).toFixed(2)}%`;
}

function show(result) {
  const plans = result.plans;
  const total = plans.reduce((sum, plan) => sum + plan[1], 0);
  element('report').textContent = result.lines.join('\n');
  const link = element('download');
  link.href = URL.createObjectURL(new Blob([result.record], { type: 'text/plain' }));
  link.download = result.name;
  link.textContent = `Download the record (${result.name})`;

  const bands = element('bands').tBodies[0];
  bands.replaceChildren();
  for (const [low, high, label] of BANDS) {
    const count = plans.filter(([, , , gap]) => (high === 0 ? gap <= 1e-12 : gap > Math.max(low, 1e-12) && gap <= high))
      .reduce((sum, plan) => sum + plan[1], 0);
    const row = bands.insertRow();
    cell(row, label, true);
    cell(row, String(count));
    cell(row, share(count, total));
  }

  element('plans-title').textContent = `Every plan measured (${plans.length})`;
  const body = element('plans').tBodies[0];
  body.replaceChildren();
  for (const [crews, count, cost, gap] of plans) {
    const row = body.insertRow();
    cell(row, crews, true);
    cell(row, String(count));
    cell(row, share(count, total));
    cell(row, cost.toFixed(3));
    cell(row, gap <= 1e-12 ? 'best plan' : `+${(100 * gap).toFixed(2)}%`);
  }
  element('result').hidden = false;
}

function clicked() {
  return new Promise((resolve) => { button.onclick = resolve; });
}

// jobs: shared with main.js, so that one job in a run holds for the whole page on IonQ backends
export async function startReport(jobs) {
  while (!jobs.used) {
    await clicked();
    if (jobs.used) break;
    const shots = Number(element('shots').value);
    const depth = Number(element('depth').value);
    const keep = Number(element('keep').value);
    button.disabled = true;
    status.textContent = `Running ${shots} shots at depth ${depth} with ${keep}% of the pair terms...`;
    try {
      // https://qollab.xyz/learn/docs/js-qiskit-projects: call Python through .callPromising, needed on IonQ backends
      const proxy = await page.run_page.callPromising(backend, shots, depth, keep);
      // https://pyodide.org/en/stable/usage/api/js-api.html: toJs turns a dict into a Map by default; destroy frees the proxy
      const result = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy();
      show(result);
      console.log(result.record);
      jobs.used = result.one_job;
      status.textContent = jobs.used
        ? 'Done. IonQ backends take one job per run: press Stop, then Run, for another.'
        : 'Done. Change the settings and press Run the circuit again for another run.';
    } catch (error) {
      status.textContent = `The run failed: ${error.message ?? error}`;
      console.log(String(error));
    }
    button.disabled = jobs.used;
  }
  button.disabled = true;
  status.textContent = 'IonQ backends take one job per run: press Stop, then Run, for another.';
}
