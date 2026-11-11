import shutil
import subprocess
from pathlib import Path

import pytest


def test_miniapp_rechecks_conflict_after_changing_time():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the miniapp request regression")
    html = Path(__file__).parents[1] / "miniapp" / "index.html"
    harness = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = new Map();
function getElementById(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: '', disabled: false, listeners: {},
    classList: {toggle(){}, add(){}, remove(){}},
    addEventListener(event, fn) { this.listeners[event] = fn; },
    appendChild(){}, replaceChildren(){}, focus(){}, reportValidity(){ return true; },
    showModal(){}, close(){}, reset(){},
  });
  return nodes.get(id);
}
const calls = [];
const context = vm.createContext({
  URL, URLSearchParams, AbortController, crypto: require('node:crypto'),
  location: new URL('https://example.test/app/'),
  window: {WebApp: {initData: 'signed-fixture'}}, setTimeout(){},
  document: {getElementById, createElement(){ return {}; }},
  fetch: async (url, options) => {
    if (options.method !== 'POST') {
      return {ok: true, json: async () => url.includes('/v1/me') ? {user_id: '1'} : []};
    }
    calls.push(JSON.parse(options.body));
    if (calls.length >= 3) return {ok: true, json: async () => ({})};
    return {ok: false, status: 409, json: async () => ({detail: {
      confirmation: 'snapshot-' + calls.length,
      conflicts: [{name: 'Занято', start_at: '2030-11-11T10:00Z', end_at: '2030-11-11T11:00Z'}],
    }})};
  },
});
vm.runInContext(source, context);
(async () => {
  await vm.runInContext('start()', context);
  const order = vm.runInContext(`
    const today = periodRange('today').from;
    sortCalendarEntries([
      {id: 'undated', when: null},
      {id: 'tomorrow', when: periodRange('tomorrow').from},
      {id: 'yesterday', when: new Date(today.getTime() - 3600000)},
      {id: 'today', when: today},
    ], 'all').map(event => event.id).join(',')
  `, context);
  assert.equal(order, 'today,tomorrow,yesterday,undated');
  getElementById('createOpen').listeners.click();
  getElementById('createType').value = 'meeting';
  getElementById('createType').listeners.change();
  getElementById('createName').value = 'Встреча';
  getElementById('createDate').value = '2030-11-11';
  getElementById('createTime').value = '10:00';
  await vm.runInContext('saveEntry(null)', context);
  assert.equal(vm.runInContext('entrySnapshot', context), 'snapshot-1');
  getElementById('createChange').listeners.click();
  getElementById('createTime').value = '10:30';
  getElementById('createForm').listeners.input();
  assert.equal(vm.runInContext('entrySnapshot', context), null);
  await vm.runInContext('saveEntry(null)', context);
  await vm.runInContext('saveEntry(entrySnapshot)', context);
  assert.deepEqual(calls.map(call => call.confirmation), [null, null, 'snapshot-2']);
  assert.notEqual(calls[0].request_id, calls[1].request_id);
  assert.equal(calls[1].request_id, calls[2].request_id);

  getElementById('createOpen').listeners.click();
  getElementById('createType').value = 'task';
  getElementById('createType').listeners.change();
  getElementById('createName').value = 'Сделать ДЗ';
  getElementById('createDate').value = '2030-11-16';
  getElementById('createTime').value = '';
  assert.equal(getElementById('createTime').required, false);
  await vm.runInContext('saveEntry(null)', context);
  assert.equal(calls[3].all_day, true);
  assert.equal(calls[3].start_at, '2030-11-16T00:00:00');
  const date = vm.runInContext(`
    const task = normalize({all_day: true, start_at: '2030-11-16T00:00:00+03:00'});
    [task.dateOnly, toDateOnly(task.when)].join(',')
  `, context);
  assert.equal(date, 'true,2030-11-16');
})().catch(error => { console.error(error); process.exit(1); });
"""
    subprocess.run([node, "-e", harness, str(html)], check=True, timeout=15)
