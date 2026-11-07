import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("types", [["meeting"], ["reminder", "task"], []])
def test_miniapp_sends_type_filters_to_search_before_result_limit(types):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the miniapp request regression")
    html = Path(__file__).parents[1] / "miniapp" / "index.html"
    harness = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const requests = [];
const element = () => ({classList:{toggle(){},add(){},remove(){}},
  addEventListener(){},appendChild(){},append(){},remove(){}});
const location = new URL('https://example.test/app/');
const context = vm.createContext({URL, URLSearchParams, AbortController, location,
  window:{WebApp:{initData:'signed-fixture',ready(){}}},
  document:{getElementById:element,createElement:element},setTimeout(){return 0;},
  fetch:async(url) => { requests.push(url);return {ok:true,status:200,
    json:async()=>url.includes('/v1/me')?{user_id:'1'}:[]};}});
vm.runInContext(source, context);
(async()=>{
  await vm.runInContext('start()', context);
  vm.runInContext('state.query="документы";state.types=new Set('+process.argv[2]+')',context);
  await vm.runInContext('fetchSearch()',context);
  console.log(JSON.stringify(requests.filter(url=>url.includes('/events/search'))));
})().catch(error=>{console.error(error);process.exit(1);});
"""
    result = subprocess.run(
        [node, "-e", harness, str(html), json.dumps(types)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    from urllib.parse import parse_qs, urlsplit

    query = parse_qs(urlsplit(json.loads(result.stdout)[-1]).query)
    assert query["q"] == ["документы"]
    assert query.get("type", []) == types
