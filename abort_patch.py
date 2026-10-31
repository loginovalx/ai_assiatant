import pathlib
p = pathlib.Path("miniapp/index.html")
s = p.read_text()
old = "AbortSignal.timeout(20000)"
count = s.count(old)
assert count > 0, "pattern not found"
# AbortSignal.timeout() is unsupported in older WebKit (pre Safari 16) —
# MAX's in-app webview appears to be one such case, causing a silent
# synchronous throw before fetch() is even called (no request ever
# reaches the server). Replace with a manual AbortController timeout,
# which works everywhere.
new = "(() => { const c = new AbortController(); setTimeout(() => c.abort(), 20000); return c.signal; })()"
s = s.replace(old, new)
p.write_text(s)
print(f"patched {count} occurrence(s)")
