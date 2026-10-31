import pathlib
p = pathlib.Path("miniapp/index.html")
s = p.read_text()
old = "showState(isLocalDev ? 'Укажите user_id для локальной разработки' : 'Откройте мини-приложение через MAX', '', false);"
new = "showState((isLocalDev ? 'Укажите user_id для локальной разработки' : 'Откройте мини-приложение через MAX') + ' [debug hasWebApp=' + (!!window.WebApp) + ' initDataLen=' + initData.length + ' origin=' + location.origin + ']', '', false);"
assert old in s, "pattern not found"
s = s.replace(old, new)
p.write_text(s)
print("patched ok")
