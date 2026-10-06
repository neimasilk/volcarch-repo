"""Minimal direct CDP client for one tab of the PI's logged-in browser (port 9333).

Playwright's connect_over_cdp hangs on this Chrome build once browser_ui targets exist, so each tab is addressed through
its own websocket. Read-only helpers: text, links, goto (navigation only).

usage: python rawcdp.py <url-substring> text [maxchars]
       python rawcdp.py <url-substring> goto <url>
       python rawcdp.py <url-substring> links [filter-regex]
       python rawcdp.py <url-substring> eval "<js expression>"
"""
import io, json, re, sys, time, urllib.request

import websocket

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def tabs():
    return [t for t in json.load(urllib.request.urlopen("http://localhost:9333/json/list", timeout=8)) if t.get("type") == "page"]


class Tab:
    def __init__(self, sub):
        m = [t for t in tabs() if sub in t.get("url", "")]
        if not m:
            raise SystemExit("no tab with %r; tabs: %s" % (sub, [t["url"][:60] for t in tabs()]))
        self.ws = websocket.create_connection(m[0]["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.n:
                if "error" in r:
                    raise RuntimeError(r["error"])
                return r.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def goto(self, url, wait=6.0):
        self.call("Page.navigate", url=url)
        time.sleep(wait)
        for _ in range(20):
            if self.js("document.readyState") == "complete":
                break
            time.sleep(1)
        return self.js("location.href"), self.js("document.title")

    def text(self):
        return self.js("document.body ? document.body.innerText : ''") or ""

    def links(self):
        return self.js("Array.from(document.querySelectorAll('a,button,input[type=submit],input[type=button]')).map(e => "
                       "[(e.innerText||e.value||'').trim().slice(0,80), e.getAttribute('title')||'', e.href||e.getAttribute('onclick')||''])") or []


def scrub(t):
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "<email>", t)


if __name__ == "__main__":
    sub, action = sys.argv[1], sys.argv[2]
    arg = sys.argv[3] if len(sys.argv) > 3 else None
    tab = Tab(sub)
    if action == "goto":
        print(tab.goto(arg))
    elif action == "text":
        print(scrub(tab.text())[: int(arg) if arg else 6000])
    elif action == "links":
        for tx, ti, h in tab.links():
            line = f"{tx} | {ti} | {h}"
            if not arg or re.search(arg, line, re.I):
                print("  ", scrub(line)[:260])
    elif action == "eval":
        print(tab.js(arg))
