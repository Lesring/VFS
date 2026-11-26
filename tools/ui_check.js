// 用 Chrome DevTools Protocol 驱动页面，验证前端交互链路（不依赖 playwright）。
// 用法: node tools/ui_check.js http://127.0.0.1:8931
const http = require("http");
const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://127.0.0.1:8931";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PORT = 9223;
const OUT = path.join(__dirname, "..", "data");

const getJSON = (url) =>
  new Promise((res, rej) => {
    http.get(url, (r) => {
      let b = "";
      r.on("data", (c) => (b += c));
      r.on("end", () => {
        try { res(JSON.parse(b)); } catch (e) { rej(e); }
      });
    }).on("error", rej);
  });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const userDir = path.join(OUT, "chrome-profile");
  const child = execFile(CHROME, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${userDir}`,
    "--window-size=1500,1200", "about:blank",
  ]);
  child.on("error", (e) => { console.error("chrome launch failed", e.message); process.exit(1); });

  let targets = null;
  for (let i = 0; i < 40; i++) {
    try { targets = await getJSON(`http://127.0.0.1:${PORT}/json/list`); break; }
    catch { await sleep(300); }
  }
  if (!targets) { console.error("CDP not reachable"); child.kill(); process.exit(1); }

  const page = targets.find((t) => t.type === "page");
  let id = 0;
  const pending = new Map();
  const events = [];
  // Node 22+ 内置 WebSocket 客户端，不需要第三方 ws 包
  const WS = globalThis.WebSocket;
  if (!WS) { console.error("当前 Node 不支持内置 WebSocket（需要 22+）"); child.kill(); process.exit(1); }
  const ws = new WS(page.webSocketDebuggerUrl);
  let wsOpen = false;
  ws.onopen = () => { wsOpen = true; };
  ws.onmessage = null;
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
    else if (m.method) events.push(m);
  };
  for (let i = 0; i < 40 && !wsOpen; i++) await sleep(150);

  const send = (method, params = {}) =>
    new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });

  await send("Runtime.enable");
  await send("Log.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: BASE });
  await sleep(1500);

  const errs = events
    .filter((e) => e.method === "Runtime.exceptionThrown" || (e.method === "Log.entryAdded" && e.params.entry.level === "error"))
    .map((e) => JSON.stringify(e.params).slice(0, 300));
  if (errs.length) console.log("页面加载错误:", errs);

  // 点击"用内置样例试跑"
  const clicked = await send("Runtime.evaluate", {
    expression: `(() => {
      const b = document.querySelector('#demo');
      if (!b) return 'no-demo-button';
      b.click();
      return 'clicked';
    })()`, returnByValue: true,
  });
  console.log("click demo:", clicked.result.value);

  // 轮询直到右侧出现"结论"卡片
  let ok = false;
  for (let i = 0; i < 60; i++) {
    await sleep(1000);
    const st = await send("Runtime.evaluate", {
      expression: `(() => {
        const r = document.querySelector('#right');
        const t = r ? r.innerText : '';
        return JSON.stringify({
          hasAnswer: t.includes('结论'),
          hasFrames: document.querySelectorAll('#right .fr').length,
          hasEvidence: t.includes('证据表'),
          hasTrace: t.includes('推理链'),
          len: t.length,
        });
      })()`, returnByValue: true,
    });
    const v = JSON.parse(st.result.value);
    if (v.hasAnswer) {
      console.log("渲染结果:", v);
      ok = v.hasFrames > 0 && v.hasEvidence && v.hasTrace;
      break;
    }
  }

  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  const p = path.join(OUT, "ui_result.png");
  fs.writeFileSync(p, Buffer.from(shot.data, "base64"));
  console.log("screenshot:", p);

  const finalErrs = events
    .filter((e) => e.method === "Runtime.exceptionThrown")
    .map((e) => (e.params.exceptionDetails?.exception?.description || "").slice(0, 200));
  if (finalErrs.length) console.log("运行期异常:", finalErrs);

  ws.close();
  child.kill();
  console.log(ok && !finalErrs.length ? "UI 验证通过" : "UI 验证未通过");
  process.exit(ok && !finalErrs.length ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
