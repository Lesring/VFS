// 验证配置面板：打开 → 渲染字段 → 改值保存 → 检查生效。
const http = require("http");
const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://127.0.0.1:8931";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PORT = 9224;
const OUT = path.join(__dirname, "..", "data");

const getJSON = (url) => new Promise((res, rej) => {
  http.get(url, (r) => { let b = ""; r.on("data", c => b += c); r.on("end", () => { try { res(JSON.parse(b)); } catch (e) { rej(e); } }); }).on("error", rej);
});
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  const child = execFile(CHROME, ["--headless=new", "--disable-gpu", "--hide-scrollbars",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${path.join(OUT, "cmp2")}`,
    "--window-size=1400,1400", "about:blank"]);
  child.on("error", e => { console.error("chrome fail", e.message); process.exit(1); });

  let targets = null;
  for (let i = 0; i < 40; i++) {
    try { targets = await getJSON(`http://127.0.0.1:${PORT}/json/list`); break; } catch { await sleep(300); }
  }
  const page = targets.find(t => t.type === "page");
  let id = 0; const pending = new Map(); const events = [];
  const ws = new globalThis.WebSocket(page.webSocketDebuggerUrl);
  let open = false;
  ws.onopen = () => { open = true; };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
    else if (m.method) events.push(m);
  };
  for (let i = 0; i < 40 && !open; i++) await sleep(150);
  const send = (method, params = {}) => new Promise(res => {
    const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
  });

  await send("Runtime.enable"); await send("Page.enable");
  await send("Page.navigate", { url: BASE });
  await sleep(2000);

  // 打开配置面板
  await send("Runtime.evaluate", { expression: "document.querySelector('#openCfg').click()" });
  await sleep(2500);

  const state = await send("Runtime.evaluate", {
    expression: `JSON.stringify({
      modalShown: document.querySelector('#cfgModal').classList.contains('show'),
      groups: [...document.querySelectorAll('#cfgBody .grp > h4')].map(e => e.textContent),
      nFields: document.querySelectorAll('#cfgBody [data-key]').length,
      hasSelect: !!document.querySelector('#cfgBody select'),
      hasSw: !!document.querySelector('#cfgBody .sw input'),
      hasPassword: !!document.querySelector('#cfgBody input[type=password]'),
      healthItems: document.querySelectorAll('#healthGrid .hitem').length,
      regionTag: (document.querySelector('#regionTag')||{}).textContent,
    })`, returnByValue: true,
  });
  const v = JSON.parse(state.result.value);
  console.log("配置面板:", v);

  // 截图
  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  fs.writeFileSync(path.join(OUT, "ui_settings.png"), Buffer.from(shot.data, "base64"));

  // 滚动截一张面板中段
  await send("Runtime.evaluate", { expression: "document.querySelector('#cfgModal').scrollTop = 900" });
  await sleep(600);
  const shot2 = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(OUT, "ui_settings2.png"), Buffer.from(shot2.data, "base64"));

  // 改一个字段并保存
  await send("Runtime.evaluate", {
    expression: `(() => {
      const el = document.querySelector('[data-key="vlm.max_frames"]');
      if (el) el.value = '9';
      const sel = document.querySelector('[data-key="retrieval.region_strategy"]');
      if (sel) sel.value = 'overseas_first';
      document.querySelector('#saveCfg').click();
      return 'clicked';
    })()`, returnByValue: true,
  });
  await sleep(2500);
  const after = await send("Runtime.evaluate", {
    expression: "JSON.stringify({msg:(document.querySelector('#cfgMsg')||{}).textContent})",
    returnByValue: true,
  });
  console.log("保存后:", after.result.value);

  const errs = events.filter(e => e.method === "Runtime.exceptionThrown")
    .map(e => (e.params.exceptionDetails?.exception?.description || "").slice(0, 200));
  if (errs.length) console.log("异常:", errs);

  ws.close(); child.kill();
  const ok = v.modalShown && v.nFields >= 20 && v.healthItems > 0 && v.hasSelect && v.hasPassword;
  console.log(ok && !errs.length ? "配置面板验证通过" : "配置面板验证未通过");
  process.exit(ok && !errs.length ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
