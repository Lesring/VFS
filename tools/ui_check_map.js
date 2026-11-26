// 验证地图卡片与检索降级提示在页面上正常渲染。
const http = require("http");
const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://127.0.0.1:8931";
const JOB = process.argv[3] || "";
const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const PORT = 9225;
const OUT = path.join(__dirname, "..", "data");

const getJSON = (url) => new Promise((res, rej) => {
  http.get(url, (r) => { let b = ""; r.on("data", c => b += c); r.on("end", () => { try { res(JSON.parse(b)); } catch (e) { rej(e); } }); }).on("error", rej);
});
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function main() {
  const child = execFile(CHROME, ["--headless=new", "--disable-gpu", "--hide-scrollbars",
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${path.join(OUT, "cmp3")}`,
    "--window-size=1500,1600", "about:blank"]);
  child.on("error", e => { console.error(e.message); process.exit(1); });

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
  await sleep(1800);

  // 直接调用前端 render() 渲染已有结果，避免重复跑分析
  const res = await send("Runtime.evaluate", {
    expression: `(async () => {
      const job = ${JSON.stringify(JOB)};
      if (!job) return 'no-job';
      const d = await (await fetch('/api/jobs/' + job + '/result')).json();
      render(d);
      await new Promise(r => setTimeout(r, 1500));
      const right = document.querySelector('#right');
      const t = right.innerText;
      const mapImg = right.querySelector('.mapwrap img');
      return JSON.stringify({
        hasMapCard: t.includes('位置地图'),
        mapProvider: (t.match(/Google Maps|OpenStreetMap|地图链接/) || [])[0],
        hasMapImg: !!mapImg,
        mapImgSrc: mapImg ? mapImg.src.slice(0, 70) : '',
        mapImgLoaded: mapImg ? (mapImg.naturalWidth > 0) : false,
        hasSearchCard: t.includes('检索记录'),
        hasConflict: t.includes('矛盾点'),
        len: t.length,
      });
    })()`, returnByValue: true, awaitPromise: true,
  });
  const v = JSON.parse(res.result.value || "{}");
  console.log("渲染结果:", v);

  const shot = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  fs.writeFileSync(path.join(OUT, "ui_map.png"), Buffer.from(shot.data, "base64"));

  const errs = events.filter(e => e.method === "Runtime.exceptionThrown")
    .map(e => (e.params.exceptionDetails?.exception?.description || "").slice(0, 200));
  if (errs.length) console.log("异常:", errs);

  ws.close(); child.kill();
  const ok = v.hasMapCard && v.hasMapImg && v.hasSearchCard && !errs.length;
  console.log(ok ? "地图渲染验证通过" : "地图渲染验证未通过");
  process.exit(ok ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
