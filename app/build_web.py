"""app/model.json 을 읽어 의존성 없는 단일 HTML 앱을 생성한다.

    python app/build_web.py            -> app/mrr_app.html

만들어진 파일은 파이썬 없이 브라우저로 바로 열어 쓸 수 있다.
"""
import json
import sys
from pathlib import Path

MODEL = Path(__file__).with_name("model.json")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("mrr_app.html")
d = json.loads(MODEL.read_text(encoding="utf-8"))

lo = min(min(p) for p in d["train_pressure"])
hi = max(max(p) for p in d["train_pressure"])
payload = json.dumps({k: d[k] for k in
                      ("radius", "x_mean", "x_std", "coef", "train_pressure", "sigma", "cv")},
                     separators=(",", ":"))

HTML = """<title>MRR Profile Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@300;400;500;600&display=swap">
<style>
:root{
  --ground:#E9EDF1; --surface:#FFFFFF; --sunken:#DFE5EB;
  --line:#CBD5DE; --line-soft:#E1E7EC;
  --ink:#0F1720; --ink-2:#33414F; --muted:#5C6A78;
  --accent:#A8481F; --accent-soft:#F2E2D9;
  --data:#166F8C; --band5:#B8791A; --band7:#9B2C2C;
  --warn-bg:#FBEEE6; --warn-ink:#8A3B14;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0C1117; --surface:#131B23; --sunken:#0A0F14;
    --line:#26333D; --line-soft:#1C262E;
    --ink:#E4ECF2; --ink-2:#BAC7D2; --muted:#8593A1;
    --accent:#DD8256; --accent-soft:#33201A;
    --data:#57B4D2; --band5:#E0A94E; --band7:#E07A72;
    --warn-bg:#2A1B13; --warn-ink:#E8A97F;
  }
}
:root[data-theme="dark"]{
  --ground:#0C1117; --surface:#131B23; --sunken:#0A0F14;
  --line:#26333D; --line-soft:#1C262E;
  --ink:#E4ECF2; --ink-2:#BAC7D2; --muted:#8593A1;
  --accent:#DD8256; --accent-soft:#33201A;
  --data:#57B4D2; --band5:#E0A94E; --band7:#E07A72;
  --warn-bg:#2A1B13; --warn-ink:#E8A97F;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:26px 22px 60px}
header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:22px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);font-weight:500}
h1{font-size:clamp(21px,2.6vw,27px);margin:6px 0 0;font-weight:600;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px;margin-top:5px;
  font-family:"IBM Plex Mono",monospace}
.grid{display:grid;grid-template-columns:340px 1fr;gap:22px;align-items:start}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);padding:20px}
.card h2{font-size:13px;margin:0 0 14px;font-weight:600;letter-spacing:.02em}
.zone{margin-bottom:16px}
.zlab{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.zname{font-size:13px;font-weight:600}
.zdesc{font-size:11px;color:var(--muted)}
.zrow{display:flex;gap:10px;align-items:center}
input[type=range]{flex:1;accent-color:var(--data);height:22px}
input[type=number]{width:74px;padding:6px 8px;border:1px solid var(--line);
  background:var(--sunken);color:var(--ink);font-family:"IBM Plex Mono",monospace;
  font-size:13px;text-align:right;border-radius:2px}
input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.hint{font-size:11px;color:var(--muted);font-family:"IBM Plex Mono",monospace;
  margin-top:2px}
.warn{background:var(--warn-bg);color:var(--warn-ink);border-left:3px solid var(--accent);
  padding:10px 12px;font-size:12px;margin-top:14px;display:none}
.warn.on{display:block}
.btnrow{display:flex;gap:8px;margin-top:16px}
button{flex:1;padding:9px 10px;border:1px solid var(--line);background:var(--sunken);
  color:var(--ink);font-family:inherit;font-size:12.5px;cursor:pointer;border-radius:2px}
button:hover{border-color:var(--accent);color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line-soft);text-align:right}
th:first-child,td:first-child{text-align:left}
th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--muted);font-weight:500;background:var(--sunken)}
td{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
.nu{font-weight:600;font-size:14px}
.tag{display:inline-block;width:9px;height:9px;margin-right:7px;vertical-align:1px}
.chartbox{background:var(--surface);border:1px solid var(--line);padding:14px}
canvas{width:100%;height:auto;display:block}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
  border:1px solid var(--line);background:var(--surface);margin-top:22px}
.stat{padding:14px 16px;border-right:1px solid var(--line-soft)}
.stat:last-child{border-right:0}
.sk{font-family:"IBM Plex Mono",monospace;font-size:10px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted)}
.sv{font-family:"IBM Plex Mono",monospace;font-size:20px;font-weight:600;margin-top:3px;
  font-variant-numeric:tabular-nums}
.sv small{font-size:11px;font-weight:400;color:var(--muted);margin-left:3px}
details{margin-top:22px;background:var(--surface);border:1px solid var(--line)}
summary{padding:12px 18px;cursor:pointer;font-size:13px;font-weight:600}
.tblwrap{max-height:340px;overflow:auto;border-top:1px solid var(--line-soft)}
footer{margin-top:26px;font-size:11.5px;color:var(--muted);
  font-family:"IBM Plex Mono",monospace;border-top:1px solid var(--line);padding-top:14px}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">SiO₂ CMP · 3-Zone Pressure Control</div>
  <h1>압력 → MRR Profile / WIWNU 예측기</h1>
  <div class="sub" id="meta"></div>
</header>

<div class="grid">
  <div>
    <div class="card">
      <h2>압력 입력 [psi]</h2>
      <div id="zones"></div>
      <div class="hint" id="range-hint"></div>
      <div class="warn" id="warn"></div>
      <div class="btnrow">
        <button id="btn-csv">CSV 저장</button>
        <button id="btn-reset">초기화</button>
      </div>
    </div>

    <div class="card" style="margin-top:18px">
      <h2>WIWNU = (표본표준편차 / 평균) × 100</h2>
      <table>
        <thead><tr><th>구간</th><th>범위</th><th>평균</th><th>σ</th><th>WIWNU</th></tr></thead>
        <tbody id="nu-body"></tbody>
      </table>
    </div>
  </div>

  <div>
    <div class="chartbox"><canvas id="cv" width="1500" height="820"></canvas></div>
    <div class="stats" id="stats"></div>
    <details>
      <summary>MRR Profile 전체 값 (43점)</summary>
      <div class="tblwrap">
        <table>
          <thead><tr><th>r (mm)</th><th>MRR</th><th>95% 하한</th><th>95% 상한</th></tr></thead>
          <tbody id="prof-body"></tbody>
        </table>
      </div>
    </details>
  </div>
</div>

<footer id="foot"></footer>
</div>

<script>
const M = __PAYLOAD__;
const LO = __LO__, HI = __HI__;
const R = M.radius, SIG = M.sigma;
const BANDS = [["all",74,"var(--data)"],["5mm",70,"var(--band5)"],["7mm",68,"var(--band7)"]];
const DEF = [2.0, 2.2, 2.0];
const ZONES = [["Zone 1","최외곽 에지"],["Zone 2","중간 링"],["Zone 3","중심"]];

/* ---- 모델: 학습 때와 똑같은 2차 특징 + 표준화 + 선형결합 ---- */
function features(z){
  const [a,b,c] = z;
  return [a,b,c, a*a,b*b,c*c, a*b,a*c,b*c];
}
function predict(z){
  const f = features(z);
  const out = new Array(R.length).fill(0);
  for(let j=0;j<R.length;j++) out[j] = M.coef[0][j];          // 절편
  for(let i=0;i<f.length;i++){
    const s = (f[i]-M.x_mean[i])/M.x_std[i];
    const row = M.coef[i+1];
    for(let j=0;j<R.length;j++) out[j] += s*row[j];
  }
  return out;
}
function wiwnu(y, rmax){
  const idx=[]; for(let i=0;i<R.length;i++) if(Math.abs(R[i])<=rmax+1e-9) idx.push(i);
  const v = idx.map(i=>y[i]);
  const n = v.length, mean = v.reduce((s,x)=>s+x,0)/n;
  const sd = Math.sqrt(v.reduce((s,x)=>s+(x-mean)*(x-mean),0)/(n-1));   // 표본표준편차
  return {n, mean, sd, nu: sd/mean*100, min:Math.min(...v), max:Math.max(...v)};
}
function extrapDist(z){
  let best = Infinity;
  for(const p of M.train_pressure){
    const d = Math.hypot(p[0]-z[0], p[1]-z[1], p[2]-z[2]);
    if(d<best) best=d;
  }
  return best;
}

/* ---- 입력 UI ---- */
const zonesEl = document.getElementById("zones");
const inputs = [];
ZONES.forEach(([name,desc],i)=>{
  const d = document.createElement("div"); d.className="zone";
  d.innerHTML = `<div class="zlab"><span class="zname">${name}</span>
      <span class="zdesc">${desc}</span></div>
    <div class="zrow">
      <input type="range" min="${(LO-0.3).toFixed(2)}" max="${(HI+0.3).toFixed(2)}"
             step="0.01" value="${DEF[i]}" aria-label="${name} 슬라이더">
      <input type="number" step="0.01" value="${DEF[i].toFixed(2)}" aria-label="${name} 값">
    </div>`;
  zonesEl.appendChild(d);
  const [rg,nb] = d.querySelectorAll("input");
  rg.addEventListener("input", ()=>{ nb.value = (+rg.value).toFixed(2); render(); });
  nb.addEventListener("input", ()=>{ if(nb.value!==""){ rg.value = nb.value; render(); }});
  inputs.push([rg,nb]);
});
document.getElementById("range-hint").textContent =
  `학습 압력 범위  ${LO} ~ ${HI} psi`;
document.getElementById("meta").textContent =
  `학습 웨이퍼 ${M.cv.n_wafer}장 · Quad Ridge (α=${M.cv.alpha}) · ` +
  `교차검증 RMSE ${M.cv.rmse.toFixed(1)} Å/min (${M.cv.mape.toFixed(1)}%)`;
document.getElementById("foot").textContent =
  `모델 ${M.cv.source} 학습 · 반경 ${R.length}점 (−74 ~ +74 mm) · ` +
  `WIWNU all: 전 구간 / 5mm: −70~+70 / 7mm: −68~+68`;

document.getElementById("btn-reset").onclick = ()=>{
  inputs.forEach(([rg,nb],i)=>{ rg.value=DEF[i]; nb.value=DEF[i].toFixed(2); });
  render();
};

let LAST = null;

/* 저장 경로는 두 가지다.
   - claude.ai 아티팩트로 열었을 때 : downloads 기능을 통해 저장 (뷰어가 확인)
   - 로컬 HTML 파일로 열었을 때     : 보통의 blob 링크
   두 환경 모두에서 동작하도록 앞의 것을 먼저 시도하고 없으면 뒤로 넘어간다. */
let dlPromise = null;
function getDownloads(){
  if(!dlPromise){
    dlPromise = (window.claude && window.claude.use)
      ? Promise.resolve(window.claude.use("downloads")).catch(()=>null)
      : Promise.resolve(null);
  }
  return dlPromise;
}
function buildCsv(z,y,w){
  let s = `Zone1,${z[0]}\\nZone2,${z[1]}\\nZone3,${z[2]}\\n\\nradius_mm,MRR_pred\\n`;
  const ord = R.map((_,i)=>i).sort((a,b)=>R[b]-R[a]);
  ord.forEach(i=> s += `${R[i]},${y[i].toFixed(4)}\\n`);
  s += `\\nband,range_mm,n_points,mean,std,WIWNU_pct\\n`;
  BANDS.forEach(([nm,rm])=>{ const q=w[nm];
    s += `${nm},-${rm} ~ +${rm},${q.n},${q.mean.toFixed(4)},${q.sd.toFixed(4)},${q.nu.toFixed(4)}\\n`; });
  return s;
}
function blobSave(name, text){
  const blob = new Blob(["\\ufeff"+text], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href), 4000);
}
document.getElementById("btn-csv").onclick = async ()=>{
  if(!LAST) return;
  const {z,y,w} = LAST;
  const text = buildCsv(z,y,w);
  const name = `MRR_${z[0]}_${z[1]}_${z[2]}.csv`;
  const dl = await getDownloads();
  if(dl){
    try { await dl.save({filename:name, data:"\\ufeff"+text}); return; }
    catch(e){
      if(e && e.code === "declined") return;                 // 사용자가 취소함
      if(e && e.code === "extension_not_enabled"){            // csv 가 막힌 뷰어
        try { await dl.save({filename:name.replace(/\\.csv$/,".txt"), data:text}); return; }
        catch(e2){ if(e2 && e2.code === "declined") return; }
      }
    }
  }
  blobSave(name, text);
};

/* ---- 그리기 ---- */
function css(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function draw(y){
  const cv = document.getElementById("cv"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const P = {l:110, r:26, t:26, b:70};
  g.clearRect(0,0,W,H);

  const band = SIG.some(v=>v>0);
  let ymin = Infinity, ymax = -Infinity;
  for(let i=0;i<y.length;i++){
    ymin = Math.min(ymin, band ? y[i]-1.96*SIG[i] : y[i]);
    ymax = Math.max(ymax, band ? y[i]+1.96*SIG[i] : y[i]);
  }
  const pad = (ymax-ymin)*0.08; ymin-=pad; ymax+=pad;
  const X = r => P.l + (r+78)/156 * (W-P.l-P.r);
  const Y = v => H-P.b - (v-ymin)/(ymax-ymin) * (H-P.t-P.b);

  g.strokeStyle = css("--line-soft"); g.lineWidth = 1.5;
  g.fillStyle = css("--muted"); g.font = "22px 'IBM Plex Mono',monospace";
  g.textAlign = "right"; g.textBaseline = "middle";
  for(let k=0;k<=5;k++){
    const v = ymin + (ymax-ymin)*k/5, yy = Y(v);
    g.beginPath(); g.moveTo(P.l,yy); g.lineTo(W-P.r,yy); g.stroke();
    g.fillText(Math.round(v), P.l-12, yy);
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for(const t of [-74,-50,-25,0,25,50,74]){
    g.beginPath(); g.moveTo(X(t),P.t); g.lineTo(X(t),H-P.b); g.stroke();
    g.fillText(t, X(t), H-P.b+12);
  }
  g.font = "24px 'IBM Plex Sans KR',sans-serif";
  g.fillText("Radius [mm]", (P.l+W-P.r)/2, H-P.b+46);
  g.save(); g.translate(30,(P.t+H-P.b)/2); g.rotate(-Math.PI/2);
  g.textBaseline="middle"; g.fillText("MRR [Å/min]",0,0); g.restore();

  const ord = R.map((_,i)=>i).sort((a,b)=>R[a]-R[b]);
  if(band){
    g.fillStyle = css("--data") + "2e";
    g.beginPath();
    ord.forEach((i,k)=>{ const x=X(R[i]), v=Y(y[i]+1.96*SIG[i]); k?g.lineTo(x,v):g.moveTo(x,v); });
    [...ord].reverse().forEach(i=> g.lineTo(X(R[i]), Y(y[i]-1.96*SIG[i])));
    g.closePath(); g.fill();
  }
  BANDS.slice(1).forEach(([nm,rm,col],k)=>{
    const c = css(col.slice(4,-1));
    g.strokeStyle = c; g.setLineDash([7,7]); g.lineWidth = 2;
    for(const s of [rm,-rm]){ g.beginPath(); g.moveTo(X(s),P.t); g.lineTo(X(s),H-P.b); g.stroke(); }
    g.setLineDash([]);
    g.fillStyle = c; g.font = "19px 'IBM Plex Mono',monospace";
    g.textAlign = "right"; g.textBaseline = "bottom";   // 아래쪽에 배치해 y축 눈금과 겹치지 않게
    g.fillText(`${nm} ±${rm}`, X(-rm)-8, H - P.b - 8 - k*24);
  });
  g.strokeStyle = css("--data"); g.lineWidth = 3.2; g.beginPath();
  ord.forEach((i,k)=>{ const x=X(R[i]), v=Y(y[i]); k?g.lineTo(x,v):g.moveTo(x,v); });
  g.stroke();
  g.fillStyle = css("--data");
  ord.forEach(i=>{ g.beginPath(); g.arc(X(R[i]),Y(y[i]),4,0,7); g.fill(); });
}

/* ---- 메인 ---- */
function render(){
  const z = inputs.map(([rg,nb]) => {
    const v = parseFloat(nb.value);
    return Number.isFinite(v) ? v : 0;
  });
  const y = predict(z);
  const w = {}; BANDS.forEach(([nm,rm]) => w[nm] = wiwnu(y, rm));
  LAST = {z,y,w};

  document.getElementById("nu-body").innerHTML = BANDS.map(([nm,rm,col])=>
    `<tr><td><span class="tag" style="background:${col}"></span>${nm}</td>
      <td>±${rm}</td><td>${w[nm].mean.toFixed(0)}</td>
      <td>${w[nm].sd.toFixed(1)}</td>
      <td class="nu">${w[nm].nu.toFixed(2)}%</td></tr>`).join("");

  const a = w.all;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="sk">평균 MRR</div>
      <div class="sv">${a.mean.toFixed(0)}<small>Å/min</small></div></div>
    <div class="stat"><div class="sk">WIWNU 5mm</div>
      <div class="sv">${w["5mm"].nu.toFixed(2)}<small>%</small></div></div>
    <div class="stat"><div class="sk">WIWNU 7mm</div>
      <div class="sv">${w["7mm"].nu.toFixed(2)}<small>%</small></div></div>
    <div class="stat"><div class="sk">최소 / 최대</div>
      <div class="sv" style="font-size:15px">${a.min.toFixed(0)} / ${a.max.toFixed(0)}</div></div>`;

  const ord = R.map((_,i)=>i).sort((b,c)=>R[c]-R[b]);
  document.getElementById("prof-body").innerHTML = ord.map(i=>
    `<tr><td>${R[i]}</td><td>${y[i].toFixed(1)}</td>
      <td>${(y[i]-1.96*SIG[i]).toFixed(1)}</td>
      <td>${(y[i]+1.96*SIG[i]).toFixed(1)}</td></tr>`).join("");

  const d = extrapDist(z), wEl = document.getElementById("warn");
  if(d > 0.3){
    wEl.className = "warn on";
    wEl.textContent = `학습한 압력 조건에서 ${d.toFixed(2)} psi 떨어진 외삽 영역입니다. `
      + `예측 신뢰도가 낮으니 검증 실험을 권장합니다.`;
  } else wEl.className = "warn";

  draw(y);
}
render();
window.addEventListener("resize", ()=>render());
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", ()=>render());
</script>
"""

OUT.write_text(HTML.replace("__PAYLOAD__", payload)
                   .replace("__LO__", f"{lo:g}").replace("__HI__", f"{hi:g}"),
               encoding="utf-8")
print(f"{OUT} 저장 ({OUT.stat().st_size/1024:.0f} KB)")
