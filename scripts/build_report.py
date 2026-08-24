"""분석 결과를 단일 HTML 리포트로 빌드 (그림은 data URI 로 내장)."""
import base64, pathlib, sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "reports/report.html")


def img(p):
    return "data:image/png;base64," + base64.b64encode(pathlib.Path(p).read_bytes()).decode()


MODELS = [
    ("대칭성 능형회귀 [quad]", 163.2, "best"),
    ("PLS 3성분 [quad]", 164.0, "lin"),
    ("반경별 능형회귀 [quad]", 168.1, "lin"),
    ("반경별 능형회귀 [zone]", 169.2, "lin"),
    ("반경별 능형회귀 [phys]", 174.1, "lin"),
    ("저계수(rank-3) 능형회귀", 178.7, "lin"),
    ("PLS 2성분 [zone+ring]", 181.4, "lin"),
    ("신경망 MLP", 195.4, "ml"),
    ("가우시안 과정 (GPR)", 202.3, "ml"),
    ("랜덤포레스트", 227.5, "ml"),
    ("Preston 스케일 (기준선)", 282.6, "base"),
    ("평균 프로파일 (기준선)", 408.7, "base"),
]
SCALE = 430.0
FLOOR = 118.4

bars = "\n".join(
    f'''      <div class="bar-row">
        <div class="bar-name{' is-best' if k == 'best' else ''}">{n}</div>
        <div class="bar-track"><div class="bar bar--{k}" style="width:{v / SCALE * 100:.1f}%"></div></div>
        <div class="bar-val">{v:.1f}</div>
      </div>''' for n, v, k in MODELS)

HTML = f'''<title>3-Zone 압력에서 MRR 프로파일 예측</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@300;400;500;600&family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,400&display=swap">
<style>
:root {{
  --ground:#E9EDF1; --surface:#FFFFFF; --sunken:#DFE5EB;
  --line:#CBD5DE; --line-soft:#E1E7EC;
  --ink:#0F1720; --ink-2:#33414F; --muted:#5C6A78;
  --accent:#A8481F; --accent-soft:#F2E2D9;
  --data:#166F8C; --data-soft:#DCEAF0;
  --good:#2C6E52; --warn:#8A5A12;
  --plate:#FFFFFF;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0C1117; --surface:#131B23; --sunken:#0A0F14;
    --line:#26333D; --line-soft:#1C262E;
    --ink:#E4ECF2; --ink-2:#BAC7D2; --muted:#8593A1;
    --accent:#DD8256; --accent-soft:#33201A;
    --data:#57B4D2; --data-soft:#122830;
    --good:#6FBF97; --warn:#D5A055;
    --plate:#E9EDF1;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0C1117; --surface:#131B23; --sunken:#0A0F14;
  --line:#26333D; --line-soft:#1C262E;
  --ink:#E4ECF2; --ink-2:#BAC7D2; --muted:#8593A1;
  --accent:#DD8256; --accent-soft:#33201A;
  --data:#57B4D2; --data-soft:#122830;
  --good:#6FBF97; --warn:#D5A055;
  --plate:#E9EDF1;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  font-weight:400; line-height:1.72; font-size:16px;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 28px 96px; }}
.col {{ max-width:70ch; }}

/* ---------- header ---------- */
header {{
  border-bottom:1px solid var(--line);
  background:var(--surface);
  padding:56px 0 40px; margin-bottom:56px;
}}
.eyebrow {{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); font-weight:500;
}}
h1 {{
  font-family:"IBM Plex Serif","IBM Plex Sans KR",serif; font-weight:600;
  font-size:clamp(30px,4.4vw,46px); line-height:1.18; letter-spacing:-.015em;
  margin:14px 0 0; text-wrap:balance;
}}
.dek {{ color:var(--ink-2); font-size:17px; margin:16px 0 0; max-width:62ch; }}

/* ---------- verdict ---------- */
.verdict {{
  display:grid; gap:0; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  border:1px solid var(--line); background:var(--surface);
  margin:36px 0 0;
}}
.stat {{ padding:22px 24px; border-right:1px solid var(--line-soft); }}
.stat:last-child {{ border-right:0; }}
.stat-k {{
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted);
}}
.stat-v {{
  font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:30px; font-weight:600; line-height:1.25; margin-top:6px;
}}
.stat-v small {{ font-size:14px; font-weight:400; color:var(--muted); margin-left:4px; }}
.stat--hero .stat-v {{ color:var(--accent); }}
.stat--floor .stat-v {{ color:var(--muted); }}

/* ---------- sections ---------- */
section {{ margin:0 0 68px; }}
h2 {{
  font-family:"IBM Plex Serif","IBM Plex Sans KR",serif; font-weight:600;
  font-size:26px; letter-spacing:-.01em; margin:0 0 6px; text-wrap:balance;
}}
h2 .num {{
  font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--accent);
  font-weight:500; margin-right:12px; vertical-align:2px;
}}
h3 {{ font-size:16px; font-weight:600; margin:32px 0 6px; letter-spacing:-.005em; }}
p {{ margin:14px 0; }}
.lede {{ color:var(--ink-2); }}
strong {{ font-weight:600; }}
em {{ font-style:normal; background:var(--accent-soft); padding:1px 5px; }}
code {{
  font-family:"IBM Plex Mono",monospace; font-size:.88em;
  background:var(--sunken); padding:1px 5px; border-radius:2px;
}}
.rule {{ border:0; border-top:1px solid var(--line); margin:0 0 44px; }}

/* ---------- bar chart ---------- */
.panel {{
  background:var(--surface); border:1px solid var(--line);
  padding:28px 26px 22px; margin:26px 0;
}}
.panel-head {{
  display:flex; justify-content:space-between; align-items:baseline;
  gap:16px; flex-wrap:wrap; margin-bottom:22px;
}}
.panel-title {{ font-weight:600; font-size:15px; }}
.panel-sub {{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--muted);
  letter-spacing:.03em;
}}
.chart {{ position:relative; }}
.floor-line {{
  position:absolute; top:-6px; bottom:16px; width:0;
  border-left:1px dashed var(--accent); pointer-events:none;
}}
.floor-tag {{
  position:absolute; top:-24px; transform:translateX(-50%);
  font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--accent);
  white-space:nowrap; letter-spacing:.04em;
}}
.bar-row {{
  display:grid; grid-template-columns:minmax(140px,200px) 1fr 54px;
  gap:14px; align-items:center; padding:3px 0;
}}
.bar-name {{ font-size:13px; color:var(--ink-2); }}
.bar-name.is-best {{ color:var(--accent); font-weight:600; }}
.bar-track {{ background:var(--sunken); height:19px; position:relative; }}
.bar {{ height:100%; }}
.bar--best {{ background:var(--accent); }}
.bar--lin  {{ background:var(--data); }}
.bar--ml   {{ background:var(--muted); opacity:.75; }}
.bar--base {{ background:var(--line); }}
.bar-val {{
  font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:12.5px; text-align:right; color:var(--ink-2);
}}
.legend {{
  display:flex; gap:20px; flex-wrap:wrap; margin-top:20px;
  padding-top:16px; border-top:1px solid var(--line-soft);
  font-size:12px; color:var(--muted);
}}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.swatch {{ width:11px; height:11px; display:inline-block; }}

/* ---------- tables ---------- */
.tbl-wrap {{ overflow-x:auto; margin:22px 0; border:1px solid var(--line); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ padding:11px 16px; text-align:left; border-bottom:1px solid var(--line-soft); }}
th {{
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:500;
  background:var(--sunken); white-space:nowrap;
}}
td.num, th.num {{ text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:0; }}
tr.hl td {{ background:var(--accent-soft); font-weight:600; }}
tr.floor td {{ background:var(--sunken); color:var(--muted); }}

/* ---------- figure plates ---------- */
figure {{ margin:26px 0; }}
.plate {{ background:var(--plate); border:1px solid var(--line); padding:10px; }}
.plate img {{ display:block; width:100%; height:auto; }}
figcaption {{
  font-size:13px; color:var(--muted); margin-top:11px; max-width:70ch;
  padding-left:2px; border-left:2px solid var(--line); padding-left:12px;
}}

/* ---------- callout ---------- */
.callout {{
  border-left:3px solid var(--accent); background:var(--surface);
  border-top:1px solid var(--line); border-right:1px solid var(--line);
  border-bottom:1px solid var(--line);
  padding:20px 24px; margin:26px 0;
}}
.callout p:first-child {{ margin-top:0; }}
.callout p:last-child {{ margin-bottom:0; }}

pre {{
  background:var(--sunken); border:1px solid var(--line-soft);
  padding:18px 20px; overflow-x:auto; font-size:13px; line-height:1.65;
  font-family:"IBM Plex Mono",monospace; margin:20px 0;
}}
ul {{ padding-left:20px; }}
li {{ margin:9px 0; }}
li::marker {{ color:var(--accent); }}
footer {{
  border-top:1px solid var(--line); margin-top:64px; padding-top:22px;
  font-size:12.5px; color:var(--muted);
  font-family:"IBM Plex Mono",monospace;
}}
</style>

<header>
  <div class="wrap">
    <div class="eyebrow">SiO₂ CMP · 3-Zone Pressure Control · n = 21 wafers</div>
    <h1>압력에서 MRR 프로파일로,<br>가장 강력한 예측 모델은 무엇인가</h1>
    <p class="dek">웨이퍼 21장 · 반경 −74~+74 mm 43지점 측정 데이터로 18종 모델을 교차검증한 결과.
      결론은 <strong>딥러닝이 아니라 반경별 선형 영향함수 모델</strong>이고, 그 이유는 데이터 구조 자체에 있습니다.</p>

    <div class="verdict">
      <div class="stat stat--hero">
        <div class="stat-k">CV RMSE (전체)</div>
        <div class="stat-v">163.2<small>A/min</small></div>
      </div>
      <div class="stat stat--hero">
        <div class="stat-k">Edge exclusion 4 mm</div>
        <div class="stat-v">67.8<small>A/min</small></div>
      </div>
      <div class="stat">
        <div class="stat-k">MAPE (|r| ≤ 70)</div>
        <div class="stat-v">4.53<small>%</small></div>
      </div>
      <div class="stat stat--floor">
        <div class="stat-k">측정 재현성 하한</div>
        <div class="stat-v">118.4<small>A/min</small></div>
      </div>
    </div>
  </div>
</header>

<div class="wrap">

<section>
  <div class="col">
    <h2><span class="num">01</span>결론</h2>
    <p class="lede">MRR 프로파일을 반경마다 하나씩, 존 압력의 선형결합으로 예측합니다.</p>
  </div>
  <pre>MRR(r) = a(r) + K₁(r)·P₁ + K₂(r)·P₂ + K₃(r)·P₃    ( + 2차·교호항 )</pre>
  <div class="col">
    <p><code>K_z(r)</code> 는 <strong>"존 z의 압력을 1 psi 올리면 반경 r 에서 MRR이 몇 A/min 오르는가"</strong>
      — 물리적으로 해석 가능한 영향함수입니다. 여기에 두 가지 물리 사전지식을 규제로 얹은 것이 최종 모델입니다.</p>
  </div>

  <div class="tbl-wrap">
    <table>
      <thead><tr><th>물리 사전지식</th><th>구현</th><th>효과</th></tr></thead>
      <tbody>
        <tr>
          <td>연마는 회전평균 → 응답은 |r| 의 함수</td>
          <td>프로파일을 대칭/반대칭 분해, 대칭 성분은 접힌 반경에서 적합</td>
          <td>반경당 표본 2배, 노이즈 1/√2</td>
        </tr>
        <tr>
          <td>좌우 비대칭은 압력이 아니라 헤드 틸트 등 장비 고유 signature</td>
          <td>반대칭 성분을 rank-1 (고정 형상 × 압력 선형) 로 강하게 규제</td>
          <td>자유 파라미터 21개 회귀 → 4개</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="callout col">
    <p><strong>두 번째는 추측이 아니라 데이터로 확인한 것입니다.</strong> 동일 압력으로 반복 연마한
      웨이퍼(#1,2,3 / #19,20,21) 사이에서 좌우 비대칭 성분의 상관계수가 <em>0.85 ~ 0.99</em> 로
      재현되고, SVD 1개 성분이 비대칭 분산의 <em>88.6 %</em> 를 설명합니다.
      즉 비대칭은 노이즈가 아니라 결정론적 신호이므로, 흔히 하듯 <strong>대칭화해서 평균내면
      실제 신호를 버리는 것</strong>입니다.</p>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">02</span>18종 모델 교차검증</h2>
    <p class="lede">동일 압력 조건의 반복 웨이퍼를 통째로 묶어서 빼는
      <strong>Leave-One-Design-Out</strong> 방식입니다. "본 적 없는 압력 조건"에 대한 정직한 성능이고,
      16개 fold에 대한 부트스트랩 표준오차는 ±24 A/min 입니다.</p>
  </div>

  <div class="panel">
    <div class="panel-head">
      <div class="panel-title">교차검증 RMSE — 낮을수록 좋음</div>
      <div class="panel-sub">A/min · 하이퍼파라미터 고정 · ±SE ≈ 24</div>
    </div>
    <div class="chart">
      <div class="floor-line" style="left:calc(140px + 14px + (100% - 140px - 54px - 28px) * {FLOOR / SCALE:.4f});">
        <div class="floor-tag">측정 재현성 하한 118.4</div>
      </div>
{bars}
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--accent)"></i>최종 채택</span>
      <span><i class="swatch" style="background:var(--data)"></i>선형 · 저차원</span>
      <span><i class="swatch" style="background:var(--muted);opacity:.75"></i>비선형 ML</span>
      <span><i class="swatch" style="background:var(--line)"></i>기준선</span>
    </div>
  </div>

  <div class="col">
    <p>읽는 법이 중요합니다.</p>
    <ul>
      <li><strong>선형·저차원 모델 7종이 1 SE 안에 몰려 있습니다.</strong> 이들 사이의 차이는
        통계적으로 의미가 없습니다 — 어느 걸 써도 됩니다.</li>
      <li><strong>랜덤포레스트·GPR·MLP는 명확히 열등합니다</strong> (1.5 ~ 4.6 SE 차이).
        표본이 21개인 문제에서 유연한 비선형 모델은 손해만 봅니다.</li>
      <li>최고 모델도 하한(118.4)의 <strong>1.38배</strong>입니다.
        이 데이터로 얻을 수 있는 성능의 대부분에 이미 도달했다는 뜻입니다.</li>
    </ul>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">03</span>왜 복잡한 모델이 이기지 못하는가</h2>

    <h3>표본이 21장, 고유 압력 조건은 16개뿐</h3>
    <p>입력 차원 3~4, 출력 차원 43, 표본 21. 트리·신경망이 학습할 데이터가 근본적으로 없습니다.</p>

    <h3>프로파일이 사실상 2~3차원</h3>
  </div>
  <div class="tbl-wrap" style="max-width:640px">
    <table>
      <thead><tr><th>주성분 개수</th><th class="num">1</th><th class="num">2</th><th class="num">3</th><th class="num">4</th></tr></thead>
      <tbody>
        <tr><td>설명분산 누적</td><td class="num">71.1%</td><td class="num">94.2%</td><td class="num">97.5%</td><td class="num">98.9%</td></tr>
        <tr><td>재구성 RMSE</td><td class="num">205</td><td class="num">91</td><td class="num">60</td><td class="num">39</td></tr>
      </tbody>
    </table>
  </div>
  <div class="col">
    <p>주성분 2개면 재구성 오차가 91 A/min 인데, 측정 재현성 한계가 118.4 A/min 입니다.
      <strong>3차원을 넘는 표현력은 노이즈를 학습하는 것</strong>과 같습니다.</p>

    <h3>측정 재현성이라는 넘을 수 없는 하한</h3>
    <p>동일 압력으로 반복 연마한 웨이퍼들(#1,2,3 / #4,5 / #19,20,21)의 순수 오차입니다.
      어떤 모델도 이보다 잘 맞출 수 없고, 더 잘 맞았다면 그건 과적합입니다.</p>
  </div>
  <div class="tbl-wrap" style="max-width:640px">
    <table>
      <thead><tr><th>구간</th><th class="num">반복실험 노이즈</th></tr></thead>
      <tbody>
        <tr><td>중심 |r| ≤ 50 mm</td><td class="num">40 A/min</td></tr>
        <tr><td>중간 55 ~ 68 mm</td><td class="num">46 A/min</td></tr>
        <tr><td>에지 |r| ≥ 70 mm</td><td class="num">162 A/min</td></tr>
        <tr><td>최외곽 r = ±74 mm</td><td class="num">287 ~ 500 A/min</td></tr>
        <tr class="hl"><td>전체</td><td class="num">118.4 A/min</td></tr>
      </tbody>
    </table>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">04</span>모델이 스스로 복원한 존 배치</h2>
    <p class="lede">학습된 영향함수 <code>K_z(r)</code> 를 물리 단위로 그린 것입니다.
      존이 어느 반경을 담당하는지 알려준 적이 없는데도, 데이터가 헤드 구조를 그대로 재현합니다.</p>
  </div>
  <figure>
    <div class="plate"><img alt="존별 영향함수: Zone 3는 중심에서 평평하고, Zone 2는 60-64 mm에서 솟고, Zone 1은 70 mm 이상에서 급상승" src="{img('figures/02_influence_functions.png')}"></div>
    <figcaption>Zone 3 는 |r| ≤ 55 mm 에서 평평하게 ~470, Zone 2 는 |r| ≈ 60~64 mm 에서 ~490 으로 솟고,
      Zone 1 은 |r| ≥ 70 mm 에서 급격히 올라 r = 74 mm 에서 2250 A/min·psi⁻¹ 에 달합니다.
      최외곽에서 Zone 2 계수가 음수로 내려가는 것은 물리가 아니라 존 압력 간 다중공선성 아티팩트로 보는 것이 안전합니다.</figcaption>
  </figure>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">05</span>오차는 어디에 남아 있는가</h2>
    <p class="lede">반경별 CV 오차와 반복실험 노이즈를 겹쳐 그리면 두 곡선이 거의 포개집니다.
      남은 오차는 사실상 최외곽 4개 점(±72~74 mm)에만 있습니다.</p>
  </div>
  <figure>
    <div class="plate"><img alt="반경별 모델 오차와 반복실험 노이즈 비교. 중심부에서 두 곡선이 거의 겹치고 최외곽에서만 벌어짐" src="{img('figures/04_error_vs_noise.png')}"></div>
  </figure>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>구간</th><th class="num">CV 오차</th><th class="num">노이즈 하한</th><th class="num">비율</th></tr></thead>
      <tbody>
        <tr><td>중심 |r| ≤ 50 mm</td><td class="num">48.9</td><td class="num">40.7</td><td class="num">1.20</td></tr>
        <tr><td>중간 55 ~ 68 mm</td><td class="num">83.1</td><td class="num">46.4</td><td class="num">1.79</td></tr>
        <tr><td>에지 |r| ≥ 70 mm</td><td class="num">318.2</td><td class="num">211.5</td><td class="num">1.50</td></tr>
        <tr><td>최외곽 r = ±74 mm</td><td class="num">589.8</td><td class="num">407.4</td><td class="num">1.45</td></tr>
      </tbody>
    </table>
  </div>
  <div class="callout col">
    <p><strong>중심부는 개선 여지가 거의 없습니다.</strong> 모델 오차가 같은 조건으로 두 번 연마했을 때의
      차이의 1.2배 수준입니다. 실무에서 흔히 쓰는 edge exclusion 4 mm 를 적용하면
      <em>RMSE 67.8 A/min · MAPE 4.53 %</em> — 공정 판단에 쓰기 충분한 정확도입니다.</p>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">06</span>예측 성능 실제 확인</h2>
    <p class="lede">각 웨이퍼를 <strong>자기 압력 조건을 학습에서 완전히 뺀 상태로</strong> 예측한 결과입니다.
      빨간 점선이 예측, 검은 실선이 실측입니다.</p>
  </div>
  <figure>
    <div class="plate"><img alt="8개 웨이퍼에 대한 교차검증 예측과 실측 프로파일 비교" src="{img('figures/03_cv_profiles.png')}"></div>
    <figcaption>#13, #18 처럼 잘 맞는 조건이 있는가 하면 #9 (Zone 3 = 1.5, 데이터에 단 하나뿐인 조건)와
      #16 (3,3,3 — 설계 공간의 모서리)은 오차가 큽니다. 모델의 한계가 아니라 설계 공간의 한계입니다.</figcaption>
  </figure>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">07</span>외삽은 어디까지 믿을 수 있나</h2>
    <p class="lede">압력 공간이 저압군(1.3~2.5 psi, 15장)과 고압군(2.7~3.2 psi, 6장) 두 덩어리로
      갈라져 있습니다. 한쪽만 보고 다른 쪽을 맞히는 것은 순수 외삽입니다.</p>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>모델</th><th class="num">저압 → 고압</th><th class="num">고압 → 저압</th><th class="num">평균 MRR 편향</th></tr></thead>
      <tbody>
        <tr class="hl"><td>반경별 능형회귀 [quad]</td><td class="num">252.6</td><td class="num">1546.7</td><td class="num">+22</td></tr>
        <tr><td>대칭성 능형회귀 [quad]</td><td class="num">255.0</td><td class="num">1847.0</td><td class="num">+51</td></tr>
        <tr><td>반경별 능형회귀 [zone]</td><td class="num">269.5</td><td class="num">1201.1</td><td class="num">−82</td></tr>
        <tr><td>가우시안 과정</td><td class="num">465.4</td><td class="num">660.2</td><td class="num">−412</td></tr>
        <tr><td>랜덤포레스트</td><td class="num">494.2</td><td class="num">641.3</td><td class="num">−436</td></tr>
      </tbody>
    </table>
  </div>
  <div class="col">
    <ul>
      <li><strong>위쪽 외삽은 선형 모델이 안전합니다.</strong> 1.3~2.5 psi 만 보고 3.2 psi 를 예측했는데
        평균 MRR 편향이 1728 대비 +22 (1.3 %) 에 불과합니다. Preston 법칙(MRR ∝ P)이 실제로
        성립한다는 증거입니다.</li>
      <li><strong>랜덤포레스트·GPR은 외삽을 아예 못 합니다.</strong> 학습 평균으로 수렴해 −436 의
        구조적 편향이 생깁니다. 압력을 새로 조합해 보는 것이 목적인 이 문제에서 치명적입니다.</li>
      <li>오른쪽 열(고압 → 저압)이 무너지는 것은 모델 탓이 아닙니다. 고압군은 웨이퍼 6장·고유조건
        4개뿐이고 세 존이 함께 움직여 기울기 자체가 결정되지 않습니다.</li>
    </ul>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">08</span>지금 병목은 모델이 아니라 설계 공간입니다</h2>
    <p>정확도를 더 올리고 싶다면 모델을 바꾸는 것이 아니라 <strong>실험을 추가</strong>해야 합니다.
      현재 데이터의 구조적 약점은 세 가지입니다.</p>
    <ul>
      <li><strong>압력 공간이 두 덩어리로 갈라져 있습니다.</strong> 2.5 ~ 2.7 psi 사이가 비어 있어,
        그 구간 예측은 보간이 아니라 사실상 외삽입니다.</li>
      <li><strong>존별 단독 변화 조건이 각 1~2개뿐입니다.</strong> Zone 3 = 1.5 는 웨이퍼 #9 하나뿐이라,
        그 조건을 빼고 예측하면 오차가 286 A/min 으로 전체 평균의 1.6배까지 뜁니다.</li>
      <li><strong>존 압력끼리 상관되어 있습니다.</strong> 고압군은 세 존이 함께 올라갑니다.
        영향함수의 최외곽 음수 계수가 그 부작용입니다.</li>
    </ul>

    <h3>다음 실험을 설계한다면</h3>
    <ul>
      <li>2.5 ~ 2.7 psi 공백 구간 채우기 (2~3 조건)</li>
      <li>Zone 3 저압(1.5) 반복 및 Zone 3 단독 수준 추가</li>
      <li>세 존을 서로 <strong>반대 방향</strong>으로 움직이는 조건 — 다중공선성 제거</li>
      <li>r = ±72~74 mm 구간 반복 측정 — 전체 오차의 대부분이 여기서 나옵니다</li>
    </ul>
  </div>
</section>

<hr class="rule">

<section>
  <div class="col">
    <h2><span class="num">09</span>사용</h2>
    <p>존 압력 3개만 입력하면 43지점 프로파일과 반경별 95 % 예측구간, 균일도 지표가 나옵니다.
      학습 설계점에서 0.3 psi 이상 떨어지면 외삽 경고를 출력합니다.</p>
  </div>
  <pre>pip install -r requirements.txt
python scripts/train.py                     # 학습 → models/mrr_model.pkl
python scripts/predict.py 2.0 2.2 2.0       # Zone1 Zone2 Zone3 [psi] → 프로파일

# 역방향: 목표 균일도를 만드는 압력 레시피 탐색
python scripts/optimize.py --mode uniform --target-mean 1200 --edge-exclude 70
  → Zone1 = 2.03  Zone2 = 2.57  Zone3 = 2.38 psi
     예상 평균 MRR 1200 A/min,  불균일도 1.58 % (1σ, |r| ≤ 70 mm)</pre>
</section>

<footer>
  학습 데이터 Train_1.csv · 웨이퍼 21장 × 반경 43지점 · 고유 압력 조건 16개<br>
  평가 방식 Leave-One-Design-Out 교차검증 (반복 웨이퍼 동시 제외) · 부트스트랩 SE 2000회
</footer>

</div>
'''

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML, encoding="utf-8")
print(f"{OUT} 저장 ({OUT.stat().st_size/1024:.0f} KB)")
