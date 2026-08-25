"""압력 -> MRR 프로파일 + WIWNU 예측 (CLI).

    python app/predict.py 2.0 2.2 2.0
    python app/predict.py 2.0 2.2 2.0 --csv out.csv --plot out.png
    python app/predict.py --batch recipes.csv        (Z1,Z2,Z3 3열 CSV 일괄 처리)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrr_model import QuadRidgeMRR, wiwnu  # noqa: E402

ap = argparse.ArgumentParser(
    description="SiO2 CMP 3-Zone 압력에서 MRR 프로파일과 WIWNU 예측",
    formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("z1", type=float, nargs="?", help="Zone 1 압력 [psi] (최외곽 에지)")
ap.add_argument("z2", type=float, nargs="?", help="Zone 2 압력 [psi] (중간 링)")
ap.add_argument("z3", type=float, nargs="?", help="Zone 3 압력 [psi] (중심)")
ap.add_argument("--model", default=str(Path(__file__).with_name("model.json")))
ap.add_argument("--csv", help="프로파일을 CSV 로 저장")
ap.add_argument("--plot", help="프로파일 그림을 PNG 로 저장 (matplotlib 필요)")
ap.add_argument("--batch", help="Z1,Z2,Z3 3열 CSV 를 읽어 일괄 예측")
ap.add_argument("--no-interval", action="store_true", help="95%% 예측구간 숨기기")
a = ap.parse_args()

m = QuadRidgeMRR.load(a.model)
r = m.radius_
order = np.argsort(-r)                                   # +74 -> -74 로 출력


def banner(z1, z2, z3):
    print()
    print("=" * 62)
    print(f" 입력 압력   Zone1 = {z1:g}   Zone2 = {z2:g}   Zone3 = {z3:g}  [psi]")
    print("=" * 62)
    d = m.extrapolation_distance(z1, z2, z3)
    if d > 0.3:
        print(f" [경고] 학습 조건에서 {d:.2f} psi 떨어진 외삽 영역입니다.")
        print("        예측 신뢰도가 낮으니 검증 실험을 권장합니다.")
        print("-" * 62)


def report(z1, z2, z3):
    banner(z1, z2, z3)
    y = m.predict_one(z1, z2, z3)
    w = wiwnu(y, r)

    print("\n [MRR Profile]")
    if a.no_interval or not m.sigma_.any():
        print(f"   {'r (mm)':>8s}{'MRR (A/min)':>14s}")
        print("   " + "-" * 22)
        for i in order:
            print(f"   {r[i]:8.0f}{y[i]:14.1f}")
    else:
        lo, hi = y - 1.96 * m.sigma_, y + 1.96 * m.sigma_
        print(f"   {'r (mm)':>8s}{'MRR (A/min)':>14s}{'95% 예측구간':>26s}")
        print("   " + "-" * 48)
        for i in order:
            print(f"   {r[i]:8.0f}{y[i]:14.1f}      [{lo[i]:8.1f}, {hi[i]:8.1f}]")

    print("\n [WIWNU]   (표본표준편차 / 평균) x 100")
    print(f"   {'구간':<8s}{'범위 (mm)':>16s}{'점수':>6s}"
          f"{'평균 MRR':>12s}{'표준편차':>11s}{'WIWNU':>10s}")
    print("   " + "-" * 63)
    for name in ("all", "5mm", "7mm"):
        v = w[name]
        print(f"   {name:<8s}{v['range_mm']:>16s}{v['n_points']:>6d}"
              f"{v['mean']:12.1f}{v['std']:11.1f}{v['wiwnu_pct']:9.2f}%")
    print("   " + "-" * 63)
    print(f"   최소 / 최대 MRR (전 구간) : {w['all']['min']:.1f} / {w['all']['max']:.1f} A/min")
    print(f"\n   모델 교차검증 정확도 : RMSE {m.cv_.get('rmse', 0):.1f} A/min "
          f"({m.cv_.get('mape', 0):.1f}%),  학습 웨이퍼 {m.cv_.get('n_wafer', 0)}장")
    return y, w


# --------------------------------------------------------------- 일괄 처리
if a.batch:
    rows = [ln.split(",") for ln in
            Path(a.batch).read_text(encoding="utf-8-sig").strip().splitlines()]
    if not rows[0][0].strip().replace(".", "").replace("-", "").isdigit():
        rows = rows[1:]                                   # 헤더 건너뛰기
    print(f"{'Z1':>6s}{'Z2':>6s}{'Z3':>6s}{'평균MRR':>11s}"
          f"{'WIWNU all':>11s}{'WIWNU 5mm':>11s}{'WIWNU 7mm':>11s}")
    print("-" * 67)
    out = []
    for row in rows:
        z = [float(v) for v in row[:3]]
        y = m.predict_one(*z)
        w = wiwnu(y, r)
        print(f"{z[0]:6g}{z[1]:6g}{z[2]:6g}{w['all']['mean']:11.1f}"
              f"{w['all']['wiwnu_pct']:10.2f}%{w['5mm']['wiwnu_pct']:10.2f}%"
              f"{w['7mm']['wiwnu_pct']:10.2f}%")
        out.append([*z, w["all"]["mean"], w["all"]["wiwnu_pct"],
                    w["5mm"]["wiwnu_pct"], w["7mm"]["wiwnu_pct"]])
    if a.csv:
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("Zone1,Zone2,Zone3,mean_MRR,WIWNU_all,WIWNU_5mm,WIWNU_7mm\n")
            for o in out:
                f.write(",".join(f"{v:.4f}" for v in o) + "\n")
        print(f"\nCSV 저장: {a.csv}")
    sys.exit(0)

# --------------------------------------------------------------- 단건 처리
if a.z1 is None or a.z2 is None or a.z3 is None:
    ap.error("Zone1 Zone2 Zone3 세 개의 압력을 입력하세요.  예)  python app/predict.py 2.0 2.2 2.0")

y, w = report(a.z1, a.z2, a.z3)

if a.csv:
    with open(a.csv, "w", encoding="utf-8") as f:
        f.write("radius_mm,MRR_pred\n")
        for i in order:
            f.write(f"{r[i]:g},{y[i]:.4f}\n")
        f.write("\nband,range_mm,n_points,mean,std,WIWNU_pct\n")
        for name in ("all", "5mm", "7mm"):
            v = w[name]
            f.write(f"{name},{v['range_mm']},{v['n_points']},"
                    f"{v['mean']:.4f},{v['std']:.4f},{v['wiwnu_pct']:.4f}\n")
    print(f"\nCSV 저장: {a.csv}")

if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = np.argsort(r)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if m.sigma_.any():
        ax.fill_between(r[o], (y - 1.96 * m.sigma_)[o], (y + 1.96 * m.sigma_)[o],
                        alpha=.2, color="C0", label="95% prediction interval")
    ax.plot(r[o], y[o], "-o", ms=3.5, color="C0", label="predicted MRR")
    ax.axhline(w["all"]["mean"], ls="--", c="gray", lw=.8)
    for b, c in ((70, "C1"), (68, "C3")):
        ax.axvline(b, ls=":", c=c, lw=1); ax.axvline(-b, ls=":", c=c, lw=1)
    ax.set_xlabel("Radius [mm]"); ax.set_ylabel("MRR [A/min]")
    ax.set_title(f"Z1={a.z1:g}, Z2={a.z2:g}, Z3={a.z3:g} psi   |   "
                 f"WIWNU all {w['all']['wiwnu_pct']:.2f}%  "
                 f"5mm {w['5mm']['wiwnu_pct']:.2f}%  "
                 f"7mm {w['7mm']['wiwnu_pct']:.2f}%")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(a.plot, dpi=130)
    print(f"그림 저장: {a.plot}")
