"""GUI 버전 — 압력 3개를 넣고 버튼을 누르면 프로파일과 WIWNU 가 나온다.

    python app/gui.py

필요: tkinter (파이썬 기본 포함), matplotlib.
tkinter 가 없으면 (리눅스에서 가끔) 아래로 설치:
    sudo apt install python3-tk
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrr_model import QuadRidgeMRR, wiwnu  # noqa: E402

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    sys.exit("tkinter 가 없습니다.  Windows/macOS 는 기본 포함, "
             "Ubuntu 는  sudo apt install python3-tk")

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  # noqa: E402
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure  # noqa: E402

MODEL_PATH = Path(__file__).with_name("model.json")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SiO₂ CMP  3-Zone 압력 → MRR Profile / WIWNU")
        self.geometry("1120x740")
        self.minsize(940, 640)

        try:
            self.model = QuadRidgeMRR.load(MODEL_PATH)
        except FileNotFoundError:
            messagebox.showerror("모델 없음",
                                 f"{MODEL_PATH} 가 없습니다.\n먼저 실행하세요:\n\n"
                                 f"    python app/train.py")
            self.destroy(); return

        self.r = self.model.radius_
        self.result = None
        self._build()
        self.run()                       # 시작하자마자 기본값으로 한 번 계산

    # ------------------------------------------------------------------ UI
    def _build(self):
        left = ttk.Frame(self, padding=14)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="압력 입력  [psi]",
                  font=("", 13, "bold")).pack(anchor="w", pady=(0, 2))
        cv = self.model.cv_
        ttk.Label(left, foreground="#666",
                  text=f"학습 웨이퍼 {cv.get('n_wafer', 0)}장 · "
                       f"CV RMSE {cv.get('rmse', 0):.0f} A/min").pack(anchor="w", pady=(0, 12))

        self.var = {}
        rows = [("Zone 1", "최외곽 에지", 2.0),
                ("Zone 2", "중간 링", 2.2),
                ("Zone 3", "중심", 2.0)]
        lo = float(self.model.train_pressure_.min())
        hi = float(self.model.train_pressure_.max())
        for name, desc, default in rows:
            box = ttk.Frame(left); box.pack(fill="x", pady=6)
            ttk.Label(box, text=f"{name}  ({desc})",
                      font=("", 10, "bold")).pack(anchor="w")
            v = tk.DoubleVar(value=default)
            self.var[name] = v
            sub = ttk.Frame(box); sub.pack(fill="x", pady=(3, 0))
            sc = ttk.Scale(sub, from_=lo - 0.3, to=hi + 0.3, variable=v,
                           command=lambda _e, vv=v: vv.set(round(vv.get(), 2)))
            sc.pack(side="left", fill="x", expand=True)
            ttk.Entry(sub, textvariable=v, width=7).pack(side="left", padx=(8, 0))
        ttk.Label(left, foreground="#666",
                  text=f"학습 범위 {lo:g} ~ {hi:g} psi").pack(anchor="w", pady=(2, 12))

        ttk.Button(left, text="예측 실행", command=self.run).pack(fill="x", pady=(0, 6))
        ttk.Button(left, text="CSV 저장", command=self.save_csv).pack(fill="x", pady=2)
        ttk.Button(left, text="그림 저장", command=self.save_png).pack(fill="x", pady=2)

        ttk.Separator(left).pack(fill="x", pady=14)
        ttk.Label(left, text="WIWNU  = (표본표준편차 / 평균) × 100",
                  font=("", 11, "bold")).pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("range", "mean", "std", "nu"),
                                 show="headings", height=3)
        for c, t, w in (("range", "범위(mm)", 96), ("mean", "평균MRR", 78),
                        ("std", "표준편차", 72), ("nu", "WIWNU", 72)):
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="e")
        self.tree.pack(fill="x", pady=8)

        self.info = tk.Text(left, height=7, width=42, relief="flat",
                            background="#f4f4f4", font=("", 9))
        self.info.pack(fill="x")

        self.fig = Figure(figsize=(7.4, 5.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()

    # ------------------------------------------------------------- 계산
    def run(self):
        try:
            z = [float(self.var[k].get()) for k in ("Zone 1", "Zone 2", "Zone 3")]
        except (tk.TclError, ValueError):
            messagebox.showwarning("입력 오류", "압력은 숫자로 입력하세요.")
            return
        y = self.model.predict_one(*z)
        w = wiwnu(y, self.r)
        self.result = (z, y, w)

        self.tree.delete(*self.tree.get_children())
        for name in ("all", "5mm", "7mm"):
            v = w[name]
            self.tree.insert("", "end", values=(
                v["range_mm"], f"{v['mean']:.1f}", f"{v['std']:.1f}",
                f"{v['wiwnu_pct']:.2f}%"))

        d = self.model.extrapolation_distance(*z)
        self.info.delete("1.0", "end")
        self.info.insert("end",
                         f"평균 MRR (전 구간) : {w['all']['mean']:.1f} A/min\n"
                         f"최소 / 최대        : {w['all']['min']:.1f} / {w['all']['max']:.1f}\n"
                         f"학습 조건까지 거리 : {d:.2f} psi\n")
        if d > 0.3:
            self.info.insert("end",
                             "\n[경고] 학습 압력 조건에서 멀리 떨어진\n"
                             "외삽 영역입니다. 예측 신뢰도가 낮으니\n"
                             "검증 실험을 권장합니다.\n")
        self.draw(z, y, w)

    def draw(self, z, y, w):
        ax, r = self.ax, self.r
        ax.clear()
        o = np.argsort(r)
        s = self.model.sigma_
        if s.any():
            ax.fill_between(r[o], (y - 1.96 * s)[o], (y + 1.96 * s)[o],
                            alpha=.18, color="C0", label="95% prediction interval")
        ax.plot(r[o], y[o], "-o", ms=3.5, lw=1.8, color="C0", label="predicted MRR")
        ax.axhline(w["all"]["mean"], ls="--", c="gray", lw=.8)
        for b, c, lb in ((70, "C1", "±70 mm (5 mm EE)"), (68, "C3", "±68 mm (7 mm EE)")):
            ax.axvline(b, ls=":", c=c, lw=1.2, label=lb)
            ax.axvline(-b, ls=":", c=c, lw=1.2)
        ax.set_xlabel("Radius [mm]")
        ax.set_ylabel("MRR [Å/min]")
        ax.set_title(f"Z1={z[0]:g}  Z2={z[1]:g}  Z3={z[2]:g} psi     "
                     f"WIWNU  all {w['all']['wiwnu_pct']:.2f}% / "
                     f"5mm {w['5mm']['wiwnu_pct']:.2f}% / "
                     f"7mm {w['7mm']['wiwnu_pct']:.2f}%", fontsize=10)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc="upper center")
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------- 저장
    def save_csv(self):
        if not self.result:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        z, y, w = self.result
        with open(p, "w", encoding="utf-8-sig") as f:
            f.write(f"Zone1,{z[0]}\nZone2,{z[1]}\nZone3,{z[2]}\n\n")
            f.write("radius_mm,MRR_pred\n")
            for i in np.argsort(-self.r):
                f.write(f"{self.r[i]:g},{y[i]:.4f}\n")
            f.write("\nband,range_mm,n_points,mean,std,WIWNU_pct\n")
            for name in ("all", "5mm", "7mm"):
                v = w[name]
                f.write(f"{name},{v['range_mm']},{v['n_points']},"
                        f"{v['mean']:.4f},{v['std']:.4f},{v['wiwnu_pct']:.4f}\n")
        messagebox.showinfo("저장 완료", p)

    def save_png(self):
        if not self.result:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.fig.savefig(p, dpi=150)
            messagebox.showinfo("저장 완료", p)


if __name__ == "__main__":
    App().mainloop()
