# MRR Profile / WIWNU 예측기

3-zone 압력을 넣으면 −74~+74 mm 43지점의 MRR Profile 과 WIWNU 를 계산합니다.
**학습 데이터가 다른 두 모델**을 같은 방식(Quad Ridge)으로 만들어 두었습니다.

| 모델 | 학습 데이터 | 브라우저 앱 | 모델 파일 |
|---|---|---|---|
| **Train_1** | 웨이퍼 21장 | `app/mrr_app_train1.html` | `app/model_train1.json` |
| **Train_2** | 웨이퍼 18장 | `app/mrr_app.html` | `app/model.json` |

Train_1 은 Train_2 에 웨이퍼 19·20·21 (2.7 / 3.2 / 3.0 psi 반복 3장) 이 추가된 것입니다.
두 모델 비교는 `python app/compare.py` 로 볼 수 있습니다.

---

## 실행 방법 3가지

### 1. 브라우저 (설치 필요 없음 — 가장 간단)

`app/mrr_app_train1.html` 또는 `app/mrr_app.html` 을 더블클릭해서 브라우저로 엽니다. 끝입니다.
파이썬도 인터넷도 필요 없고, 슬라이더를 움직이면 실시간으로 다시 계산됩니다.
두 파일을 나란히 열어두면 같은 압력에서 두 모델을 바로 비교할 수 있습니다.

### 2. 명령줄

```bash
pip install numpy                 # 필수
pip install matplotlib            # --plot 쓸 때만

python app/predict.py 2.0 2.2 2.0                              # 기본 = Train_2 모델
python app/predict.py 2.0 2.2 2.0 --model app/model_train1.json  # Train_1 모델
python app/predict.py 2.0 2.2 2.0 --csv out.csv --plot out.png
python app/predict.py --batch recipes.csv --csv summary.csv    # 여러 조건 일괄

python app/compare.py                    # 두 모델 요약 비교
python app/compare.py 2.0 2.2 2.0        # 특정 압력에서 상세 비교
```

`--batch` 에 넣는 CSV 는 `Zone1,Zone2,Zone3` 3열이면 됩니다 (헤더 있어도 됨).

### 3. 데스크톱 GUI

```bash
pip install numpy matplotlib
python app/gui.py                                  # Train_2 모델
python app/gui.py --model app/model_train1.json    # Train_1 모델
```

슬라이더로 압력을 조절하면 그래프와 WIWNU 표가 바로 갱신됩니다.
리눅스에서 `tkinter` 오류가 나면 `sudo apt install python3-tk` 로 설치하세요.
(윈도우·맥은 파이썬에 기본 포함되어 있습니다.)

---

## 출력 예시

```
$ python app/predict.py 2.0 2.2 2.0

 [WIWNU]   (표본표준편차 / 평균) x 100
   구간               범위 (mm)    점수      평균 MRR       표준편차     WIWNU
   ---------------------------------------------------------------
   all            -74 ~ +74    43      1206.9      443.3    36.73%
   5mm            -70 ~ +70    35      1028.0       61.2     5.96%
   7mm            -68 ~ +68    33      1015.1       30.2     2.97%
```

### WIWNU 정의

```
WIWNU = (표본표준편차 / 평균) × 100          [%]
```

| 구간 | 반경 범위 | 점 개수 | 의미 |
|---|---|---|---|
| `all` | −74 ~ +74 mm | 43 | 측정 전 구간 |
| `5mm` | −70 ~ +70 mm | 35 | edge exclusion 5 mm |
| `7mm` | −68 ~ +68 mm | 33 | edge exclusion 7 mm |

표준편차는 세 구간 모두 **표본표준편차**(Excel `STDEV.S`, numpy `ddof=1`)입니다.

---

## 모델

반경 r 마다 독립적으로 능형회귀를 적합합니다.

```
MRR(r) = a(r) + Σ_j B_j(r) · f_j(P)

f(P) = [Z1, Z2, Z3, Z1², Z2², Z3², Z1·Z2, Z1·Z3, Z2·Z3]
```

- 특징은 학습셋 평균·표준편차로 표준화 후 사용
- 절편에는 규제를 걸지 않고 기울기 계수에만 `alpha` 적용
- `alpha = 1.0` — 동일 압력 조건 웨이퍼를 통째로 빼는
  Leave-One-Design-Out 교차검증으로 선택

### 학습 성능 (본 적 없는 압력 조건 기준)

두 모델 모두 `alpha = 1.0` 이 선택되었습니다.

| 항목 | Train_1 (21장) | Train_2 (18장) |
|---|---:|---:|
| 전 구간 RMSE | **163.6** Å/min | 168.4 Å/min |
| MAPE | **5.89 %** | 6.11 % |
| edge exclusion 5 mm 기준 RMSE | 69.0 Å/min | **67.8** Å/min |
| 측정 재현성 한계 (반복 웨이퍼) | 108.7 Å/min | 123.9 Å/min |

WIWNU 예측 절대오차 (교차검증 평균):

| 구간 | Train_1 | Train_2 |
|---|---:|---:|
| all | 6.15 %p | 6.70 %p |
| 5mm | **2.61 %p** | 2.65 %p |
| 7mm | **1.98 %p** | 1.89 %p |

웨이퍼 3장 차이라 성능 차이는 작습니다. 다만 추가된 3장이 모두 고압
(2.7 / 3.2 / 3.0 psi) 반복 조건이라, **고압 영역 예측에서 차이가 벌어집니다.**

| 압력 (Z1,Z2,Z3) | 평균 MRR T1 → T2 | 프로파일 차이 RMS |
|---|---|---:|
| 2.0, 2.2, 2.0 | 1205.8 → 1206.9 | 4.7 |
| 2.0, 1.5, 2.0 | 1189.5 → 1180.4 | 19.8 |
| 2.5, 2.5, 2.5 | 1479.3 → 1492.1 | 26.5 |
| **3.0, 3.0, 3.0** | 1799.2 → 1830.5 | **63.0** |
| **2.7, 3.2, 3.0** | 1700.1 → 1739.6 | **77.0** |

저압~중압 조건은 두 모델이 사실상 같고, 고압 조건은 Train_1 쪽이 실측 반복
데이터를 갖고 있으므로 **고압 예측은 Train_1 모델을 쓰는 것이 낫습니다.**

`all` 구간 오차가 큰 이유는 최외곽 ±72~74 mm 의 측정 산포 자체가 크기 때문입니다
(그 지점 반복 측정 편차가 300~500 Å/min). **실무 판단에는 `5mm` / `7mm` 를 쓰세요.**

### 주의

- 학습 압력 범위는 **Z1 1.3~3.0, Z2 1.5~3.2, Z3 1.5~3.0 psi** 입니다.
- 학습 조건에서 0.3 psi 이상 떨어지면 세 프로그램 모두 **외삽 경고**를 띄웁니다.
- 2.5 ~ 2.7 psi 구간은 학습 데이터가 비어 있어 보간이 아니라 사실상 외삽입니다.

---

## 다시 학습하기

데이터를 추가했거나 다른 CSV 로 학습하려면:

```bash
python app/train.py --csv data/Train_1.csv --out app/model_train1.json --label Train_1
python app/build_web.py app/mrr_app_train1.html --model app/model_train1.json

python app/train.py --csv data/Train_2.csv --out app/model.json --label Train_2
python app/build_web.py app/mrr_app.html     --model app/model.json
```

`train.py` 는 alpha 후보를 교차검증으로 훑어 최적값을 자동 선택하고,
학습된 계수를 사람이 읽을 수 있는 JSON (`app/model.json`) 으로 저장합니다.

---

## 파일 구성

```
app/
  mrr_model.py    모델 본체 — 특징변환 / 학습 / 예측 / WIWNU / 교차검증
  train.py        학습 스크립트 (alpha 자동 선택)
  predict.py      명령줄 예측기 (단건 + 일괄)
  gui.py          데스크톱 GUI (tkinter)
  build_web.py    model.json -> 단일 HTML 앱 생성기
  compare.py      Train_1 / Train_2 두 모델 비교

  mrr_app_train1.html   브라우저 앱 — Train_1 (21장)   [생성물, 의존성 없음]
  model_train1.json     학습된 계수 — Train_1          [생성물]
  mrr_app.html          브라우저 앱 — Train_2 (18장)   [생성물, 의존성 없음]
  model.json            학습된 계수 — Train_2          [생성물]
```
