## 전체 표 목록 (2026-07-30 확정 번호 기준 — §4.2 신설로 밀린 최종 절 번호)

### 본문

| # | 위치 | 내용 | 상태 |
| --- | --- | --- | --- |
| Table 1 | §4.1 | calib(6행: 단일언어 5 + Balanced) × eval(5열) Δbpb% 격자, 30칸, 대각선 볼드 | ✅ 확정 |
| Table 2 | §4.1 | 판정표: seed별 own_gain / mean / 부호 일관성 / noise_floor / 라벨 | ✅ 확정 — BN(+ZH)는 SIGN_CONSISTENT_NO_FLOOR로 최종 확정(2026-07-30 확장 포기, 밤 큐 ⑦ 취소 — chinese_only_b/bengali_only_b 신규 실행 안 함) |
| Table 3 | §4.2(downstream) | Belebele retention — 현재 korean_only × EN/KO/ZH 1행 구조 | 🔄 korean_only분 확정(아래 실측), english_only/mixed 완주 시 3조건 격자로 확장(TODO-D1) |
| Table 4 | §6.2 | balanced vs expert-balanced vs 제안 배분의 Δbpb% 비교 | ⏳ SWITCH-4 대기 — 버전 A면 이 표, 버전 B면 "변형별 worst-language vs balanced" 표로 형태 변경 |

### 부록

| # | 위치 | 내용 | 상태 |
| --- | --- | --- | --- |
| 표 A1 | 부록 A | calibration 조건별 실측 토큰 길이(raw/realized/cap도달률/fertility) | ✅ 확정(06_논문_구성.md에 원문) |
| 표 C1 | 부록 C | 언어별 tokens/byte(n_tokens, n_bytes) | ✅ 확정 — 번호 없는 인라인 표였음, 이번에 C1 번호 부여 |
| 표 C2 | 부록 C | 언어별 baseline bpb + tokens/byte 병기 | ✅ 확정(BN 0.8857) |
| 표 D1 | 부록 D | Table 1 30칸 per-seed Δbpb% + std | ✅ 확정 — std ddof=1로 교체 완료, Balanced×ben std 0.08→**0.09**로 잘림 복구 완료(2026-07-30) |
| 표 D2 | 부록 D | placebo 3조건 대각선 per-seed Δbpb% + 본 조건 대비 | ✅ 확정(Δbpb% 버전) |
| 표 E1 | 부록 E | flagged 36쌍 (layer, expert, 언어) 전체 목록 | ✅ 확정(`phase1_52b_deficiency_pairs_list_result.json`) |
| 표 E2 | 부록 E | deficiency 임계값 민감도(threshold 15/20/25/30 × 언어) | ✅ 확정(`phase1_52c_deficiency_robustness_check_result.json`) |
| 표 F1 | 부록 F | 언어쌍 10개 + placebo 3쌍의 Fisher 순위 ρ(mean±std, 27-layer 평균, seed0) | ✅ 확정 |
| 표 F2 | 부록 F | §5.1 헤드라인 수치의 seed 0/1/2 일관성 | ✅ 확정 |
| 표 G1 | 부록 G | whitening 지표별 between/within ratio — 전체/tertile | ✅ 확정 |
| B.3 인라인 | 부록 B | 판정 라벨 4종(SUPPORTED/INCONCLUSIVE/NOT_SUPPORTED/SIGN_CONSISTENT_NO_FLOOR) 정의 | ✅ 내용 확정, 표/텍스트 형식만 선택 |

**참고**: 이전에 "Table 3"으로 부르던 §4.3 Fisher×Scale 2×2 + pruning ablation 내용(원래 완성본)은 위 4개 목록에 없음 — §4.2 신설로 번호가 밀리면서 아직 새 번호를 못 받았음. 내용 자체는 완성돼 있으니(아래 "§4.3 Fisher×Scale 경로 분리" 절 참고) 최종 번호만 확정해서 알려주면 이 문서에 반영.

## 전체 그림 목록

| # | 내용 | 실물 상태 | 참조 위치 | 판단 |
| --- | --- | --- | --- | --- |
| Figure 1(기존) | layer-locality 진단(proxy divergence, 구간 ρ=0.85, ratio=14.4, Qwen3-30B-A3B) | ✅ 확인함 — `0726_results/1_layer_locality.png` | §4.1 도입부(설계 동기), §5.1 구판 | **삭제 확정 권장**: §5.1 관련 문서 전체 검색 결과 이 파일을 데이터-의존적으로 참조하는 곳 없음(전부 "layer-locality" 개념·역사만 서술, §4.1 도입 동기로만 인용). Figure 3 panel A가 같은 주장을 DeepSeek 실측으로 대체. 파일 자체는 `0726_results/`에 보존, 논문 Figure 번호에서는 제외 |
| Figure 2 | proxy–real Fisher 검증(ρ=0.870, Fisher Pilot A) | ✅ 확인함 — `figures/figure2_fisher_pilot_a_validation.png`(원본 `0726_results/2_fisher_pilot_a_validation.png`에서 복사) | §6.1(프록시 정당화) | 본문/부록 배치 재확인 필요하나 실물 확정, 재생성 불필요 |
| Figure 3 | 3-panel 통합(§5.1 ρ / §5.2 결핍 / §5.3 ratio) | ✅ 있음 | §5.1, §5.2, §5.3 | 본문 확정. 생성: `scripts/make_figure3_whitening_evidence.py` → `figures/figure3_whitening_layer_evidence.png` |
| Figure 4 | between vs within 분포 바이올린 | ✅ 있음 | §5.3 | 본문 확정. 생성: `scripts/make_figure4_whitening_distributions.py` → `figures/figure4_whitening_distributions.png` |
| 그림 G1 | whitening raw 절대값 곡선 | ✅ 있음 | 부록 G | 확정. `scripts/make_figure_whitening_geometry.py` → `figures/figureG1_whitening_layer_curve_raw.png` |
| 그림 G2 | whitening heatmap | ✅ 있음 | 부록 G | 확정. `scripts/make_figure_whitening_geometry.py` → `figures/figureG2_whitening_heatmap.png` |
| Figure A(별도) | §4.3 Fisher/whitening/pruning 세 경로 크기 막대 | ✅ 있음 | §4.3(번호 미확정, 위 참고) | `scripts/make_figure_a_path_decomposition.py` → `figures/figure_a_path_decomposition.png` |

---

## 본문 상세

### **Table 1 — §4.1 calib × eval 격자** [완성 — `phase1_41_headline_gate.py` 공식 출력]

6행(5 단일언어 + Balanced) × 5열, 3-seed 평균 Δbpb% (baseline 대비, 낮을수록 좋음). 대각선(own-language) 칸 볼드 처리.

| Calib \ Eval | eng_Latn | kor_Hang | zho_Hans |  swh_Latn | ben_Beng |
| ------------ | -------: | -------: | -------: | --------: | -------: |
| English      | **7.28** |    88.76 |    13.80 |     33.09 |    89.02 |
| Korean       |    12.68 | **3.86** |    12.68 |     38.52 |    25.24 |
| Chinese      |    10.29 |    79.28 | **4.24** |     35.10 |    87.41 |
| Swahili      |    12.83 |    66.65 |    15.20 | **-0.05** |    75.06 |
| Bengali      |    17.17 |    23.91 |    18.31 |     35.30 | **1.33** |
| Balanced     |     9.29 |     5.36 |     7.10 |      2.83 |     2.90 |

* 캡션: "whitening-ON 전체 파이프라인, 3-seed 평균, 낮을수록 좋음. Placebo는 미포함(부록 D2 참고), per-seed 값은 부록 D1."
* 데이터 출처: `phase1_41_headline_gate_result.json`(`grid` 필드).
* Bengali 자기 행(1.33)이 30칸 중 가장 낮은 값 — own-language 보호 효과가 가장 강하지만, 대응 placebo(bengali_only_b)가 없어 Table 2에서는 SIGN_CONSISTENT_NO_FLOOR로 보수적으로 처리됨.

### **Table 2 — §4.1 headline gate 판정표** [완성 — `phase1_41_headline_gate.py` 공식 출력]

| 언어     | seed별 own_gain          | mean own_gain | 부호 일관성 |  noise_floor | 판정                     |
| -------- | ------------------------ | ------------: | :---------: | -----------: | ------------------------ |
| eng_Latn | +5.56 / +6.64 / +5.69    |     **+5.96** |     ✓     |         4.05 | INCONCLUSIVE             |
| kor_Hang | +60.23 / +60.04 / +62.08 |    **+60.78** |     ✓     |         0.79 | **SUPPORTED**            |
| zho_Hans | +11.29 / +10.40 / +10.59 |    **+10.76** |     ✓     | (no placebo) | SIGN_CONSISTENT_NO_FLOOR |
| swh_Latn | +36.35 / +34.33 / +35.97 |    **+35.55** |     ✓     |         0.91 | **SUPPORTED**            |
| ben_Beng | +66.70 / +67.43 / +69.42 |    **+67.85** |     ✓     | (no placebo) | SIGN_CONSISTENT_NO_FLOOR |

* own_gain = 자기 언어 calibration이 자기 언어 평가에서 다른 단일언어 calibration들 평균보다 얼마나 나은가.
* noise_floor는 placebo(\_b) 있는 언어(EN/KO/SW)만 계산 가능 — ZH/BN은 SUPPORTED 승격이 구조적으로 불가능(SIGN_CONSISTENT_NO_FLOOR 상한).
* **Bengali는 mean own_gain=+67.85로 30칸 중 최대 효과**이나 placebo 부재로 SUPPORTED 미승격 — chinese_only_b/bengali_only_b 확장은 **2026-07-30 포기 확정(밤 큐 ⑦ 취소)**, SIGN_CONSISTENT_NO_FLOOR로 최종.
* 데이터 출처: `phase1_41_headline_gate_result.json`(`own_gain` 필드).

### **Table 3 — §4.2 Belebele downstream retention** [🔄 korean_only 확정, 확장 대기]

**full-pipeline(Fisher+whitened SVD+pruning) retention, condition=korean_only, pp_ratio=0.2, 5-shot n=200×3seed**

| 평가 언어 | baseline acc | retention(OFF, no pruning) | retention(ON, pruning) |
| --------- | ------------: | --------------------------: | -----------------------: |
| eng_Latn  | 0.525 | 0.625 | 0.603 |
| kor_Hang  | 0.360 | **0.819** | **0.708** |
| zho_Hans  | 0.470 | 0.663 | 0.660 |
| swh_Latn  | 0.285 | 0.906 [FLAGGED-AT-CHANCE] | 0.848 [FLAGGED-AT-CHANCE] |
| ben_Beng  | 0.285 | 0.977 [FLAGGED-AT-CHANCE] | 0.965 [FLAGGED-AT-CHANCE] |
| **macro-avg(비flagged만)** | | **0.703** | **0.657** |
| **worst-language(비flagged만, =EN)** | | **0.625** | **0.603** |

* **해석**: pruning(ON) 추가 시 macro 0.703→0.657, worst-language 0.625→0.603로 유의미하게 하락 — "pruning까지 포함한 전체 파이프라인에서도 own-language 보호 효과가 살아있는가"에 대한 답은 **부분적으로만 그렇다**(효과 자체는 존재하나 pruning이 갉아먹음).
* swh_Latn/ben_Beng은 `phase1_belebele_floor_check.py`에서 chance(25%) 근방으로 flagged되어 macro/worst 계산에서 제외.
* **TODO-D1**: english_only, mixed_5lang(Balanced) 조건도 같은 방식으로 돌려서 "3(또는 4)조건 × 5언어" 격자로 확장하면 Table 3이 완성됨 — 현재는 korean_only 1행뿐.
* 데이터 출처: `/mnt/HDD/minjeong/d2moe_results/phase1/phase1_belebele_gate_korean_only_result.json`.

### §4.3 Fisher×Scale 경로 분리 [완성 — 번호 미확정, 위 "전체 표 목록" 참고]

**패널 A. Fisher × Scale 2×2 (KO 평가 기준, 3-seed 평균 bpb 증가율 %, baseline 대비)**

| Fisher 조건 | Scale=EN | Scale=KO | own_scale_gain (EN−KO) |
| ----------- | -------: | -------: | ------------------------: |
| Fisher=EN   |    88.76 |     4.84 |               **83.92%p** |
| Fisher=KO   |    42.81 |     3.86 |               **38.95%p** |

Placebo(Fisher=KO 고정, Scale을 동일 언어·다른 표본으로 대체):

|               | Scale=EN_b | Scale=KO_b |
| ------------- | ----------: | ----------: |
| bpb 증가율(%) |       48.64 |        4.42 |

* `noise_floor_scale` = max(\|Scale=KO−KO_b\|, \|Scale=EN−EN_b\|), 3-seed 중 최댓값 = **6.88%p**. threshold(2×floor) = 13.76%p.
* mean own_scale_gain(KO | Fisher=KO) = 38.95%p → **VERDICT: SUPPORTED (noise floor 대비 5.66배)**.
* Path-independence: Fisher=EN 고정 시 own_scale_gain = 83.92%p(Fisher=KO보다 큼) → whitening 경로가 KO 보호의 주범이라는 결론이 Fisher 계산 언어와 무관하게 재현됨.
* **해석**: Scale(=whitening) 축만 EN→KO로 바꿔도 KO bpb 증가율이 ~43%p에서 ~4%p로 떨어짐. §4.1 Fisher merge 단독 이득(+5.60%p)의 7배 이상 — KO 보호 효과 대부분이 whitening 단계에서 발생.
* per-seed 원자료: Fisher=KO, Scale=EN [42.94, 42.47, 43.02] / Scale=KO [3.66, 4.07, 3.86] / Scale=EN_b [47.75, 49.35, 48.81] / Scale=KO_b [4.50, 4.67, 4.10]; Fisher=EN, Scale=EN [88.68, 85.38, 92.21] / Scale=KO [4.65, 4.98, 4.89].

**패널 B. Pruning on/off ablation — korean_only 단일 조건** (Fisher/Scale=KO 고정, pp_ratio=0.2)

| 평가 언어 | seed0 |  seed1 |  seed2 |  mean Δpp | 부호 일관성 | 판정           |
| --------- | ----: | -----: | -----: | ---------: | :---------: | -------------- |
| EN        | +0.08 | −0.09 | −1.58 | **−0.53** |     ✗     | PRUNING_HELPS |
| KO        | +2.26 |  +2.18 |  +1.92 |  **+2.12** |     ✓     | PRUNING_HURTS |
| ZH        | +0.96 |  +0.96 |  +0.81 |  **+0.91** |     ✓     | PRUNING_HURTS |

* Δpp = pruning on − off (bpb 기준). 데이터 출처: `phase1_pruning_gate_korean_only_result.json`.
* **해석**: KO/ZH는 3-seed 모두 일관되게 손해(PRUNING_HURTS), EN은 부호 불일치(결론 불가) — 계산 조건이 korean_only뿐이라 언어별 pruning 축 비교는 아직 못함.

**Figure A — 세 경로 크기 비교 막대그래프** [완성 — `figures/figure_a_path_decomposition.png`]

* Fisher 5.60%p / whitening 38.95%p / pruning 2.12%p 나란히 + noise floor 선(6.88%p). 생성: `scripts/make_figure_a_path_decomposition.py`.

### **Table 4 — §6.2 배분 비교** [⏳ SWITCH-4 대기]

* Balanced / 무제약 / cap50 × seed별 worst + mean-of-worst + mean-of-mean, NOT_SUPPORTED 판정.
* "최선 단일 언어" 열: **Korean**, mean_of_worst=**23.93**(worst=swh_Latn 3-seed 일관) — Balanced(22.03, worst=ben_Beng)보다 나쁨, 즉 단일 최선 언어로도 Balanced를 못 이김.
* 참고: Track A/B/C 후보(vulnerability_targeted 계열, interference_minimax)는 아직 실제 GPU 파이프라인 실행 전(`candidates` 전부 `ready:false`) — §6 전체 판정(NOT_SUPPORTED)은 재평가 필요(`03_experiment_prompt/0730_1258_gpu_schedule.txt` 참고).
* 데이터 출처: `phase1_6_budget_gate_v2_result.json`.
* **SWITCH-4**: 버전 A(이 표 그대로) vs 버전 B("변형별 worst-language vs balanced" 표로 재구성) 결정 대기.

**Figure B — §6 언어별 예산–성능 곡선** [미생성, 데이터 완비]

* x축 언어별 예산 비중(20%/35~50%/73%), y축 Δbpb%. Swahili 단조 개선(19.50→17.06→15.41), Bengali는 예산 늘려도 안 됨(cap50 반증).

**Table 5(§6 예산 배분 가중치)** [완성] — Balanced/무제약/cap50 × 5언어. 작아서 본문 인라인 또는 Figure B 주석으로 흡수 가능.

### **Figure C — §4.2 취약도-비례 산점도** [업데이트 필요]

* x축 취약도(타 언어 calibration 시 평균 저하), y축 own_gain. 현재 KO/Swahili(+부분 ZH) 확정 — Bengali 완주 시 4~5번째 포인트 추가.

### **Figure 3 — §5 layer-aligned 통합 그림** [완성 — 본문 대표 그림 ★]

* 3-panel(레이어축 1-27 공유): panel A=§5.1 cross-language rho vs placebo, panel B=§5.2 deficient pair 히스토그램, panel C=§5.3 whitening between/within ratio.
* **주목할 정렬**: 10-18 중반보다 **layer 22-27(후반)**에서 세 지표가 동시에 움직임(§5.1 rho 최저, §5.2 Swahili 결핍 집중, §5.3 cov ratio 급등) — 어느 구간을 핵심 서술로 할지 재검토.
* 생성: `scripts/make_figure3_whitening_evidence.py` → `figures/figure3_whitening_layer_evidence.png`.

### **Figure 4 — §5.3 between/within 분포 비교** [완성]

* mean principal angle / covariance Frobenius, within(n=1458)/between(n=10935) 바이올린 + ratio + p<0.0001(순열검정, 10000회) 주석. max_angle은 포화(≈90°)라 제외.
* 생성: `scripts/make_figure4_whitening_distributions.py` → `figures/figure4_whitening_distributions.png`.

### 부록 그림 G1/G2

* **G1**(`figures/figureG1_whitening_layer_curve_raw.png`): §5.3 raw within/between 절대값 곡선(ratio 아님), late-layer(21-27) 음영 + 캡션.
* **G2**(`figures/figureG2_whitening_heatmap.png`): 6×6 언어쌍 heatmap(mean_angle_deg_k64, 대각선=within-language).

---

## 부록 상세

**표 A1 — calibration 조건별 실측 토큰 길이** [확정 — 원문은 `06_논문_구성.md`]

| Calibration 조건 | mean raw tokens(절단 전) | mean realized tokens | 512-cap 도달 비율 | fertility(chars/token) |
| --- | ---: | ---: | ---: | ---: |
| (5개 조건, 원문 표 그대로 06_논문_구성.md:319 참고) | | | | |

* English만 낮은 fertility로 평균 459.9토큰(상한의 ~90%) 실현, 나머지는 전부 512-cap 도달.

**표 C1/C2 — 언어별 tokens/byte + baseline bpb**

| 언어         |     EN |     KO |     ZH |     SW |         BN |
| ------------ | -----: | -----: | -----: | -----: | ---------: |
| baseline bpb | 0.9137 | 1.3697 | 1.0776 | 2.2429 | **0.8857** |
| tokens/byte  | 0.2116 | 0.7018 | 0.2383 | 0.3929 | **0.6854** |

* 데이터 출처: `baseline/eval_ppl.json`. BN도 KO처럼 tokens/byte가 높음(UTF-8 byte-fallback 토크나이징 아티팩트, KO와 동일 원인).
* 주의: 이 "baseline bpb"는 §4.2 "취약도"(Table 1 grid 기반 저하율)와 다른 지표 — BN은 baseline bpb 최저(0.8857)지만 취약도는 최대(EN/ZH calibration 시 87~89% 저하).

**표 D1 — Table 1 원자료(30칸×3seed, std=ddof=1)**

| Calib    | Eval     | seed0 | seed1 | seed2 |  mean | std(ddof=1) |
| -------- | -------- | ----: | ----: | ----: | ----: | ----------: |
| English  | eng_Latn |  7.69 |  6.91 |  7.23 |  7.28 |        0.39 |
| English  | kor_Hang | 88.68 | 85.38 | 92.20 | 88.76 |        3.41 |
| English  | zho_Hans | 13.79 | 12.93 | 14.67 | 13.80 |        0.87 |
| English  | swh_Latn | 34.78 | 30.25 | 34.25 | 33.09 |        2.48 |
| English  | ben_Beng | 89.73 | 87.10 | 90.22 | 89.02 |        1.68 |
| Korean   | eng_Latn | 12.84 | 12.69 | 12.51 | 12.68 |        0.17 |
| Korean   | kor_Hang |  3.66 |  4.07 |  3.86 |  3.86 |        0.21 |
| Korean   | zho_Hans | 13.01 | 12.32 | 12.73 | 12.68 |        0.35 |
| Korean   | swh_Latn | 39.83 | 37.30 | 38.43 | 38.52 |        1.27 |
| Korean   | ben_Beng | 25.50 | 24.56 | 25.65 | 25.24 |        0.59 |
| Chinese  | eng_Latn |  9.93 | 10.80 | 10.13 | 10.29 |        0.46 |
| Chinese  | kor_Hang | 77.24 | 81.69 | 78.90 | 79.28 |        2.25 |
| Chinese  | zho_Hans |  3.82 |  4.33 |  4.57 |  4.24 |        0.38 |
| Chinese  | swh_Latn | 35.45 | 35.22 | 34.64 | 35.10 |        0.42 |
| Chinese  | ben_Beng | 83.70 | 88.47 | 90.06 | 87.41 |        3.31 |
| Swahili  | eng_Latn | 12.70 | 13.05 | 12.73 | 12.83 |        0.19 |
| Swahili  | kor_Hang | 65.04 | 66.23 | 68.68 | 66.65 |        1.86 |
| Swahili  | zho_Hans | 15.80 | 15.09 | 14.70 | 15.20 |        0.56 |
| Swahili  | swh_Latn |  0.02 |  0.06 | -0.22 | -0.05 |        0.15 |
| Swahili  | ben_Beng | 73.43 | 75.00 | 76.74 | 75.06 |        1.65 |
| Bengali  | eng_Latn | 17.54 | 17.68 | 16.30 | 17.17 |        0.76 |
| Bengali  | kor_Hang | 24.57 | 23.16 | 23.98 | 23.91 |        0.71 |
| Bengali  | zho_Hans | 17.83 | 18.56 | 18.54 | 18.31 |        0.41 |
| Bengali  | swh_Latn | 35.43 | 34.80 | 35.67 | 35.30 |        0.45 |
| Bengali  | ben_Beng |  1.39 |  1.35 |  1.25 |  1.33 |        0.07 |
| Balanced | eng_Latn |  8.63 |  9.72 |  9.53 |  9.29 |        0.58 |
| Balanced | kor_Hang |  5.66 |  5.06 |  5.37 |  5.36 |        0.30 |
| Balanced | zho_Hans |  7.21 |  7.44 |  6.66 |  7.10 |        0.40 |
| Balanced | swh_Latn |  2.75 |  2.95 |  2.80 |  2.83 |        0.10 |
| Balanced | ben_Beng |  2.97 |  2.79 |  2.93 |  2.90 |        **0.09** |

* std = 표본표준편차(ddof=1, n-1로 나눔). Balanced×ben_Beng: population stdev(구버전, ddof=0)로는 0.0757→"0.08"로 반올림됐었으나, ddof=1 정밀값은 0.0927→**0.09**(잘림 복구 완료). 데이터 출처: `phase1_41_headline_gate_result.json`.

**표 D2 — placebo 대각선 Δbpb%**

| Calib × Eval (placebo)    |  seed0 |  seed1 |  seed2 |    mean |
| -------------------------- | -----: | -----: | -----: | ------: |
| english_only_b × eng_Latn | 10.8396 | 10.9625 | 10.8437 | **10.8819** |
| korean_only_b × kor_Hang  |  4.4490 |  4.5846 |  4.3553 |  **4.4630** |
| swahili_only_b × swh_Latn |  0.5931 |  0.3489 |  0.6826 |  **0.5415** |

* placebo 조건은 대각선만 표시. Δbpb% = 100×(bpb/baseline_bpb − 1), whitening-ON 전체 파이프라인 — Table 1과 동일 정의·동일 스크립트(`phase1_41_headline_gate.py`)로 재계산. ZH/BN은 대응 placebo(_b)가 없어 제외.
* 데이터 출처: `{english_only_b,korean_only_b,swahili_only_b}/seed{0,1,2}/scale_{condition}_seed{seed}/eval_ppl.json`. Table 2 noise_floor의 원천 대각선 값(예: korean_only_b mean 4.4630이 kor_Hang noise_floor 0.79의 한쪽 항).

**표 E1 — flagged 36쌍 전체 목록** [확정]

* (layer, expert, lang, hits, n_tok, rate, expected_hits_at_baseline) 36행. 데이터 출처: `phase1_52b_deficiency_pairs_list_result.json`(`flagged` 필드). by_lang: swh_Latn=31, ben_Beng=5.

**표 E2 — deficiency 임계값 민감도**

| threshold | eng | kor | zho | swh | ben |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 | 0 | 0 | 0 | 22 | 4 |
| 20(기준) | 0 | 0 | 0 | 31 | 5 |
| 25 | 0 | 0 | 0 | 36 | 6 |
| 30 | 0 | 0 | 0 | 48 | 6 |

* 20→25 경계 borderline 6건(front 1건, mid 0건, back 5건) — 결론(Swahili/Bengali만 결핍, 나머지 0)은 threshold 15~30 전 구간에서 안정적. 데이터 출처: `phase1_52c_deficiency_robustness_check_result.json`.

**표 F1 — 언어쌍별 평균 ρ (27-layer 평균, seed0)**

| 언어쌍 | mean rho | std |
| --- | ---: | ---: |
| korean↔bengali | 0.682 | 0.088 |
| korean↔chinese | 0.507 | 0.109 |
| swahili↔bengali | 0.451 | 0.149 |
| english↔swahili | 0.434 | 0.116 |
| korean↔swahili | 0.344 | 0.168 |
| english↔chinese | 0.273 | 0.439 |
| chinese↔bengali | 0.197 | 0.132 |
| chinese↔swahili | 0.165 | 0.235 |
| english↔korean | 0.080 | 0.255 |
| english↔bengali | 0.006 | 0.116 |
| **placebo: swahili↔swahili_b** | **0.977** | 0.010 |
| **placebo: korean↔korean_b** | **0.957** | 0.021 |
| **placebo: english↔english_b** | **0.921** | 0.041 |

* std는 population stdev(27-layer 기준, ddof=0) — Table D1과 ddof 컨벤션이 다름, 필요시 ddof=1로 통일 가능. 데이터 출처: `phase1_51_fisher_rank_correlation_result_seed0.json`.

**표 F2 — §5.1 헤드라인 수치의 seed 일관성**

| seed | 전체 언어쌍 ρ | 전체 placebo ρ | 전반(1–13) | 후반(14–27) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.314 | 0.952 | 0.390 | 0.243 |
| 1 | 0.341 | 0.941 | 0.419 | 0.268 |
| 2 | 0.332 | 0.946 | 0.406 | 0.263 |
| 범위 | 0.027 | 0.011 | 0.029 | 0.025 |

* 실측 검증 완료(2026-07-30) — seed0/2 전부 일치, seed1 전반 셀은 0.420이 아니라 **0.419**(범위 행이 이 값을 전제로 계산돼 있어 내부 일관성으로 확인). 데이터 출처: `phase1_51_fisher_rank_correlation_result_seed{0,1,2}.json`(`summary` 필드).

**표 G1 — §5.3 지표×구간 ratio**

| 지표 | all-layer | 1-9 | 10-18 | 19-27 |
| --- | ---: | ---: | ---: | ---: |
| mean_angle_deg_k64 | 1.936 | 2.120 | 1.817 | 1.893 |
| cov_reldist_k64 | 4.854 | 4.822 | 4.422 | 5.364 |
| whitening_reldist_k64 | 1.388 | 1.440 | 1.332 | 1.395 |

* 데이터 출처: `whitening_geometry_pairs.csv`(within=같은 조건 다른 seed, between=다른 조건 — mixed_5lang 포함, placebo 미포함).

**B.3 — 판정 라벨 4종 정의** [내용 확정, 표/텍스트 형식 미정]

* **SUPPORTED**: mean own_gain > 2×noise_floor (placebo 존재).
* **INCONCLUSIVE**: 0 < own_gain ≤ 2×noise_floor.
* **NOT_SUPPORTED**: own_gain ≤ 0.
* **SIGN_CONSISTENT_NO_FLOOR**: placebo(noise_floor) 자체가 없어 위 기준 적용 불가, 부호만 일관되게 양수인 경우(SUPPORTED 상한 캡).

**App: hyperparameter/재현 표** — delta_ratio 0.8(rank 667/1408), pp_ratio 0.2, 64샘플×512, seed 0/1/2, 4×3090 + merge_eval 2-GPU 페어링.

**App: Fisher-merge lossless 단위 테스트** — mean rel. Frobenius 0.303%, max 0.309%, 192쌍 — §4.3 리뷰어 방어("분해 오류 아님")의 근거.

**App: proxy 검증** — Figure 2(`figures/figure2_fisher_pilot_a_validation.png`, ρ=0.870) — §5.1과 §6.1 둘 다 참조하므로 appendix 배치 + 본문 참조 권장.

---

## 컷/보류 (만들지 말 것)

* §5.3 principal angle 그림, §5.4 인과 사슬 상관 그림: 이번에 §5.3은 Figure 3/4/G1/G2로 완성됨(이 항목 해소). §5.4는 여전히 데이터 미착수 — future work 한 문장으로 압축.
* §7 일반화 관련 일체: 컷 확정.

## 확인 필요/미결 항목 요약 (2026-07-30)

1. **Figure 1 최종 삭제 여부**: 문서 검색상 §3 등 다른 곳의 데이터-의존 참조 없음 확인 — 삭제해도 안전해 보이나 최종 확정은 사용자 판단.
2. **§4.3(Fisher×Scale 2×2, 옛 Table 3) 번호 재부여**: 내용은 완성, 번호만 미정.
3. **Table 3(Belebele) 확장**: english_only/mixed_5lang 완주 필요(TODO-D1, 밤 큐 진행 중이던 4.3 grid와는 별개 작업).
4. **Table 4 SWITCH-4**: 버전 A/B 형식 결정 대기.
5. **표 F1 std 컨벤션**: ddof=0(population) — Table D1의 ddof=1과 다름, 통일 여부 결정 필요.
