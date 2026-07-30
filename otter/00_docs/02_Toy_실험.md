# Toy 실험: Toy0 + Phase 0.5

> 전체 연구 설계는 [01_연구설계.md](01_연구설계.md), 현재 상태 한 줄 요약은
> [04_전체요약.md](04_전체요약.md), GPU 정책/실행 명령/폴더 구조 같은 운영 정보는
> [03_기술노트.md](03_기술노트.md) 참고. 이 문서는 이 저장소에서 실제로 돌린 두
> 실험(Toy0, Phase 0.5)의 설계와 결과만 다룬다.

## 왜 압축부터 하지 않았는가

이 문서가 다루는 Toy0/Phase 0.5 시점에는 01_연구설계.md §22의 체크리스트 1번
("공식 D²-MoE repo 설치")이 아직 안 되어 있었다. 공식 구현이 없으면 Phase 0
(재현성 검증)도 Phase 1(실제 압축)도 시작할 수 없었다. (그 뒤 Phase 0/Fisher
Pilot A/Phase 1이 모두 완료됐다 — 현재 상태는 [04_전체요약.md](04_전체요약.md),
세부 내용은 [03_기술노트.md](03_기술노트.md) 참고.) 그런데 압축과 무관하게 그
시점에도 바로 검증 가능한 좁은 전제조건이 하나 있었다:
**calibration 언어를 바꾸면, D²-MoE 압축 절차가 실제로 소비하는 두 가지 통계
(routing coverage, per-expert importance)가 애초에 바뀌기는 하는가?**

이게 바뀌지 않는다면, 이후 Fisher-weighted merging이든 SVD든 pruning이든 어떤
D²-MoE 세부 단계를 봐도 calibration 언어 효과가 나올 수 없다 — 압축 파이프라인을
설치하기 전에 먼저 확인해 볼 가치가 있다. 이것이 Toy0이다. FERRET-R의 이전 toy
체인(현재는 `../../archive_failed/`에 보관, 각 폴더가 왜 게이트를 통과하지 못했는지는
`archive_failed/README.md` 참고)의 Toy0가 "D²-MoE decomposition이 이 모델에서
성립하는가"를 확인하는 게이트였던 것과 같은 논리를, "calibration 언어 신호가 이
모델에서 존재하는가"에 적용한 것이다.

## 이미 재사용 가능한 자원

이 호스트(4x3090)에서 이미 검증된 자원을 새로 만들지 않고 그대로 가져다 썼다.

| 자원 | 출처 | 비고 |
|---|---|---|
| Qwen3-30B-A3B 로딩/device_map | `archive_failed/ferret_kr/scripts/00_config.py` | bf16, GPU별 free memory probing |
| MoE expert forward hook | `archive_failed/ferret_kr/scripts/moe_hooks.py` | `Qwen3MoeExperts.forward` monkeypatch, 그대로 복사 |
| FLORES-200 다국어 문장 로딩 | `archive_failed/ferret_kr/scripts/data_utils.py` | `israel/flores-parallel` (ungated 재업로드본) |
| conda 환경 / 라이브러리 버전 | `archive_failed/ferret_toy0_qwen3_ver1/configs/environment.txt` | torch 2.5.1+cu121, transformers 5.13.1 |

**이 시점(Toy0/Phase 0.5)에 아직 없었던 것:** 공식 D²-MoE repo, 실제 gradient
기반 Fisher 계산, 실제 압축 — Toy0/Phase 0.5는 이 중 아무것도 필요로 하지
않았다. 셋 다 이후 Phase 0/Fisher Pilot A/Phase 1에서 확보됐다.

---

## Toy0 설계 (2026-07-23)

**질문:** English-only / Korean-only / Chinese-only / Balanced 네 조건의
calibration 문장을 Qwen3-30B-A3B에 forward pass만 시켰을 때, (a) 어떤 expert가
얼마나 라우팅되는지, (b) expert별 중요도 proxy가, 언어에 따라 "같은 언어를
다시 샘플링했을 때의 노이즈"보다 크게 달라지는가?

**Calibration 조건** (`data/languages.yaml`):

| id | 언어 구성 | 역할 |
|---|---|---|
| `en_only` | English 30문장 | baseline |
| `en_only_control` | English 30문장 (다른 offset, 겹치지 않음) | **같은 언어 내 noise floor** |
| `ko_only` | Korean 30문장 | |
| `zh_only` | Chinese(Simplified) 30문장 | |
| `balanced` | English/Korean/Chinese 각 10문장 | 언어당 동일 배분 (01_연구설계.md §6) |

`en_only_control`이 핵심이다 — 언어 라벨을 아예 안 바꾸고 문장만 바꿔도 라우팅
분포는 조금은 달라진다(표집 노이즈). 언어를 바꿔서 생기는 차이가 이 노이즈보다
뚜렷이 커야만 "calibration 언어 효과"라고 주장할 수 있다. 01_연구설계.md §16의
"cliff-capture 위장 통과 방지" 교훈과 같은 이유다.

**측정 (레이어 8/22/38 — early/mid/late 서브셋, `00_config.py`):**

1. **Routing coverage** — expert별 hit_count를 분포로 정규화하고, 조건 쌍마다
   Jensen–Shannon divergence + top-25% expert Jaccard overlap (01_연구설계.md
   §11.A/B의 축소판)
2. **Fisher-proxy 중요도** — 실제 gradient 기반 Fisher가 아니라
   `(router_weight * ||y_t||)^2`의 합. 이 forward-only proxy가 언어 간에 갈라지는지
   Spearman rank correlation으로 확인 (01_연구설계.md §11.C의 축소판, §12-3에 대응)

**Gate 판정:** 레이어 과반수에서 `cross-language 평균 JS divergence >= 1.5 x
(en_only vs en_only_control) noise floor`이면 PASS.

### Toy0 실행 결과 (Qwen3-30B-A3B, 30문장/조건, seed 1개)

GPU 1/2에서 실행 (GPU 0/3은 다른 사용자 작업이 있어 자동으로 제외됨, 03_기술노트.md
"GPU 안전 정책" 참고). 레이어 3개(8/22/38) 중 2개에서 게이트 통과.

| layer | noise floor (JS) | cross-language mean (JS) | ratio | 판정 |
|---|---|---|---|---|
| 8  | 0.217 | 0.295 | 1.36 | FAIL |
| 22 | 0.193 | 0.300 | 1.56 | PASS |
| 38 | 0.200 | 0.356 | 1.78 | PASS |

**전체 판정: PASS** (3개 중 2개 레이어 통과). Layer 8(초반)에서는 언어 효과가
noise floor와 거의 구분되지 않지만, layer 22/38(중반~후반)로 갈수록 언어 간
routing divergence가 같은 언어 재표집 노이즈보다 뚜렷이 커진다.

Fisher-proxy rank correlation도 같은 패턴을 보인다: layer 8/22에서는 조건 간
상관이 대체로 0.8대(언어를 바꿔도 중요도 순위가 비교적 안정적)이지만, layer
38에서는 EN 계열(EN-only/control, rho=0.90) vs KO/ZH 계열 사이 상관이 0.48~0.68로
뚜렷이 낮아진다 — "영어로만 calibration하면 비영어에서 중요한 expert를 다르게
평가할 수 있다"는 H4/RQ4의 방향과 일치하는 초기 신호였다 (단, 아래 Phase 0.5에서
"레이어가 깊을수록"이라는 해석은 기각되고 "마지막 레이어 특이 현상"으로 정정된다).

산출물: `results/toy0_gate_summary.json`, `results/toy0_condition_divergence.csv`,
`results/toy0_fisher_rank_correlation.csv`, `results/toy0_routing_fisher_stats.json`

**Figures** (`figures/`, `scripts/03_make_figures.py`):
- `toy0_gate_bars.png` — 레이어별 PASS/FAIL 게이트 판정
- `toy0_divergence_heatmap.png` — 조건 쌍 routing JS divergence 히트맵
- `toy0_expert_routing.png` — 조건별 expert 히트 분포 (early/mid/late 3개 레이어)
- `toy0_fisher_correlation.png` — 조건 쌍 Fisher-proxy Spearman 상관 히트맵

**한계:** 30문장/조건은 매우 작은 표본이고 seed 1개뿐이다. layer 8의 FAIL과
layer 22/38의 PASS 차이가 진짜 "초반 vs 후반 레이어" 효과인지, 표본 크기에서의
우연인지는 이 결과만으로 구분할 수 없었다 — 그래서 Phase 0.5를 실행했다.

---

## Phase 0.5 설계 (재현성 검증, Toy0 다음 단계)

Toy0에서 관찰된 패턴이 seed·표본을 늘려도 재현되는지만 확인한다 — **새 언어,
새 분석, 새 모델은 추가하지 않는다.**

**Toy0 대비 바뀐 것:**
- **문장 수 → token budget 통제.** 가장 위험한 confound는 언어 자체가 아니라
  tokenization/문장 길이이므로, 조건당 총 5000 tokens로 통제한다 (balanced는
  언어당 1666 tokens). 각 조건은 seed마다 언어 pool을 셔플해서 disjoint chunk를
  뽑는다 (`data_utils.token_budget_chunks`).
- **seed 3개** (0, 1, 2) — 각각 독립적으로 언어 pool을 재셔플.
- **레이어 3개 → 6개** (4, 8, 14, 22, 30, 38) — Toy0의 "레이어가 깊을수록"이라는
  해석이 3점만으로 만든 착시인지 확인하려면 중간 점이 더 필요하다.
- **en_only/en_only_control → english_a/english_b.** 같은 역할(같은 언어 noise
  floor placebo)이지만, 그림/표에서 바로 "placebo pair"로 읽히도록 이름을 바꿨다.

**사전 등록 게이트 (실행 전에 고정, `data/phase0_5_config.yaml` `gate:` 블록):**

3개 이상 충족 시 본 실험(Phase 0/1) 진행:
1. layer 22와 38의 median ratio(seed 통합)가 모두 1.3 이상
2. layer 38이 seed 3개 중 최소 2개에서 개별 PASS (ratio >= 1.5)
3. layer 38의 ratio가 layer 8보다 큰 seed가 과반수
4. within-English Fisher 상관이 EN-KO/EN-ZH 상관보다 큰 경우가 layer 38에서 과반수 seed
5. bootstrap CI(seed 3개 resampling, layer 38 ratio)의 하한이 1.0을 초과

기준 2만 안정적으로 만족하면 CONDITIONAL(가설을 후반 레이어로 좁혀서 진행), 그마저
아니면 STOP_OR_NARROW.

### Phase 0.5 실행 결과 (Qwen3-30B-A3B, 3 seeds, token budget 5000/조건)

GPU는 매 실행 시점의 `nvidia-smi`+`ps` 확인으로 자동 배정됐다 (dongjin이 실행
도중 GPU0→GPU1로 옮겨서 자동으로 GPU0/2 사용, 이후 GPU1도 비어서 GPU0/1/2 사용
— 하드코딩 없이 매번 재확인).

**사전 등록 게이트: 5/5 기준 충족 → PROGRESS_PASS.** 18개 (seed, layer) 셀 전부
개별 PASS (ratio 최저값도 2.49, Toy0 margin 1.5를 전부 크게 상회). Bootstrap 95%
CI (layer 38, seed 3개 resampling): **[4.82, 5.73]**, 1.0을 명확히 배제.

| layer | median ratio (3 seed) |
|---|---|
| 4  | 5.81 |
| 8  | 3.55 |
| 14 | 4.11 |
| 22 | 4.50 |
| 30 | 2.66 |
| 38 | 5.31 |

**중요한 수정 사항 — Toy0의 "레이어가 깊을수록 커진다"는 가설은 기각됨.**
레이어를 6개로 늘려 다시 보니 ratio는 단조 증가가 아니라 **비단조(W자형)** 패턴이다
(layer 4 최댓값 → layer 8 급락 → layer 22까지 완만한 회복 → layer 30 재차 급락 →
layer 38 재상승). layer-depth vs ratio Spearman은 seed 3개 모두 **음수, 유의하지
않음** (rho ≈ -0.20 ~ -0.26, p ≈ 0.33~0.70; pooled rho = -0.24, p = 0.33), seed별
slope도 거의 0이거나 음수. Toy0의 layer 8→22→38 3점만으로는 우연히 증가 구간을
붙잡은 것이었다 — 사전에 경고됐던 정확히 그 함정이다.

**대신 더 구체적이고 여전히 유효한 신호가 있다: layer 38(마지막 근처 레이어)에서만
Fisher-proxy의 EN vs non-EN 상관이 뚜렷이 갈라진다.** within-English 상관
(english_a vs english_b, placebo)은 모든 레이어에서 seed 3개 내내 0.94~0.99로
안정적인데, EN-KO/EN-ZH 상관은 layer 8/14/22/30에서는 0.7~0.95로 within-EN과 큰
차이가 없다가, **layer 38에서만 0.49~0.52로 뚝 떨어진다** (3 seed 모두 일관).
즉 "언어에 따라 어떤 expert가 중요한지가 갈린다"는 신호는 전 레이어에 퍼진 general한
현상이 아니라, 마지막 레이어에 국소적으로 집중된 현상으로 보인다.

**논문 가설 축소 제안**: "후반 레이어로 갈수록 calibration-language sensitivity가
커진다"가 아니라, **"마지막 레이어의 Fisher-relevant 통계가 특히 calibration
언어에 민감하다"**로 좁히는 것이 이 데이터에 더 부합한다. routing divergence
ratio 자체(criterion 1~3, 5)는 모든 레이어에서 강하고 안정적으로 재현되므로
"calibration 언어가 routing/importance 통계를 바꾼다"는 핵심 주장(RQ4)은 그대로
유지되지만, "깊이가 원인"이라는 메커니즘 설명은 빼야 한다.

산출물: `results/phase0_5_stats_seed{0,1,2}.json`, `results/phase0_5_seed_layer_table.csv`,
`results/phase0_5_gate_summary.json`

**Figures** (`scripts/03b_make_figures_repro.py`):
- `phase0_5_ratio_trend.png` — 레이어별 ratio, seed별 선 + 중앙값 (비단조 W자형 확인)
- `phase0_5_fisher_trend.png` — within-EN vs EN-KO vs EN-ZH Fisher 상관, 레이어별 (min-max band, 3 seeds) — layer 38에서만 갈라지는 지점이 바로 보임
- `phase0_5_bootstrap_ci.png` — layer 38 ratio bootstrap 95% CI
- `phase0_5_gate_scorecard.png` — 5개 사전 등록 기준 PASS/FAIL 요약

**아직 통제하지 못한 것**: FLORES-200은 이미 문장 길이가 비교적 균일하고
code/URL/표가 없는 curated MT 벤치마크라 별도 필터링은 하지 않았음. calibration과
평가 문장의 contamination 제거는 아직 별도 downstream 평가셋이 없어 해당 없음
(Phase 1에서 평가셋을 붙일 때 처리). Domain-controlled ablation(동일 출처 vs
다른 출처 다국어)은 이번 범위에 없음 — 범위를 넓히지 않기로 한 지침에 따름.

---

---

## 0) Layer-38 국소성 검증 (2026-07-23, Qwen3-30B-A3B, 3 seeds)

Phase 0.5의 핵심 발견은 "layer 38에서만 Fisher-proxy의 EN vs non-EN 상관이
갈라진다"였는데, 그때 측정한 가장 깊은 레이어가 38이었다. Qwen3-30B-A3B는 실제로
**48개 레이어(0~47)**이므로 38 이후로 9개 레이어가 안 보이는 채 남아 있었다.
이 실험은 압축 없이, 같은 probe·같은 seed(0/1/2)·같은 token budget(5000/조건)
그대로 **레이어만 34/42/45/47을 추가**해서 (기존 4/8/14/22/30/38은 재현성
재확인 겸 유지) "정말 최종 레이어 1~2개 국소 현상인지, 후반 블록 전체에 퍼진
현상인지"를 갈랐다. 새 언어·새 분석·새 모델 없음 — 반나절짜리 forward-only 실험.

`data/layer_locality_config.yaml` · `scripts/01b_phase0_5_stats.py --config ... --prefix layer_locality`
(새 스크립트를 만들지 않고 Phase 0.5 스크립트에 `--config`/`--prefix`를 추가해서
재사용) · `scripts/02c_analyze_layer_locality.py` · `scripts/03c_make_figures_layer_locality.py`.

### 결과: **SPREAD_BACK_HALF** — 국소 현상이 아니라 후반 블록 전체의 점증 현상

| layer | mean fisher_gap (±std, 3 seed) | mean ratio |
|---|---|---|
| 4  | 0.258 (±0.010) | 5.82 |
| 8  | 0.078 (±0.001) | 3.60 |
| 14 | 0.234 (±0.004) | 4.08 |
| 22 | 0.134 (±0.005) | 4.36 |
| 30 | 0.015 (±0.001) | 2.78 |
| 34 | 0.215 (±0.007) | 4.47 |
| **38** | **0.478 (±0.004)** | 5.29 |
| **42** | **0.829 (±0.013)** | 7.66 |
| **45** | **0.672 (±0.020)** | 7.65 |
| **47** | **0.784 (±0.024)** | **14.42** |

(fisher_gap = within-English Spearman rho − mean(EN-KO rho, EN-ZH rho); 임계값 0.3)

**layer 38/42/45/47이 하나의 연속 구간으로 임계값을 넘는다** (`[[38, 42, 45, 47]]`).
게다가 이 구간 안에서는 (Phase 0.5가 전체 네트워크에서 본 비단조 W자형과 달리)
**뚜렷한 단조 증가 경향**이 있다 — layer 22부터 47까지만 놓고 본 Spearman(layer,
fisher_gap)은 **rho = 0.85, p = 1.1e-6**로 강하게 유의하다. 3 seed 간 표준편차도
전 구간에서 매우 작다(최대 0.024) — 이 패턴은 우연이 아니다.

가장 극적인 지점은 **layer 47(진짜 마지막 레이어)의 routing ratio = 14.4**로,
다른 어떤 레이어보다도 훨씬 크다 (그 다음으로 큰 42/45도 7.6대).

산출물: `results/layer_locality_stats_seed{0,1,2}.json`,
`results/layer_locality_seed_layer_table.csv`, `results/layer_locality_gate_summary.json`

**Figures** (`scripts/03c_make_figures_layer_locality.py`):
- `layer_locality_ratio_trend.png` — 전체 10개 레이어의 routing ratio, 마지막에서 급등
- `layer_locality_fisher_gap.png` — Fisher gap, layer 30 이후 뚜렷한 상승, 3 seed가 거의 겹칠 정도로 일치

### 논문 가설 재조정 (두 번째 수정)

Toy0의 "깊을수록"(3점) → Phase 0.5의 "layer 38만 특이"(6점, 오해였음) →
**이번의 "후반 블록(약 38~47, 네트워크 마지막 ~20%)에서 calibration-language
sensitivity가 점증하며 마지막 레이어에서 정점"**(10점, 지금까지 가장 근거가
탄탄함)으로 좁혀진다. 앞쪽 3/4(레이어 4~34)는 임계값 아래에서 특별한 추세 없이
등락하는 반면, 뒤쪽 1/4은 seed 간 거의 완벽히 재현되는 단조 증가를 보인다 —
"마지막 레이어 하나"라고 쓰면 과소진술이고, "레이어가 깊을수록 전체적으로
증가한다"고 쓰면 (Phase 0.5가 이미 반박한 대로) 과대진술이다.

---

## Fisher-proxy 한계 (반드시 읽을 것)

Toy0/Phase 0.5가 계산하는 "Fisher-proxy"는 D²-MoE 논문이 실제로 쓰는 gradient
기반 Fisher information이 아니다. forward pass만으로 얻을 수 있는
`(router_weight * ||down_proj output||)^2`를 대신 쓴 것 — 순전히 "언어를 바꾸면
뭐라도 신호가 갈라지는가"를 압축 파이프라인 없이 값싸게 확인하기 위한 대용물이다.
게이트가 PASS해도 이것이 "진짜 Fisher가 언어에 민감하다"는 증거는 아니다. 진짜
Fisher는 공식 D²-MoE repo를 설치한 뒤 Phase 0/1에서 직접 계산해야 한다
(01_연구설계.md §11.C, §21).
