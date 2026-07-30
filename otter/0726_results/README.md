# 0726 결과 — 검증된 결과 3가지 + 예비 결과 1가지

이 폴더는 OTTER 프로젝트(D²-MoE 다국어 calibration 연구)에서 **실제로 검증되어
방어 가능한 결과 3가지**(Figure 1~3)와, 2026-07-26에 막 나온 **아직 검증 전인
예비 결과 1가지**(Figure 4, PRELIMINARY로 명시)를 PNG로 정리한 것이다. 가설과
무관하거나 노이즈와 구분이 안 되는 결과는 의도적으로 뺐다 — 어떤 걸 왜 뺐는지는
맨 아래 "제외한 것" 절 참고.

전체 연구 설계·전체 실험 로그는 `../00_docs/`에 있다. 이 문서는 그중 이 네
결과만 뽑아 그림과 함께 자세히 설명한다.

Figure 1~4 외에, §4.2 downstream-task 재확인용으로 진행 중인 **Belebele 표
하나**(Table 1, PNG 없이 표만)도 맨 아래에 추가했다 — 아직 조건 3개 중 1개
(korean_only)만 완결됐고 나머지 둘은 seed 부분 결측 상태라 PRELIMINARY보다도
이른 **IN PROGRESS**로 표시한다.

---

## Figure 1 — `1_layer_locality.png`

### 무엇을 보여주는가
Qwen3-30B-A3B에서, calibration 언어를 바꿨을 때 라우팅/Fisher-proxy 통계가
레이어별로 얼마나 갈라지는지(divergence ratio, y축)를 레이어 인덱스(x축, 0~47)에
따라 그렸다. 배경 음영 두 겹: 연한 색 = 네트워크 후반 ~20%(레이어 22~47), 진한
색 = 사전 등록한 게이트 임계값을 실제로 넘은 구간(레이어 38, 42, 45, 47).

### 어떤 실험에서 나왔나
세 단계로 반복 확인됐다:
1. **Toy0** — 레이어 3개(8/22/38)만 보고 "깊을수록 커지는 것 같다"는 가설을 세움.
2. **Phase 0.5** — 레이어를 6개로 늘리고 seed 3개로 재현성 검증 → 실제로는
   비단조(W자형) 패턴이라 "깊이 효과" 가설은 기각됨(Toy0의 3점짜리 결론이
   틀렸다는 걸 스스로 뒤집은 사례).
2. **0) 레이어 국소성 검증** — 레이어를 10개(4~47)로 확장해서 Qwen3의 진짜 마지막
   레이어(47)까지 포함 → "전체 네트워크는 비단조지만, 마지막 ~20% 블록
   안에서는 뚜렷한 단조 증가가 있다"는 세 번째, 최종적으로 안정된 해석에 도달.

### 실험 설정
- **모델:** Qwen/Qwen3-30B-A3B (bf16, 48레이어)
- **Calibration 데이터:** `israel/flores-parallel`(HF, FLORES-200 parallel), test split
- **언어/조건 (5개):** `english_a`/`english_b`(같은 영어 pool에서 겹치지 않는 두
  표본 — same-language 노이즈 기준선), `korean`, `chinese`, `balanced`(영/한/중 균등 혼합)
- **표본 규모:** 조건당 token budget **5,000 tokens**(balanced는 3개 언어에
  1,667씩 분할) — 문장 수가 아니라 tokenizer 토큰 수 기준으로 정확히 맞춤
- **측정 레이어:** 10개 — 4, 8, 14, 22, 30, 34, 38, 42, 45, 47
- **Seed:** 0/1/2 총 3개, seed마다 FLORES pool을 독립적으로 재표집
- **측정값:** 레이어별 (expert별 routing hit_count, forward-only Fisher-proxy)
  → 조건 간 Jensen-Shannon divergence ratio, seed 3개 평균/표준편차로 집계

### 핵심 수치
- 후반 구간(레이어 22~47) Spearman ρ = **0.85**, p = 1.1×10⁻⁶
- 앞쪽 3/4 구간(레이어 4~34)은 뚜렷한 추세 없음
- 가장 극적인 지점: **레이어 47**(Qwen3의 진짜 마지막 레이어)의 ratio = **14.4** — 다른 모든 레이어보다 훨씬 큼
- seed 3개 표준편차 최대 0.024 — 매우 재현성 높음

### 이게 증명하는 것
RQ4의 전제조건("calibration 언어가 압축이 참조할 routing/Fisher 통계를
바꾸는가")을 세 번 반복 확인했고, **어디서(WHERE) 그 효과가 나타나는지**까지
특정했다: 네트워크 전체가 아니라 마지막 ~20% 블록에 집중된다.

**원본 데이터:** `../results/layer_locality_gate_summary.json`,
`../results/layer_locality_seed_layer_table.csv` (레이어 × seed 3개, raw ratio/gap 전부 포함)
**세부 문서:** `../00_docs/02_Toy_실험.md`

---

## Figure 2 — `2_fisher_pilot_a_validation.png`

### 어떤 실험에서 나왔나
**Fisher Pilot A** — Figure 1의 발견은 전부 forward-only proxy 기반이었다.
"이 proxy가 진짜 gradient Fisher를 잘 대변하는가?"와 "후반 블록 민감도가
**다른 모델**에서도 재현되는가?"를 확인하려고, DeepSeek-MoE-16B(Qwen3와 완전히
다른 모델)에서 4개 레이어 × 4조건(English 자기 자신 재표집 포함) × 진짜
backward pass로 real gradient Fisher를 계산해 비교했다. 원 논문 규모
(1024 samples, 전체 파라미터 backward)는 이 호스트에서 반복 OOM이 나서,
레이어 하나씩 순차적으로 freeze-and-accumulate하는 방식으로 다시 설계해서
GPU 2장으로 해결했다. (a), (b) 두 패널은 **같은 실행에서 나온 같은 원본 데이터**를
다른 방식으로 집계한 것 — 아래 "공통 실험 설정"이 둘 다에 적용된다.

### 공통 실험 설정
- **모델:** deepseek-ai/deepseek-moe-16b-base (bf16, 28레이어 — 0번 dense, 1~27 MoE)
- **Calibration 데이터:** `israel/flores-parallel`(HF), test split — Figure 1과 동일 소스
- **언어/조건 (4개):** `english_a`/`english_b`(같은 영어 pool의 서로 다른 표본),
  `korean`, `chinese`
- **표본 규모:** 조건당 token budget **768 tokens**, 문장당 최대 길이(seqlen)
  **192 tokens**, batch size 1
- **측정 레이어:** 4개 — 4(early, ~14% 깊이) / 16(transition, ~61%) /
  24(sensitive, ~86%) / 27(final, 100%, 실제 마지막 레이어)
- **Seed:** 0 하나만 (pilot 성격 — 전체 5-layer×5-condition×3-seed 설계를
  축소한 4×4×1)
- **Fisher 계산 방식:** 레이어 1개씩 순차 처리, 해당 레이어 expert 파라미터만
  `requires_grad=True`(나머지 전부 freeze), expert별 scalar
  `sum(grad^2)`으로 누적(element-wise 아님) — GPU 2장(16GiB cap씩)으로 실행
- **Proxy 계산 방식:** 같은 forward를 `(router_weight * ||expert_output||)^2`
  캡처 로직으로 재구성, gradient 없이 계산

### (a) proxy ↔ real Fisher 상관 — "이 proxy를 믿어도 되는가"

**무엇을 보여주는가:** 레이어별로, forward-only Fisher-proxy가 매긴 expert
중요도 순위가 진짜 gradient Fisher가 매긴 순위와 얼마나 일치하는지
(Spearman ρ, 4조건 평균, 레이어 역할별 막대).

**계산 방법:** 각 (레이어, 조건) 조합마다 그 레이어의 모든 expert를 (1) 진짜
Fisher 값 (2) proxy 값으로 각각 정렬해 순위를 매기고, 두 순위 사이 Spearman
상관을 구한다. 이걸 4조건(english_a/b, korean, chinese)에서 각각 구한 뒤
평균낸 값이 막대 높이다.

**핵심 수치:**
- 레이어·조건 전체 범위 **0.82~0.91**, 평균 **0.870** — 뚜렷한 양의 상관
- 레이어 역할(early~final)에 따른 큰 차이 없이 전 구간에서 고르게 높음

**증명하는 것:** forward-only proxy가 실제로 유효한 대용물이다 — 지금까지의
모든 Toy0/Phase 0.5 결과가 이 proxy에 의존했는데, 그 신뢰성이 여기서 진짜
gradient Fisher와 대조해 처음으로 확보됨.

### (b) EN vs 비영어 상관 gap — "후반 레이어 민감도가 진짜 Fisher에도 있는가"

**무엇을 보여주는가:** 레이어별로, "같은 언어(영어)끼리 비교한 순위 상관"과
"영어 vs 한국어/중국어로 비교한 순위 상관"의 차이("gap") — 이 값이 클수록
해당 레이어가 calibration 언어에 민감하다는 뜻.

**계산 방법:** 진짜 Fisher 기준으로, `within_en_rho`(english_a vs
english_b 순위 상관)에서 `mean(en_ko_rho, en_zh_rho)`(영어 vs 한국어·중국어
순위 상관의 평균)를 뺀 값 = `real_gap`. Figure 1의 layer-locality와 같은
지표를, forward-only proxy가 아니라 이번엔 진짜 gradient Fisher로 다시 계산한 것.

**핵심 수치:**
- early/transition(레이어 4, 16) 평균 gap **0.475** → sensitive/final(레이어
  24, 27) 평균 gap **1.09~1.19** (약 2.3~2.5배 증가)
- sensitive/final 레이어의 EN-KO/EN-ZH 상관은 **음수**(-0.10~-0.38)까지 떨어짐 —
  영어로만 calibration하면 후반 레이어의 중요 expert를 "다르게 평가"하는 정도가
  아니라 사실상 "반대로 평가"한다는 뜻

**증명하는 것:** Figure 1(Qwen3, forward-only proxy)이 찾은 "후반 블록
민감도"가 Qwen3 하나의 우연이나 proxy의 착시가 아니라, **다른 모델(DeepSeek)·
진짜 gradient Fisher**에서도 같은 방향으로, 오히려 더 강하게 재현되는 진짜
패턴이다.

**원본 데이터:** `/mnt/HDD/minjeong/d2moe_results/fisher_pilot_a/pilot_a_analysis.json`
(레이어별 proxy-real 상관 및 gap 전체), `pilot_a_results.json`(원본 expert별 Fisher/proxy 값)
**세부 문서:** `../00_docs/03_기술노트.md` "Fisher Pilot A" 절

---

## Figure 3 — `3_phase1_placebo_verified.png`

### 무엇을 보여주는가
두 패널:
- **(a)** 세 언어(Korean, Swahili, English)의 own-language 이득(자기 언어로
  calibration했을 때 그 언어가 얼마나 더 잘 보존되는지, 가로 막대)과 각 언어의
  **noise floor 2배 기준선**(세로 눈금) — 막대 끝이 기준선을 넘으면 SUPPORTED.
- **(b)** 압축 전 baseline bpb(그 언어가 원래 얼마나 취약한지, x축)와
  own-language 이득(y축)의 관계 — 우상향 패턴이 보이면 "취약한 언어일수록
  자기 언어 calibration의 효과가 크다"는 뜻.

### 어떤 실험에서 나왔나 (가장 긴 검증 사슬)
1. **Phase 1 1차 pilot** — DeepSeek-MoE-16B를 실제로 압축(Fisher-weighted
   merge + plain SVD delta)하고 4개 calibration 조건(EN/KO/ZH-only, Balanced)
   +baseline을 FLORES PPL로 평가. Seed 1개, 축소 예산(16 samples/seqlen 256).
2. **tokenizer 아티팩트 발견·수정** — baseline KO PPL이 이상하게 낮았는데,
   DeepSeek tokenizer가 한글 음절을 UTF-8 byte-fallback으로 쪼개서 생긴 착시로
   판명. bits-per-byte로 재계산해서 언어 간 정당하게 비교 가능한 지표로 교체.
3. **예산 정상화 + seed 3개** — 64 samples/seqlen 512, seed 3개로 재실행.
   own-language 이득이 n=3 부트스트랩 CI로는 세 언어 모두 0을 배제했지만,
   이 CI가 통계적으로 약하다는 지적에 따라 다음 단계로 넘어감.
4. **진짜 placebo(noise floor) 검증** — Toy0의 english_a/b와 같은 설계로
   `english_only_b`/`korean_only_b`(같은 언어, 겹치지 않는 pool 절반, 3 seed)를
   새로 돌려서 "같은 언어를 다른 표본으로 calibration해도 이 정도 차이가
   나는지"를 직접 측정. **KO는 SUPPORTED, EN은 INCONCLUSIVE**로 판정 —
   "세 언어 모두 보편적"이라던 1차 판정은 폐기.
5. **저자원 언어(Swahili) 추가** — "취약한 언어일수록 효과가 크다"는 KO 하나의
   사례였으므로, 사용자가 선택한 Swahili로 같은 파이프라인(placebo 포함)을
   반복. GPU 4장 병렬로 3시간 50분 만에 완료. **Swahili도 SUPPORTED, KO보다도
   더 결정적**으로 나옴 — 패턴이 두 번째 독립 언어로 확인됨.

### 실험 설정
- **모델:** deepseek-ai/deepseek-moe-16b-base (bf16) — 실제 D²-MoE 압축
  (Fisher-weighted expert merge + whitening 없는 plain SVD delta,
  `delta_ratio=0.8`, `share_ratio=1.0`) 적용
- **Calibration 데이터:** `israel/flores-parallel`(HF), test split — Fisher
  계산은 조건별로 **64 samples × seqlen 512**(최종 정상화 예산; 1차 pilot은
  16 samples/seqlen 256이었다가 상향)
- **Calibration 조건:** `english_only`/`korean_only`/`swahili_only`(각 언어
  전용) 등, own-language 이득 측정용. Placebo(노이즈 기준선)로
  `english_only_b`/`korean_only_b`(같은 언어, 겹치지 않는 pool 절반)도 별도 계산
- **평가 데이터:** FLORES devtest, 언어당 **held-out 60문장**
  (English/Korean/Chinese/Swahili), bits-per-byte(bpb)로 측정 — tokenizer의
  한글 byte-fallback 아티팩트 때문에 per-token PPL 대신 bpb로 전환
- **Seed:** own-language 이득·noise floor 모두 **seed 3개**(`per_seed_abs_diff`)
  기준 평균
- **Swahili 실행 환경:** GPU 4장 병렬, 총 소요 3시간 50분

### 핵심 수치
| 언어 | own-language 이득 | noise floor | 2×floor 기준 | 판정 |
|---|---|---|---|---|
| Korean | 5.598%p | 0.504%p | 1.008%p | **SUPPORTED (5.6배)** |
| Swahili | 7.845%p | 0.535%p | 1.069%p | **SUPPORTED (14.6배)** |
| English | 0.745%p | 0.562%p | 1.124%p | 기준 미달 (INCONCLUSIVE) |

baseline bpb(취약도): English 0.914 < Korean 1.370 < Swahili 2.243 —
취약도 순서와 own-language 이득 순서가 정확히 일치한다.

### 이게 증명하는 것
- "자기 언어로 calibration하면 그 언어가 더 잘 보존된다"(H2)는 **KO·Swahili
  두 언어에서 진짜 placebo로 검증된, 이 프로젝트에서 가장 방어 가능한 결과**다.
- "압축에 더 취약한(사전학습 데이터가 적은) 언어일수록 own-language
  calibration의 실질 효과가 크다"는 취약도-비례 패턴이 **KO 하나의 우연이
  아니라 두 번째 독립 언어(Swahili)로 확인된 패턴**이다.
- 단, 이 결과는 `delta_ratio=0.8` + whitening 미적용(plain SVD) + pp_ratio
  미적용 상태에서 나온 것 — "D²-MoE가 민감하다"가 아니라 **"D²-MoE의
  Fisher-merge 단계가 민감하다"**로 한정해서 읽어야 한다 (whitening 경로의
  기여는 별도의 Fisher×Scale 2×2 실험에서 확인 중, 아직 진행 중이라 이 폴더에
  포함하지 않음).

**원본 데이터:** `/mnt/HDD/minjeong/d2moe_results/phase1/phase1_placebo_gate_result.json`,
`phase1_swahili_gate_result.json`, `baseline/eval_ppl.json`
**세부 문서:** `../00_docs/03_기술노트.md` "1) 예산 정상화", "1.5) placebo", "1.7) 저자원 언어" 절,
`../00_docs/04_전체요약.md` 핵심 발견 5~7번

---

## Figure 4 — `4_2x2_whitening_preliminary.png` (⚠️ PRELIMINARY, 미검증)

### 무엇을 보여주는가
두 패널:
- **(a)** Fisher calibration 언어(EN/KO) × Scale(whitening) calibration
  언어(EN/KO) 2×2 히트맵 — 값은 KO bpb 증가율(%). 색이 진할수록(빨강) KO가
  더 많이 나빠졌다는 뜻.
- **(b)** 같은 4개 셀에서 EN/KO/ZH/Swahili 4개 언어 전부의 bpb 증가율을
  나란히 비교 — "다른 언어들은 셀이 바뀌어도 안정적인데 KO만 요동치는가"를
  직접 확인하는 sanity check 패널.

### 어떤 실험에서 나왔나
Figure 3까지의 모든 결과는 `svd_scale=None`(whitening 없이 plain SVD)
상태였다 — D²-MoE의 두 언어-민감 경로(Fisher-merge / activation-aware
whitening) 중 Fisher-merge 쪽만 켜져 있었던 것이다. "이 효과가 Fisher
경로 때문인지 whitening 경로 때문인지"를 분리하려고, get_scale.py(whitening
통계를 계산하는 D²-MoE 원본 스크립트)를 켜려다가 **버그 3개**를 연달아
발견했다:
1. 메모리 누수 — hook이 쌓은 누적 텐서를 실제로 안 지우는 죽은 코드
   (`raw_scaling_diag_matrix`라는, 어디서도 설정 안 된 속성을 지우고 있었음)
2. 두 번째 메모리 누수 — 반환값(`profiling_mat`)이 전체 27개 레이어의
   결과를 GPU에 올려둔 채로 계속 쌓는 구조
3. 완전히 별개의 버그 — DeepSeek 자체 vendored 모델 코드(`moe_infer`의
   `scatter_reduce_`)에서 나는 결정론적 "illegal memory access", 레이어 13에서
   항상 재현됨

1, 2번은 고쳤고(검증 완료), 3번은 원인이 vendored 모델 코드 안이라 고치는
대신 **`phase1_fisher.py`가 이미 문제없이 쓰고 있던 표준 forward 방식
(`model(input_ids=...)`)으로 whitening 통계 계산을 처음부터 재구현**했다
(`phase1_svd_scale.py`) — get_scale.py를 더 파는 대신 이미 검증된 다른
코드 경로를 재사용하는 쪽을 택함. 재구현한 코드로 EN/KO 두 언어의 whitening
통계를 새로 계산한 뒤, 기존에 이미 계산해둔 EN/KO Fisher 통계와 2×2로
교차시켜 4개 조합을 압축·평가했다.

### 실험 설정
- **모델:** deepseek-ai/deepseek-moe-16b-base — Figure 3과 동일 압축 파이프라인
- **Fisher 데이터:** Figure 3의 `english_only`/`korean_only` seed 0 결과를
  그대로 재사용(새로 계산 안 함)
- **Scale(whitening) 데이터:** `phase1_svd_scale.py`로 새로 계산, EN/KO
  각각 **64 samples × seqlen 512**(Fisher와 동일 예산), seed 0 하나만
- **4개 셀:** (Fisher=EN,Scale=EN) / (Fisher=EN,Scale=KO) /
  (Fisher=KO,Scale=EN) / (Fisher=KO,Scale=KO)
- **평가:** Figure 3과 동일, FLORES devtest 60문장 × 4언어, bpb

### 핵심 수치
| Fisher | Scale | EN | KO | ZH | SW |
|---|---|---|---|---|---|
| EN | EN | 7.7% | **88.7%** | 13.8% | 34.8% |
| EN | KO | 10.6% | **4.7%** | 10.8% | 36.9% |
| KO | EN | 9.5% | **42.9%** | 16.4% | 36.4% |
| KO | KO | 12.8% | **3.7%** | 13.0% | 39.8% |

(표는 baseline bpb 대비 증가율. EN/ZH/SW 열은 4개 셀에서 거의 안 변하는데
**KO 열만 Scale이 EN이냐 KO이냐에 따라 3.7~4.7% ↔ 42.9~88.7%로 극단적으로
갈린다** — Fisher가 EN이든 KO이든 이 패턴은 거의 안 바뀜.)

```
own_scale_gain(KO | Fisher=KO) = 39.28%p   (참고: Fisher 단독 이득은 5.60%p였음 — 7배 이상)
own_scale_gain(KO | Fisher=EN) = 84.03%p
VERDICT: WHITENING_DOMINANT
```

### 이게 (잠정적으로) 시사하는 것
- KO 보존에는 **Fisher calibration 언어보다 whitening calibration 언어가
  훨씬 더 크게 작용하는 것으로 보인다** — whitening이 KO와 매칭되면
  Fisher-단독 결과(17.51%)보다도 좋아지고(3.7~4.7%), 안 맞으면 재앙 수준
  (42.9~88.7%)으로 나빠진다.
- EN/ZH/Swahili가 4개 셀에서 안정적이라는 건(sanity check), 이게 무작위
  계산 오류가 아니라 **KO에 타겟된 진짜 효과일 가능성**을 뒷받침한다.

### ⚠️ 왜 아직 "증명된 결과"가 아닌가
Figure 1~3과 똑같은 검증(seed 3개 반복 + 진짜 placebo)을 아직 하나도
안 거쳤다:
- **seed 1개** — 이 정도로 큰 효과(7배)가 seed noise인지 진짜인지 전혀 모름
- **whitening 통계 자체가 표본 부족에 취약할 수 있음** — 64 samples로
  2048차원 공분산 행렬을 추정하는 건(expert 하나당 실제로 받는 토큰은
  더 적음) Fisher(스칼라/벡터 통계)보다 훨씬 더 많은 표본이 필요할 수 있는
  통계량이라, 극단적인 수치가 진짜 언어 효과인지 추정 불안정성인지 구분이
  안 됨
- **placebo 없음** — Figure 3처럼 "같은 언어, 다른 표본"으로 noise floor를
  재는 과정을 아직 안 거침

**다음 단계:** 이 발견이 진짜인지 확인하려면 whitening도 Fisher처럼
seed 3개 + placebo(`english_only_b`/`korean_only_b`용 whitening도 계산)
검증이 필요하다 — 진행 여부는 사용자 확인 후 결정.

**원본 데이터:** `/mnt/HDD/minjeong/d2moe_results/phase1/phase1_2x2_gate_result.json`
**세부 문서:** `../00_docs/03_기술노트.md` "2) whitening 복원" 절

---

## Table 1 — Belebele downstream 재확인 (⚠️ IN PROGRESS, 3조건 중 1개만 완결)

### 무엇을 보여주는가
Figure 1~4는 전부 FLORES bpb(언어모델링 likelihood 지표)로 측정한 결과다.
`04_전체요약.md`가 논문 핵심 주장의 미해결 전제조건으로 지목한 것 — "이
own-language 효과가 실제 downstream task(4지선다 독해)에서도 재확인되는가"
— 를 Belebele 정확도로 직접 측정한 표. Fisher-merge + own-language whitened
SVD만 적용(pruning 미적용, pp_ratio OFF)한 상태에서, calibration 조건별로
eng/kor/zho 3개 언어 정확도가 baseline 대비 얼마나 보존되는지(retention)를
조건×언어 3×3 격자로 본다.

### 어떤 실험에서 나왔나
2026-07-30 저녁 §4.2 확장 프롬프트로 시작 — 기존에는 `korean_only` 조건
하나만 Belebele로 평가돼 있었고(pruning ON/OFF 둘 다, 5개 언어), 이걸
`english_only`/`mixed_5lang`(+best-effort `chinese_only`)까지 넓혀 own-language
대각 비교가 가능한 격자를 만드는 게 목표였다. swh_Latn/ben_Beng은 baseline
5-shot 정확도 자체가 chance(0.25) 근접으로 사전 flag돼 있어(``belebele_floor_check_result_fewshot5.json``)
이번 확장에서는 처음부터 평가 대상에서 뺐다(eng/kor/zho 3개만).

**중간에 사용자 지시로 중단됨** — `english_only`는 seed 0/1 성공, seed 2는
GPU1을 다른 사용자(kahyeon, 이후 dongjin도)가 상시 점유 중이던 것과
레이스컨디션이 겹쳐 CUDA OOM으로 실패(`Tried to allocate 1.55 GiB`,
merge는 끝났고 eval 진입 직후 사망). `mixed_5lang`은 seed 0/1 성공 후
seed 2가 돌던 중 정지 지시로 kill(출력 없음). `chinese_only`는 큐에
넣었으나 시작 전에 정지. 이후 `safe_gpus.sh`에 `MANUAL_EXCLUDE_GPUS=1`을
추가해 GPU1을 앞으로의 모든 실행에서 하드 제외해뒀다(재개 시 이 문제는
재발하지 않을 것으로 예상).

### 실험 설정
- **모델:** deepseek-ai/deepseek-moe-16b-base — Figure 3/4와 동일 압축 파이프라인
- **평가:** `phase1_belebele_eval.py`(이번에 `--langs`/`--off-only` 옵션 추가),
  5-shot, 언어당 n=200, own-language whitened SVD(`--scale-condition`=조건 자신),
  **pp_ratio OFF만**(구조적 pruning 미적용 — ON은 §4.4 ablation 범위)
- **평가 언어:** eng_Latn, kor_Hang, zho_Hans만 (swh_Latn/ben_Beng은 baseline
  chance-level로 사전 제외)
- **Baseline:** 5-shot, n=200/언어, 기존값 재사용 (eng=0.525, kor=0.360, zho=0.470)
- **Seed:** korean_only는 3개(0/1/2) 전부, english_only/mixed_5lang은 **2개(0/1)만**
  (seed2는 OOM/수동중단으로 결측), chinese_only는 **0개**(미실행)

### 핵심 수치 — retention(=compressed acc / baseline acc), 사용 가능한 seed 평균

| 조건 (Fisher calib 언어) | eng_Latn | kor_Hang | zho_Hans | own-language 칸이 행 최대? |
|---|---|---|---|---|
| korean_only (n=3 seed) | 0.625 | **0.819** | 0.663 | 예 (kor_Hang 최대) |
| english_only (n=2 seed) | 0.686 | 0.681 | **0.729** | 아니오 (zho_Hans가 더 높음) |
| mixed_5lang (n=2 seed) | 0.686 | **0.819** | 0.739 | 해당 없음(단일 own-language 없음) |
| chinese_only | — (미실행) | — | — | — |

(굵게 표시한 값이 각 행의 최대 retention 언어. mixed_5lang은 5개 언어를
섞은 조건이라 "own-language" 대각 비교 대상이 아니라 참고용으로만 넣음.)

원시 acc 값(참고):

| 조건 | eng_Latn seed accs | kor_Hang seed accs | zho_Hans seed accs |
|---|---|---|---|
| korean_only | [0.330, 0.315, 0.340] | [0.320, 0.265, 0.300] | [0.345, 0.285, 0.305] |
| english_only | [0.315, 0.405] | [0.245, 0.245] | [0.340, 0.345] |
| mixed_5lang | [0.345, 0.375] | [0.255, 0.335] | [0.335, 0.360] |

### 지금까지 이게 시사하는 것 (판정 아님, 수치만)
- **korean_only**: own-language(kor_Hang) 칸이 그 행에서 확실히 최대 —
  Figure 3의 bpb 기반 own-language 효과와 방향이 일치.
- **english_only**: own-language(eng_Latn) 칸이 최대가 **아님** — zho_Hans
  retention이 더 높게 나옴. seed 2개뿐이라 noise인지 실제 패턴인지 구분 불가.
- **mixed_5lang**: kor_Hang retention(0.819)이 korean_only의 kor_Hang
  retention(0.819)과 사실상 동일 — 흥미롭지만 seed 2개 + chinese_only
  결측 상태라 해석은 보류.

### ⚠️ 왜 아직 "결과"가 아닌가
- 3조건×3언어 중 1개 조건(korean_only)만 seed 3개 완결, 나머지 2개는 seed
  2개(성공분만), 1개 조건(chinese_only)은 아예 미실행 — 애초 목표였던
  3×3 완전 격자(대각 비교)가 완성되지 않음
- english_only/mixed_5lang은 placebo(noise floor) 비교 없음 — Figure 3
  수준의 검증을 전혀 거치지 않은 원시 수치
- pruning(pp_ratio ON) 미적용 — full pipeline 재확인이라는 원래 목적(§4.1
  전제조건)에는 아직 못 미침, Fisher+whitened-SVD 단계까지만 반영

**다음 단계:** english_only seed2 재시도(GPU1 배제 상태로), mixed_5lang
seed2 재시도, chinese_only 3 seed 신규 실행 — 큐잉 스크립트
`../scripts/wait_and_run_belebele_extended_0730.sh`는 남겨뒀으니 재개 시
그대로 쓰거나 참고 가능.

**원본 데이터:** `/mnt/HDD/minjeong/d2moe_results/{baseline,korean_only,english_only,mixed_5lang}/.../fewshot_5/eval_belebele.json`
**세부 로그:** `../logs/belebele_extended_0730.log`, `../scripts/logs/belebele_extended_milestones.log`

---

## 제외한 것 (의도적)

| 제외한 것 | 이유 |
|---|---|
| per-token PPL 기준 1차 Phase 1 결과 | tokenizer(byte-fallback) 아티팩트로 판명 — bpb로 대체됨 |
| n=3 부트스트랩만으로 낸 "세 언어 모두 H2_HELD" 판정 | 통계적으로 약한 근거(조합 10개뿐)임이 드러나 폐기, Figure 3의 placebo 판정으로 대체 |
| 상대 비율(규모 효과) 확인 | KO 이득이 규모 효과가 아님을 확인하는 **진단용 계산**이지, 그 자체가 독립된 결과는 아님 |
| ZH(Chinese)의 own-language 효과 | placebo를 안 돌려서 SUPPORTED/INCONCLUSIVE 여부를 모름 — 판단 보류 상태 |
| Toy0/Phase 0.5의 개별 중간 단계 그림 | Figure 1이 세 단계를 압축한 최종 버전이라 중복 |

## 재현 방법

```bash
cd ../scripts
conda run -n d2moe_env python make_figures_0726.py
```

네 함수(`fig1_layer_locality`, `fig2_fisher_pilot_a`, `fig3_phase1_placebo`,
`fig4_2x2_whitening`)가 위에 적은 원본 JSON을 그대로 읽어서 그린다 — 새로
계산하는 값은 없다.
