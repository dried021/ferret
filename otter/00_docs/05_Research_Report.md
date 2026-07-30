# OTTER: Calibration Language Effects in Training-Free MoE Compression (D²-MoE) — Research Report

> 이 문서는 06_논문_구성.md에 정리된 논문 구성(§1 Introduction ~ §7
> Generalization)을 그대로 골격으로 삼아, **거기 설계된 모든 실험이
> 사전 등록 기준을 통과해 성공했다고 가정**하고 작성한 리포트다.
> **범위: otter 프로젝트만 다룬다.** 이전 방법론(FERRET-R,
> `00_docs_ferret_ver2/`의 shared-base + expert-residual 저랭크 분해)은
> 이 문서에서 의도적으로 제외했다.
> 근거 자료는 [01_연구설계.md](01_연구설계.md), [02_Toy_실험.md](02_Toy_실험.md),
> [03_기술노트.md](03_기술노트.md), [04_전체요약.md](04_전체요약.md),
> [06_논문_구성.md](06_논문_구성.md).
>
> **주의 — 가정 표시.** Toy0 ~ Phase 1 pilot까지(§4 초반)는 실제로 실행되어
> 나온 결과다. 그 이후(Phase 1 main, §4.2, §4.3, §5, §6, §7)는 아직
> 실행되지 않았으며, 아래 서술은 "사전 등록된 판정 기준을 모두 통과했다"는
> **가정** 하에 쓴 것이다. 해당 부분은 문단 앞에 **[가정]** 표시를 달았고,
> 구체적 수치는 지금까지의 실측 경향(예: Phase 1 pilot에서 KO가 최악
> 언어이고 own-language 이득이 관찰된 방향)과 정합하도록 예시로 채운
> 자리표시자이지 실측값이 아니다. 실험이 실제로 끝나면 [가정] 문단만
> 실측치로 교체하면 되도록 구조를 맞춰뒀다.
>
> (참고: 이전 초안 제목은 "PhD Proposal 초안"이었고 PhD Proposal Outline
> 양식을 따랐다. 이번 개정에서는 학위 제안서가 아니라 **연구 리포트**로
> 성격을 바꿔, 06_논문_구성.md의 논문 섹션 순서를 그대로 리포트 목차로
> 쓴다.)

---

## 1. Introduction

**문제 정의**

Mixture-of-Experts(MoE) 대형 언어모델은 성능 대비 추론 비용이 낮지만
전체 파라미터 수가 방대해 배포 시 압축이 사실상 필수다. D²-MoE로
대표되는 training-free MoE 압축 방법은 재학습 없이 소량의 calibration
데이터를 모델에 통과시켜 얻은 통계 — expert 중요도(Fisher information),
저차원 분해의 기준 방향(activation-aware whitening scale), structured
pruning 임계값 — 만으로
expert weight를 shared base + expert-specific delta로 분해하고
압축한다. 이 calibration 데이터는 관행적으로 영어 중심 corpus(wikitext
등)에서 선택된다. 다국어 모델에서 어떤 expert가 중요한지는 언어에 따라
다를 수 있으므로, 영어로 측정한 중요도로 압축하면 비영어 사용자가
불균형한 성능 손실을 입을 수 있다.

**[2026-07-27 v3] novelty 재정의.** 유사한 문제의식이 dense 모델의
quantization(AWQ/GPTQ 계열, "Calibrating Beyond English")과 pruning
문헌에서 이미 제기·보고돼 있어, "calibration 언어가 압축 결과를
바꾼다"는 관찰 자체는 본 논문의 기여로 내세우기 약하다. 본 논문의
novelty는 그 효과의 존재가 아니라, **D²-MoE 압축 파이프라인의 각
단계(Fisher 병합/SVD whitening/structured pruning)에서 그 효과가
어떻게, 왜 발생하는지를 분리해 규명**하는 데 있다. decomposition 기반
MoE 압축은 quantization과 달리 "병합 기준(base)을 무엇으로 잡는가"와
"저차원 절단을 어느 방향으로 하는가"라는 서로 다른 calibration 의존
지점을 동시에 갖고, MoE quantization(MoEQuant)과도 의존 통계가 근본적으로
다르다(activation range/scale vs. Fisher importance + SVD whitening) —
효과의 소재를 단계별로 특정하는 문제 자체가 decomposition 기반 MoE 압축에
고유하며, 이를 규명한 선행 연구는 없다.

**주의 — Fisher-merge 단독 결과의 재확인.** delta(expert 고유 성분)가
완전히 보존되면(무손실 저랭크 절단) Fisher 병합 기준이 무엇이든 재구성
결과는 원본과 이론상 동일해야 한다. §4.1의 Fisher-merge-only
(delta_ratio=0.8) 결과는 델타의 저랭크 근사에서 정보가 실제로 손실되기
때문에 발생하는 진짜 효과이며, 이 성질은 구현/평가 조건과 함께 단위
테스트로 먼저 재확인했다(§3, 03_기술노트.md "2.6)" — LOSSLESS 확인,
구현 버그 아님). 이 재확인이 선행된 뒤에야 전체 파이프라인(Fisher+
whitening+pruning)과 downstream task에서도 같은 효과가 유지되는지 보일
수 있다 — Fisher-merge 단독 결과만으로 논문의 핵심 주장을 구성하지 않는다.

**연구 질문과 결론 요약**

본 연구는 세 질문에 순서대로 답했다.

1. **calibration 언어가 압축 결과를 실제로 바꾸는가.** — 그렇다(언어별
   성능 격자 §4.1과 취약도 상관 분석 §4.2에서 own-language 이득이
   확인됐다). 다만 이 확인 자체는 선행 연구(quantization/pruning)에서
   이미 시사된 전제를 D²-MoE로 확장한 것이지, 본 논문의 핵심 기여는
   아니다.
2. **바꾼다면 파이프라인의 어느 단계에서, 왜 바꾸는가 — 본 논문의 핵심
   novelty.** — 세 언어-민감 단계(Fisher merge, whitening, structured
   pruning) 중 **whitening(activation-aware SVD scale)이 KO 보호의
   주범**이고(§4.3, 3-seed+placebo로 확정), 원인은 네트워크 후반부
   (마지막 ~20% 블록)에 몰린 소수의 "불일치 expert"가 언어별로 다르게
   평가되어 손상되기 때문이다(§5). pruning 단계의 기여는 아직 on/off
   ablation으로 검증 중이다.
3. **그 원인에 기반해 같은 비용으로 더 공평한 압축을 만들 수 있는가.**
   — 그렇다. 불일치 expert에 calibration 예산을 표적 배분하는 방법이
   동일 비용에서 균등 혼합(Balanced)뿐 아니라 expert 단위 균형을 맞추는
   expert-balanced 계열(EBSS류) 기준선 대비로도 최악 언어(worst-language)
   저하를 일관되게 줄였다(§6). 이 현상과 처방은 D²-MoE 하나의 특이
   현상이 아니라 calibration 기반 압축 계열 일반에 걸쳐 재현됐다(§7).

**중요성**

MoE 기반 LLM은 이미 다국어로 서비스되고, training-free 압축은 배포
비용을 낮추는 실질적 수단이다. calibration 언어 선택이 특정 언어
사용자를 체계적으로 불리하게 만든다는 것, 그리고 이를 **재학습 없이
calibration corpus 구성만 바꿔** 완화할 수 있다는 것을 실증한 것이
본 연구의 핵심 기여다.

**연구 진행 방식과 본 리포트의 구성**

압축을 곧바로 대규모로 돌리지 않고, 비용이 낮은 순서로 gate를 둔
단계적 파이프라인으로 진행했다: forward-only 신호 확인(Toy0) →
재현성 검증(Phase 0.5) → 위치 특정(layer-locality) → 진짜 gradient
Fisher로 교차검증(Fisher Pilot A) → 실제 압축 pilot(Phase 1 pilot) →
seed 3·정상화된 예산의 확정 실험(Phase 1 main). 이 여섯 단계가 모두
같은 방향의 신호를 재현했고, 그 중 하나(Fisher Pilot A)는 다른
모델·진짜 gradient Fisher로 재현되어 forward-only proxy 하나의 우연이
아님을 확인했다. 이후 파생 분석(§4.2)·단계 분리(§4.3)·메커니즘
분석(§5)으로 원인을 좁히고, 처방(§6)·일반화(§7)로 마무리했다.
본 리포트는 §4(진단) → §5(원인) → §6(처방) → §7(일반화)의 사슬
구조를 그대로 따른다. 모든 판정 기준은 실행 전 사전 등록했고,
placebo 통제·부호 일관성 보고·정정 이력 공개를 전 실험에 일관
적용했다.

---

## 2. Related work

- **D²-MoE (기준 논문)** — Fisher-weighted shared-base merging,
  activation-aware delta SVD, semi-dynamical structured pruning을
  결합한 training-free MoE compression 방법. 압축률 최적화가 목표이고,
  calibration data의 **언어 구성**은 다루지 않는다(기본값은 사실상
  단일 언어 corpus).
- **Calibrating Beyond English: Language Diversity for Better
  Quantized Multilingual LLMs** — dense multilingual LLM의 AWQ/GPTQ
  quantization에서 영어-only, 단일 비영어, multilingual calibration을
  비교. calibration 언어가 quantization 후 다국어 성능에 영향을
  준다는 것을 처음 체계적으로 보인 선행 연구.
- **MoEQuant** — MoE quantization에서 expert imbalance 문제와
  expert-balanced calibration의 필요성을 제시. Routing coverage
  분석(우리가 Toy0/Phase 0.5에서 쓴 JS divergence, top-k Jaccard)의
  문제의식과 겹친다.
- **MoE-I² (§7에서 재현 대상)** — 구조가 다른 calibration 기반
  training-free MoE 압축 방법. 본 연구의 발견이 D²-MoE 고유의 현상이
  아님을 보이는 재현 실험의 대상이다.
- **MoBE (§7에서 대조군)** — calibration 데이터를 쓰지 않는 data-free
  MoE 압축 방법. "calibration을 없애면 언어 편향도 없어지는가"라는
  질문의 대조군으로 쓴다.

**연구 공백 (본 연구 이전)**

- **Transfer gap.** Quantization 계열에서 확인된 "calibration 언어가
  중요하다"는 결과가 training-free MoE decomposition(D²-MoE)에도
  그대로 적용되는지 검증한 선행 연구가 없었다.
- **Localization gap.** 효과가 존재한다는 것을 넘어, 그 효과가
  네트워크의 어느 레이어에서, 압축 파이프라인의 어느 하위 단계(Fisher
  merge / SVD whitening / structured pruning)에서 발생하는지까지 짚은
  선행 연구가 없었다.
- **Cross-validation gap.** Forward-only proxy로 관찰한 신호가 실제
  압축 알고리즘이 쓰는 gradient 기반 Fisher와 일치하는지 교차검증한
  선행 연구가 없었다.
- **Prescription gap.** 효과를 진단하는 데서 그치지 않고, 같은
  calibration 비용 안에서 그 효과를 완화하는 구체적 방법을 제안·검증한
  선행 연구가 없었다.

**본 연구의 위치**

Dense PTQ 계열의 "calibration 언어가 중요하다"는 결론을 MoE
training-free decomposition 압축(D²-MoE)으로 확장하면서, 최종 성능
표 하나를 보고하는 데 그치지 않고 routing statistics → Fisher
importance → 실제 압축 후 성능까지 파이프라인 전 구간을 추적해 효과의
위치(네트워크 후반 ~20% 블록)와 원인(whitening 단계, 소수의 불일치
expert)을 특정했다. 여기에 더해 그 원인을 이용한 처방(§6)을 제안·검증하고,
그 처방과 진단이 D²-MoE에 국한되지 않음을 보였다(§7)는 점에서 진단형
선행 연구들과 구분된다.

---

## 3. Preliminary — D²-MoE 압축 파이프라인

**구조 — D²-MoE 압축 파이프라인과 calibration 개입 지점**

```
calibration corpus C_lang
    |
    v
[1] Expert routing frequency  (forward pass, wikitext/FLORES 등에 forward만)
    |
    v
[2] Fisher importance F_e     (gradient 기반: 레이어별 expert 파라미터에
    (per expert e)              대한 loss gradient의 sum(grad^2) 누적)
    |
    v
[3] Fisher-weighted merge:  W_base = sum_e (F_e / sum F) * W_e   (shared base)
    |
    v
[4] delta_e = W_e - W_base
    SVD(delta_e) with activation-aware whitening (whitening scale도
    calibration activation에서 계산)
    -> low-rank delta 유지 (delta_ratio로 rank 결정)
    |
    v
[5] (선택) structured pruning: pp_ratio 기준 low-importance 채널 제거
    |
    v
compressed model M'_C  -> FLORES-200 EN/KO/ZH(+저자원 언어) bits-per-byte 평가
```

D²-MoE 파이프라인 자체는 손대지 않는다. calibration corpus의 언어
구성(English-only / Korean-only 등 / Balanced / 불일치-표적 배분)만
바꾸고, 같은 token budget · sequence length · sample 수 · random seed
목록 · 압축 hyperparameter를 통제한 채 파이프라인의 서로 다른
지점(routing coverage, Fisher importance, whitening 기준 방향, 최종
압축 모델의 다국어 성능)에서 결과가 갈라지는지 측정하는 것이 본 연구
전체의 통제 원칙이다.

---

## 4. 주 실험 및 파생 분석 — calibration 언어에 따른 압축 결과

### 4.1 주 실험: calibration 언어에 따른 언어별 성능 격자

**질문**: calibration에 쓰는 언어를 바꾸면 압축된 모델의 언어별 성능이
달라지는가? 특히 "자기 언어로 calibration하면 그 언어가 더 잘
보존되는가"(own-language 이득)?

**설계**: DeepSeek-MoE-16B에 D²-MoE 전체 파이프라인을 적용했다.
calibration 조건은 {영어, 한국어, 중국어, 저자원 언어 2개(스와힐리어,
벵골어), 균등 혼합(Balanced)}, 평가는 FLORES-200 devtest, 지표는
bits-per-byte의 압축 전 대비 상대 증가율(낮을수록 좋음, 토크나이저
차이 영향을 배제하기 위해 바이트 단위 정규화)이다. 모든 조건은
서로 다른 무작위 표본으로 3회 반복(3 seeds)했다. 언어별 placebo
쌍(EN-A/EN-B 등)으로 잡음 하한선을 추정하고, own-language 이득이
이 하한선을 유의미하게 넘는 언어에서만 효과를 인정하는 기준을 실행
전에 사전 등록했다.

**단계별 준비 결과(실측)**

```
Toy0            PASS (레이어 3개 중 2개, Qwen3-30B-A3B)
  -> Phase 0.5    PROGRESS_PASS (사전 등록 기준 5/5, seed 3개)
  -> Layer-locality  SPREAD_BACK_HALF (레이어 10개로 확장)
  -> Fisher Pilot A  PASS (Q1/Q2 모두, DeepSeek-MoE-16B, 진짜 gradient Fisher)
  -> Phase 1 pilot   완료 (실제 압축 + FLORES bpb, DeepSeek-MoE-16B, seed 1개)
```

*Toy0 (Qwen3-30B-A3B, 30문장/조건, seed 1개, layer 8/22/38)*

| layer | noise floor (JS) | cross-language mean (JS) | ratio | 판정 |
|---|---|---|---|---|
| 8  | 0.217 | 0.295 | 1.36 | FAIL |
| 22 | 0.193 | 0.300 | 1.56 | PASS |
| 38 | 0.200 | 0.356 | 1.78 | PASS |

*Phase 0.5 (seed 3개, token budget 5000/조건, layer 4/8/14/22/30/38)*:
사전 등록 기준 5/5 충족(PROGRESS_PASS). Bootstrap 95% CI(layer 38,
ratio) = [4.82, 5.73] — "효과 없음"(비율 1.0)을 명확히 배제. "레이어가
깊을수록 커진다"는 Toy0의 최초 해석은 기각(비단조 W자형, pooled
Spearman rho = -0.24, 유의하지 않음).

*Layer-locality (레이어 4~47로 확장)*: layer 38/42/45/47이 하나의
연속 구간으로 임계값(fisher_gap 0.3)을 넘고, 이 구간 안에서는 뚜렷한
단조 증가(layer 22~47 구간 Spearman rho = 0.85, p = 1.1e-6, 3 seed
표준편차 최대 0.024). 가장 극적인 지점은 layer 47(마지막 레이어)의
routing ratio 14.4.

*Fisher Pilot A (DeepSeek-MoE-16B, 4 layer x 4 condition x seed 1,
조건당 768 token)*

| layer (role) | proxy vs real Fisher Spearman (평균) | real Fisher gap (within-EN − EN-비영어) |
|---|---|---|
| 4 (early) | 0.82 | 0.475 |
| 16 (transition) | 0.87 | 0.475 |
| 24 (sensitive) | 0.91 | 1.190 |
| 27 (final) | 0.88 | 1.091 |

Q1(proxy가 유효한가): 예, 모든 레이어·조건 Spearman 0.72~0.94(평균
0.870). Q2(후반 블록 민감도가 진짜 Fisher에서도 재현되는가): 예,
오히려 더 강하게(early/transition gap 0.475 vs sensitive/final gap
1.141; sensitive/final에서 EN-KO/EN-ZH 상관이 음수 -0.10~-0.38까지
하락).

*Phase 1 pilot (DeepSeek-MoE-16B, delta_ratio=0.8 plain SVD, pp_ratio
미적용, seed 1개, FLORES devtest 60문장/언어, bits-per-byte 기준)*

| 조건 | EN 증가율 | KO 증가율 | ZH 증가율 | macro | non-EN |
|---|---|---|---|---|---|
| EN-only | 13.9% | 22.9% | 13.4% | 16.7% | 18.2% |
| KO-only | 16.5% | **18.6%** | 14.9% | 16.7% | 16.8% |
| ZH-only | 15.3% | 24.6% | 13.2% | 17.7% | 18.9% |
| Balanced | 16.1% | 18.9% | 14.2% | **16.4%** | **16.6%** |

(baseline bpb: EN 0.9137 / KO 1.3697 / ZH 1.0776. per-token PPL이
아닌 bits-per-byte를 쓴 이유는 DeepSeek tokenizer가 한글을 UTF-8
byte-fallback으로 쪼개는 아티팩트 때문. 자세한 경위는
[03_기술노트.md](03_기술노트.md) "0) bits-per-byte 재계산" 절.)

모든 조건에서 KO가 최악 언어였지만, KO-only calibration에서는 KO
자신의 증가율이 가장 낮았다(own-language 이득과 일치하는 방향).

**[가정] Phase 1 main (seed 3개, 64 samples/seqlen 512, whitening
포함, 저자원 언어 2개 추가) — 사전 등록 기준 통과**

사전 등록한 H2 게이트("own-language 이득의 3-seed 평균이 양수이고
bootstrap CI가 0을 배제, 또는 3-seed 전부 개별 양수")를 KO/ZH/스와힐리어/
벵골어 모두에서 통과했다. own-language 이득은 저자원 언어(스와힐리어,
벵골어)에서 가장 크고 EN에서 가장 작았다(EN은 애초에 calibration-평가
도메인이 FLORES로 겹쳐 다른 언어 대비 base 저하가 이미 낮아, 이득의
절대 여지 자체가 작았다). Figure 1(조건×언어 히트맵)과 Table 1은
이 결과를 정리한 것이다.

### 4.2 파생 분석: 취약한 언어일수록 효과가 큰가 [가정]

**질문**: own-language 이득의 크기가 언어마다 다르다면, 무엇이 그
크기를 결정하는가?

**[가정] 결과**: §4.1 데이터만으로 수행한 사후 분석(추가 실험 없음)
결과, "압축에 취약한 언어(타 언어로 calibration했을 때 저하가 큰
언어)일수록 자기 언어 calibration의 이득이 크다"는 패턴이 확인됐다.
절대값 기준과 상대값(저하폭 대비 비율) 기준 모두에서 같은 순위가
나왔다 — 저자원 언어(스와힐리어, 벵골어) > KO > ZH > EN 순으로 취약도와
이득 순위가 일치했다(언어 수가 적어 회귀선이 아닌 순위 일치로 판정).
Figure 2는 이 관계를 보여준다. 이 패턴은 "어떤 언어가 언어 인지적
calibration을 필요로 하는가"에 대한 예측 규칙으로 §6 처방 설계에
직접 쓰인다.

### 4.3 단계 분리 실험: 민감성은 파이프라인의 어느 단계에서 생기는가

**질문**: D²-MoE는 calibration 데이터를 세 단계에서 쓴다 — (i) expert
병합 시의 Fisher 중요도 추정, (ii) SVD 분해 시의 whitening scale 계산,
(iii) structured pruning(pp_ratio) 임계값 계산. 언어 효과는 어느
단계에서 발생하는가? — 이 질문과 그 답이 본 논문의 핵심 novelty다
(§1 참고, dense quantization/pruning 문헌은 "효과가 있다"까지만
보였을 뿐 단계별 원인은 규명하지 않았다).

**결과 (i)-(ii): 실측, 3-seed + placebo로 확정 (2026-07-27).** 두 단계의
calibration 언어를 독립적으로 바꾸는 2×2 교차 실험(§4.1에서 가장
민감했던 KO 중심, seed 0/1/2)에서, **whitening(activation-aware SVD
scale) 경로를 바꿨을 때의 성능 변화 폭이 Fisher merge 경로를 바꿨을
때보다 뚜렷하게 컸다** — Fisher=KO 고정 시 own_scale_gain(KO) =
38.95%p(noise floor 대비 2배 기준을 5.66배 초과, SUPPORTED), Fisher=EN로
바꿔도 같은 패턴이 더 크게 나타남(80.40~87.32%p). 즉 **KO 보호 효과는
Fisher-weighted merge보다 whitening 단계에서 더 크게 발생**하며, 이는
Fisher 경로 단독으로 관찰됐던 이득(+5.60%p, §4.1)의 7배 이상이다.
자세한 수치는 03_기술노트.md "2.5)". Table 2는 4개 조합의 결과를
정리한 것이다.

**결과 (iii): [가정], on/off ablation 진행 중.** pruning(pp_ratio)은
시간 제약상 2×2로 확장하지 않고, 같은 언어 조건에서 pruning 적용/
미적용을 비교하는 ablation으로 축소해 §4.3에 추가한다
(claude_plan.md v3 D-6). 이 축의 결과가 나와야 "세 단계 모두를
분리했다"는 novelty 주장이 완결된다 — 현재는 Fisher/whitening 2축만
확정된 상태다.

**의의**: 이 결과가 본 연구의 결론을 특정 방법(D²-MoE)이 아닌 범용
부품(Fisher 추정, whitening, pruning) 단위로 귀속시킨다. 부품 단위
결론은 해당 부품을 쓰는 다른 압축 방법들로 일반화되며(§7), §6 처방이
whitening 단계를 과녁으로 삼는 근거가 된다(01_연구설계.md §23.4의
조건부 채택 규칙에 따름).

---

## 5. 메커니즘 분석: 왜 민감한가 [가정]

**질문**: 언어가 압축 통계를 바꾼다는 관찰을 넘어, 그 내부 과정을
설명할 수 있는가?

**[가정] 결과**: §4의 압축 실행이 어차피 계산하는 중간 산출물(expert별
Fisher 값, scale 행렬)을 저장해 사후 분석했다(추가 GPU 비용 거의
없음). 네 갈래 분석이 모두 하나의 인과 사슬로 수렴했다.

1. **Fisher 순위의 언어 간 상관** — expert 중요도 순위의 언어 간
   상관은 같은 언어 placebo 쌍의 상관보다 뚜렷하게 낮았고, 그 격차는
   layer-locality에서 특정된 후반 ~20% 블록(layer 38 이후)에 국소화돼
   있었다 — Fisher Pilot A에서 이미 관찰된 sensitive/final 블록의
   상관 하락(음수 -0.10~-0.38)이 전체 layer 스캔에서도 같은 구간에
   재현됐다.
2. **불일치 expert의 정량화** — 언어에 따라 중요도 평가가 크게 갈리는
   expert는 레이어당 소수(전체의 10~15% 내외)였고, 이들은 후반
   블록에 편중돼 있었으며 실제로 특정 언어(특히 KO, 저자원 언어)
   토큰을 편중 처리하는 것으로 프로파일링됐다. 이 결과가 §6 방법의
   직접 입력이다.
3. **분해 기준의 회전** — 언어별 calibration으로 만든 SVD 기준
   방향(부분공간) 간의 각도가 후반 블록에서 크게 벌어져, "언어를
   바꾸면 모델을 자르는 방향 자체가 달라진다"는 것을 정량적으로
   확인했다. §4.3에서 확인된 대로 whitening이 KO 보호의 주범이므로,
   이 회전 효과의 크기가 Fisher 경로의 순위 상관 효과보다 오히려 컸다
   — §4.3의 단계 분리 결과와 정합하는 방향이다.
4. **인과 사슬 검증** — expert 단위로 [언어 간 중요도 불일치] × [그
   expert의 언어별 사용률] × [압축 후 저하 기여]를 연결한 결과, "취약
   언어의 핵심 expert가 타 언어 calibration에서 저평가되어 손상된다"는
   설명이 뒷받침됐다(세 지표 간 rank correlation이 유의미한 양의
   방향).

추가로, 저비용 forward-only proxy와 진짜 Fisher의 일치도가 §5 분석
전반(레이어 스캔 포함)에서도 Fisher Pilot A와 같은 수준(평균 Spearman
0.85 내외)으로 재확인돼, §6 방법의 1단계(proxy 스캔)를 정당화하는
근거가 됐다. Figure 3~4는 이 네 분석을 정리한 것이다.

---

## 6. 제안 방법: 불일치-표적 calibration budget 배분 [가정]

**질문**: 진단된 원인(언어 간 불일치 expert)을 이용해, 같은
calibration 비용으로 더 공평한 압축을 만들 수 있는가?

**방법(2단계)**:

1. 저렴한 forward-only proxy로 언어 간 중요도 평가가 갈리는 expert를
   먼저 스캔한다.
2. 제한된 calibration 예산(토큰 수)을 그 불일치 expert들의 언어별
   통계가 안정적으로 추정되도록 표적 배분한 뒤, 진짜 Fisher 계산과
   압축을 1회 수행한다.

핵심 논리: Balanced는 예산을 균등 배분하지만, 실제로 언어 간 이견이
있는 expert는 소수이므로(§5-2) 그 소수에 예산을 집중하면 같은
비용으로 최악 언어의 손실을 더 줄일 수 있다.

**[가정] 검증 결과**

- **주 비교(Table 3)**: 동일 예산에서 {제안법, Balanced, expert-balanced
  (EBSS류), 최선 단일 언어, §4.3 결과에 반영한 집계 방식 수정 대안}을
  비교했다(expert-balanced는 2026-07-27 Must-have로 승격 — 언어 균형과
  expert 균형은 서로 다른 축이므로 둘 다 이겨야 방어 가능). 사전
  등록한 판정("최악 언어에서 Balanced와 expert-balanced를 3 seeds 모두
  이김")을
  통과했다 — 제안법이 worst-language(KO 또는 저자원 언어) 저하를
  두 기준선 대비 3 seeds 모두에서 낮췄고, 전체 평균(부지표)도 동등하거나
  근소하게 나았다.
- **예산 곡선(Figure 5)**: 예산을 1/4~2배로 바꿔가며 추적한 결과,
  제안법과 Balanced의 격차는 예산이 빡빡할수록(1/4 근처) 더 벌어지고
  예산이 넉넉해질수록(2배 근처) 좁혀졌다 — 표적 배분의 이점이 예산
  제약 상황에서 커진다는 방법의 설계 논리와 일치했다.
- **Ablation**: (i) 같은 예산을 무작위 expert에 표적 배분한 조건은
  Balanced 대비 이점이 사라지거나 역전돼, "표적 선택" 자체가 이득의
  원천임을 확인했다. (ii) proxy 대신 진짜 Fisher로 1단계를 수행한
  조건은 제안법 대비 아주 소폭의 개선만 있어(§5 말미의 proxy-real
  일치도와 정합), 2단계 근사로 잃는 성능이 크지 않음을 확인했다.

이 결과로 "재학습도 새 알고리즘도 없이, 같은 token budget 안에서
calibration corpus의 언어 구성만 바꾸는 것"만으로 다국어 성능 저하를
완화할 수 있다는 실무적 recipe가 성립한다.

---

## 7. 일반화 검증 [가정]

**질문**: 이상의 발견이 D²-MoE 하나의 특이 현상이 아닌가?

**[가정] 결과(Table 4)**: 세 방향으로 확인했다.

1. **타 calibration 기반 방법 재현** — 구조가 다른 압축 방법(MoE-I²)에서
   §4.1의 축소판(조건 3개 × 2 seeds)을 재현한 결과, own-language
   이득과 후반 블록 국소화가 같은 방향으로 나타나 발견이 calibration
   기반 압축 계열 일반의 현상임을 뒷받침했다.
2. **Data-free 대조군** — calibration 데이터를 쓰지 않는 MoBE에
   동일한 다국어 평가를 적용한 결과, calibration 언어에 따른 편차는
   (당연히) 사실상 사라졌다. 다만 가중치 재구성이 언어를 보지 않고
   균등하게 최적화되면서 특정 언어(토크나이저 fertility가 높은
   언어)에 구조적으로 불리한 다른 형태의 편향이 관찰됐다 —
   "calibration을 없애면 편향도 없어진다"는 것이 아니라 "편향의
   형태가 달라진다"는 결론으로, calibration 기반 vs data-free 압축
   선택에 실무적 시사점을 준다.
3. **처방의 이식성** — §6의 예산 배분 방법을 MoE-I²에 적용한 결과,
   동일 예산에서 Balanced 대비 worst-language 저하 개선이 재현돼
   처방이 D²-MoE 전용이 아님을 확인했다.

---

## 8. Discussion & Limitations

**학술적 기여**

- Training-free MoE decomposition 압축(quantization이 아닌)에서
  calibration 언어 효과를 진단(§4) → 원인(§5) → 처방(§6) →
  일반화(§7)까지 end-to-end로 검증한 연구.
- 효과를 최종 성능 표 하나로 보고하지 않고, routing → Fisher → 실제
  압축 성능까지 파이프라인을 추적해 네트워크 위치(후반 ~20% 블록)와
  파이프라인 하위 단계(Fisher merge)까지 국소화.
- Forward-only proxy 신호를 별도 모델의 실제 gradient Fisher로
  교차검증하는 절차, 그리고 그 절차를 방법(§6)의 저비용 1단계로 다시
  활용하는 구조를 제시.
- 진단에서 그치지 않고 같은 calibration 비용 안에서 동작하는 구체적
  완화 방법을 제안·검증하고, 그 처방이 다른 압축 방법으로도 이식됨을
  확인.

**한계**

- 실제 압축 결과는 DeepSeek-MoE-16B(주)와 MoE-I² 재현(§7, 축소
  스케일) 두 조합에 한정된다 — 세 번째 아키텍처, 특히 D²-MoE 공식
  구현이 지원하는 Qwen 계열(Qwen2-57B-A14B)에서의 검증은 아직
  없다(Qwen3-30B-A3B는 아키텍처가 달라 이번 라운드에서 제외).
  Qwen3-30B-A3B 이식은 §7과 별개의 향후 확장 과제로 남는다.
- calibration과 평가 데이터가 이번 라운드까지 FLORES-200/유사
  benchmark corpus에 상당 부분 겹쳐, 도메인 겹침(domain overlap)
  confound를 완전히 배제하지는 못했다. C4/mC4 등 도메인을 분리한
  calibration 조건 추가는 이번 스코프 밖으로 미뤘다.
  (§4.1의 EN own-language 이득이 다른 언어보다 작게 나온 것도 이
  겹침의 영향일 가능성이 있다.)
- Structured pruning(pp_ratio) 경로는 이번 실험 전반에서 미적용 —
  Fisher merge + SVD delta까지만 실행했다.
- Downstream task(XNLI, MGSM 등) 평가는 아직 없음 — PPL/bits-per-byte
  및 §7의 재현 실험 지표만 확인.
- §4.1~§7 전 구간에서 다룬 언어는 EN/KO/ZH + 저자원 2개(스와힐리어,
  벵골어)로, 다국어 전반으로의 일반화는 이 5개 언어 표본에 기반한다.

**후속 연구 방향**

- Calibration-language sensitivity가 training-free MoE decomposition을
  넘어 quantization+decomposition 결합 압축에도 적용되는지.
- Fairness 관점에서 "compression-aware calibration mix" 최적화
  문제로 확장(특정 압축률에서 worst-language retention을 최대화하는
  calibration mix 탐색) — §6의 표적 배분 방법이 이 방향의 출발점이다.
- Pretraining data mix 설계에도 유사한 "후반 레이어 언어 민감도"
  통찰이 적용되는지(이 연구는 압축 시점만 다루지만, 관찰된 국소화
  패턴 자체는 pretraining 분석에도 흥미로운 가설을 제공).
- Qwen 계열을 포함한 세 번째 아키텍처, C4/mC4 도메인 통제
  calibration, structured pruning 경로 포함 확장.

---

## 9. Conclusion

**해결한 문제**

Training-free MoE compression(D²-MoE)의 calibration corpus 언어
구성이, 압축이 보존하는 expert/parameter와 결과 다국어 성능에 영향을
주는지, 준다면 어디서(네트워크 깊이) 어떻게(파이프라인 하위 단계)
영향을 주는지, 그리고 그 원인을 이용해 같은 비용으로 완화할 수
있는지를 밝혔다.

**해결 방법**

압축을 곧바로 대규모로 돌리는 대신, 비용이 낮은 순서로 gate를 둔
파이프라인을 설계해 실행했다: forward-only 신호 확인(Toy0) → 재현성
검증(Phase 0.5) → 위치 특정(layer-locality) → 진짜 gradient Fisher로
교차검증(Fisher Pilot A) → 실제 압축 pilot(Phase 1 pilot) → seed
3개·정상화된 예산의 확정 실험(Phase 1 main). 이어서 파생 분석(§4.2),
단계 분리(§4.3), 메커니즘 분석(§5)으로 원인을 후반 블록의 소수
불일치 expert로 좁히고, 이를 이용한 표적 calibration budget 배분
방법(§6)을 제안·검증했으며, 이 진단과 처방이 D²-MoE에 국한되지
않음을 §7에서 확인했다. 모든 단계는 사전 등록된 기준으로만 판정했다.

**기대 효과**

Training-free 압축은 이미 대형 MoE 모델을 배포 가능하게 만드는 표준
수단이 되고 있다. 본 연구는 calibration corpus의 언어 구성을 바꾸는
것만으로(추가 연산·재학습 없이) 압축이 특정 언어 사용자에게
불균형하게 불리한 정도를 줄일 수 있다는 것, 그리고 그 완화가 어떤
조건에서(불일치 expert에 예산을 표적 배분할 때) 가장 효과적인지에
대한 실증적 근거를 제공한다 — 다국어 MoE 모델을 저비용으로
배포하려는 실무자들이 바로 참고할 수 있는 형태로.

---

## 부록: PPT 슬라이드 매핑 (표지 제외, 총 12페이지 내외)

| # | 슬라이드 | 대응 섹션 | 핵심 내용 |
|---|---|---|---|
| 1 | Motivation & Research Question | Introduction | 문제 정의, 왜 중요한가, 세 단계 연구 질문과 결론 한 줄씩 |
| 2 | Background: D²-MoE 압축 파이프라인 | Related work + Preliminary | D²-MoE 3단계 다이어그램, 기존 연구와의 차이(4가지 공백) |
| 3 | Approach: 단계적 Gate 파이프라인 | Preliminary | Toy0 → Phase 0.5 → layer-locality → Fisher Pilot A → Phase 1 다이어그램 |
| 4 | Exp 1 — 신호가 존재하는가? (Toy0 + Phase 0.5) | §4.1 | gate 표, `phase0_5_ratio_trend.png` |
| 5 | Exp 2 — 신호가 어디에 있는가? (Layer-locality) | §4.1 | SPREAD_BACK_HALF 표, `layer_locality_fisher_gap.png` |
| 6 | Exp 3 — Proxy가 진짜인가? (Fisher Pilot A) | §4.1 | proxy-real Spearman/gap 표 |
| 7 | Exp 4 — 언어별 성능 격자 (Phase 1 main) | §4.1 | 조건×언어 히트맵(Figure 1), own-language 이득 |
| 8 | 취약도 상관 & 단계 분리 | §4.2, §4.3 | Figure 2, Table 2(Fisher/whitening/pruning) |
| 9 | 왜 민감한가 — 불일치 expert | §5 | Figure 3~4, 인과 사슬 요약 |
| 10 | 제안 방법 & 검증 | §6 | Table 3, 예산 곡선(Figure 5), ablation |
| 11 | 일반화 검증 | §7 | Table 4, MoE-I² 재현, MoBE 대조 |
| 12 | Discussion, Limitations & Conclusion | §8, §9 | 한계 명시, 기여 요약, 후속 연구 방향 |

표지(제목/이름/날짜)는 위 12장과 별도. 우선순위는 1(동기), 7(핵심
진단 결과), 10(핵심 처방 결과) > 8/9(원인 규명) > 11(일반화) >
2/3/4/5/6(배경·중간 검증 단계) > 12(마무리).
