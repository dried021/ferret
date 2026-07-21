# 15. FERRET Toy0 실험 결과 — Qwen3-30B-A3B

**실행 위치**: `ferret_toy0_qwen3/` (모델·트레이스·SVD 팩터·결과 CSV/Parquet·그림 전체 포함)
**실행일**: 2026-07-21 · **GPU**: 물리 GPU 0,1 (4-GPU 공유 호스트, `CUDA_VISIBLE_DEVICES=0,1`
— 물리 GPU 3이 다른 사용자 작업으로 15GB/96% 사용 중이라 배정) · **백본 정밀도**: INT8
(bitsandbytes, `llm_int8_enable_fp32_cpu_offload=True`) · **전문가 원본 가중치**: BF16
(safetensors 샤드에서 직접 읽음, 전체 모델 미로드) · **최대 GPU 메모리 사용**: cuda:0
22.36GB / cuda:1 21.3GB (budget 22.5GB 이내)

[13_Toy0_방법론_단일GPU.md](13_Toy0_방법론_단일GPU.md)의 방법론을 따르되, 모델이
2xRTX 3090(48GB)에도 bf16 풀프리시전으로는 들어가지 않는 30B급이라 INT8 백본 +
2-GPU `balanced` device_map으로 확장했다. [14_Toy0_실험결과.md](14_Toy0_실험결과.md)
(Granite MoE 1B-A400M)와 같은 파이프라인·같은 Go/No-Go 기준을 그대로 적용한
두 번째 데이터 포인트다.

---

## 1. 요약 (TL;DR)

| 질문 | 답 |
|---|---|
| **Gate1** 전문가가 낮은 랭크(≤50%)로도 잘 압축되는가? | **아니오** — activation-aware 재시도 후에도 실패 (Gate1 **FAIL**, 0/4) |
| **RQ1 / Gate2** 토큰별 필요 랭크가 이질적인가? | **아니오** — 4개 전문가 중 3개는 오라클 랭크의 100%(또는 99.96%)가 단일 값(768=full rank)에 몰림 (Gate2 **FAIL**, 0/4) |
| **RQ2 / Gate3** 동일 평균 랭크에서 Oracle이 최선의 전문가별 정적 랭크를 이기는가? | **부분적** — 4개 중 2개만 통과, 그마저도 아래 §5에서 설명하듯 해석에 주의 필요 (Gate3 **FAIL**, 2/4) |
| **Gate4** 전체 모델 NLL 확인 | **미실행** — Gate3 종합 판정이 실패해 방법론 Section 19 규정에 따라 진행하지 않음 |

**한 줄 결론**: Qwen3-30B-A3B의 `down_proj` (2048×768, 128 experts/layer, top-8
라우팅)는 Granite MoE보다도 훨씬 강하게 "저랭크 친화적이지 않다" — naive
weight-SVD에서는 **4개 전문가 전원, 캡처된 모든 토큰이 예외 없이 오라클 랭크
768(=최대 랭크, 100%)** 을 요구했고, activation-aware 재시도 후에도 대부분
그대로였다. **Go/No-Go 판정은 No-Go**: 이 4개 전문가 샘플에서는 FERRET의
토큰별 저랭크 압축 가설이 지지되지 않는다.

---

## 2. 설정 및 스펙 대비 편차

### 2.1 모델·정밀도

- **`Qwen/Qwen3-30B-A3B`**
  - 전체 파라미터 약 30B, 활성 파라미터 약 3B
  - 48개 레이어, `num_experts=128`, `num_experts_per_tok=8` (top-8, Granite와
    동일하게 파인그레인drained/고밀도 라우팅)
  - `hidden_size=2048`, `moe_intermediate_size=768` (SwiGLU 게이트 전문가,
    실행 시점엔 `gate_up_proj`+`down_proj` 3D 배치 텐서 — 단, 체크포인트
    자체는 전문가별 2D 텐서로 저장되어 있어 `model.layers.{L}.mlp.experts.{E}
    .down_proj.weight` 키로 직접 읽을 수 있었다)
  - bf16 체크포인트 약 57GB (다운로드/디스크 사용량)

- **정밀도 전략**: 60GB급 모델이라 Granite처럼 전체를 bf16로 올릴 수 없어
  두 단계로 분리했다 — Stage A/C(라우팅 카운트, 트레이스 수집)는 INT8
  백본(2-GPU `balanced` + `max_memory` 재배분 + `llm_int8_enable_fp32_cpu_offload`),
  Stage B(전문가 가중치 추출/SVD)는 safetensors 샤드에서 원본 BF16
  `down_proj` 텐서를 직접 읽어 전체 모델 로드 없이 처리했다.

### 2.2 GPU 배분

원래 config는 물리 GPU 2,3을 지정했으나 실행 시점에 물리 GPU 3이 다른
사용자 프로세스(96% 사용률)로 점유돼 있어 물리 GPU 0,1로 변경했다
(`configs/device_map.json`, `configs/experiment_config.yaml` 갱신). Phase 0
스모크 테스트에서 `max_memory`를 `{0:22GiB,1:22GiB}` → `{0:23GiB,1:21GiB}`로
재배분해 budget(22.5GB) 초과 문제(cuda:1 22.53GB)를 해결했다 — 두 GPU 간
가중치 배치가 균등하지 않아 생긴 문제였다.

### 2.3 데이터

- Wikitext-2-raw-v1 (`Salesforce/wikitext`)
- `split_by_document=true` — Granite 실험(전체 코퍼스를 이어붙여 seq_len으로
  균등 절단)과 달리, 빈 줄 경계로 문서를 나누고 각 문서에서 하나의 seq_len
  청크만 취해 캘리브레이션/평가 세트가 절대 같은 문서를 공유하지 않도록 함
- 캘리브레이션 16 시퀀스 + 평가 16 시퀀스, 시퀀스 길이 256, 배치 크기 1,
  `use_cache=False`

### 2.4 레이어·전문가 선정 (Phase 1)

방법론과 달리 이 config는 후보 레이어를 미리 4개(중간 2, 후반 2)로 좁혀뒀다
(`layer_selection.candidates`: middle=[22,26], late=[38,42], minimal run은
이 4개만 사용). 각 포지션에서 상위-2 전문가 합산 카운트가 더 큰 후보를 채택:

| 포지션 | 후보 레이어(합산 카운트) | 채택 | 전문가(캘리브레이션 카운트) |
|---|---|---|---|
| middle | 22(4831) vs 26(4768) | **layer 22** | expert 18 (2858), expert 30 (1973) |
| late | 38(6206) vs 42(3493) | **layer 38** | expert 55 (3793), expert 3 (2413) |

4개 전문가 모두 최소 기준(500) 대비 4~8배. 평가 세트에서도 671~3535 토큰
확보 (최소치인 layer22_expert30만 권장 범위 1,000에 못 미침).

### 2.5 개입 대상

Toy0-A 규정대로 `down_proj`만 저랭크로 근사 (`z_t` = SwiGLU 활성화 이후
입력, `y_t = W_down z_t`, 편향 없음 — `Qwen3MoeExperts.forward` 소스와 정확히
일치).

---

## 3. 구현 중 발견한 버그: accelerate 오프로드 훅 우회

Phase 1 첫 실행 때 `get_moe_layers`가 반환한 MoE 블록 전체(`layer.mlp`)에
직접 훅을 걸어 `TypeError`가 발생했다 — Granite 스크립트의 `.experts` 접근을
누락한 실수로, `block.experts`를 명시해 해결했다.

더 미묘한 두 번째 버그는 Phase 2(트레이스 수집)에서 나타났다. layer38의
`.experts` 모듈은 `max_memory` 제약으로 인해 accelerate의
`AlignDevicesHook(offload=True)`가 걸려 있어, 파라미터가 평소엔 meta 텐서로
비어 있다가 `pre_forward`/`post_forward` 훅이 실행되는 동안만 실제 데이터로
채워진다. `experts_module.forward`를 직접 덮어써 이 훅을 우회하면, 우리
캡처 코드는 **초기화되지 않은 meta 텐서**를 읽게 된다 — 실제로 layer38
두 전문가의 `z_t`/`y_t`에 NaN/Inf가 섞여 나왔다 (`z` NaN 최대 25,321개,
`y` 최대 절댓값 36,608 — 이후 최댓값 8.1로 정상화됨). `moe_hooks.py`에
`_hf_hook` 존재 여부를 확인해 있으면 `pre_forward`/`post_forward`를 우리가
직접 구동하도록 수정한 뒤 재수집해 해결했다 (`install_capture_hook` 참고).

---

## 4. Phase 3~4: SVD 분해와 랭크 스윕

### 4.1 1차 시도 — 가중치 전용(naive) SVD → Gate1 즉시, 전면 실패

`down_proj` (2048×768)를 그대로 SVD한 뒤 그리드
`{64,96,128,160,192,256,320,384,448,512,544,576,640,704,768}`로 스윕한 결과,
**eps=0.05 기준 오라클 랭크가 4개 전문가·캡처된 9,247개 토큰 전부 예외 없이
768(최대 랭크, 100%)** 이었다. Granite("100% 512에 집중")보다도 더 극단적인
평탄 스펙트럼이다.

### 4.2 재시도 — activation-aware(whitened) SVD (Section 19 규정)

캘리브레이션 `z_t` 분포로 가중된 SVD (`W`가 아니라 `W @ L`을 분해,
`L L^T = E[z z^T]`)를 구현하고, 그리드를
`{16,32,48,64,96,128,160,192,224,256,320,384,448,512,544,576,640,704,768}`로
세분화했다. 결과는 소폭 개선됐지만 구조적 결론은 그대로였다.

| 랭크 r (비율) | 중앙값 상대오차 (4개 전문가 범위) |
|---|---|
| 64 (8.3%) | 60~85% |
| 256 (33%) | 37~58% |
| 384 (50%) | 27~45% |
| 512 (67%) | 18~32% |
| 704 (92%) | 6~13% |
| 768 (100%) | ~0.17% |

**Gate1 판정 (50% 랭크=384에서 중앙값<5%, p90<10%)**: 4개 전문가 **전부
실패** — 중앙값 26.5~44.9%, p90 32.3~52.5% (목표의 5~10배).
→ `results/gate1_lowrank_feasibility.csv`

---

## 5. Gate2: 전문가 내부 이질성 — **FAIL (0/4)**

| 태그 | 지배 랭크 | 지배 비중 | 점유 구간 수 | 10%↑ 구간 수 | 평균 오라클 랭크 |
|---|---|---|---|---|---|
| layer22_expert18 | 768 | **100.0%** | 1 | 1 | 768.0 (100%) |
| layer22_expert30 | 768 | **100.0%** | 1 | 1 | 768.0 (100%) |
| layer38_expert03 | 768 | 99.96% | 2 | 1 | 767.97 (99.997%) |
| layer38_expert55 | 768 | 82.3% | 4 | 2 | 754.6 (98.3%) |

기준(지배 비중<80%, 점유 구간≥3, 10%이상 구간≥2)을 4개 전문가 **모두
실패**. layer22의 두 전문가는 캡처된 모든 토큰이 문자 그대로 랭크 768 하나로
쏠려 이질성이 전무하다. layer38_expert55만 4개 구간에 걸쳐 약간의 분산을
보였지만 여전히 지배 비중 82.3%로 기준(80%)을 살짝 넘긴다. Granite에서
관찰됐던 "좁지만 실재하는 이질성"조차 이 모델에서는 거의 사라졌다.

[Figure 2 — 오라클 랭크 히스토그램](../ferret_toy0_qwen3/figures/oracle_rank_histograms.png)

---

## 6. Gate3: Oracle vs 최선의 전문가별 정적 — **FAIL (2/4), 해석 주의**

| 태그 | 평균오차(정적, r) | 평균오차(Oracle) | 상대 감소율 | 부트스트랩 95% CI | Gate3 |
|---|---|---|---|---|---|
| layer22_expert18 | 0.00172 (r=768) | 0.00172 | 0.0% | [0, 0] | FAIL |
| layer22_expert30 | 0.00173 (r=768) | 0.00173 | 0.0% | [0, 0] | FAIL |
| layer38_expert03 | 0.1283 (r=704) | 0.00174 | **98.6%** | [0.119, 0.133] | **PASS** |
| layer38_expert55 | 0.0614 (r=704) | 0.00933 | **84.8%** | [0.048, 0.056] | **PASS** |

**해석 주의**: layer22 두 전문가는 오라클 평균 랭크 자체가 정확히 768(최대치)
이라 "정적 랭크"도 그대로 768이 선택돼 정적=오라클, 비교 자체가 성립하지
않는다 (개선의 여지가 원천적으로 없음). layer38 두 전문가는 오라클 평균
랭크가 767.97 / 754.6으로 768에 매우 가깝지만 그리드상 768 미만이라, "오라클
평균 랭크 이하의 최대 그리드 값"으로 정적 랭크가 **704**로 선택된다. 즉
Gate3 통과는 "704와 768 사이의 벼랑 같은 오차 급락"(§4.2 표에서 704→768 구간
오차가 6~13%→0.17%로 급감)을 오라클이 정확히 붙잡아낸 것일 뿐, 방법론이
원래 의도한 "완만한 중간 랭크 대역에서의 토큰별 적응 이득"과는 성격이
다르다. 종합 판정 기준(4개 중 3개 이상)에도 못 미쳐 **Gate3 전체 FAIL**.

[Figure 3 — 품질-연산량 곡선](../ferret_toy0_qwen3/figures/quality_compute_curve.png)
[Figure 4 — 전문가별 Oracle 이득](../ferret_toy0_qwen3/figures/per_expert_gain.png)

---

## 7. Gate4: 미실행

방법론 Section 19 — "Gate 3이 실패하면 랭크 예측기를 학습하지 않는다",
"Gate 4는 로컬 결과가 긍정적일 때만 실행한다" — 규정에 따라, Gate3 종합
판정이 실패(2/4)한 이 시점에서 전체 모델 NLL 검증(Phase 6)은 진행하지
않았다.

---

## 8. 종합 Go/No-Go 판정 — **No-Go**

| Gate | 판정 | 근거 |
|---|---|---|
| Gate1 (저랭크 타당성) | **FAIL** (0/4) | activation-aware 재시도 후에도 50% 랭크에서 중앙값 오차 26.5~44.9% (목표 <5%) |
| Gate2 (전문가 내 이질성) | **FAIL** (0/4) | 지배 비중 82.3~100% (목표 <80%), 2개 전문가는 지배 비중 정확히 100% |
| Gate3 (Oracle 우위) | **FAIL** (2/4) | 통과한 2개도 "704→768 급락 구간"을 붙잡은 것일 뿐, layer22 두 전문가는 비교 자체가 무의미 |
| Gate4 (모델 수준 확인) | **미실행** | Gate3 종합 실패로 방법론 규정상 건너뜀 |

**이 4개 전문가 샘플에서 FERRET의 토큰별 저랭크 압축 가설은 지지되지
않는다.** [14_Toy0_실험결과.md](14_Toy0_실험결과.md)(Granite)에서는 최소한
Gate2/Gate3가 강하게 통과해 "좁지만 실재하는 이질성"이 확인됐던 반면, Qwen3-
30B-A3B에서는 그 좁은 이질성마저 대부분 사라지고 "거의 모든 토큰이 거의
항상 풀랭크를 요구"하는 훨씬 비관적인 양상으로 나타났다.

---

## 9. 해석: 왜 Granite보다도 더 저랭크 친화적이지 않은가

두 모델 모두 파인그레인drained(fine-grained) MoE(작은 `intermediate_size`,
높은 top-k)라는 공통점이 있지만, Qwen3-30B-A3B는 그 경향이 더 심하다:

- 전문가 수가 128개(Granite 32개)로 4배 많고, 전문가당 `moe_intermediate_size`
  (768)는 `hidden_size`(2048)의 37.5%로 Granite(512/1024=50%)보다 상대적으로
  더 작다 — 전문가 하나가 표현할 수 있는 함수의 폭이 이미 좁게 설계돼 있다.
- 30B급 모델은 사전학습 과정에서 각 전문가의 파라미터를 정보 이론적으로
  거의 포화 상태까지 활용하도록 압박받았을 가능성이 높다 — Granite(1.3B)
  보다 총 학습 토큰 수/파라미터 비율이 유리했을 것이므로, 잘라낼 수 있는
  잉여(redundancy)가 더 적다는 가설이 성립한다.
- INT8 백본으로 라우팅/트레이스를 수집했다는 방법론적 차이가 결과에 미친
  영향은 크지 않아 보인다 — `down_proj` 자체는 §3의 버그 수정 후 BF16
  연산으로 깨끗하게 캡처됐고, SVD는 원본 BF16 가중치를 safetensors에서
  직접 읽어 수행했으므로 양자화 노이즈가 저랭크성 자체를 왜곡했을 가능성은
  낮다.

이는 스펙의 **Risk 1**("전문가가 충분히 저랭크가 아니다 → 동적 실행의
이점이 거의 없다")이 Granite보다 더 강하게 현실화된 두 번째 사례다.

---

## 10. 권장 다음 단계

1. **"성긴" MoE로 재현이 더욱 시급해졌다** — 전문가당 `intermediate_size`가
   `hidden_size` 대비 훨씬 큰 모델(Mixtral류, top-1/2 라우팅)에서 동일
   파이프라인을 재실행하는 것이 이제 두 번 연속 실패 이후의 필수 다음
   단계다. 파인그레인drained 라우팅 계열(Granite, Qwen3-MoE, DeepSeekMoE
   계열)에서는 FERRET의 토큰별 저랭크 가설이 반복적으로 기각될 가능성을
   염두에 둬야 한다.
2. **다른 레이어/전문가 표본 확장** — 이번 4개 전문가(중간 1개 레이어 +
   후반 1개 레이어, 각 상위 2개)가 우연히 비전형적일 가능성을 배제하기
   위해, early 레이어(config의 `layer_selection.candidates.early=[8,12]`,
   recommended run)와 저빈도 전문가를 추가로 표본화해 같은 결론이
   재현되는지 확인한다.
3. **예측기 학습(Toy3~Toy5)은 착수하지 않는다** — Gate1·Gate2·Gate3가 모두
   종합 실패한 상태이므로, master 스펙의 "Oracle이 Static을 명확히 능가하기
   전까지 예측기 학습 단계로 진행하지 않는다"는 문구가 이번에는 더 명확하게
   적용된다.
4. **down_proj 단일 행렬 대신 Toy0-B(전체 전문가 함수 근사)를 시도할
   유인은 낮다** — 가장 관대한 activation-aware 조건에서도 실패했으므로,
   같은 표본에 대한 추가 근사 시도보다는 (1)의 아키텍처 축을 바꾸는 쪽이
   우선순위가 높다.

---

## 11. 산출물 위치

```
ferret_toy0_qwen3/
    configs/{experiment_config.yaml, device_map.json, environment.txt}
    routing_counts/{routing_counts.json, selection.json}
    traces/_sequences.pt, layerXX_expertYY_{z,y,z_calib,metadata}.pt
    factors/layerXX_expertYY_downproj_svd{,_actaware}.pt, manifest{,_actaware}.json
    results/
        phase0_smoke_test.json
        token_rank_errors{,_actaware}.parquet
        oracle_ranks{,_actaware}.parquet
        rank_grid{,_actaware}.json
        heterogeneity_stats.csv, gate1_lowrank_feasibility.csv,
        gate2_heterogeneity.csv, static_baselines.csv,
        quality_compute_curve.csv, gate_summary.json
    figures/
        rank_error_curve.png, oracle_rank_histograms.png,
        quality_compute_curve.png, per_expert_gain.png
    scripts/
        data_utils.py, moe_hooks.py, phase0_smoke_test.py,
        phase1_routing_count.py, phase2_trace_collection.py,
        phase2b_calib_trace.py, phase3_svd.py, phase3b_svd_actaware.py,
        phase4_rank_sweep.py, phase4b_rank_sweep_actaware.py,
        phase5_analysis.py
    logs/
        model_download.log, phase{0,1,2,2b,3,3b,4,4b,5}_run.log
```
