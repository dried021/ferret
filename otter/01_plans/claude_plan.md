# EACL 제출 계획 (v3) — 2026.07.27(월) 랩 피드백 반영판 (기준 v2 11:13 KST)

**마감**: D-0 = 2026.08.04(화), 제출은 마감 3시간 전 목표
**핵심 변경 (v1 → v2)**:

- Fisher-merge lossless 검증 완료 → 구현 버그 아님, 진짜 언어 효과로 확정
- **A100 불필요로 정정** (아래 "GPU/A100 참고" 섹션 참조) — 전 일정 3090 2장으로 진행
- Calibration 언어 6개 확정
- Phase1 2×2(경로 분리)가 원래 D-6 예정이었으나 D-8 저녁부터 선행 진행 중 → 일정 하루 앞당김

**핵심 변경 (v2 → v3, 2026-07-27 10:20 KST 랩 피드백 반영)**:

- **novelty 재정의**: calibration 언어가 압축 결과를 바꾼다는 효과 자체는
  dense quantization/pruning 문헌에 이미 보고돼 있음 → 본 논문의 기여를
  "효과의 존재"가 아니라 "D²-MoE 파이프라인 단계별(Fisher 병합/SVD
  whitening/structured pruning) 원인 규명"으로 재정의(06_논문_구성.md §1
  갱신, Intro/Related Work 집필 시 이 프레이밍을 따를 것).
- **Fisher-merge only 결과 재확인이 최우선**: delta가 보존되면 병합 기준과
  무관하게 원본과 이론상 동일해야 한다는 성질을 구현/평가 조건과 함께
  가장 먼저 재확인한다 — 오늘 완료한 Fisher-merge lossless 단위 테스트가
  이 요구의 첫 사례("구현 버그 아님" 확정). 같은 검증 원칙을 whitening/
  pruning 단계가 확정되는 대로 동일하게 적용할 것(각 단계도 "정보 보존
  시 원본과 동일"이 성립하는지 별도 확인 없이 결과를 확정 취급하지 않는다).
- **핵심 주장은 Fisher-merge 단독이 아니라 full pipeline + downstream
  task에서 재확인되어야 함**: §4.1/§4.3 그리드를 whitening과 pruning까지
  포함한 전체 파이프라인으로 완주하고, 경량 downstream(Belebele/XNLI)
  결과도 반드시 곁들여야 최종 결론으로 인정한다.
- **8월 싸이클 집중 항목(시간 제약)**: full D²-MoE(Fisher+whitening+
  pruning), calibration 언어 6개(+Balanced), placebo, 단계 분리, 그리고
  disagreement-aware calibration을 **Balanced와 expert-balanced 양쪽
  모두**에 대해 비교하는 데 집중한다. 이에 따라 expert-balanced 비교를
  Nice-to-have에서 Must-have로 승격(아래 컷라인 갱신 참고).

---

## GPU / A100 참고 (중요 — 별도 표기)

> **결론: 현재 계획 전체에서 A100이 반드시 필요한 항목 없음.**

이전 버전에서 "Fisher 계산은 weight 33GB + gradient 33GB + activation으로 48GB(3090×2) 초과 → A100 필수"로 판단했으나, 실제 구현(`phase1_fisher.py`)이 `freeze_all_but_layer`로 관심 레이어의 expert 파라미터만 `requires_grad=True`로 열고 나머지는 전부 얼리는 방식이라 gradient 저장 공간이 전체 모델이 아닌 레이어 1개 분량만 필요함. English/Korean/Swahili/placebo 전 조건이 이미 3090 2장에서 real gradient Fisher로 완료된 것이 실증.

**⚠️ 그럼에도 A100이 필요해질 수 있는 예외 상황 (발생 시에만 별도 확보)**:

- Full-rank(delta_ratio≈1.0 근방) SVD를 **전체 27레이어**에 대해 한 번에 돌리는 경우 (delta weight만 50.5GB로 OOM 확인됨 — 오늘 lossless 테스트에서 실제로 겪음). 레이어 단위로 쪼개서 하면 3090으로도 가능하므로, 이 경우도 사실 A100이 "필수"는 아니고 "쪼개기 번거로움을 피하고 싶을 때의 지름길" 정도.
- Qwen3-30B-A3B(61GB)를 **새로** 돌려야 하는 상황이 생기는 경우 (현재 계획엔 없음 — §4.1 layer-locality 그림은 기존 결과 재사용)
- 위 두 경우가 실제로 발생하면 그때 가서 세션 단위로 A100을 짧게 확보. 현재 일정상 발생 예정 없음.

---

## 확정된 실험 조건

**Calibration 언어 (6개 + Balanced)**


| 언어     | 역할                                                      |
| -------- | --------------------------------------------------------- |
| English  | 고자원 앵커, D²-MoE 원논문 default                       |
| Chinese  | 고자원, 비영어권 대조군 (base 모델 주력 언어)             |
| Korean   | 중자원, 교착어 — §4.2에서 이미 own-language gain 검증됨 |
| Swahili  | 저자원, 라틴 스크립트 — 이미 검증됨                      |
| Bengali  | 저자원, 비라틴(Indic) 스크립트 — 스크립트 축 보강        |
| Balanced | 다국어 혼합 — §6 비교군과 공유                          |

Bengali가 tokenizer fragmentation으로 bpb가 비정상적으로 튀면 Vietnamese/Indonesian으로 스왑 (D-8 스모크 테스트에서 즉시 판단).

**⚠️ "Balanced" 정의 확정 필요 (리뷰 발견, 2026-07-27):** 코드(`phase1_calib_data.py`)에는 `balanced`(EN/KO/ZH 3언어, 7/24 결과 이미 있음)와 `mixed_5lang`(EN/KO/ZH/Swahili/Bengali 5언어, **아직 한 번도 실행 안 됨** — 결과 디렉토리 없음)이 재현성 보존 목적으로 의도적으로 분리돼 있다. §4.1/§6에서 저자원 언어(Swahili/Bengali) 축을 포함하는 조건들과 비교할 "Balanced"는 `mixed_5lang`이어야 논리적으로 맞다 (3언어 balanced는 애초에 Swahili/Bengali를 본 적이 없어 비교가 unfair). D-7 그리드 시작 전에 **`mixed_5lang`을 명시적으로 채택하고 새로 돌린다** — 기존 `balanced` 결과 재사용 금지. (추가 GPU 시간 1회분을 D-7 예산에 반영할 것.)

**평가**: FLORES-200 devtest, bpb(압축 전 대비 상대 증가율), calibration pool은 `[EVAL_N_SENTENCES:]`로 eval 앞 60개와 완전 분리(leakage 없음, 확인 완료). Downstream 2개(Belebele, XNLI 등 경량).

**Calibration 예산**: 64 samples / seqlen 512 — 이미 "예산 정상화" 단계에서 확정, placebo로 noise floor 대비 검증 완료. 근사가 아니라 논문 최종 표에 쓸 수 있는 수준.

---

## D-8 · 오늘 (월 7/27) — ✅ 완료 / 진행 중

- [X]  Fisher-merge lossless 검증: 192개 (projection, expert) 쌍 unit test, 재구성 오차 평균 0.30%/최대 0.31%(bf16 반올림 수준) → **구현 버그 아님 확정**
- [X]  delta_ratio=0.8이 진짜 압축임을 확인 (no-op 아님, rank 1408→667)
- [X]  calibration/eval leakage 없음 확인
- [X]  Scale(whitening) 10개 조건 계산 완료
- [X]  Merge+eval 14개 완료 (2026-07-27 08:17 UTC, `phase1_2x2_verify_gate_result.json`) — Fisher×Scale 2×2, seed 1~3. `mean_own_scale_gain` 38.95%p > threshold(2×noise floor) 13.76%p → **VERDICT: SUPPORTED**
- [ ]  **남은 것**: 전체 27레이어 스케일의 ratio≈1.0 sanity check는 아직 안 함 (OOM으로 레이어 단위 unit test로 대체됨 — 이미 논문 신뢰도엔 충분하나, 시간 되면 D-6 부근에 레이어 일부만 골라 저비용으로 배선(plumbing) 재확인 권장, blocking 아님)

  - **확인 완료 (2026-07-27 리뷰):** 코드/로그 상 지금까지 실행된 건 `phase1_lossless_unit_test.py`의 `LAYER_IDX=5` 고정, `english_only`/`seed0` 단일 조건 1건뿐 (`phase1_lossless_unit_test_result.json`). "레이어 일부"(복수) 재확인은 아직 착수 전 — 위 체크박스 상태가 정확하며 D-6 항목으로 그대로 유효함.
- [X]  **Bengali 스모크 테스트 완료 (2026-07-27):** ① 토크나이저 fragmentation 체크(`phase1_bengali_fragmentation_check.py`, CPU-only) — ben_Beng 1.453 bytes/token, 이미 검증된 kor_Hang(1.433)과 유사한 수준으로 유독 낮지 않음. ② GPU 스모크 런(`run_phase1_bengali_smoke.py`, baseline 재평가 + `bengali_only` seed0 full run) — baseline bpb: ben_Beng 0.8857로 5개 언어 중 최저(English 0.9137보다도 낮음). `bengali_only` 계산 후 bpb 증가율: ben_Beng +17.68%로 English(+16.95%)와 비슷한 중간값, Swahili(+25.71%)보다 훨씬 낮음 — fragmentation으로 인한 bpb 이상 급등 징후 없음. **판단: 스왑 불필요, Bengali 유지.** (own-language gain 정식 계산은 `english_only`의 ben_Beng 재평가 + `bengali_only_b` + seed 2·3이 더 필요하며 이는 스모크 범위가 아니라 D-7 그리드 몫)
- [X]  문서화: `03_기술노트.md` 2.6절, `04_전체요약.md` 핵심 발견 8번 반영 완료

## D-7 (화 7/28) — 주 실험 그리드 발사 (계획대로, 3090만 사용)

- §4.1 full-pipeline 그리드: 6개 언어 + Balanced + placebo(En-b, Ko-b 최소 2개) × seed 1개 먼저 전 조건 완주. seed 2·3은 백그라운드 순차 추가
- (원래 A100 배정했던 "그리드 전체 Fisher 계산"도 3090에서 그대로 진행 — layer-freeze 트릭 적용)
- GPU 여유 시간에 downstream 평가도 같은 체크포인트에 걸기
- 낮: Intro/Related Work 초고. novelty 문구를 "단계별 원인 규명(localization) + disagreement-aware 처방"으로 정리

## D-6 (수 7/29) — 주 결과 확정 + 단계 분리 (2×2가 이미 하루 앞서가 있으므로 분석 위주로 전환)

- §4.1 결과 1차 분석: full pipeline에서도 own-language 효과가 placebo noise floor 넘는지 확인
- §4.3 2×2 (Fisher 언어 × whitening 언어) — D-8 저녁부터 선행 진행 중이라 이 시점엔 seed 3개까지 채운 정식 gate 결과 분석에 집중. pruning 축은 on/off ablation으로 축소해 추가
- §4.2 재분석 (검증된 새 데이터로, 반나절)
- 여유 있으면: lossless sanity check(전체 배선) 레이어 일부로 저비용 재확인

## D-5 (목 7/30) — 단계 분리 결과 + §5 최소 버전

- 2×2 결과 → "효과가 어느 단계에서 발생하는가" 확정 (핵심 figure)
- §5: ①Fisher 순위 언어 간 상관(이미 있음) + ②불일치 expert 정량화(§6 입력이라 필수). ③④는 시간 남으면
  - ②에서 expert별 실제 라우팅 토큰 수 히스토그램도 같이 확인 (64-sample 예산에서 인기 없는 expert의 Fisher 추정이 노이즈에 가까울 가능성 배제)
- 밤: §6 구현 시작 — proxy 스캔 → 불일치 expert에 토큰 예산 표적 배분 → 압축 1회

## D-4 (금 7/31) — §6 제안 방법

- §6 본 실험: {제안, Balanced, expert-balanced(EBSS류), 최선 단일 언어} 4조건. EBSS 재현 안 되면 간이 구현으로 대체·명시
- 낮: §4 결과 섹션 집필 + figure 병행

## D-3 (토 8/1) — §6 마무리 + 집필 총력

- §6 budget sweep 2~3 포인트
- §5, §6 집필 — 본문 8할 완성 목표
- 백그라운드: seed 2·3 error bar 마무리

## D-2 (일 8/2) — 전체 초고 완성

- 드래프트 조립, Abstract, Limitations(§7 일반화는 future work로, 본문에서 삭제)
- 표 전체 숫자 로그 대조 검산

## D-1 (월 8/3) — 퇴고와 내부 리뷰

- 오전 지도교수/랩원 전달 → 오후 반영
- ARR 체크리스트, 익명화, 재현성 문서, 그림 캡션

## D-0 (화 8/4) — 버퍼 (계획상 작업 없음, 반드시 비워둘 것)

---

## Must-have / Nice-to-have 컷라인 (변동 없음)

**Must**: 검증된 §4.1 full-pipeline 그리드, placebo, §4.3 단계 분리(Fisher×
whitening 2×2 + pruning on/off), §6 vs Balanced **및 vs expert-balanced**
(2026-07-27 승격 — 언어 균형/expert 균형 두 축 모두 이겨야 기여 인정)
**Nice**: seed 3개 error bar, §5 ③④, downstream 2개째
**Cut 이미 확정**: §7 일반화 전체 (future work로)

목요일 저녁까지 §4.3 안 끝나면 §5를 ①만으로, §6 조건 수를 3개로 축소 — 컷은 실험 범위에서, 일정에서 하지 않는다.
