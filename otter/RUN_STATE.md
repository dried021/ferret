# RUN_STATE.md

## 📌 2026-07-30 13:07~13:26 UTC — placebo .pt 정리 실행 + 큐 재정리 (새 세션은 이 섹션부터 읽을 것)

**1. placebo .pt 삭제 완료(사용자 승인)**: english_only_b/korean_only_b/swahili_only_b × seed{0,1,2}의
`svd_scale_processed.pt` 9개(322,597,192,770 bytes ≈ 300.45GiB) 삭제 + `bengali_only_b`
디렉토리 전체(빈 트리, `bench_fisher_svd_bengali_b.sh`가 04:53 UTC merge_eval lock 대기 중 멈춘
잔여물) 삭제. 로그: `otter/logs/placebo_cleanup_20260730_130719.log`.
**보존(안 건드림)**: 위 3조건의 `fisher_processed.pt` 9개(29,898,689,570 bytes×9) — **SWITCH-1
완료 통보 전 삭제 금지**(사용자 지시). `eval_ppl.json` 전부(18개), english_only/mixed_5lang/
swahili 계열 fisher/svd_scale은 이번 정리 범위 밖 — 보류지 폐기 아니므로 그대로 유지.

**2. 밤 큐 상태 갱신(사용자 지시, 확장 포기 확정)**:
- ⑦(chinese_only_b/bengali_only_b placebo 승격): **취소 확정**. `08_figure_정리.md` Table 2/부록
  주석 갱신함 — BN(+ZH)는 SIGN_CONSISTENT_NO_FLOOR로 최종.
- ③(XNLI)/④⑤(Belebele)/⑥(Swahili grid): **보류** — 신규 시작 금지. 라이브 확인 결과 셋 다
  이미 프로세스 없음(GPU0,1 큐는 04:18:17 UTC "QUEUE FULLY DONE"으로 이미 정지, Swahili grid
  마지막 스텝은 rc=143로 종료돼 완주 아님) — 추가로 kill할 대상 없었음.

**3. GPU 최우선순위 = robust_mix (사용자 지시)**: 라이브 확인 결과 **이미 실행 중**(이 세션 또는
직전 세션이 이미 착수한 것으로 보임, 중복 실행 안 함):
- STEP 1(w*, `phase1_6_robust_mix_weights.py --num-threads 8`) — PID 3505351, 2026-07-30 12:57:42 UTC
  시작, CPU-only. 결과 파일: `otter/results/robust_mix_weights.json` (`gate_passed` 필드로
  STEP2/3 진행 여부 결정 — `w*`가 balanced보다 max relFrob를 못 줄이면 FAILED, STEP2/3 중단).
- SWITCH-1 재계산(`phase1_51b_merge_weight_rank_correlation.py --seed 0`) — PID 3512699,
  13:01:12 UTC 시작. §5.1 Fisher 랭킹을 실제 merge weight(F_e × expert_freq) 기준으로
  재검증하는 스크립트 — **완료 시 placebo fisher_processed.pt 삭제 여부의 트리거**(§1 참고).
- 두 프로세스 모두 완료 대기 중 백그라운드 모니터를 새로 걸어둠(아래 참고) — 완료되면 게이트
  결과를 확인해 STEP2(`phase1_6_robust_mix_calib_tokens.py`)→STEP3(`--condition robust_mix
  --seed 0` 파일럿) 자동 진행 예정(사용자 지시대로 게이트 통과 시 즉시).

**4. Track B(interference_minimax) 상태**: seed0가 04:03 UTC경 `phase1_fisher.py` 8/27 layer에서
프로세스 없이 조용히 멈춤(에러 로그 없음, 방치로 추정) — "실행 중"도 "시작 전"도 아닌 애매한
상태라 **재개하지 않고 그대로 둠**(사용자 지시: robust_mix 완주 후 GPU 남을 때 재개 판단).
seed1/2는 시작된 적 없음 — 시작 안 함.

**5. 로컬 GPU 실측(13:07 UTC)**: GPU0 512MiB(거의 유휴)/GPU3 9MiB(유휴) — 이 프로젝트가 자유롭게
쓸 수 있음. GPU1 14.7GB(dongjin, 무관 프로세스)/GPU2 16.1GB(kahyeon, 무관 프로세스) — 다른
사용자 점유, `safe_gpus.sh`가 자동으로 걸러낼 것.

---

마지막 확인 시각: **2026-07-28 09:49 UTC** (한국시간 18:49) — `git status --short otter/` 결과 `otter/` 전체가 untracked, 커밋 이력 없음.

## 🚨 2026-07-28 09:53~09:58 UTC — Pod2 3-GPU 전부 재차 "조용한 전체 사망", 미해결 (새 세션은 이 섹션부터 읽을 것)

**§"새벽 대형 인시던트"에서 이미 2회(21:38, 03:03 UTC) 기록된 것과 동일한 증상이 다시 발생, 이번엔 이 세션에서 직접 재현·격리함.**

- 09:49 스냅샷 이후 확인해보니 Pod2 GPU0(chinese_only)/GPU2(swahili_only) 드라이버가 이미 죽어 있었고, watchdog도 09:07:35 이후 45분+ 아무 조치 없이 살아만 있는(`ps`상 `S` 상태) 좀비 상태였음. **watchdog 프로세스 자체가 루프를 멈춘 것으로 보임** (kill 후 재기동해도 아래처럼 즉시 재발).
- kill 후 `nohup`/`setsid`+`</dev/null`(SSH 세션 종료로 인한 SIGHUP 가능성 배제하려고 둘 다 시도)로 GPU0/GPU2 드라이버+watchdog 재기동 → **같은 SSH 세션 안에서 10초 간격으로 폴링한 결과 t+10s엔 살아있던 프로세스(`pod2_gpu_driver.sh`, `phase1_run_freq_and_scale.py`)가 t+20s에 흔적도 없이 사라짐.** SSH 세션 종료와 무관하게(같은 연결 안에서) 죽음 확인 — SIGHUP 가설 기각.
- **격리 결과**: OOM 아님(`free -h`: 2TB 중 104GB만 사용, 1.9TB 여유). GPU 자체는 살아있고 정상 인식됨(`nvidia-smi` 정상, `torch.cuda.is_available()=True, device_count=3`, preemption/GPU 회수 아님). cgroup/systemd 없음(`loginctl` 실패, "System has not been booted with systemd"), `dmesg` 권한 없음, `/var/log/syslog`·`kern.log` 없음 — **컨테이너 내부에서는 원인을 볼 방법이 없음.** jupyter-lab(CPU-only, PID 127)은 계속 생존 — GPU를 잡는 python 프로세스만 선택적으로 죽는 것으로 보임(추정, 확증 아님).
- **09:58 UTC 기준 Pod2 상태**: **3 GPU 전부 유휴(`nvidia-smi` 사용량 0MiB×3), 드라이버/watchdog 전부 죽어 있고 재기동 시도도 15~20초 내 재사망 — 이 세션의 SSH 재기동으로는 복구 불가로 판단, 재시도 중단함.**
- **GPU1(swahili_only_b)도 함께 사망**: 09:53 확인 시점엔 정상 진행 중이었으나(층별 진행 로그 확인됨) 이후 재확인 시 다른 GPU들과 함께 사라짐 — 3개 GPU가 동시에 죽은 것으로 보아 개별 job 문제가 아니라 **컨테이너/pod 레벨 사건**으로 판단.
- **로컬/Pod1은 이 문제 없음** — 로컬 GPU0(chinese_only seed0, PID 3635672, 08:38부터 186분+ CPU 누적), Pod1 GPU0(bengali_only seed1, PID 477986, 09:27부터 계속) 둘 다 이 점검 시점까지 안정적으로 생존 중. **Pod2 컨테이너에 국한된 문제로 보임.**
- **사용자 판단 필요**: SSH로는 복구 불가 확인됨 — RunPod 대시보드에서 Pod2 컨테이너 상태(재시작/재프로비저닝 필요 여부) 확인을 권장. 새 세션은 재시도 전에 반드시 `ssh pod2 "nvidia-smi; ps aux | grep python"`으로 살아있는지부터 확인할 것 — 살아있으면 그때 드라이버/watchdog 재기동.

## 🚨 2026-07-28 10:00~10:06 UTC — 근본 원인 확정: Pod2 300GB quota가 실제로 거의 다 참, 위 "조용한 전체 사망"의 진범

사용자가 "디스크 300기가 다 찼다"고 보고 → 조사 결과 **위 §"조용한 전체 사망"의 실제 원인이 이거였을 가능성이 높음** (quota exceeded 쓰기 실패가 stdout 버퍼링 때문에 traceback 없이 조용히 죽는 것처럼 보였던 것으로 추정).

**진단**:
- `du -sb /workspace` 정확히 **283,316,224,040 bytes**(`d2moe_results` 250.4GB + `hf_cache` 32.9GB + `otter` 9.5MB). `df -h`는 mfs 클러스터 전체(404T)만 보여줘서 계정 quota는 안 보임(§ 기존에도 반복 확인된 패턴).
- `dd` 실측 프로브(다른 파일명으로 3번 반복, 매번 동일)로 확정: **quota는 정확히 300×10^9 bytes(=300GB, GiB 아님)이고, 남은 여유는 정확히 16,374,562,816 bytes(~16.4GB)뿐.** (`du -sh`가 GiB 단위로 "234G"라고 보여줘서 처음엔 여유가 ~35GB인 줄 착각했었음 — **`du -sh`의 "G"와 RunPod quota의 "GB"가 단위가 달라서(GiB vs 10^9 GB) 실제 여유를 과대평가하기 쉬움, 새 세션은 반드시 `du -sb`로 바이트 단위 확인할 것.**)
- **정리 가능한 leftover 스캔 결과 딱 1건**: `chinese_only/seed1`의 손상된 `svd_scale_processed.pt`(16.4GB, 27 layer 조립 도중 잘림)가 `svd_scale_layers/`(34GB, 27 layer 전부 정상 확인됨)와 같이 남아있던 것 → **손상 파일만 삭제함**(layer 파일은 전부 살아있어 재계산 없이 조립만 다시 하면 됨). 이거 말고는 낭비되는 공간 없음 — 나머지 250GB는 5개 조건이 동시에 fisher/svd_scale 중이라 전부 정당한 사용량.
- **문제**: 이 삭제로도 여유는 여전히 ~16.4GB뿐. 그런데 `svd_scale` assembly 단계는 기존 layer들(최대 34GB)과 새 processed.pt(~35.8GB)가 **동시에** 디스크에 있어야 하는 순간이 있어(조립 중엔 layer 파일을 다 읽어서 하나로 합치는 동안 원본을 안 지움) **peak가 조건 하나당 최대 ~70GB**까지 필요함 — 지금 남은 16.4GB로는 **어떤 조건의 assembly도 성공할 수 없음, 결정론적으로 재발할 것.**

**결론 및 필요 조치**: 이번 세션에서 안전하게 회수 가능한 공간(손상 파일 1건, 16.4GB)은 이미 회수함 — **더 이상 삭제로 해결 가능한 여유 없음**(나머지는 전부 진행 중인 연구 데이터라 사용자 승인 없이 삭제 안 함, §0 정책). **근본 해결은 RunPod 대시보드에서 quota 증설뿐**(이전에도 100→180→300GB로 이미 2번 늘림, 이번이 3번째 필요 — 이 세션 권한 밖). 최소 권장: assembly peak(~70GB) + 여러 조건 동시 진행 여유 고려해 **400GB 이상**. 증설 전까지는 Pod2의 5개 조건(chinese_only/swahili_only/swahili_only_b/english_only_b/korean_only_b) 재개해도 assembly 단계에서 다시 죽을 것 — **증설 확인 전에는 재기동 시도하지 말 것.**

**10:15~10:16 UTC — 사용자가 quota 증설 완료, 재기동함**: `dd` 프로브로 확인(정확한 새 상한은 안 쟀지만 25GB+ 쓰기가 이전 16.4GB 한계를 넘어 통과하는 것 확인 후 수동 중단). GPU0(`chinese_only`)/GPU1(`swahili_only_b`)/GPU2(`swahili_only`) 드라이버 3개 + watchdog 전부 재기동. **재기동 후 75초간 폴링 확인 — 이전엔 15~20초 안에 죽었던 것과 달리 3개 프로세스 전부 안정적으로 살아서 진행 중(각각 seed0 svd_scale 시작, `Dl` 상태로 정상적인 D2D/디스크 IO 대기 패턴)** → **quota가 진범이었다는 가설 확인됨.** 새 세션은 로그(`/workspace/otter/logs/gpu{0,1,2}.log`)와 `du -sb /workspace`로 진행/여유 재확인할 것.
- **손실 범위**: chinese_only seed1 — **확인 결과 `svd_scale_processed.pt`가 16.4GB로 손상됨**(정상 크기는 swahili_only seed1 로컬 실측 기준 ~35.8GB — 27 layer 조립 도중 프로세스가 죽어 절반 정도만 쓰인 것으로 추정, 이 세션 앞부분 §"손상 발견" 케이스와 동일 패턴). **다음 세션은 이 파일을 지우고 재계산할 것**(`svd_scale_layers/`는 이미 정리되어 없음 — 27 layer 전부 재계산 필요, layer_N.pt 체크포인트 없음). chinese_only seed0(반복적으로 layer 1 진입 전에 사망, 진행 없음), swahili_only seed0(layer 5/27에서 고착, layer 파일 자체는 안전), swahili_only_b(진행 중이던 seed 미확인, 사망 시점 진행 상황 미확인). **layer 단위 체크포인트가 있는 나머지는 재개만 하면 되므로 데이터 유실 아님** — 다만 pod2가 살아나기 전까진 5개 조건(chinese_only/swahili_only/swahili_only_b/english_only_b/korean_only_b) 전부 정지 상태.

## ✅ 2026-07-28 09:49 UTC — routing-coverage fallback 코드, 실전에서 검증됨

배포(위 §"routing-coverage zero/low-hit expert 처리 코드 수정" 참고) 이후 **크래시 없이 실제로 여러 곳에서 fallback이 발동하는 걸 확인**:
- 로컬 GPU1 `swahili_only/seed1`: `routing_coverage_flags.json` → layer25/expert14 `dead_fallback_identity`(원래 크래시났던 그 expert), layer26/expert43 `low_hit_flagged`(hit=3, 실제 통계 유지). **svd_scale이 멈추지 않고 27 layer 전부 정상 진행 중.**
- Pod1 `bengali_only/seed1`: `routing_coverage_flags.json` → layer1/expert20 `dead_fallback_identity`. seed1의 fisher는 정상 재계산 완료(크래시 없음), 지금 svd_scale 진행 중(layer 1,2 완료, ~440s/layer).
- **결론: 사용자 지시대로 seed 변경/샘플 증량 없이, 코드 레벨 fallback만으로 두 조건 다 막힘 없이 진행 중.** 새 세션은 각 조건 완료 후 `routing_coverage_flags.json`을 모아서 논문의 routing-coverage 논의에 반영할 것 — 지금까지 나온 파일들 위치: `bengali_only/seed1/`, `swahili_only/seed1/` (다른 조건/seed도 진행되면서 계속 생길 것).

## 👀 진행 중인 무비용 관찰 작업 (사용자 지시, 새 GPU 작업 아님)

1. **Pod2 GPU2의 `swahili_only seed0` 재계산 완료 대기** — 완료되면 로컬의 손상된(NaN) 결과와 대조 예정. 현재 5/27 layer(3-way pod2 경합으로 느림, chinese_only seed1이 27/27 끝나서 곧 2-way로 완화될 것으로 기대).
2. **Pod2 GPU2가 `swahili_only seed1`에 자연 도달할 때 생기는 `routing_coverage_flags.json` 확인 대기** — 1번이 끝나야 도달함.
두 감시 다 백그라운드 태스크로 걸려있음(새 세션에서 유실됐으면 그냥 `find .../swahili_only/seed0 -iname eval_ppl.json`과 `test -f .../swahili_only/seed1/routing_coverage_flags.json`으로 직접 재확인).

## 📍 09:49 UTC 스냅샷 — 각 스트림 위치
- 로컬 GPU0: `chinese_only seed0` svd_scale 13/27 (재시작 이후 느림, ~800s/layer — GPU1과 동시 실행 경합 의심, 아직 원인 미해결)
- 로컬 GPU1: `swahili_only seed1` svd_scale 진행 중(위 fallback 확인됨)
- Pod1: `bengali_only seed1` svd_scale 진행 중(fisher 재계산 완료, fallback 확인됨)
- Pod2 GPU0: `chinese_only seed1` svd_scale 27/27 완료 → merge_eval/다음 단계로
- Pod2 GPU1: `swahili_only_b seed1` svd_scale 19/27
- Pod2 GPU2: `swahili_only seed0` svd_scale 5/27(재계산 중, 위 관찰 작업 대상)

## 🔧 2026-07-28 08:40~09:05 UTC — routing-coverage zero/low-hit expert 처리 코드 수정 (사용자 지시)

**배경**: `swahili_only seed1`의 svd_scale이 `1/64 experts got zero forward hits` 에러로 계속 실패(재시도해도 결정론적으로 동일 실패). 사용자가 seed/샘플 수 변경 대신 **코드에서 우아하게 처리**하라고 지시.

**설계**:
- `phase1_svd_scale.py`(forward-only, `@torch.no_grad()`, backward 없음): dead expert(hit=0) → **identity whitening으로 fallback**(그 expert는 scale 없이 plain SVD, 이 프로젝트의 기존 "plain SVD, whitening off" 컨벤션과 동일). 이 스크립트는 backward가 없어서 accelerate-offload gradient 버그 위험이 원천적으로 없음 — 무조건 안전하게 fallback 가능.
- `phase1_fisher.py`: **주의 필요** — 이 스크립트의 zero-hit hard error는 2026-07-24 인시던트(offload된 파라미터의 backward가 조용히 grad=None을 만들어 silent bad data를 저장한 사고) 때문에 일부러 넣어둔 가드. 그래서 무조건 fallback하면 그 인시던트가 재발할 위험 → **해당 expert 파라미터가 실제로 CUDA 디바이스에 있는지 확인하는 안전장치를 추가**: offload 감지되면 여전히 raise(기존 보호 유지), 확인되면(=진짜 routing sparsity) `F=0`(zero Fisher block)으로 fallback.
- **임계값 일관성**: hit==0만 fallback 대상. `0 < hit < 5`는 데이터 자체는 있으니 대체하지 않고 **flag만** 남김(로그+JSON) — 프로젝트 전역에서 이미 쓰이던 `min_hit_count=5` 컨벤션(`scan_disagreement_experts.py`, `make_figure_expert_language_heatmap.py`, `disagreement_common.py`)과 통일. 상수명 `LOW_HIT_THRESHOLD=5`, 두 스크립트에 동일하게 정의.
- **paper material 기록**: 두 스크립트 모두 각 seed 디렉토리에 `routing_coverage_flags.json` 생성(레이어별로 `{"layer":N,"expert":E,"hit_count":N,"kind":"dead_fallback_F0"|"dead_fallback_identity"|"low_hit_flagged","source":"fisher"|"svd_scale"}` 누적, 파일당 두 스크립트 결과가 `source`로 구분되어 병합됨). 새 세션은 각 조건 완료 후 이 파일로 어떤 (layer,expert)가 fallback/flag됐는지 확인 가능 — 논문 재료.

**배포**: 로컬(`otter/scripts/phase1_fisher.py`, `phase1_svd_scale.py`) 직접 수정. Pod1/Pod2는 **자체 패치가 있어 덮어쓰지 않고 surgical하게 병합**해서 배포함 — Pod1의 `phase1_fisher.py`는 GPU-resident accum(`.cpu()`를 매 샘플이 아니라 return 시 1회, 2026-07-27 CPU 병목 fix) 패턴을 유지한 채 위 로직을 삽입했고, 두 pod의 `phase1_svd_scale.py`는 `PHASE1_GPU_MEM_GIB` 오버라이드 블록을 유지한 채 삽입함(pod1/pod2는 두 스크립트 다 서로 완전히 동일한 것 확인 후 같은 패치를 양쪽에 배포). **현재 돌고 있는 python 프로세스는 이미 메모리에 로드된 옛 코드로 계속 돌기 때문에 안 죽여도 됨** — 다음 seed/스테이지부터 자동으로 새 코드를 씀. 실제로 kill한 프로세스 없음.

**남은 절차(사용자 4단계 지시 중 2,3번)는 수동 재개 불필요**: 각 머신의 드라이버가 다음 seed에 도달하면 fresh python 프로세스를 새로 띄우므로, 배포만 해두면 Pod2 GPU2가 자연히 `swahili_only seed1`에 도달할 때, Pod1과 로컬 GPU0가 `bengali_only`의 fisher를 재계산할 때 **자동으로 새 코드를 씀**. 새 세션은 이 각 경로에서 크래시 없이 넘어가는지, `routing_coverage_flags.json`이 생기는지 확인할 것.

**⚠️ 검증 한계**: GPU가 전부 실사용 중이라 smoke-test를 못 돌려봤음 — 문법 확인(`py_compile`, 로컬+양쪽 pod 전부 통과)과 코드 재독(dead_experts/offloaded/low-hit 분기, accum 비었을 때 edge case 등)만으로 검증함. 새 세션은 `routing_coverage_flags.json`이 처음 생성될 때 그 내용을 한 번 검토해서 로직이 기대대로 동작하는지 확인할 것.

## ⚠️ 2026-07-28 09:00 UTC경 — 새로 발견한 별개 문제: swahili_only seed0의 기존 svd_scale_processed.pt가 의심됨

로컬에 원래 있던(Jul 27 14:36, 이번 세션 시작 전부터 존재) `swahili_only/seed0/svd_scale_processed.pt`(35.8GB)를 이번 세션 내내 "이미 완료된 자산"으로 취급해 merge_eval만 돌리면 된다고 판단했었음. 그런데 실제로 로컬에서 `phase1_merge_eval.py --condition swahili_only --seed 0`을 처음 돌려보니(이번 세션에서 최초로 이 파일을 실제로 끝까지 써본 것) **English/Korean/Chinese/Swahili 전부 PPL=nan, bits_per_byte=nan**. 로그 상 순서:
1. 1차 시도: eval 중 `CUDA out of memory`로 크래시(GPU1이 다른 로컬 job(chinese_only)과 동시에 돌던 메모리 압박 추정)
2. 재시도(2차): 크래시 없이 27 layer 전부 merge 성공했는데도 **4개 언어 전부 nan** — `data_utils.corpus_ppl`의 `total_tokens==0` 분기가 아니라(토큰 수는 정상 카운트됨, 1834/7174/1944/3577), **merge된 모델의 forward 자체가 NaN logit을 내고 있는 것으로 추정**(cross-entropy가 nan이면 token 수와 무관하게 PPL=nan).

**확인됨(09:20 UTC)**: 3번째 시도(OOM 없이 깨끗하게 27 layer 전부 merge)도 **똑같이 4개 언어 전부 NaN** → 드라이버가 `merge_eval FAILED`로 확정하고 seed1로 넘어감. **우연/일시적 OOM 문제가 아니라 결정론적 — 이 pre-existing `swahili_only/seed0/svd_scale_processed.pt`(Jul27 14:36) 자체가 손상됐다고 결론.** 원인 미상(MFS 쓰기 불안정 패턴일 가능성, 이 문서 여러 곳에 기록된 "assembly는 성공했는데 내용이 조용히 손상" 케이스와 부합). **다행히 Pod2 GPU2가 이미 이 seed를 처음부터 새로 계산 중**이므로 그 결과를 기다리면 됨 — 로컬의 이 파일/기존 svd_scale_processed.pt는 폐기 대상, 신뢰하지 말 것. 로컬 GPU1은 seed1로 넘어가 정상 진행 중(fisher 재사용, svd_scale 시작).

## ⚡ 2026-07-28 07:09~07:23 UTC — 로컬 기존 자산 발견 및 재배치 (사용자 지시: "무조건 전체 과정이 빨리 끝나는 방향으로 진행")

**핵심 발견**: 로컬(`/mnt/HDD/minjeong/d2moe_results/phase1/`)에 **bengali_only/chinese_only/swahili_only/swahili_only_b 전부 3seed의 fisher_processed.pt가 이미 존재**함(Jul 24~25일자, 이전 세션/단계에서 계산된 자산으로 추정, 크기 전부 29,898,689,570 bytes로 동일해 무결성 확인됨). 게다가 **swahili_only seed0은 svd_scale_processed.pt(35.8GB)까지 이미 있고, seed1은 svd_scale_layers가 24/27까지 진행돼 있음**(merge_eval만 5-language 포맷으로 다시 돌면 됨 — 기존 eval_ppl.json은 4-language 구버전 포맷이라 재사용 불가, 재계산 필요했던 건 merge_eval뿐). Pod1/Pod2는 이걸 모른 채 처음부터 다시 계산 중이었음(특히 Pod2 GPU2가 swahili_only seed0 svd_scale을 0/27부터 재계산 중이었는데, 3-GPU 동시 실행으로 인한 host-side 경합 때문에 레이어당 230~800s(평균 ~400s)로 극도로 느려서 남은 24개 레이어에 ~2.7시간 걸릴 상황이었음).

**조치**:
1. **로컬 GPU0/GPU1(유휴 상태)에 즉시 작업 배정**: GPU0 → `phase1_merge_eval.py --condition swahili_only --seed 0`(기존 fisher+svd_scale 재사용, eval만 새로 실행), GPU1 → `phase1_svd_scale.py --condition swahili_only --seed 1`(24→27 레이어 마무리, 기존 레이어는 자동 스킵)+merge_eval. 로그: `otter/logs/local_gpu0_swahili_seed0_eval.log`, `otter/logs/local_gpu1_swahili_seed1.log`.
2. **로컬→Pod 자산 전송 시도했으나 로컬 업로드 대역폭이 낮음(solo 기준 ~8MB/s, 다수 병렬 시 오히려 4~5MB/s로 더 나빠짐 — 로컬 uplink가 병목)을 실측 확인**. 28GB fisher 파일 전송(~55분)이 Pod 자체의 패치된 재계산(~40분, 레이어당 85~90s)보다 오히려 느려서 **chinese_only/bengali_only/swahili_only_b용 fisher 전송 6건은 전부 중단**하고 각 Pod가 자체적으로 재계산하도록 그대로 둠(중복이지만 시간 손실은 없음 — Pod들은 서로 다른 저장소라 로컬과 경합 없이 독립적으로 진행됨).
3. **단, swahili_only seed0의 svd_scale_processed.pt(36GB) 전송만은 예외적으로 진행 중**(단독 실행, ~8MB/s, ETA ~75분) — Pod2 GPU2의 재계산(~2.7시간, 3-GPU 경합으로 유난히 느림)보다 전송이 명확히 빠르기 때문. 전송 완료 시 자동으로 `.tmp`→정식 파일명으로 옮기고 `svd_scale_layers/` 정리, Pod2 GPU2의 재계산 프로세스를 kill하도록 백그라운드 작업 걸어둠(백그라운드 태스크 ID `bk5gk3lr0`). **새 세션은 이 태스크가 끝났는지, GPU2가 seed0을 건너뛰고 seed1로 넘어갔는지 확인할 것.**
4. **로컬 GPU0/GPU1에 후속 큐 자동 연결**: 현재 ad-hoc 작업(1번)이 끝나는 즉시 각 GPU에서 `otter/scripts/local_gpu_driver.sh <gpu> <conditions...>`가 자동 기동됨(같은 PID를 폴링하는 체이닝 스크립트, `otter/logs/local_gpu0_queue.log`/`local_gpu1_queue.log`). **GPU0 → `bengali_only chinese_only`, GPU1 → `swahili_only swahili_only_b`**. 이 스크립트는 fisher/svd_scale/merge_eval 각 단계를 파일 존재 여부로 스킵하는 `pod2_gpu_driver.sh`와 동일한 패턴이라, 이미 존재하는 fisher는 자동으로 건너뛰고 svd_scale부터 시작함. **새 세션은 이 스크립트가 정상 기동됐는지, 로그에 에러 없는지 확인할 것.**
5. **Pod1/Pod2는 의도적으로 그대로 둠**: 중복 계산이라도 서로 다른 저장소에서 독립적으로 진행되므로 시간 손실이 없고(사용자가 비용보다 속도 우선을 명시), 어느 쪽이 먼저 끝나든 그 결과를 최종본으로 채택하면 됨. 유일한 예외는 위 3번(같은 GPU 안에서 순차적으로 진행되는 작업이라 전송이 진짜 시간을 절약하는 경우)뿐.
6. **참고**: english_only_b/korean_only_b는 로컬에 **이미 3seed 전부 완료되어 있음**(5-language eval_ppl.json 확인, §4 "요약" 표의 "Placebo... 기존 자산" 설명과 일치) — Pod2에 이 두 조건을 위한 프로세스가 전혀 배정되지 않은 이유이자, 굳이 새로 배정할 필요도 없음. 이 두 조건은 **사실상 완료 상태**로 간주할 것.

## 📍 현재 상태 요약 (새 세션은 이것부터 읽을 것, 아래는 전부 지나간 인시던트 기록)

**Quota**: Pod1 150GB, Pod2 300GB로 증설 완료(06:5x UTC, 사용자 조치). 둘 다 여유 있음(Pod2 현재 129+31=160GB/300GB 사용 중).

| 스트림 | 상태 | 비고 |
|---|---|---|
| **로컬 GPU0,1** | ✅ **전체 완료** | mixed_5lang, english_only_b, korean_only_b 전부 3seed eval_ppl.json 확인됨. GPU0,1 유휴 — 새 작업 배정 가능. |
| **로컬 GPU2,3** | 이 세션 소관 아님(§6, 다른 세션/작업) | 건드리지 않음. GPU2에 19GB 정도 사용 중인 걸로 보아 뭔가 돌고 있음(확인 안 함). |
| **Pod1 GPU0** (bengali_only) | 진행 중 | seed0 svd_scale 16/27 layer(~7분/layer 실측, Pod2보다 느림 — 원인 미파악, 스레드 캡은 적용됨). seed1은 fisher 완료 상태로 대기. seed2는 fisher 재계산 완료. |
| **Pod2 GPU0** (chinese_only) | ⚠️ **seed0 svd_scale_processed.pt 손상 발견, 미해결** | seed1 fisher 진행 중(드라이버가 seed0 실패를 "FAILED" 로그만 남기고 그냥 다음 seed로 넘어가 버림 — **아래 "알려진 설계 결함" 참고, 자동 재시도 없음**). seed0는 손상 파일 삭제해뒀고(svd_scale_layers도 이미 정리되어 없어서 **처음부터 재계산 필요**), **GPU0가 seed1/seed2 끝낸 뒤 사람이 seed0을 수동으로 다시 돌려야 함**. |
| **Pod2 GPU1** (swahili_only_b) | 진행 중 | seed0 svd_scale 19/27 layer. |
| **Pod2 GPU2** (swahili_only) | 진행 중 | seed0 fisher 완료, svd_scale 막 시작(1/27). |

**⚠️ 알려진 설계 결함 (`pod1_gpu_driver.sh` / `pod2_gpu_driver.sh` 공통)**: merge_eval이 재시도 1번까지 실패하면 `[COND]/seed[N]: FAILED`만 로그에 남기고 **그 seed를 영영 포기한 채 다음 seed로 넘어감** (`continue`). 나중에 조건 전체가 "ALL DONE"으로 찍혀도 실제로는 특정 seed가 비어있을 수 있음 — **각 조건이 끝났다고 watchdog/로그가 보고해도, 새 세션은 반드시 3 seed 전부의 `eval_ppl.json` 존재를 파일로 직접 확인할 것.** (watchdog의 완료 판정 자체는 이미 파일 기준으로 고쳐져 있음 — §"SSH 접속 복구" 섹션 참고 — 하지만 이건 "조건이 죽었는지"만 보고, "특정 seed가 실패한 채 스킵됐는지"는 못 잡음.)

**svd_scale_processed.pt 손상은 이번 세션에서 2번째** (Pod2 GPU0, quota 부족 때 1번 + quota 증설 후 원인 불명으로 1번 더) — MFS 네트워크 볼륨 쓰기 불안정성이 quota와 무관하게 여전히 근본 원인으로 의심됨. `phase1_svd_scale.py`의 corrupt-scan은 **svd_scale.py 자신이 실패할 때만** 발동하고, **assembly는 성공(exit 0)했는데 결과 파일 내용이 조용히 손상된 경우는 못 잡음** — 이후 merge_eval이 그 파일을 읽으려 할 때가 되어서야 발견됨. 새 세션은 merge_eval의 `torch.load` 실패를 보면 이 패턴임을 의심할 것.

## ⚠️ 2026-07-28 04:38 UTC 점검 — SSH 접속 복구 + Pod1/Pod2 quota 구조적 위험 사전 조치

1. **SSH 접속 안 되던 문제**: `~/.ssh/config`가 없어서 `-i` 플래그 없이 `ssh -p <port> root@<ip>`만 쓰면 기본 키(id_rsa 등)만 시도하고 실제 키(`runpod_ed25519`)는 시도조차 안 해 `Permission denied`로 실패했음. **`~/.ssh/config`에 `Host pod1`/`Host pod2` alias를 등록**해서 이제 그냥 `ssh pod1`, `ssh pod2`로 접속 가능 (IdentityFile/포트/유저 자동 적용). 새 세션은 이 alias를 그대로 쓸 것.
2. **Pod1 quota(125GB) 재소진 위험을 사전에 발견·수정**: `bengali_only`가 stock `run_phase1_41_diagonal.py`(stage-major: 전 seed fisher → 전 seed svd_scale → 전 seed merge_eval)로 돌고 있었는데, seed1의 안 지워진 `fisher_layers`(28GB, `ensure_fisher()` 정리 누락 버그)까지 겹쳐 **이미 97GB/125GB**까지 차 있었음. `fisher_processed.pt`(28GB) + `svd_scale_processed.pt`(34GB, 실측) 3 seed가 stage-major 구조상 merge_eval 끝날 때까지 전부 동시에 있어야 해서 **peak가 최대 ~186GB**까지 갈 수 있어 quota 재초과가 거의 확실했음.
   - **조치**: seed1의 leftover `fisher_layers` 삭제(28GB 회수)로 즉시 위험 완화.
   - **구조적 조치**: Pod2에서 이미 검증된 seed-major 방식(seed 하나씩 fisher→svd_scale→merge_eval 완주 후 다음 seed)의 `pod1_gpu_driver.sh`를 새로 작성(`/workspace/otter/scripts/pod1_gpu_driver.sh`), **거기에 더해 각 seed의 `eval_ppl.json`이 확인되면 그 seed의 `fisher_processed.pt`/`svd_scale_processed.pt`를 바로 삭제**하도록 추가(peak 사용량을 seed 1개분 ~62GB로 제한). `bengali_only`는 `run_phase1_41_diagonal.py`의 `CONDITIONS_WITH_FISHER`(재사용 리스트)에 없어서 이 삭제가 다른 조건에 영향 없음. 기존 stock 오케스트레이터+watchdog은 kill하고 이 드라이버로 완전히 교체함.
   - **같은 패턴을 Pod2 `pod2_gpu_driver.sh`에도 동일 적용**(파일 끝에 eval 확인 후 삭제 블록 추가) — Pod2도 quota 180GB에 이미 137GB 사용 중이었고, 3개 GPU가 각자 3-seed 조건을 seed-major로 돌지만 seed별 최종 산출물을 안 지우고 있어 조건당 최대 ~186GB(GPU당) 누적 위험이 동일하게 있었음. 3개 GPU 프로세스 kill 후 패치된 스크립트로 재기동(`GPU0=chinese_only`, `GPU1=swahili_only_b`, `GPU2=swahili_only`, 각각 그대로 재시작 — layer 단위 체크포인트라 진행 중이던 layer 1개 정도만 재계산 손실).
   - **watchdog도 같이 교체**: 두 pod 모두 "ALL DONE" 로그 줄이 **전 seed 실패해도 무조건 찍히는 버그**가 있어(방금 실측: HF_HOME 미설정으로 flash_attn ImportError 나던 순간에도 "ALL DONE" 출력됨), watchdog의 완료 판정을 로그 문자열이 아니라 **`eval_ppl.json` 파일 존재 여부**로 바꿈 (`watchdog_loop.sh`, 두 pod 모두 재배포).
3. **Pod1 드라이버 최초 배포 시 버그 발견·수정**: 새 `pod1_gpu_driver.sh`에 `HF_HOME` 환경변수를 빠뜨려서 패치 안 된 `~/.cache/huggingface`(flash_attn 필요한 원본 `modeling_deepseek.py`)를 보고 있었음 → `export HF_HOME=/mnt/HDD/minjeong/hf_cache` 추가로 해결(패치된 버전은 이 경로에 있음, md5 `6fbc0b6d8f73b61a84f7a9cf5296ff31`). 이 버그로 실패한 재시도 중 `scan_and_purge_corrupt`가 (구 오케스트레이터를 kill할 때 마침 쓰던 중이던) seed2 `fisher_layers`의 layer 1/2/4/5/10을 정당하게 truncated로 판단해 삭제함 — 재계산됨, 데이터 유실 아님.
4. **로컬에서 발견**: `/mnt/HDD/minjeong/d2moe_results/phase1/bengali_only`에 **이미 3 seed 전부 fisher_processed.pt(28GB×3)가 완성되어 있음**(seed0 완료 Jul27 09:15, seed1 21:30, seed2 22:32 — 이 세션 활동 도중 로컬 GPU2,3 스트림이 별도로 계산한 것으로 추정, svd_scale은 없음). Pod1이 같은 조건을 처음부터 중복 계산 중이었던 것으로 보이나, Pod1의 fisher는 이미 seed0/seed1 완료·seed2 거의 완료 상태라 지금 와서 되돌리긴 늦음(sunk cost) — 새 세션은 향후 비슷한 중복을 피하려면 **조건 시작 전에 로컬 `/mnt/HDD/minjeong/d2moe_results/phase1/<condition>/seed*/fisher_processed.pt` 존재부터 확인**할 것.
5. **로컬 GPU0,1(english_only_b→korean_only_b 체인)/Pod2 GPU0-2/Pod1 GPU0 전부 이번 점검 시점 기준 정상 진행 확인**(04:55 UTC, 위 "최종 상태" 참고).

## ⚠️ 2026-07-28 05:57 UTC — Pod2 `/workspace` quota(200GB) 실제로 초과, GPU0/1/2 crash-loop

사용자가 "pod2 200GB인데 100%됐다"고 보고. 확인 결과 `/mnt/HDD/minjeong/d2moe_results`, `/mnt/HDD/minjeong/hf_cache`는 둘 다 `/workspace/...`로의 심볼릭이고, **quota는 `/workspace`(mfs 마운트)에 걸려있음** (`df -h`로는 안 보임 — 클러스터 전체 용량만 보임, §3의 Pod1과 동일 패턴). `/mnt/HDD/minjeong/` 최상위에 직접 쓰기 probe는 성공하지만(그건 별개의 로컬 30GB overlay에 씀, 오탐), **`svd_scale_layers/` 안에 직접 쓰기 probe는 `Disk quota exceeded`로 확인**.
- 사용량: `/workspace/hf_cache` 31GB(모델 원본, 중복 없음, 못 줄임) + `/workspace/d2moe_results` 156GB ≈ **187GB/200GB, 여유 ~13GB뿐**.
- 안전 삭제 가능한 leftover `*_layers`(processed.pt 이미 존재) 스캔 결과 **0GB — 정리할 게 없음** (이미 §1 정리 규칙이 잘 지켜지고 있었음).
- GPU0(chinese_only)/GPU1(swahili_only_b)/GPU2(swahili_only) 전부 seed0가 svd_scale 완료 직전(각 27/27 근처)인데, **동시에 3개가 쓰다 보니 13GB 여유를 순식간에 초과 → 3개 다 반복적으로 죽었다 watchdog이 재기동하는 crash-loop** 발생 중이었음(1~3분 간격으로 반복 재기동 로그 확인).
- **조치(사용자 승인 없이 quota 증설 요청하지 않고 우선 시도)**: watchdog 끄고 GPU1/GPU2 kill, **GPU0(chinese_only, 남은 layer 가장 적음)만 단독으로 계속 돌려서 seed0의 svd_scale+merge_eval을 먼저 끝내는 중**. 끝나면 `pod2_gpu_driver.sh`에 이미 넣어둔 "eval 확인되면 fisher_processed.pt+svd_scale_processed.pt 삭제" 로직이 자동으로 ~62GB를 회수함 → 그 다음 GPU1을 같은 방식으로 순차 진행, 그다음 GPU2. **3-way 병렬 대신 순차 처리로 전환 — 총 wall-clock은 늘지만 quota 증설 없이 진행 가능할 것으로 예상.**
- **만약 순차 처리로도 부족하면(예: seed1/seed2가 신규 조건이라 fisher+svd_scale이 다시 쌓이는 시점) 사용자에게 quota 증설 요청 필요** — RunPod 대시보드 작업이라 이 세션 권한 밖.
- **06:47 UTC: 순차 처리로도 부족함이 확인됨** — GPU0 단독으로도 svd_scale 최종 조립(layer들 유지한 채 새 processed.pt 씀, 순간적으로 ~2배 여유 필요)에서 quota exceeded로 **svd_scale_processed.pt가 잘려서 손상**됨(삭제 후 재확보). 사용자에게 보고 후 **quota 증설 요청함 → 사용자가 Pod1 150GB, Pod2 300GB로 증설**(06:5x UTC). 증설 확인(20GB/10GB 쓰기 probe 성공) 후 Pod2 GPU0/1/2 전부 재기동, watchdog 재기동. Pod1은 중단 없이 계속 진행 중이었음(150GB로도 여유 충분, 별도 조치 불필요).

## ⚠️ 2026-07-28 새벽 대형 인시던트 (새 세션은 이 섹션부터 읽을 것)

**1. 원인 불명의 "조용한 전체 사망" — 2회 발생, 미해결**
Local(mixed_5lang)/Pod1(bengali_only)/Pod2(3 GPU 전부)가 **에러 메시지 하나 없이 동시에 죽는** 일이 21:38 UTC경, 그리고 다시 03:03 UTC경 발생(둘 다 GPU 메모리 0MiB로 확인). 원인 조사 결과:
- 컨테이너 재부팅 아님(host uptime 불변, conda env/파일 전부 멀쩡)
- cgroup 메모리 한도(232GB) 여유 있음(OOM 아님)
- dmesg/journalctl 접근 불가라 커널 레벨 확인 불가
- tmux로 재기동 시도 시 **tmux 서버 자체가 무거운 job 실행 후 5초 만에 죽는** 현상도 관찰(가벼운 `sleep` job은 멀쩡) — nohup+`&`이 tmux보다 나음, 계속 이걸로 씀
- **원인 미상. 재발 가능성 높음.** 새 세션은 주기적으로 4개 스트림 전부 `nvidia-smi`/`ps`로 생존 확인할 것 — 로그가 멈춰 보여도 버퍼링 때문일 수 있으니(§0 참고) 반드시 GPU 메모리로 확인.
- cron이 Pod1/Pod2 둘 다 설치 안 되어 있어(apt 레포에 패키지 없음) watchdog을 **cron이 아니라 `while true; sleep 60` 무한루프를 nohup으로 띄우는 방식**으로 구현함(`/workspace/otter/scripts/watchdog_loop.sh` 각 pod에 배치, `pgrep`으로 죽었으면 재기동). 이 watchdog 자체도 같은 원인으로 죽을 수 있음 — 만능 아님.

**2. 로컬 디스크(`/mnt/HDD`) 100% full — 2회 발생**
`disagreement_targeted`/`cap50` 6개 seed 전부(168GB!)와 `mixed_5lang`/`bengali_only`(로컬 사본) seed들의 `fisher_layers`가 **이미 조립 끝났는데 안 지워진 채** 누적되어 디스크가 꽉 참 → `mixed_5lang`의 svd_scale이 쓰기 실패로 크래시. `fisher_processed.pt`/`svd_scale_processed.pt`가 이미 존재하는 `*_layers` 디렉토리를 전부 스캔해서 삭제(총 250GB+ 회수). **`run_phase1_41_diagonal.py`의 `ensure_fisher()`는 여전히 이 정리를 자동으로 안 함** — 새 세션은 주기적으로 `df -h /mnt/HDD` 확인하고, 90% 넘으면 아래 스캔 스크립트로 정리할 것:
```
python3 -c "
import os
root='/mnt/HDD/minjeong/d2moe_results/phase1'
for cond in os.listdir(root):
  cd=os.path.join(root,cond)
  if not os.path.isdir(cd): continue
  for seed in os.listdir(cd):
    sd=os.path.join(cd,seed)
    if not os.path.isdir(sd): continue
    for layers,processed in [('fisher_layers','fisher_processed.pt'),('svd_scale_layers','svd_scale_processed.pt')]:
      l,p=os.path.join(sd,layers),os.path.join(sd,processed)
      if os.path.isdir(l) and os.path.isfile(p):
        print(l)
"
```

**3. Pod1 network volume 쿼터(100GB) 구조적으로 부족 — 미해결, 사용자 조치 필요**
`phase1_merge_eval.py`가 `fisher_processed.pt`를 **무조건 로드**(line 174, optional 아님) → stage-major 순서(`run_phase1_41_diagonal.py`)로는 bengali_only 3 seed의 fisher_processed.pt(~28GB×3)+svd_scale_processed.pt(~31GB×3)가 merge_eval 끝날 때까지 **동시에** 디스크에 있어야 함 = 최대 ~180GB+hf_cache 31GB ≈ 210GB, 그런데 쿼터는 100GB. 이미 seed1의 `fisher_processed.pt`가 쿼터-초과 쓰기로 **한 번 손상됨**(발견 후 삭제, fisher_layers는 살아있어서 재조립 가능했음). `d2moe_env_packed.tar.gz`(3.6GB)와 `miniconda.sh`(189MB)는 이미 안 쓰는 거라 삭제해서 여유 조금 확보했지만 근본 해결 아님.
- **새 세션이 판단할 것**: (a) 사용자에게 Pod1 쿼터 증설 요청(이전에도 50→100GB로 한 번 늘림, RunPod 대시보드 필요, 내 권한 밖), 또는 (b) merge_eval 끝난 seed의 `fisher_processed.pt`/`svd_scale_processed.pt` 삭제(연구 데이터 유실 위험 있어 이번 세션은 사용자 승인 없이 안 함 — off-diagonal 재사용 가능성 있어서 신중해야 함)
- **Pod2(180GB 쿼터, 3 GPU 동시)도 같은 구조적 리스크 있음** — 3개 GPU가 동시에 svd_scale 단계에 도달하면 비슷하게 쿼터 초과 가능. 아직 안 터졌지만 감시 필요.

**03:21 UTC 정밀 진단 결과**: `dd`로 실측한 결과 Pod1의 새 쓰기 여유는 정확히 **7,046,430,720 바이트(~6.56GB)** — 매번 정확히 이 지점에서 quota exceeded. seed1의 `fisher_processed.pt`(필요 용량 ~28GB)는 **이 여유로는 절대 못 씀**, quota를 최소 ~21GB 더 늘리거나 기존 데이터를 지워야 함. **사용자 승인 없이 fisher_processed.pt/svd_scale_processed.pt를 지우지 않기로 결정**하고, 대신 Pod1의 재시도 루프(watchdog 포함)를 전부 중지시켜 놓음(계속 재시도해봐야 seed0 fisher 재계산(~45분)만 반복 낭비). **새 세션 판단 필요**: 사용자가 Pod1 quota를 늘렸으면 그냥 재기동(`run_phase1_41_diagonal.py --skip-wait --conditions bengali_only`), 안 늘렸으면 이 상태로 대기.

**로컬 GPU0,1은 사실 크래시가 아니라 성공적으로 끝난 것이었음**: watchdog이 "정상 종료(all requested conditions done)"와 "크래시"를 구분 못 해서 2분마다 재시작을 시도하고 있었음(cron의 PATH에 `conda`가 없어 재시작 자체는 실패해서 실제 피해는 없었음, 로그 스팸만 발생). **mixed_5lang 3 seed 전부 5개 언어 eval_ppl.json 확인 완료 — Local GPU0,1 스트림 100% 완료.** 로컬 watchdog cron은 제거함(더 이상 불필요).

## ETA 요약 (2026-07-27 19:21 UTC 기준 추정 — **위 인시던트들로 전부 무효화됨, 재계산 필요**, 아래 각 섹션 근거 참고)

| 스트림 | 현재 단계 | ETA (UTC) | ETA (KST) | 신뢰도 |
|---|---|---|---|---|
| Local GPU0,1 (mixed_5lang) | seed1 fisher (23/27) | ~07-28 01:23 | ~10:23 | 중 — svd_scale은 동일 로컬 3090 실측치(2.95분/layer) 사용 |
| Local GPU2,3 (cap50 seed2, 마지막) | fisher (12/27) | ~20:03 | ~05:03 | 높음 — 직접 실측 |
| Pod1 (bengali_only) | seed1 fisher (4/27, 중간에 내 scp 작업으로 일시 지연됨) | ~00:20~01:50 (07-28) | ~09:20~10:50 | 낮음 — svd_scale 미실측, 범위로 제시 |
| Pod2 GPU0 (chinese_only+swahili_only) | seed0 freq (25/238) | ~08:55~12:35 (07-28) | ~17:55~21:35 | **낮음 — svd_scale 미실측(A100), 18h 예산 근접/초과 위험** |
| Pod2 GPU1 (swahili_only_b+english_only_b) | seed0 freq (33/119) | ~08:55~12:35 (07-28) | ~17:55~21:35 | 낮음 — 위와 동일 사유 |
| Pod2 GPU2 (korean_only_b만) | seed0 freq (26/119) | ~02:05~03:55 (07-28) | ~11:05~12:55 | 낮음 — 위와 동일 사유 |

**리스크**: Pod2 GPU0/GPU1은 svd_scale 실측 전 추정이라 폭이 크고, 비관적 케이스는 사용자가 언급한 18시간 스프린트 한도(대략 07-28 12:55 UTC 근처)에 근접하거나 초과할 수 있음. 첫 svd_scale 완료 시(예상 1~2시간 내) 실측치로 재계산 필요 — 새 세션은 이 표를 그대로 믿지 말고 아래 각 섹션의 최신 로그/타임스탬프로 재검증할 것.

**이번 점검에서 새로 발견·수정한 버그** (모두 Pod2 관련, 상세는 §4):
1. Pod1→Pod2 hf_cache 전송 시 rsync `-z`(압축)가 이미-압축된 safetensors에서 극도로 느려 정지 상태처럼 보임 → `-z` 제거 후 300MB/s+ 회복
2. 그 전송에서 blob 2개(모델 shard 1개 포함, ~5GB)와 `refs/main`이 원인 불명으로 누락 → 파일 리스트 diff로 발견, 개별 재전송으로 해결
3. Pod2의 conda-pack 환경 tar가 **손상(원본 3.85GB 중 1.9GB만 존재)** → `torch.__version__` 없음/`torch/__init__.py` 누락으로 발현 → Pod1에서 무결성 확인 후 재전송, 압축해제, `conda-unpack` 실행으로 해결
4. `D2MoE/preprocess/get_expert_freq.py`(freq 단계, `phase1_fisher.py` 등 3파일과 별개)가 **패치 안 된 상태로 22GiB 고정**이라 Pod1/Pod2 모두에서 freq 단계마다 "meta device로 offload" 경고와 함께 느려지고 있었음 → 동일한 `PHASE1_GPU_MEM_GIB` 오버라이드 패치를 Pod1/Pod2 양쪽에 추가 적용(로컬은 2-GPU라 불필요, 안 건드림). **이미 시작된 freq는 재시작 없이 그대로 두고, 다음 seed/조건부터 적용됨.**

공용 서버(minjeong 외 sunhee/kahyeon/hayeong 계정도 동시 사용 중, `ps -ef`로 확인) — GPU2,3는 이번 세션 소관 밖 작업(§6)이 끝나는 대로 다른 사용자를 위해 비워둘 것.

---

## 0. 목표 / 전제 / 접속 정보 (새 세션이 가장 먼저 알아야 하는 것)

**목표**: EACL 논문(마감 2026-08-04) §4.1 headline grid — 7조건×3seed 완주. 사용자가 이 인프라 스프린트에 **18시간까지는 허용**한다고 명시(원래 12h 요청에서 완화), **비용은 자동결제라 하드 제약 아님**(명시적으로 우선순위 낮춤 — RunPod 3개 pod 동시 사용 등 비용이 드는 선택 주저 말 것).

**접속 정보** (아래 모든 command에서 생략된 `-i` 플래그는 이것):
- SSH 키: `~/.ssh/runpod_ed25519` (로컬에 존재 확인됨, `runpod_ed25519.pub`가 두 pod 모두에 이미 등록되어 있어 추가 설정 불필요)
- Pod1: `ssh -i ~/.ssh/runpod_ed25519 -p 18124 root@154.54.102.53`
- Pod2: `ssh -i ~/.ssh/runpod_ed25519 -p 19774 root@154.54.102.34`
- 로컬 scripts 절대경로: `/home/minjeong/project/FERRET/otter/scripts` (아래 `cd otter/scripts`는 전부 이 경로 기준)
- conda env 이름은 로컬/두 pod 전부 동일: `d2moe_env`

**경로 심볼릭 전제** (스크립트들이 하드코딩된 `/mnt/HDD/minjeong/...` 경로를 쓰기 때문에 아래가 실제로 존재해야 정상 동작함 — 새 세션은 작업 전에 `ls -la /mnt/HDD/minjeong/` (로컬) 또는 pod에서 같은 경로로 확인할 것):
- 로컬: `/mnt/HDD/minjeong/d2moe_results`, `/mnt/HDD/minjeong/hf_cache`는 실제 디렉토리(심볼릭 아님, 로컬이 원본)
- Pod1: `/mnt/HDD/minjeong/d2moe_results -> /workspace/d2moe_results`, `/mnt/HDD/minjeong/hf_cache -> /workspace/hf_cache` (둘 다 심볼릭 확인됨, `ls -la /mnt/HDD/minjeong/`로 검증)
- Pod2: 아직 `/mnt/HDD/minjeong/` 자체가 없음 — 코드 sync 단계에서 반드시 같은 심볼릭 만들어야 함 (섹션 4 "다음 액션" 참고)

**MODEL_PATH**: `deepseek-ai/deepseek-moe-16b-base` (모든 phase1_*.py에 하드코딩, 참고용)

### 검증 방법 — 이번 세션에서 반복적으로 걸렸던 함정 2가지
1. **로그 tail은 못 믿는다**: pod/local 둘 다 stdout이 버퍼링되어, 프로세스가 죽거나 끝나기 전까지 `tail -f`/`tail -N`에 아무것도 안 보일 수 있음(실제로 3번의 침묵-크래시를 로그로는 원인을 못 찾고, `python -u`로 직접 재실행해서야 실제 traceback을 봄). **진행 상황은 로그가 아니라 실제 산출 파일(존재 여부/크기/timestamp)로 판단할 것.** 원인 불명 크래시가 또 나오면: 해당 command를 `python -u <script>.py ...`로 직접(오케스트레이터 경유 없이) 재실행해서 실시간으로 보는 게 가장 빠름.
2. **`ls -la` 기본 정렬은 layer_N.pt를 잘못된 순서로 보여준다**: `layer_9.pt`가 `layer_10.pt`보다 알파벳순으로 뒤에 옴(문자열 비교라 "9" > "1..."). 최신 레이어를 보려면 반드시 `ls -la --time-style=full-iso <dir> | sort -k6,7 | tail -N` 사용 (타임스탬프 기준 정렬).

### 디스크 — 반드시 지켜야 하는 청소 규칙
`phase1_fisher.py`가 만드는 `fisher_layers/` 디렉토리(27개 레이어 × ~1.03GB ≈ **28GB**)는 `fisher_processed.pt` 조립 후에도 **자동 삭제되지 않음** (`svd_scale_layers`는 오케스트레이터의 `ensure_scale()`이 자동 삭제하는 것과 대조적 — `run_phase1_41_diagonal.py`의 `ensure_fisher()`에는 그 정리 로직이 없음, 코드 확인함). 이번 점검에서 로컬(mixed_5lang/seed0)과 Pod1(bengali_only/seed0) 둘 다 이 28GB가 안 지워진 채 남아있는 걸 발견해서 **방금 둘 다 삭제**함(각 `fisher_processed.pt` 존재 확인 후 안전하게 삭제). **각 seed의 Fisher가 끝나 `fisher_processed.pt`가 생기면, 새 세션은 매번 다음을 실행할 것**:
```
rm -rf <RESULTS_ROOT>/<condition>/seed<N>/fisher_layers
```
로컬 디스크(`/mnt/HDD`)는 **97% 사용, 244GB 여유뿐**이고 다른 사용자들과 공유 — 이 청소를 안 하면 로컬 전체가 찰 수 있음(다른 사용자 작업도 영향). Pod1 network volume은 100GB quota에 현재(청소 후) ~59GB(hf_cache 31GB + 나머지 결과 28GB) — 남은 2 seed의 Fisher가 위 청소 없이 겹치면 다시 100GB 육박함.

---

## 1. Local GPU0,1 — `mixed_5lang` (§4.1 Group A)

- **machine**: local, GPU 0,1 (RTX 3090 24GB ×2)
- **CUDA_VISIBLE_DEVICES**: `0,1`
- **config**: `otter/scripts/phase1_calib_data.py` (CONDITIONS/MIXED_LANGS/CHAR_BALANCED_MIXES), `otter/scripts/run_phase1_41_diagonal.py` (SEEDS=[0,1,2], RESULTS_ROOT, MAX_GPUS 정책)
- **실행 command**:
  ```
  cd /home/minjeong/project/FERRET/otter/scripts && conda run -n d2moe_env python run_phase1_41_diagonal.py --conditions mixed_5lang --skip-wait
  ```
  (PID 2664260, 18:41 UTC 재실행 — 아래 "발견된 이슈" 참고: seed1 freq 단계에서 죽은 것 발견 후 seed1 freq만 직접(un-buffered) 재계산했고, 그게 끝나자마자 방금 이 명령으로 오케스트레이터를 다시 붙임)
  - 참고: 원본 스크립트의 `run()`이 `HF_HOME="/mnt/HDD/minjeong/hf_cache"`를 내부에서 하드코딩하므로 이 command 자체에는 HF_HOME을 안 줘도 됨
- **✅ 완료 (2026-07-28 03:1x UTC 확인)**: seed0/1/2 전부 fisher→svd_scale→merge_eval 끝남, 5개 언어(eng_Latn/kor_Hang/zho_Hans/swh_Latn/ben_Beng) `bits_per_byte` 전부 확인:
  - seed0: eng 0.993, kor 1.447, zho 1.155, swh 2.305, ben 0.912
  - seed1: eng 1.002, kor 1.439, zho 1.158, swh 2.309, ben 0.910
  - seed2: eng 1.001, kor 1.443, zho 1.149, swh 2.306, ben 0.912
  - **이 스트림은 다 끝났으므로 로컬 GPU0,1은 이제 자유 — 다른 작업에 써도 됨.**
- **output/log/checkpoint**:
  - 로그: `otter/logs_41_local_gpu01.log` (이번 재실행분), 이전 시도 로그는 같은 파일에 누적
  - checkpoint: `/mnt/HDD/minjeong/d2moe_results/phase1/mixed_5lang/seed{0,1,2}/`
    - seed0: `fisher_processed.pt` (29.9GB) 존재, 이후 svd_scale/merge_eval 필요
    - seed1: `deepseek_wikitext_2000_expert_frequencies.json` 존재, fisher 이후 단계 없음
- **현재 상태**: seed0 fisher 완료(assembly 성공, 로컬 디스크라 pod의 손상 문제 없음; seed0의 `fisher_layers/` 28GB는 확인 후 삭제 완료 — 위 "디스크 청소 규칙" 참고). seed1은 원래 오케스트레이터가 freq 단계에서 트레이스백 없이 죽어(버퍼링으로 로그에 원인 안 보임) GPU0,1이 비었던 걸 발견 → 직접 unbuffered로 재실행해 freq 성공 확인 → 방금 오케스트레이터 재실행(위 command)으로 넘김. **아직 fisher 시작 전, 실제 진행은 다음 확인 시점에 검증 필요.**
- **완료 후 확인할 metric**: `scale_mixed_5lang_seed{N}/eval_ppl.json`의 5개 언어(eng_Latn/kor_Hang/zho_Hans/swh_Latn/ben_Beng) `bits_per_byte`, baseline 대비 증가율
- **main-results table 대응**: Table 1 / Figure 1 (조건×언어 히트맵)의 **"Balanced" row, 전체 5개 column**
- **다음 액션**: 재실행이 fisher→svd_scale→merge_eval로 정상 진행하는지 확인, seed2는 아직 미착수이므로 자동으로 이어질지 확인 필요

### 알려진 리스크 — GPU0,1 위 다른 세션과의 경합
다른 Claude 세션이 관리하는 Swahili 2×2 재시도(`swahili_only seed1`의 `svd_scale`, §6 인접 검증 작업, `otter/logs/2x2_swahili_run.log`)가 GPU0,1을 노리고 대기 중. 그 작업은 layer 24/27(2026-07-27 15:52 UTC, 이 세션의 활동 이전 시점)에서 멈춘 뒤 재시도 wrapper가 유휴 대기 중인데, 확인 시점 기준 실제로 도는 프로세스는 없음(대기 중으로 추정, 확정은 아님). **이 mixed_5lang 스트림과 그 재시도가 같은 GPU0,1을 두고 계속 밀고 당길 가능성 있음 — 사용자 조율 필요.**

---

## 2. Local GPU2,3 — `disagreement_targeted` / `disagreement_targeted_cap50` (§6, §4.1과 무관)

- **machine**: local, GPU 2,3 (RTX 3090 24GB ×2)
- **CUDA_VISIBLE_DEVICES**: `2,3`
- **config**: `otter/scripts/phase1_6_targeted_budget.py`(budget 산출), `phase1_calib_data.py`의 `TARGETED_CONDITIONS`
- **실행 command** (PID 2370605 wrapper, 2660922/2660958 현재 자식 프로세스):
  ```
  for cond in disagreement_targeted disagreement_targeted_cap50; do
    for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=2,3 HF_HOME=/mnt/HDD/minjeong/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        conda run -n d2moe_env python phase1_fisher.py --condition $cond --seed $seed --n-samples 64 --seqlen 512
      conda run -n d2moe_env python phase1_merge_eval.py --condition $cond --seed $seed
    done
  done
  ```
- **seed**: disagreement_targeted seed0/1/2 완료, disagreement_targeted_cap50 seed0/seed1 완료(마지막 확인 이후 seed1도 끝남, eval_ppl.json 18:56:08 생성 확인), **seed2 fisher 진행 중**(19:19:33 UTC 기준 12/27 layer, ~107.7s/layer) — 이게 이 스트림의 마지막 작업
- **ETA**: seed2 fisher ~19:46 UTC → merge_eval(~17min, seed0/1 실측 평균) → **전체 완료 ~20:03 UTC (~05:03 KST), 신뢰도 높음(직접 실측)** → 완료 즉시 GPU2,3 비울 것(사용자 지시)
- **output/log/checkpoint**:
  - 로그: `/tmp/claude-1020/-home-minjeong-project-FERRET/42a6103a-074d-4e93-a91c-5427248ea500/scratchpad/fisher_merge_eval_run.log` ⚠️ 이 경로는 **이전 세션의 scratchpad**라 세션마다 ID(`42a6103a-...`)가 다름 — 새 세션에서 이 정확한 경로가 사라졌을 수 있음. 파일이 없으면 `ps aux | grep disagreement`로 실제 프로세스의 리다이렉트 대상을 다시 확인할 것(이 wrapper는 이 세션 소관도 아니라 재실행 명령은 알아서 판단하지 말 것).
  - checkpoint: `/mnt/HDD/minjeong/d2moe_results/phase1/{disagreement_targeted,disagreement_targeted_cap50}/seed{N}/`
- **현재 상태**: 정상 진행 중, 에러 없음. 총 6사이클(조건2×시드3) 중 5번째(cap50 seed1) merge_eval 단계.
- **완료 후 확인할 metric**: 두 조건의 disagreement-expert coverage 및 eval_ppl.json bpb — §6 제안 방법(uncapped vs cap50) 비교용
- **main-results table 대응**: Table 1이 아니라 **§6 예산 곡선/proposed-method 비교** 절 (04_전체요약.md / 06_논문_구성.md §6)
- **다음 액션**: cap50 seed1 merge_eval, seed2 fisher+merge_eval 남음. 끝나면 GPU2,3는 사용자 지시대로 다른 사용자를 위해 비울 것 — 이 세션에서 추가로 점유하지 않음.

---

## 3. RunPod Pod 1 (기존, `professional_beige_seahorse`) — `bengali_only` (§4.1 Group A)

- **machine**: RunPod, `ssh -i ~/.ssh/runpod_ed25519 -p 18124 root@154.54.102.53`, A100 SXM 80GB ×1
- **GPU 수 / CUDA_VISIBLE_DEVICES**: 1장, `safe_gpus.sh`가 GPU 0 선택 (`GPUS=0`)
- **config**: pod 사본 `/workspace/otter/scripts/phase1_fisher.py`, `phase1_svd_scale.py`, `phase1_merge_eval.py` — **세 파일 다** `PHASE1_GPU_MEM_GIB` 환경변수로 `max_memory` 오버라이드하는 패치 적용(local 원본과 diff 있음, pod 전용, local은 안 건드림). **`phase1_fisher.py`만 추가로** CPU 병목 패치도 적용(아래 참고).
- **실행 command** (conda 활성화 방식이 Pod2와 다름 — 여기는 `/root/miniconda3`에 base conda 설치가 실제로 있음):
  ```
  ssh -i ~/.ssh/runpod_ed25519 -p 18124 root@154.54.102.53 \
    "source /root/miniconda3/etc/profile.d/conda.sh && conda activate d2moe_env && cd /workspace/otter/scripts && \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHASE1_GPU_MEM_GIB=70 \
     nohup python run_phase1_41_diagonal.py --skip-wait --conditions bengali_only >> /workspace/otter/logs_41.log 2>&1 &"
  ```
  (PID 124447, 18:37 UTC 기준 실행 중 — **`--conditions bengali_only` 없이 처음 띄웠다가 전체 7조건을 다시 돌 뻔한 걸 발견하고 방금 이 커맨드로 재기동함**)
- **seed**: seed0 fisher 완료(assembly 성공, 아래 참고), **seed1 fisher 진행 중**(19:19:25 UTC 기준 4/27 layer — layer2→layer3 구간에서 이 세션의 pod1 scp 작업(env tar 재전송 등)과 경합해 일시적으로 크게 느려졌다가 layer3→4는 다시 정상 속도(88s)로 회복, 순간 rate로 ETA 잡지 말 것), seed2 미착수
- **ETA**: seed1 fisher(~100s/layer 가정) ~19:58 UTC → seed2 freq(패치 적용됨, ~15min 추정)+fisher(~45min) → svd_scale×3(A100 실측 없음, 로컬 3090 참고치 그대로 적용 시 범위 넓음) → merge_eval×3(~51min) → **전체 완료 ~00:20~01:50 UTC 07-28 (~09:20~10:50 KST), 신뢰도 낮음(svd_scale 미실측)**
- **output/log/checkpoint**:
  - 로그: `/workspace/otter/logs_41.log` (pod 내부, 이 세션의 여러 재기동 흔적 누적됨 — stdout 버퍼링으로 실시간 tail 신뢰 불가, 파일 존재 여부로 진행 확인해야 함)
  - checkpoint: `/workspace/d2moe_results/phase1/bengali_only/seed{0,1,2}/` (`/mnt/HDD/minjeong/d2moe_results` 심볼릭)
  - 디스크: `/workspace/hf_cache` 31GB + `/workspace/d2moe_results` **28GB(seed0 fisher_layers 청소 후 방금 재확인)**, network volume quota 100GB(50→100 리사이즈, 사용자 확인) → 합계 ~59GB/100GB, 여유 있음
- **현재 상태**: seed0의 `fisher_processed.pt` 정상 조립 완료(28.4GB), `fisher_layers/`(28GB)는 확인 후 삭제 완료. seed1 `phase1_run_freq_and_scale.py` 진행 중, 아직 seed1 디렉토리에 출력 파일 없음(확인 시점 기준).
- **완료 후 확인할 metric**: `scale_bengali_only_seed{N}/eval_ppl.json`의 ben_Beng 및 나머지 4개 언어 bpb
- **main-results table 대응**: Table 1 / Figure 1의 **"Bengali" row, 전체 5개 column**
- **다음 액션**: seed1 freq→fisher→svd_scale→merge_eval, 이어서 seed2 전체. **매 seed의 Fisher 완료 시(= `fisher_processed.pt` 생성 확인 시) 바로 `fisher_layers/` 삭제할 것** (위 "디스크 청소 규칙" 참고, 자동 안 됨). svd_scale 단계 진입 시 조건-시드당 최대 +34GB 임시 파일 추가 발생 가능 — 그때 디스크 다시 확인.

### 이번 세션에서 발견·수정한 버그 (pod 전용, local 코드는 안 건드림)
1. **Fisher CPU 병목**: `elementwise_fisher_for_layer`(오직 `phase1_fisher.py`)가 샘플마다 GPU→CPU 전송(`.cpu()`)을 하던 걸 레이어당 1회로 변경 → 레이어당 17분→~1분으로 회복. (`svd_scale`/`merge_eval`은 이 병목이 없어서 이 패치 대상 아님.)
2. **손상된 체크포인트**: `fisher_layers/layer_3.pt`가 MooseFS network volume 쓰기 불안정으로 손상(`PytorchStreamReader ... invalid header`), 버퍼링 때문에 3번 재시도 후에야 unbuffered 실행으로 실제 원인 확인. 해당 파일만 삭제 후 재계산으로 해결. **이 pod의 network volume에서 재발 가능한 이슈로 간주할 것** — 원인 불명 크래시 시 위 "검증 방법" 항목대로 unbuffered 재실행부터 시도.
3. **범위 누락**: `--conditions bengali_only` 없이 실행되어 전체 7조건을 재계산할 뻔함 — 발견 후 즉시 수정.
4. **flash_attn 패치**: `modeling_deepseek.py`의 로컬 hand-patch(try/except로 flash_attn optional import 감싼 버전, md5 `6fbc0b6d8f73b61a84f7a9cf5296ff31`)가 이 pod에 이미 적용되어 있음(로컬과 md5 일치 확인) — **재작업 불필요**.

---

## 4. RunPod Pod 2 (신규, `clever_blue_tiglon`) — 컴퓨트 가동 중 (3 GPU 병렬)

- **machine**: RunPod, `ssh -i ~/.ssh/runpod_ed25519 -p 19774 root@154.54.102.34`, A100 SXM 80GB ×3
- **GPU 배분**: GPU0=`chinese_only`+`swahili_only`, GPU1=`swahili_only_b`+`english_only_b`, GPU2=`korean_only_b` (각각 `CUDA_VISIBLE_DEVICES=<N>`으로 완전히 분리된 프로세스)
- **왜 오케스트레이터(`run_phase1_41_diagonal.py`)를 안 쓰고 직접 스크립트를 도는가**: 그 스크립트의 `safe_gpus()`가 `nvidia-smi`로 물리 GPU 전체를 다시 조회해서 `MAX_GPUS`(기본 2, 가장 낮은 인덱스부터)로 고정하기 때문에, 이 pod에서 3개를 동시에 띄우면 전부 GPU0,1을 잡으려고 충돌함(같은 유저 소유 프로세스는 "다른 사용자 점유"로 안 걸러짐). 대신 로컬 §2(GPU2,3 disagreement 작업)와 동일한 패턴으로 각 stage 스크립트(`phase1_run_freq_and_scale.py`→`phase1_fisher.py`→`phase1_svd_scale.py`→`phase1_merge_eval.py`)를 직접 호출하는 드라이버 스크립트를 만들어 GPU마다 하나씩 백그라운드로 돌림.
- **드라이버 스크립트**: `/workspace/otter/scripts/pod2_gpu_driver.sh <gpu_idx> <condition...>` — seed 0,1,2 각각에 대해 freq→fisher→svd_scale→merge_eval을 순서대로 실행하되, 각 단계 출력 파일이 이미 있으면 스킵(오케스트레이터의 ensure_fisher/ensure_scale/ensure_diagonal_merge_eval과 동일한 로직을 직접 구현), fisher/svd_scale 완료 직후 `fisher_layers`/`svd_scale_layers` 자동 삭제(§0 디스크 청소 규칙 반영). 로컬 원본은 `/tmp/claude-1020/-home-minjeong-project-FERRET/80861fc6-ac1c-406c-8286-53501972bc1e/scratchpad/pod2_gpu_driver.sh` (세션 전용 경로 — 사라졌으면 pod2의 `/workspace/otter/scripts/pod2_gpu_driver.sh`가 원본이므로 거기서 다시 받으면 됨).
- **실행 command** (2026-07-27 19:13:33 UTC 시작, PID GPU0=3293/GPU1=3294/GPU2=3295):
  ```
  ssh -i ~/.ssh/runpod_ed25519 -p 19774 root@154.54.102.34
  cd /workspace/otter/scripts
  nohup bash pod2_gpu_driver.sh 0 chinese_only swahili_only > /workspace/otter/logs/gpu0.log 2>&1 &
  nohup bash pod2_gpu_driver.sh 1 swahili_only_b english_only_b > /workspace/otter/logs/gpu1.log 2>&1 &
  nohup bash pod2_gpu_driver.sh 2 korean_only_b > /workspace/otter/logs/gpu2.log 2>&1 &
  ```
- **seed**: 전 조건 seed0 진행 중 (freq 단계, 19:21 UTC 기준 GPU0 25/238, GPU1 33/119, GPU2 26/119 batch). seed1/seed2는 미착수(seed0 뒤에 순차 진행).
- **output/log/checkpoint**:
  - 로그: `/workspace/otter/logs/gpu{0,1,2}.log` (pod2 내부)
  - checkpoint: `/mnt/HDD/minjeong/d2moe_results/phase1/{chinese_only,swahili_only,swahili_only_b,english_only_b,korean_only_b}/seed{0,1,2}/`
  - 모니터링: 이 세션에서 백그라운드 Monitor(`bsgvfkrs1`)가 3개 로그를 45초 간격으로 폴링해 시작/완료/에러 라인을 감시 중 — 세션이 끊기면 이 감시도 끊기므로 새 세션은 재확인 필요.
- **디스크**: hf_cache 31GB, network volume quota 180GB — fisher/svd_scale 레이어 임시 디렉토리는 드라이버 스크립트가 완료 직후 자동 삭제하므로 5조건×3seed를 순차로 돌아도 피크 사용량은 조건 1개치(~수십GB) 수준에서 안정적일 것으로 예상(직접 확인은 아직 안 함 — 첫 svd_scale 완료 후 `du -sh`로 검증 권장).
- **완료 후 확인할 metric**: `scale_{condition}_seed{N}/eval_ppl.json`의 5개 언어 bits_per_byte
- **main-results table 대응**: Table 1 / Figure 1의 **"Chinese", "Swahili" row 전체**, 그리고 **noise-floor 계산용 english_only_b/korean_only_b/swahili_only_b 대각선 셀** (own_gain 판정에 필요, `phase1_41_headline_gate.py` 참고)
- **이번 세션에서 겪은 문제와 해결 (시간순)**:
  1. **hf_cache를 로컬 경유 relay 대신 Pod1→Pod2 직접 전송으로 전환**: 이전 세션이 시작한 relay(pod1→로컬→pod2, 17GB/31GB 진행 중이던 것)를 발견하고 kill. 대신 로컬 ssh-agent에 키를 추가하고 `ssh -A`로 pod1에 접속해 pod1이 직접 pod2(공인 IP:포트)로 rsync하도록 전환(로컬 대역폭을 안 거쳐 훨씬 빠름, `bash -c 'echo > /dev/tcp/154.54.102.34/19774'`로 pod1→pod2 직접 도달 가능한 것 먼저 확인).
  2. **rsync `-z`가 병목**: 이미 압축된 safetensors/모델 blob에 gzip 압축을 걸면 CPU가 병목이 되어 사실상 멈춘 것처럼 보임(수 분간 2MB만 전송). `-z` 제거(plain `-av`)로 재시작하니 300MB/s+ 로 회복.
  3. **전송 후 파일 누락 발견**: `find -type f`로 pod1/pod2 파일 목록을 diff한 결과 blob 2개(`d7da53a8...`, 5GB짜리 모델 shard 하나 포함)와 `refs/main`이 빠져 있었음(원인 불명 — 첫 번째 `-z` 시도를 kill한 직후라 그 잔여물과의 경합 가능성). 개별 `scp`로 재전송, 사이즈 diff로 전체 재검증 완료.
  4. **conda-pack 환경이 손상돼 있었음**: `/root/miniconda3/envs/d2moe_env`가 이미 존재해서 "ENV READY"로 넘어갔었지만, `torch.__version__` 접근 시 `AttributeError`(→ `torch/__init__.py`가 아예 없음)를 발견. 원인은 `d2moe_env_packed.tar.gz`가 pod2에 **1.9GB만 전송된 상태**(원본 3.85GB)로 남아 있었던 것(`gzip -t` corrupt 확인). Pod1의 원본 tar를 gzip 무결성 확인 후 pod1→pod2 직접 전송 → 압축해제 → `conda-unpack` 실행까지 완료, `torch 2.3.1+cu121, cuda available, device_count=3` 확인함.
  5. **freq 단계 미패치 발견**: `phase1_fisher.py`/`svd_scale.py`/`merge_eval.py`는 패치했지만, freq 단계가 실제로 호출하는 `D2MoE/preprocess/get_expert_freq.py`는 별도 파일이라 패치가 안 되어 있어 "meta device offload" 경고와 함께 느리게 진행 중이었음(Pod1도 동일하게 미패치 상태였음). 동일한 `PHASE1_GPU_MEM_GIB` 오버라이드를 Pod1/Pod2 양쪽의 `get_expert_freq.py`에 추가 적용(로컬 사본은 2-GPU라 불필요, 안 건드림). 이미 시작된 seed0 freq는 재시작하지 않고 그대로 두었음 — seed1부터 효과 있을 것.
- **다음 액션**: seed0 freq 완료 대기 → fisher(패치 적용돼 있어 90~110s/layer 예상) → svd_scale(A100 실측 없음, 로컬 3090 기준 ~2.95분/layer를 참고치로만 사용) → merge_eval. 각 GPU가 자기 조건 리스트를 끝까지 순차로 돎. **svd_scale 첫 완료 시 실측 rate로 위 ETA 표 갱신 필요.**

---

## 5. (참고, 이 세션 소관 아님) 다른 Claude 세션 — Swahili 2×2 재시도

- **machine**: local, GPU0,1 목표 (mixed_5lang과 동일 자원 — 위 "알려진 리스크" 참고)
- **config/command**: `otter/scripts/run_phase1_bengali_grid.py`, `run_phase1_mixed5lang.sh`, `run_phase1_2x2_verify.py` 존재하나 **이번 세션 판단으로 미실행 지시**함 (§4.1 grid와 중복 우려)
- **seed**: `swahili_only` seed1의 `svd_scale`, layer 24/27까지 완료(2026-07-27 15:52 UTC 파일 timestamp — 이 세션 활동 이전), 이후 진행 없음
- **output/log**: `otter/logs/2x2_swahili_run.log`, checkpoint `/mnt/HDD/minjeong/d2moe_results/phase1/swahili_only/seed1/svd_scale_layers/layer_{1..24}.pt`
- **현재 상태**: 확인 시점 기준 활성 프로세스 없음 — GPU0,1 유휴 대기 wrapper로 추정되나 프로세스 목록에서 직접 확인 안 됨 (미확정)
- **다음 액션 (다른 세션 소관)**: GPU0,1 유휴 시 layer 25부터 재개 예정이라고 보고받음. **이 세션의 mixed_5lang과 자원 경합 중이므로 사용자 조율 필요.**

---

## 요약: main results table 커버리지 현황

| Row (calibration 조건) | 상태 | 담당 |
|---|---|---|
| English | 기존 완료 (2×2 verify 자산) | - |
| Korean | 기존 완료 (2×2 verify 자산) | - |
| Chinese | **미착수, Pod2 대기 중** | Pod2 |
| Swahili | **미착수, Pod2 대기 중** (2×2 재시도와 별개) | Pod2 |
| Bengali | **진행 중, seed1** | Pod1 |
| Balanced(mixed_5lang) | **✅ 완료 (3 seed 전부)** | 로컬 GPU0,1 (이제 자유) |
| Placebo(EN-B/KO-B/SW-B) | Pod2 대기(merge_eval만 필요, Fisher/scale 기존 자산) | Pod2 |
