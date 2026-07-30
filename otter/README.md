# OTTER

**OTTER: On Tongue-sensitive Truncation and Expert Reweighting**

Does the language composition of calibration data change which experts and
parameters training-free MoE compression (D²-MoE) preserves, and therefore
its multilingual performance retention?

**현재 상태 (2026-07-25): Toy0 PASS → Phase 0.5 PROGRESS_PASS (5/5 기준) →
0) 레이어 국소성 검증 SPREAD_BACK_HALF → Fisher Pilot A PASS (다른 모델·진짜
gradient Fisher로 재현) → Phase 1 완료, seed 3개 재현까지 마침, H2_HELD.**
DeepSeek-MoE-16B를 Fisher-weighted merge로 압축한 뒤 FLORES EN/KO/ZH PPL을
bits-per-byte로 비교(baseline KO PPL이 tokenizer byte-fallback 아티팩트였음을
발견하고 정정)했고, 64 samples/seqlen 512 예산 + seed 3개로 재현한 결과
사전 등록한 기준(own-language 이득의 3-seed 부트스트랩 95% CI가 0 배제)을
EN/KO/ZH 세 언어 모두 통과했다 — "자기 언어로 calibration하면 그 언어가 더
잘 보존된다"는 핵심 주장(H2)이 이 pilot 안에서는 유지된다. 남은 제약은
activation-aware whitening 미적용(plain SVD)과 pp_ratio 미적용. 자세한 내용은
아래 문서 참고.

## 문서

| 문서 | 내용 |
|---|---|
| [00_docs/04_전체요약.md](00_docs/04_전체요약.md) | **여기부터 읽기.** 현재 상태, 핵심 발견, 다음 단계 한 페이지 요약 |
| [00_docs/01_연구설계.md](00_docs/01_연구설계.md) | 전체 연구 기획 원본 (RQ/가설/MVP/확장/평가지표/통계검정/리스크) |
| [00_docs/02_Toy_실험.md](00_docs/02_Toy_실험.md) | Toy0 + Phase 0.5 실험 설계와 실제 실행 결과 (수치/그림 포함) |
| [00_docs/03_기술노트.md](00_docs/03_기술노트.md) | GPU 안전 정책, 실행 명령어, 산출물/폴더 구조 |
| [00_docs/07_실행_리소스_계획.md](00_docs/07_실행_리소스_계획.md) | 3090 vs A100 작업 분담, real Fisher 실행 횟수/비용 계획 |

## 빠른 시작

```bash
cd scripts
conda run -n torch_env python 01_calibration_stats.py --smoke   # Toy0 배선 확인
conda run -n torch_env python 02_analyze_toy0.py --smoke
conda run -n torch_env python 03_make_figures.py --smoke

conda run -n torch_env python 01b_phase0_5_stats.py --smoke     # Phase 0.5 배선 확인
conda run -n torch_env python 02b_analyze_repro.py --smoke
conda run -n torch_env python 03b_make_figures_repro.py --smoke

conda run -n torch_env python 01b_phase0_5_stats.py --smoke --config ../data/layer_locality_config.yaml --prefix layer_locality   # 0) 레이어 국소성 검증 배선 확인
conda run -n torch_env python 02c_analyze_layer_locality.py --smoke
conda run -n torch_env python 03c_make_figures_layer_locality.py --smoke

# Phase 0 (공식 D²-MoE repo, DeepSeek-MoE-16B, d2moe_env -- torch_env와 별개 환경)
cd ../D2MoE && source ../scripts/safe_gpus.sh
cd ../scripts && conda run -n d2moe_env python fisher_pilot_a.py --smoke   # Fisher Pilot A 배선 확인
```

전체(non-smoke) 실행 명령과 각 스크립트가 무엇을 하는지는
[00_docs/03_기술노트.md](00_docs/03_기술노트.md)에 정리했다.

**GPU 안전 정책 한 줄 요약:** 이 호스트(4x3090)는 다른 사용자와 공유한다.
`scripts/00_config.py`는 매 실행 시작 시 `nvidia-smi`+`ps`로 GPU별 프로세스
소유자를 확인해서, 현재 사용자가 아닌 프로세스가 있는 GPU는 하드코딩 없이 자동
제외한다. 자세한 내용은 [00_docs/03_기술노트.md](00_docs/03_기술노트.md).

## 폴더 구조

```
otter/
  README.md       # 이 문서 -- 인덱스
  00_docs/        # 연구설계 / 실험 설계·결과 / 기술노트 / 전체요약
  configs/        # device_map.json, environment.txt
  data/           # languages.yaml (Toy0), phase0_5_config.yaml (Phase 0.5), layer_locality_config.yaml (0번)
  scripts/        # 00_config.py 공용 + Toy0(01~03) + Phase 0.5(01b~03b) + 레이어 국소성(01b 재사용+02c/03c)
  results/, logs/, figures/   # 실행 산출물
```
