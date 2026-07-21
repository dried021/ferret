# 10. Go / No-Go 기준

## Stage 1

**Oracle Dynamic Rank**는 동일한 평균 FLOPs 조건에서 **최선의 Static Rank**를
능가해야 한다.

이것이 실패하면 → **STOP(중단)**.

## Stage 2

**Router Threshold(라우터 임계값)** 방식이 의미 있는 오라클 이득(gain)을 회복해야
한다. 그렇지 못하면 추가적인 특징(feature)을 조사해야 한다.

## Stage 3

**학습된 예측기(Learned Predictor)**는 경량성을 유지하면서 오라클에 근접해야 한다.
