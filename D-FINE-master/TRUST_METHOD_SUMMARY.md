# Trust-weighted Tiny/Object Localization Summary

이 문서는 지금까지 구현된 제안기법을 빠르게 이해할 수 있도록 정리한 메모다.

핵심 결론만 먼저 말하면:

- 현재 본체 기법은 `tiny/small` 객체에 대해 `matched query의 localization quality`를 예측하는 `trust` 보조 헤드다.
- 목적은 `검출 개수`를 크게 늘리는 것이 아니라 `박스 정렬(box alignment)`과 `high-IoU` 성능을 끌어올리는 것이다.
- 현재까지 가장 설득력 있는 결과는 `AP50`, `AP75`의 상승이다.
- 반면 고정 threshold 기준의 `FP`는 늘었고, `FN`은 총량 기준으로는 거의 비슷하거나 아주 조금 줄었다.
- `scale-aware matched-KD` 확장안은 baseline보다 나빠졌기 때문에, 현재 이야기의 중심은 `minimal trust-weight` run이어야 한다.

---

## 1. 문제 설정

대상 데이터셋과 프로토콜은 filtered AI-TOD tiny/small 설정이다.

- 학습/검증 모두 `tiny`와 `small` GT만 유지한다.
- 빈 이미지는 제외한다.
- 평가의 핵심 지표는 `mAP@50:95`, `AP50`, `AP75`, `AP_tiny`, `AP_s`다.
- 예전 unrestricted protocol의 `AP_very_tiny`, `AP_m`, `AP_l`은 이 단계의 주 지표가 아니다.

즉, 이 프로젝트는 일반적인 범용 detector 개선이 아니라 `tiny/small 객체의 localization 품질 개선`을 검증하는 실험이다.

---

## 2. trust의 본래 목적

trust는 `맞은 박스가 얼마나 믿을 만한가`를 따로 학습시키는 auxiliary signal이다.

구현 의도는 다음과 같다.

- decoder 내부에서 `trust score`를 예측한다.
- 이 score는 단순 confidence가 아니라 `box quality`를 반영한다.
- matched query에 대해서만 supervision을 준다.
- 작은 객체에서는 약간의 위치 오차가 IoU를 크게 흔드므로, trust가 high-IoU 박스를 더 잘 구분하도록 만든다.

즉, trust는 `classification confidence`를 대체하는 것이 아니라 `localization quality estimator`에 가깝다.

---

## 3. 구현 요약

현재 구현은 다음 구조를 따른다.

### 3.1 Trust feature

decoder가 다음 신호들을 합쳐 trust feature를 만든다.

- entropy
- top-1 / top-2 / top-k mass
- class confidence
- box drift
- log area

관련 코드:

- [decoder.py](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_decoder.py)

### 3.2 Trust target

trust의 학습 타깃은 다음과 같다.

```text
target = trust_alpha * IoU + (1 - trust_alpha) * center_term
```

여기서:

- `IoU`는 matched GT와의 overlap
- `center_term`은 GT 중심과 예측 중심의 거리 기반 항
- `trust_alpha = 0.5`

즉, trust는 `박스가 잘 맞는지`와 `중심이 얼마나 가까운지`를 같이 본다.

### 3.3 Size-aware weighting

loss는 size bin에 따라 weight를 줄 수 있게 되어 있다.

- very_tiny
- tiny
- small

현재 완료된 minimal trust-weight run은 이 부분을 강하게 건드리지 않은 상태다.

관련 코드:

- [criterion.py](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/src/zoo/dfine/dfine_criterion.py)

### 3.4 현재 활성 설정

실험용 활성 config는 아래다.

- [dfine_hgnetv2_s_aitod.yml](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_s_aitod.yml)

주요 값:

- `trust_enabled: True`
- `trust_topk: 4`
- `trust_hidden_dim: 64`
- `losses: ['vfl', 'boxes', 'local', 'trust']`
- `loss_trust: 1.0`
- train/val 모두 `tiny/small`만 사용

---

## 4. 현재까지 달성한 내용

### 4.1 Baseline vs trust-weight 결과

비교 요약은 다음과 같다.

| metric | baseline | trust-weight | delta |
|---|---:|---:|---:|
| mAP@50:95 | 0.214030 | 0.214885 | +0.000855 |
| AP50 | 0.469210 | 0.491472 | +0.022262 |
| AP75 | 0.154222 | 0.168967 | +0.014745 |
| AP_tiny | 0.175111 | 0.175616 | +0.000505 |
| AP_s | 0.281033 | 0.284889 | +0.003856 |

해석:

- 전체 평균 AP는 거의 그대로지만, `AP50`과 `AP75`가 분명히 오른다.
- 따라서 trust는 `전체 성능을 크게 올리는 장치`라기보다 `high-IoU 쪽을 밀어주는 장치`로 보는 것이 맞다.

비교 요약 파일:

- [comparison_summary.json](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/trust_weight_comparison/comparison_summary.json)

### 4.2 정성 예시

대표 예시 중 하나는 다음과 같다.

- image 2240
- class: ship
- size: tiny
- scene: 1-2
- baseline IoU: 0.549
- trust IoU: 0.806
- score 변화: -0.006

이 예시는 trust가 `confidence`를 크게 바꾼 것이 아니라, `박스 위치와 크기 정렬`을 더 잘 맞췄다는 점을 보여준다.

비슷한 예시들은 `output/trust_weight_comparison/` 아래에 저장돼 있다.

### 4.3 raw FP/FN 관점

여기서는 사용자가 헷갈릴 수 있는 지점을 분명히 적어둔다.

`summary.json`을 보면 다음과 같다.

| metric | baseline | trust-weight | delta |
|---|---:|---:|---:|
| precision | 0.523952 | 0.515571 | -0.008381 |
| recall | 0.804736 | 0.805053 | +0.000318 |
| TP | 20256 | 20264 | +8 |
| FP | 18404 | 19040 | +636 |
| FN | 4915 | 4907 | -8 |

즉:

- `FP`는 늘었다.
- `FN`은 총량 기준으로는 아주 조금 줄었다.
- 다만 `FN reasons` 안에서는 `localization`이 늘었고, `low_confidence` 류는 줄었다.

중요한 점:

- 이 `FP/FN reasons`는 heuristic post-hoc label이다.
- 그래서 카테고리별 증감은 참고용으로 봐야 한다.
- `conf_thresh = 0.3`에서 본 raw count는 trust의 최종 성공 여부를 전부 설명하지 못한다.

관련 summary 파일:

- [baseline summary](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/dfine_s_baseline_filtering/qualitative_analysis/summary.json)
- [trust summary](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/D-Fine_S_filter_trust_weight/qualitative_analysis/summary.json)

---

## 5. 해석

현재 결과를 가장 안전하게 해석하면 다음과 같다.

- trust는 `true positive`의 박스 품질을 개선한다.
- 특히 `tiny/small` 객체에서 `IoU 0.5 근처`의 박스를 `0.75 이상`으로 끌어올리는 데 의미가 있다.
- 그래서 `AP50`, `AP75` 개선이 나온다.
- 하지만 `background FP`를 충분히 줄이거나, fixed-threshold 기준의 `precision`을 확실히 올리지는 못했다.
- 따라서 이 기법은 `error reduction`보다 `box alignment refinement`로 설명하는 것이 맞다.

논문/발표에서 써도 되는 표현:

- `trust weighting sharpens box alignment for tiny/small objects`
- `trust improves high-IoU localization`
- `trust converts borderline matches into higher-quality matches`

피해야 하는 과장 표현:

- `FP/FN을 전반적으로 해결했다`
- `모든 지표를 크게 개선했다`
- `범용 detector로서 완전한 성능 향상`

---

## 6. 실패한 확장안

`scale-aware matched-KD`는 현재 이야기의 본체가 아니다.

결과적으로는 baseline보다 낮아졌다.

따라서 현재 남겨야 할 스토리는 다음 한 줄이다.

- `minimal trust-weight`는 tiny/small 객체의 high-IoU localization을 개선했지만, `scale-aware matched-KD`까지 일반화되지는 않았다.

---

## 7. 관련 산출물

시각화와 요약 산출물은 아래 경로에 있다.

- [comparison metrics figure](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/trust_weight_comparison/comparison_metrics.png)
- [comparison examples figure](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/trust_weight_comparison/comparison_examples.png)
- [comparison summary](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/output/trust_weight_comparison/comparison_summary.json)

비교 스크립트:

- [compare_trust_weight_runs.py](/home/com_2/suan/TOD/D-FINE-master/D-FINE-master/tools/analysis/compare_trust_weight_runs.py)

---

## 8. 한 줄 요약

`trust`는 tiny/small 객체에서 matched query의 박스 품질을 학습해 high-IoU localization을 밀어주는 보조 기법이며, 현재까지는 `AP50/AP75` 개선이 핵심 성과이고 `FP/FN` 감소가 핵심 성과는 아니다.
