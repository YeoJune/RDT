# RDT 사용 가이드

## 프로젝트 개요

Recursive Denoising Transformer (RDT)는 재귀적 디노이징을 통해 마스크된 텍스트를 점진적으로 복원하는 모델입니다.

### 핵심 설계 원칙

1. **Deep Encoder, Shallow Decoder**:

   - Encoder(6층): 노이즈 제거 및 Latent 정제
   - Decoder(1층 또는 Linear): Latent → Token 변환

2. **재귀적 처리**:

   - 하나의 Encoder를 여러 번 반복 사용
   - 파라미터 효율적이면서 깊은 연산 가능

3. **적응형 연산**:
   - Gate MLP가 남은 스텝 수 예측
   - 쉬운 입력은 빨리 끝나고, 어려운 입력은 더 많이 처리

## 설치

```bash
# 프로젝트 디렉토리로 이동
cd RDT

# 패키지 설치 (pyproject.toml에 정의된 모든 의존성 자동 설치)
pip install -e .

# 또는 개발 도구 포함 설치
pip install -e ".[dev]"
```

## 구성 요소

### 1. 설정 파일 (configs/)

#### base.yaml - 문서 추천 세팅

```yaml
model:
  d_model: 512 # Hidden dimension
  n_encoder_layers: 6 # Shared encoder depth
  n_decoder_layers: 1 # Shallow decoder
  decoder_type: "linear" # Extreme lightweight

training:
  max_chain_length: 5 # Training segment length
  total_steps: 10 # Total denoising steps (100% → 0%)
  batch_size: 32
  learning_rate: 0.0001

data:
  dataset_name: "wikitext-2" # or 'wikitext-103'
  max_seq_length: 512
```

#### experiment.yaml - 실험용 오버라이드

더 큰 모델이나 다른 데이터셋으로 실험할 때 사용:

```yaml
model:
  d_model: 768
  n_encoder_layers: 12

data:
  dataset_name: "wikitext-103"
```

### 2. 모델 구조 (src/model.py)

#### RDT 클래스

```python
class RDT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 1,
        decoder_type: str = 'linear',  # 'linear' or 'transformer'
        ...
    )
```

주요 메서드:

- `forward()`: 단일 스텝 처리
- `recursive_forward()`: 학습용 여러 스텝 처리
- `inference()`: Gate 기반 적응형 추론

### 3. 데이터 처리 (rdt/data/)

#### MaskingStrategy

```python
# Step k → k*10% 마스킹
# Step 0: 원본 (0% 마스크)
# Step 5: 50% 마스크
# Step 10: 100% 마스크
```

#### WikiTextDataset

- WikiText-2, WikiText-103 자동 다운로드
- BERT tokenizer 사용
- 랜덤 세그먼트 샘플링으로 메모리 효율적 학습

### 4. 학습 로직 (rdt/training/)

학습 시퀀스:

```
1. 랜덤 시작점 i와 길이 L 선택
2. L번 반복:
   - Encoder 통과: H_k = Encoder(H_{k+1})
   - Decoder 통과: 복원 Loss 계산
   - Gate 통과: 스텝 예측 Loss 계산
3. Loss 합산 후 역전파
```

## 학습 실행

### 기본 학습 (WikiText-2)

```bash
rdt-train --config rdt/configs/base.yaml
```

### 큰 데이터셋으로 학습 (WikiText-103)

```bash
python train.py --config configs/base.yaml --override configs/experiment.yaml
```

### GPU 지정

```bash
python train.py --config configs/base.yaml --device cuda
```

### 체크포인트에서 재개

```bash
python train.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_epoch_5.pt
```

### TensorBoard로 모니터링

```bash
# 새 터미널에서
tensorboard --logdir runs/

# 브라우저에서 http://localhost:6006 접속
```

로그되는 메트릭:

- `train/total_loss`: 전체 Loss
- `train/recon_loss`: 복원 Loss
- `train/gate_loss`: Gate 예측 Loss
- `train/lr`: Learning rate
- `val/*`: 검증 메트릭

## 추론 실행

### 단일 텍스트 추론

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "The quick brown fox jumps over the lazy dog"
```

출력 예시:

```
Input:  The quick brown fox jumps over the lazy dog
Output: The quick brown fox jumps over the lazy dog
Steps:  3/20
```

### 인터랙티브 모드

```bash
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

사용 예:

```
RDT Interactive Inference
Enter text to denoise (or 'quit' to exit)

Input text: hello world
Running inference...

Input:  hello world
Output: hello world
Steps taken: 2/20
```

### 파라미터 조정

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --text "Sample text" \
    --max_steps 30 \        # 최대 재귀 스텝 증가
    --threshold 0.05        # Gate 임계값 낮춤 (더 정확하게)
```

## 하이퍼파라미터 튜닝 가이드

### 모델 크기 조정

#### 작은 모델 (빠른 실험용)

```yaml
model:
  d_model: 256
  n_heads: 4
  n_encoder_layers: 4
  d_ff: 1024
```

#### 중간 모델 (권장)

```yaml
model:
  d_model: 512
  n_heads: 8
  n_encoder_layers: 6
  d_ff: 2048
```

#### 큰 모델 (성능 중시)

```yaml
model:
  d_model: 768
  n_heads: 12
  n_encoder_layers: 12
  d_ff: 3072
```

### 학습 전략 조정

#### Chain Length

- `max_chain_length: 3`: 빠른 학습, 낮은 메모리
- `max_chain_length: 5`: 균형 (권장)
- `max_chain_length: 8`: 긴 컨텍스트, 높은 메모리

#### Total Steps

- `total_steps: 5`: 적은 마스크 단계 (빠름)
- `total_steps: 10`: 균형 (권장)
- `total_steps: 20`: 섬세한 디노이징 (느림)

#### Loss Weights

Gate가 너무 빨리 수렴하면:

```yaml
training:
  loss_weight_recon: 1.0
  loss_weight_gate: 0.1 # Gate weight 줄임
```

### Decoder 타입 선택

#### Linear Decoder (권장)

```yaml
model:
  decoder_type: "linear"
  n_decoder_layers: 1 # 무시됨
```

- 장점: 가장 가볍고 빠름
- 단점: 표현력 제한
- 추천: 초기 실험 및 대부분의 경우

#### Transformer Decoder

```yaml
model:
  decoder_type: "transformer"
  n_decoder_layers: 1 # 또는 2
```

- 장점: 더 나은 표현력
- 단점: 느리고 무거움
- 추천: Linear가 부족할 때만

## 체크포인트 관리

### 저장 위치

```
checkpoints/
├── best_model.pt              # 최고 검증 Loss 모델
├── checkpoint_epoch_1_step_1000.pt
├── checkpoint_epoch_2_step_2000.pt
└── ...
```

### 자동 정리

설정에서 관리:

```yaml
training:
  save_every_n_epochs: 1 # 매 1 epoch마다 저장
  keep_last_n_checkpoints: 3 # 최근 3개만 유지
```

### 체크포인트 내용

```python
checkpoint = {
    'epoch': int,
    'step': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'loss': float,
    'config': dict
}
```

## 문제 해결

### Out of Memory (OOM)

```yaml
training:
  batch_size: 16 # 줄이기
  max_chain_length: 3 # 줄이기

data:
  max_seq_length: 256 # 줄이기
```

### 학습이 불안정함

```yaml
training:
  learning_rate: 0.00005 # 줄이기
  max_grad_norm: 0.5 # 더 강한 clipping
  warmup_ratio: 0.2 # warmup 늘리기
```

### Gate가 수렴 안 함

```yaml
training:
  loss_weight_gate: 2.0 # Gate weight 높이기

model:
  gate_hidden_dim: 512 # Gate MLP 크게
```

### 학습이 너무 느림

```yaml
data:
  num_workers: 8 # 늘리기
  pin_memory: true

training:
  batch_size: 64 # VRAM 허용 범위에서 최대로
```

## 실험 추천 순서

### Phase 1: 기본 검증

```bash
# 작은 모델로 빠른 검증
python train.py --config configs/base.yaml
```

- 목표: 모델이 학습되는지 확인
- 기대: 3-5 epochs 후 Loss 감소

### Phase 2: 스케일업

```bash
# 더 큰 모델/데이터셋
python train.py --config configs/base.yaml --override configs/experiment.yaml
```

- 목표: 성능 향상 확인
- 기대: 더 낮은 최종 Loss

### Phase 3: 하이퍼파라미터 탐색

여러 설정 파일 만들어서 실험:

```yaml
# configs/exp1.yaml
model:
  d_model: 768
training:
  max_chain_length: 8

# configs/exp2.yaml
model:
  decoder_type: 'transformer'
training:
  loss_weight_gate: 0.1
```

### Phase 4: 추론 테스트

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

- 다양한 입력으로 테스트
- Gate 동작 확인 (적응형 스텝)

## 확장 아이디어

### 1. 다른 마스킹 전략

`src/data.py`의 `MaskingStrategy` 수정:

```python
# 현재: Linear (step * 10%)
# 추가 가능: Exponential, Random, Scheduled
```

### 2. 다른 데이터셋

`src/data.py`에 새 Dataset 클래스 추가:

```python
class CustomDataset(Dataset):
    # 커스텀 데이터 로딩 로직
```

### 3. 멀티태스크 학습

`src/trainer.py`에 추가 Loss 구현:

```python
# Classification, NER 등 추가 태스크
```

### 4. 분산 학습

PyTorch DDP 사용:

```bash
torchrun --nproc_per_node=4 train.py --config configs/base.yaml
```

## 참고 문헌

- **MAE**: Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)
- **ALBERT**: A Lite BERT (Lan et al., 2020)

## 지원

문제가 발생하면:

1. TensorBoard 로그 확인
2. 체크포인트 로딩 상태 확인
3. Config 파일 검증
4. GitHub Issues에 보고
