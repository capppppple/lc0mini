# lc0mini

Lc0/AlphaZero 스타일 구조를 작게 따라가보는 미니 체스 엔진 프로젝트입니다.

목표는 처음부터 강한 엔진을 만드는 것이 아니라, 아래 흐름을 직접 돌릴 수 있게 만드는 것입니다.

```text
self-play 데이터 생성
-> MCTS 방문 횟수 policy + value 학습
-> 후보 모델 평가
-> 더 강하면 best checkpoint 승격
-> UCI 엔진으로 GUI에서 실행
```

## 구조

```text
engine/
  encoding.py     # chess.Board -> PyTorch tensor
  network.py      # policy/value network
  search.py       # neural-guided MCTS
  uci.py          # 체스 GUI와 연결할 UCI 루프

training/
  self_play.py    # MCTS self-play 데이터 생성
  train.py        # PyTorch 학습
  evaluate.py     # 후보 모델 vs 기존 best 평가전
  replay.py       # 최근 self-play 데이터 섞기
  pipeline.py     # self-play -> train -> eval -> promote 자동 루프
  hardware.py     # 현재 런타임 확인

notebooks/
  colab_train.ipynb
```

## 로컬 실행

```powershell
pip install -r requirements.txt
python -m training.hardware
python -m engine.uci --simulations 64
```

UCI 테스트 입력:

```text
uci
isready
position startpos
go nodes 64
quit
```

학습된 모델을 쓸 때:

```powershell
python -m engine.uci --model checkpoints\best.pt --simulations 128
```

## Colab GPU/TPU 추천

현재 코드는 PyTorch + CUDA 중심입니다. 그래서 지금은 **GPU 추천, TPU 비추천**입니다.

```text
CPU        디버그만 가능. 학습용 비추천.
T4 GPU     무료/저렴한 빠른 테스트용.
L4 GPU     추천 기본값. 속도/가용성/비용 균형이 좋음.
A100 GPU   진지한 학습 추천. batch와 모델을 키우기 좋음.
G4 GPU     강력하지만 현재 작은 모델에는 과함. 큰 실험용.
H100 GPU   가장 빠르지만 현재 코드 규모에는 과함. 큰 모델/배치 MCTS용.
v5e-1 TPU  지금은 비추천. torch_xla/JAX 포팅 전에는 장점이 작음.
v6e-1 TPU  매우 강하지만 지금은 비추천. TPU용 코드로 바꾼 뒤 고려.
```

Colab에서 먼저 확인:

```python
!python -m training.hardware
```

## Colab 자동 학습

Colab에서 repo를 가져옵니다.

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/capppppple/lc0mini.git
%cd lc0mini
!pip install -r requirements.txt
```

L4 추천 시작값:

```python
!python -m training.pipeline \
  --work-dir /content/drive/MyDrive/lc0mini/runs \
  --best /content/drive/MyDrive/lc0mini/checkpoints/best.pt \
  --iterations 5 \
  --games 100 \
  --simulations 64 \
  --eval-games 12 \
  --eval-simulations 64 \
  --epochs 3 \
  --batch-size 128 \
  --replay-window 5 \
  --max-replay-positions 50000 \
  --amp \
  --resume-from-best
```

A100 이상에서 더 크게:

```python
!python -m training.pipeline \
  --work-dir /content/drive/MyDrive/lc0mini/runs \
  --best /content/drive/MyDrive/lc0mini/checkpoints/best.pt \
  --iterations 10 \
  --games 300 \
  --simulations 128 \
  --eval-games 24 \
  --eval-simulations 128 \
  --epochs 5 \
  --batch-size 256 \
  --channels 96 \
  --blocks 6 \
  --replay-window 8 \
  --max-replay-positions 200000 \
  --amp \
  --resume-from-best
```

## 학습량 조절

```text
--iterations        self-play/train/eval 반복 횟수
--games             iteration마다 self-play 게임 수
--simulations       self-play 한 수당 MCTS 탐색 횟수
--eval-games        후보 모델 평가전 판수
--eval-simulations  평가전 한 수당 MCTS 탐색 횟수
--epochs            생성된 데이터를 반복 학습하는 횟수
--batch-size        한 번에 학습하는 포지션 수
--channels          네트워크 너비
--blocks            residual block 수
--replay-window     최근 몇 iteration 데이터를 섞을지
--max-replay-positions replay buffer 최대 포지션 수
--max-plies         한 게임 최대 ply 수
--adjudicate-threshold max-plies 도달 시 material 판정 최소 차이
--adjudicate-scale  material 차이를 value로 압축하는 강도
--amp               GPU mixed precision 사용
```

초기에는 게임이 체크메이트로 잘 끝나지 않기 때문에 `max-plies`에 자주 걸립니다. 기본값은 이때 기물 점수로 결과를 부드럽게 판정합니다.

```text
termination=max_plies_material  기물 차이로 value target 생성
termination=max_plies_draw      기물 차이가 작아서 draw 처리
termination=game_over           실제 체크메이트/무승부 결과 사용
```

이 기능을 끄고 싶으면:

```powershell
python -m training.pipeline --no-material-adjudication ...
```

빠른 테스트:

```powershell
python -m training.pipeline --iterations 1 --games 2 --simulations 8 --eval-games 2 --eval-simulations 4 --epochs 1 --batch-size 16
```

평가 결과는 각 iteration 폴더의 `eval.json`과 `summary.json`에 저장됩니다.

```text
runs/
  iter_0001/
    selfplay.jsonl
    replay.jsonl
    candidate.pt
    eval.json
    summary.json
  pipeline_summary.json
```

`eval.json`에는 다음 값들이 들어갑니다.

```text
win_rate   후보 모델 점수율
elo_diff   기존 best 대비 Elo 추정치
wins/draws/losses
white_score/black_score
```

## GitHub에 올리기

이 PC에서 `git` 명령이 바로 안 잡히면 전체 경로로 실행하면 됩니다.

```powershell
& "C:\Program Files\Git\cmd\git.exe" status
& "C:\Program Files\Git\cmd\git.exe" add .
& "C:\Program Files\Git\cmd\git.exe" commit -m "Update lc0mini"
& "C:\Program Files\Git\cmd\git.exe" push
```

## 다음 개발 목표

1. MCTS neural inference batching
2. 더 빠른 self-play worker 병렬화
3. Elo 추정 리포트
4. TPU/JAX 버전 실험
