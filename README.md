# lc0mini

Lc0/AlphaZero 스타일을 작게 따라가보는 미니 체스 엔진 프로젝트입니다.

목표는 아래 흐름을 직접 돌릴 수 있게 만드는 것입니다.

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
```

## Colab 저장 구조

Colab 런타임의 `/content`는 꺼지면 사라집니다. 그래서 진행상황은 Google Drive에 저장합니다.

```text
/content/drive/MyDrive/lc0mini/checkpoints/best.pt
/content/drive/MyDrive/lc0mini/runs/
/content/drive/MyDrive/lc0mini/runs/pipeline_summary.json
```

`training.pipeline`은 기본적으로 기존 `runs/iter_XXXX`를 보고 다음 번호부터 이어서 실행합니다. 예를 들어 Drive에 `iter_0008`까지 있으면 다음 실행은 자동으로 `iter_0009`부터 시작합니다.

새로 1번부터 시작하고 싶을 때만:

```powershell
python -m training.pipeline --restart ...
```

특정 번호부터 시작하고 싶으면:

```powershell
python -m training.pipeline --start-iteration 25 ...
```

## Colab 자동 학습

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/capppppple/lc0mini.git
%cd lc0mini
!pip install -r requirements.txt
!python -m training.hardware
```

L4 추천 시작값:

```python
!python -m training.pipeline \
  --work-dir /content/drive/MyDrive/lc0mini/runs \
  --best /content/drive/MyDrive/lc0mini/checkpoints/best.pt \
  --preset fast \
  --iterations 8 \
  --amp \
  --resume-from-best
```

A100 이상에서 더 크게:

```python
!python -m training.pipeline \
  --work-dir /content/drive/MyDrive/lc0mini/runs \
  --best /content/drive/MyDrive/lc0mini/checkpoints/best.pt \
  --preset strong \
  --iterations 10 \
  --amp \
  --resume-from-best
```

## 학습량 조절

```text
--iterations        이번 실행에서 추가로 돌릴 iteration 수
--preset            debug / fast / balanced / strong 기본값 묶음
--games             iteration마다 self-play 게임 수
--simulations       self-play 한 수당 MCTS 탐색 횟수
--mcts-batch-size   MCTS 중 neural inference를 몇 개씩 묶을지
--eval-games        후보 모델 평가전 판수
--eval-simulations  평가전 한 수당 MCTS 탐색 횟수
--eval-mcts-batch-size 평가전 MCTS batch size
--eval-interval     몇 iteration마다 평가전을 할지
--epochs            생성된 데이터를 반복 학습하는 횟수
--batch-size        한 번에 학습하는 포지션 수
--channels          네트워크 너비
--blocks            residual block 수
--replay-window     최근 몇 iteration 데이터를 섞을지
--max-replay-positions replay buffer 최대 포지션 수
--max-plies         한 게임 최대 ply 수
--store-visits      디버그용 방문 횟수 저장. 느리고 파일이 커짐
--amp               GPU mixed precision 사용
```

프리셋 감각:

```text
debug     코드 테스트용
fast      초반 추천. 훨씬 빠르게 반복
balanced  품질/속도 균형
strong    A100 이상에서 긴 학습용
```

MCTS batch size 추천:

```text
T4/L4: 8~16
A100: 16~32
H100: 32~64
CPU: 1~4
```

## 결과 확인

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

`eval.json`에서 볼 값:

```text
win_rate
elo_diff
wins / draws / losses
white_score / black_score
```

## 로컬 GUI 실행

Colab에서 만든 `best.pt`를 로컬의 `checkpoints/best.pt`로 가져온 뒤 실행합니다.

```powershell
cd "C:\Users\shine\OneDrive\문서\New project 2\lc0mini"
python -m engine.uci --model checkpoints\best.pt --simulations 128 --mcts-batch-size 16
```

GUI 설정 예:

```text
Executable:
C:\Python314\python.exe

Working directory:
C:\Users\shine\OneDrive\문서\New project 2\lc0mini

Arguments:
-m engine.uci --model checkpoints\best.pt --simulations 128 --mcts-batch-size 16
```
