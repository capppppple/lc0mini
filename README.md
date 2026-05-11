# lc0mini

Lc0 스타일 구조를 작게 따라가보는 미니 체스 엔진 프로젝트입니다.

목표는 처음부터 강한 엔진을 만드는 것이 아니라, 아래 흐름을 직접 돌릴 수 있게 만드는 것입니다.

```text
self-play 데이터 생성
-> policy/value network 학습
-> 체크포인트 저장
-> UCI 엔진으로 bestmove 출력
```

## 구조

```text
engine/
  encoding.py     # chess.Board -> PyTorch tensor
  network.py      # 작은 policy/value network
  search.py       # 현재는 policy 기반 수 선택
  uci.py          # 체스 GUI와 연결할 UCI 루프

training/
  self_play.py    # self-play jsonl 데이터 생성
  train.py        # PyTorch 학습 스크립트

notebooks/
  colab_train.ipynb
```

## 로컬 실행

처음 한 번만 패키지를 설치합니다.

```powershell
pip install -r requirements.txt
```

self-play 데이터 생성:

```powershell
python -m training.self_play --games 5 --out data/selfplay.jsonl
```

학습:

```powershell
python -m training.train --data data/selfplay.jsonl --out checkpoints/latest.pt
```

UCI 엔진 실행:

```powershell
python -m engine.uci --model checkpoints/latest.pt
```

모델 없이 실행하면 랜덤에 가까운 수를 둡니다.

```powershell
python -m engine.uci
```

## GitHub에 올리기

이 PC에서 `git` 명령이 바로 안 잡히면 아래처럼 전체 경로로 실행하면 됩니다.

```powershell
& "C:\Program Files\Git\cmd\git.exe" status
& "C:\Program Files\Git\cmd\git.exe" add .
& "C:\Program Files\Git\cmd\git.exe" commit -m "Add initial lc0mini scaffold"
& "C:\Program Files\Git\cmd\git.exe" push
```

## Colab에서 학습

Colab에서는 repo를 clone하고 GPU로 학습합니다.

```python
!git clone https://github.com/capppppple/lc0mini.git
%cd lc0mini
!pip install -r requirements.txt
!python -m training.self_play --games 20 --out data/selfplay.jsonl
!python -m training.train --data data/selfplay.jsonl --out checkpoints/latest.pt
```

큰 학습 데이터와 모델 파일은 GitHub가 아니라 Google Drive에 저장하는 것을 추천합니다.

## 다음 개발 목표

1. policy 기반 선택을 진짜 MCTS로 교체
2. self-play 결과를 더 좋은 학습 타깃으로 저장
3. 체크포인트 이어서 학습하기
4. Cute Chess나 BanksiaGUI에서 UCI 엔진 테스트
