# Shooting PVP game AI project

## running scene

## note

- stochastic world
- 편의상 극좌표계 ($r$,$\theta$)를 사용한다.

## more things to make 

- ~~발사 대기시간~~ 
- ~~player gravitation~~

## rule

플레이어가 상대 플레이어의 총알에 맞으면 게임이 끝난다.

## action

- move position :  위치를 이동할 수 있다. (상, 하, 좌, 우, 정지) 5가지 행동이 가능하다.
- adjust angle : 총구의 위치를 조정한다. (좌, 우, 정지) 3가지 행동이 가능하다.
- shoot bullet : 총알을 발사한다. (발사, 정지) 2가지 행동이 가능하다. 발사에는 대기시간이 존재한다.

## reward

- 적을 맞춰서 승리했을 경우 : +1
- 총알에 맞아 패배했을 경우 :  -1

## difference of pattern between AI and human

- 인간은 AI만큼 연산처리가 빠르지 않다 (computer vision).
- 인간은 어느 방향으로 쏘아야할지 대충 가늠할 수 있다.
- 인간은 총알에 맞으면 지고, 총알을 맞추면 승리한다는 사실을 안다.

## feature of this game

- 적과의 거리가 $d$일때, 맞을 확률이 $\alpha$라면, 적과의 거리가 $kd$이면,  맞을 확률은 $\frac{\alpha}{k}$이다.

## Model

### network의 설정

이 게임은 총 2개의 목표를 가지고있다. 

1. 총알에 맞지 않기
2. 상대방을 총알에 맞추기

그리고 이 게임에서는 위에 나온것 처럼 3개(move, rotate, shoot)의 action을 가지고있다 .

1번은 action:move와 직접적으로 영향받는 reward이며, 2번은 action:rotate와 action:shoot와 직접적으로 영향을 받는 reward이다.

action:shoot은 쿨타임을 계산하기 힘드므로 auto shoot(쿨타임이 찰때 바로 사격)을 사용한다.

pvp 게임 특성상, 한 플레이어는 승리하고, 한 플레이어는 패배한다. 

때문에 승리의 reward와 패배의 reward를 이용하여, 네트워크의 구성이 상호 연결적이어야 한다고 생각할 수 있다.

하지만 사격 실패 또는 사격 성공을 reward로 갖는 rotation network와, 총알 회피 또는 총알 맞음의 reward로 나누어지는 move network로 상호 독립적인 network를 구축할 수 있다.



해당 논문에서는 DQN(Deep Q-Network)을 이용하여 강화학습을 진행한다.

move network와 rotate network는 서로 다른 DQN이며, state는 input layer, action은 output layer을 의미한다.

또한, 본 게임은  극좌표계를 사용한다.

### move network

move network는 현재의 state가 주어졌을 때 어떤 방향으로 움직여야 하는지를 결정하는 network이다.

#### state

- upside bullet distance
-  

#### action

- UP
- DOWN
- LEFT
- RIGHT
- NONE



### Action:rotate network

move network와 마찬가지로 주어진 network만 optimize하면 된다.



해당 network에서는 상대방을 맞추기 위하여 어떤 행동을 해야하는지를 결정한다.

#### state

- $d$ : player와 enemy 사이의 거리
- $\theta$ : 2차원 좌표상 player와 enemy 사이의 각도

#### action

- LEFT
- RIGHT
- NONE



###The way that configure environment as human level

당연한 소리일수도 있으나, input에 noise를 첨가하면 된다. 인간은 vision을 통하여 학습하지만, 그것이 얼추 비슷한 정보일 수는 있으나, 완벽한 정보는 아니기 때문이다.



### Policy network

앞서 말했듯이 vision을 통한 정보 손실을 고려한 network로, AI가 선택해야하는 2가지 action(move,rotate)중 어떤 것이 더 현재 state에 중요한 action인지를 선택하는 network이다.

만약 move가 더 중요하다면, 다음 턴에 move를 위한 input을 갱신한다. 반대로, rotate가 더 중요하다면, 다음 턴에는 rotate를 위한 input을 갱신한다. 

갱신되지 않은 input은 이전 state와 같은 input을 가지게 된다.

갱신되지 않는 state 때문에 학습에 이상이 생길 것을 예측하여 input으로 t를 주었다.

- 학습에 문제가 생길 경우 살펴봐야 하는 변수 : t

#### input

- t:  policy를 바꾼 시점으로부터 경과 시간

#### output

- move_net : move network의 input status를 update한다.
- rotate_net : rotate network의 input status를 update한다.

![31550473118541](C:\Users\USER10\AppData\Roaming\Typora\typora-user-images\1550473118541.png)

#### Input의 설정

input으로는 크게 frame을 주는 것과, 인간이 알 수 있는 state를 최대한 주는 것, 둘 중 한 가지로 설정할 것이다.

전자의 input은 앞의 모델에서 적용했기 때문에 생략한다.

만약 후자를 선택한다면, 기계적이지 않으면서, 인간적인 state를 얻어야한다. (difference of pattern between AI and human과 같은 항목이 생긴 이유이다.)

하지만, 아무리 다양하게 준다고 해도, AI와 인간이 플레이 하고 있는 상황은 다르다.

AI는 개발자(나)가 제공해준 몇 개의 input을 이용하여 수치적으로 게임을 진행하는 반면, 인간은 자신이 스스로 정보를 획득한다.

때문에 인간의 vision은 일정하지 않고, 중간중간 정보의 손실이 일어날 수 있다.

예를 들면 가까이 다가오는 총알을 피하기 위하여 vision을 가까이 있는 총알에 고정하여 rotate가 허술하게 된다거나, rotate를 하기 위하여 상대방의 위치가 있는 방향으로 vision을 돌리면서, 다가오는 총알에 대해 위험해질 수 있다는 것이다.

나는 이러한 2가지 vision의 특성이 2가지의 목표와 관련이 깊다는 것을 알아냈고,

이러한 특성을 이용하여 크게 policy network, move network, rotate network 3가지의 network를 구축하기로 했다.