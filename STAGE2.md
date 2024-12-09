# Description

This document is intended to share the progress of Stage 2.

**Contributor: Minku, Yunjeong**

# Code Review

- scripts/train_ae.sh
    - +dataset: hydra config setup. training file (train_ae, train_ddpm)의 hydra.main decorator에 config path가 상위 경로이고, +dataset flag가 하위 경로

- main/train_ae.py
    - VAE는 resnet 구조
    - train_kwargs: pytorch_lightning trainer의 parameter set

- main/train_ddpm.py
    - decoder
        - \hat{x_0}는 decoder를 forward하기 전에 x에 concatenate 되는 방식으로 condition으로 활용됨
        - z는 time step embedding에 더해지는 방식으로 condition으로 활용됨
    - ddpm_type
        - form1: 3.4.1 in diffuseVAE paper.
        - form2: 3.4.2 in diffuseVAE paper
            - train 과정에서 noise input을 만들 때 x_hat을 그대로 더하는 방식으로 condition으로 활용됨
            - def sample: 생성하는 함수
                - post_mean = \tilde{mu} + coeff3 * x_hat
                - p_variance = \tilde{beta}
        - uncond:  \hat{x_0}를 사용하지 않음
    - DDPMWrapper
        - online ddpm: trainable decoder
        - target ddpm: non-trainable decoder. online ddpm과 exponential moving average를 통해서 업데이트 됨. 테스트 

# Progress

## 241207

- 윤정
    1. 코드 리뷰
    2. VAE, DDPM 코드 합쳐보기
- 민구
    1. 코드 리뷰
    2. 활용할만한 joint training 기법 찾아보기
 
## 241209 (1210까지)

- VAE & DDPM 합치기
- VAE가 현재 pl.LightningModule인데 nn.Module로 정의를 바꿔야할지 아니면 그대로 가도 될지

# Joint Training Candidate (This is just Minku's idea)

1. \beta training like \beta-VAE.
2. Iterative training like EM algorithm. VAE -> DDPM -> VAE -> DDPM
3. Gradually use VAE output as condition. Use noise 
4. PCGrad
5. Curriculum learning (https://arxiv.org/pdf/2403.10348)

# Discussion

1. 중요한건 아니지만, 왜 UNET의 self attention은 x+h를 return 할까요?

# Reference

1. [DDPM code review](https://kyujinpy.tistory.com/123)
   
