# Description

This document is intended to share the progress of Stage 2.

**Contributor: Minku, Yunjeong**

# Code Review

# Progress

## 241208

- 윤정
    1. 코드 리뷰
    2. VAE, DDPM 코드 합쳐보기
- 민구
    1. 코드 리뷰
    2. 활용할만한 joint training 기법 찾아보기

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
   