from eval.metric import *

if __name__ == '__main__':
    fid_score = FID(
        '/home/dj/course-diffusevae-20241212/datasets/cifar10-jpg/',
        '/home/dj/course-diffusevae-20241212/samples/ddpm-cifar10-cond-stage1-form2-50000/1000/images/',
    )
    print(fid_score)

    is_score = IS(
        None,
        '/home/dj/course-diffusevae-20241212/samples/ddpm-cifar10-cond-stage1-form2-50000/1000/images/',
        False,
    )
    print(is_score)
