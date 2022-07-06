# CelebAMaskHQ training
# python train_ddpm.py +dataset=celebamaskhq128/train \
#                      dataset.ddpm.data.root='/data/kushagrap20/vaedm/reconstructions_celebahq' \
#                      dataset.ddpm.data.name='recons' \
#                      dataset.ddpm.data.norm='False' \
#                      dataset.ddpm.training.type='form2' \
#                      dataset.ddpm.training.batch_size=10 \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data/kushagrap20/ddpm_celebamaskhq_26thOct_form2_scale[01]\' \
#                      dataset.ddpm.training.restore_path=\'/data/kushagrap20/ddpm_celebamaskhq_26thOct_form2_scale[01]/checkpoints/ddpmv2-celebamaskhq_26thOct_form2_scale01-epoch=07-loss=0.0017.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix='celebamaskhq_26thOct_form2_scale01'

# AFHQ training
# python main/train_ddpm.py +dataset=afhq128/train \
#                      dataset.ddpm.data.root='/data1/kushagrap20/reconstructions/afhq_reconsv2/' \
#                      dataset.ddpm.data.name='recons' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.batch_size=12 \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/ddpm_afhq_16thDec_form1_scale[-11]\' \
#                      dataset.ddpm.training.restore_path=\'/data1/kushagrap20/ddpm_afhq_13thDec_form1_scale[-11]/checkpoints/ddpmv2-afhq_13thDec_form1_scale[-11]-epoch=402-loss=0.0045.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix=\'afhq_16thDec_form1_scale[-11]\'

# CelebA-64 training
python main/train_ddpm.py +dataset=celeba64/train \
                     dataset.ddpm.data.root='/data1/kushagrap20/datasets/img_align_celeba/' \
                     dataset.ddpm.data.name='celeba' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.ddpm.model.dim=128 \
                     dataset.ddpm.model.dropout=0.1 \
                     dataset.ddpm.model.attn_resolutions=\'16,\' \
                     dataset.ddpm.model.n_residual=2 \
                     dataset.ddpm.model.dim_mults=\'1,2,2,2,4\' \
                     dataset.ddpm.model.n_heads=8 \
                     dataset.ddpm.training.type='form1' \
                     dataset.ddpm.training.cfd_rate=0.0 \
                     dataset.ddpm.training.epochs=500 \
                     dataset.ddpm.training.z_cond=False \
                     dataset.ddpm.training.batch_size=32 \
                     dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/celeba64/vae_celeba64_alpha=1.0/checkpoints/vae-celeba64_alpha=1.0-epoch=245-train_loss=0.0000.ckpt\' \
                     dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
                     dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/\' \
                     dataset.ddpm.training.restore_path=\'/data1/kushagrap20/diffusevae_celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1/checkpoints/ddpmv2-celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1-epoch=427-loss=0.0145.ckpt\' \
                     dataset.ddpm.training.workers=1 \
                     dataset.ddpm.training.chkpt_prefix=\'celeba64_rework_form1__21stJune_sota_nheads=8_dropout=0.1\'

# # CIFAR-10 training
# python main/train_ddpm.py +dataset=cifar10/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/\' \
#                      dataset.ddpm.data.name='cifar10' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.3 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.0 \
#                      dataset.ddpm.training.epochs=2850 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3/\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3\'

# # CIFAR-10 training (with conditional dropout)
# python main/train_ddpm.py +dataset=cifar10/train \
#                      dataset.ddpm.data.root=\'/data1/kushagrap20/datasets/\' \
#                      dataset.ddpm.data.name='cifar10' \
#                      dataset.ddpm.data.norm=True \
#                      dataset.ddpm.data.hflip=True \
#                      dataset.ddpm.model.dim=128 \
#                      dataset.ddpm.model.dropout=0.3 \
#                      dataset.ddpm.model.attn_resolutions=\'16,\' \
#                      dataset.ddpm.model.n_residual=2 \
#                      dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
#                      dataset.ddpm.model.n_heads=8 \
#                      dataset.ddpm.training.type='form1' \
#                      dataset.ddpm.training.cfd_rate=0.1 \
#                      dataset.ddpm.training.epochs=2850 \
#                      dataset.ddpm.training.z_cond=False \
#                      dataset.ddpm.training.batch_size=32 \
#                      dataset.ddpm.training.vae_chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
#                      dataset.ddpm.training.device=\'gpu:0,1,2,3\' \
#                      dataset.ddpm.training.results_dir=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance/\' \
#                      dataset.ddpm.training.restore_path=\'/data1/kushagrap20/diffusevae_cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance/checkpoints/ddpmv2-cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance-epoch=2450-loss=0.0219.ckpt\' \
#                      dataset.ddpm.training.workers=2 \
#                      dataset.ddpm.training.chkpt_prefix=\'cifar10_rework_form1__17thJune_sota_nheads=8_dropout=0.3_clffree_guidance\'