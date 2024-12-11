import copy
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from models.callbacks import EMAWeightUpdate
from models.diffusion import DDPM, DDPMv2, DDPMWrapper, SuperResModel, UNetModel, JointWrapper
# from models.vae import VAE
from models.vae_joint import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="configs")
def train(config):
    # Get config_ddpm and setup
    config_ddpm = config.dataset.ddpm
    config_vae = config.dataset.vae
    logger.info(OmegaConf.to_yaml(config_ddpm))

    # Set seed
    seed_everything(config_ddpm.training.seed, workers=True)

    # Dataset
    root = config_ddpm.data.root
    d_type = config_ddpm.data.name
    image_size = config_ddpm.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config_ddpm.data.norm, flip=config_ddpm.data.hflip
    )

    N = len(dataset)
    batch_size = config_ddpm.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    lr = config_ddpm.training.lr
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)
    dim_mults = __parse_str(config_ddpm.model.dim_mults)
    ddpm_type = config_ddpm.training.type

    # Use the superres model for conditional training
    decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
        in_channels=config_ddpm.data.n_channels,
        model_channels=config_ddpm.model.dim,
        out_channels=3,
        num_res_blocks=config_ddpm.model.n_residual,
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config_ddpm.model.dropout,
        num_heads=config_ddpm.model.n_heads,
        z_dim=config_ddpm.training.z_dim,
        use_scale_shift_norm=config_ddpm.training.z_cond,
        use_z=config_ddpm.training.z_cond,
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
    )
       
    vae = VAE(
        input_res=image_size,
        enc_block_str=config_vae.model.enc_block_config,
        dec_block_str=config_vae.model.dec_block_config,
        enc_channel_str=config_vae.model.enc_channel_config,
        dec_channel_str=config_vae.model.dec_channel_config,
    )

    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config_ddpm.data.norm}")
    
    ddpm_wrapper = JointWrapper(
        online_ddpm,
        target_ddpm,
        vae,
        lr=lr,
        cfd_rate=config_ddpm.training.cfd_rate,
        n_anneal_steps=config_ddpm.training.n_anneal_steps,
        loss=config_ddpm.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config_ddpm.training.grad_clip,
        z_cond=config_ddpm.training.z_cond,
        alpha=config_vae.training.alpha,
        alpha_max=config_vae.training.alpha_max,
        alpha_anneal_method=config_vae.training.alpha_anneal_method,
        alpha_anneal_max_steps=config_vae.training.alpha_anneal_max_steps,
        beta=config_ddpm.training.beta,
        beta_max=config_ddpm.training.beta_max,
        beta_anneal_method=config_ddpm.training.beta_anneal_method,
        beta_anneal_max_steps=config_ddpm.training.beta_anneal_max_steps
    )

    # Trainer
    train_kwargs = {}
    restore_path = config_ddpm.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config_ddpm.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config_ddpm.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config_ddpm.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config_ddpm.training.epochs
    train_kwargs["log_every_n_steps"] = config_ddpm.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config_ddpm.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config_ddpm.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config_ddpm.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config_ddpm.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config_ddpm.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config_ddpm.training.grad_clip

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloader=loader)


if __name__ == "__main__":
    train()
