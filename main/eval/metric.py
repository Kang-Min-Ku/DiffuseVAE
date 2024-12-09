from typing import Union
import os
import tempfile
from PIL import Image
from packaging import version
from .util import load_images_from_dir, CustomDataset
import torchmetrics
from torchmetrics.image.inception import InceptionScore
from cleanfid import fid
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

def FID(
    real_source: Union[torch.Tensor, str],
    gen_source: Union[torch.Tensor, str],
    batch_size = 32,
    custom_feat_extractor = None,
    custom_image_transform = None,
    num_workers = 3,
    device = "cuda",
):
    """
    This function relies entirely on off-the-shelf modules.

    real_source: torch.Tensor or str
        real images or path to the real images
        str mode is recommended because of the speed. Tensor mode makes temporary directory for the images
        in-training evaluation is enable by passing the tensor
    gen_source: torch.Tensor or str
        generated images or path to the generated images
        str mode is recommended because of the speed. Tensor mode makes temporary directory for the images
        in-training evaluation is enable by passing the tensor
    batch_size: int
        batch size for the inception network
        FID is sensitive to batch size. Tune this parameter if you get weird results
    custom_feat_extractor: nn.Module
        custom feature extractor
    custom_image_transform: nn.Module
        custom image transform
    num_workers: int
        number of workers for the data loader
    device: str
        device to use
    """
    assert torch.is_tensor(real_source) or isinstance(real_source, str), "real_source must be torch.Tensor or str"
    assert torch.is_tensor(gen_source) or isinstance(gen_source, str), "gen_source must be torch.Tensor or str"

    real_path = ""
    gen_path = ""
    real_temp_dir = None
    gen_temp_dir = None

    if torch.is_tensor(real_source):
        real_temp_dir = tempfile.TemporaryDirectory()
        real_path = real_temp_dir.name
        for i, img in enumerate(real_source):
            img = ToPILImage()(img)
            img.save(os.path.join(real_path, f"{i}.png"))
    else:
        real_path = real_source
    
    if torch.is_tensor(gen_source):
        gen_temp_dir = tempfile.TemporaryDirectory()
        gen_path = gen_temp_dir.name
        for i, img in enumerate(gen_source):
            img = ToPILImage()(img)
            img.save(os.path.join(gen_path, f"{i}.png"))
    else:
        gen_path = gen_source

    fid_score = fid.compute_fid(real_path, gen_path,
                    batch_size=batch_size,
                    custom_feat_extractor=custom_feat_extractor,
                    custom_image_tranform=custom_image_transform,
                    num_workers=num_workers,
                    device=device)

    if real_temp_dir is not None:
        real_temp_dir.cleanup()
    if gen_temp_dir is not None:
        gen_temp_dir.cleanup()

    return fid_score.item()


def IS(
    images = None,
    image_path = None,
    use_off_the_shelf = True,
    inception_network = None,
    resize = (299, 299),
    batch_size = 32,
    splits = 10,
    normalize = True,
    device = "cuda",
    image_file_extension = None,
    num_workers = 3
):
    """
    This function assumes all data is torch.FloatTensor in the range [0, 1]
    If you use images, make sure they are in the range [0, 1]
    Else if you use image_path, they will be scaled automatically

    Off-the-shelf IS and custom IS are different now X(. I recommend you to use off-the-shelf IS, if you do post-hoc evaluation!

    Now normalization transform is fitted for inception network. In other words, this is not general.
    If you want to use other datasets, you should change the mean and std

    images: torch.Tensor
        tensor with images feed to the feature extractor
    image_path: str
        path to the images
    use_off_the_shelf: bool
        whether to use off-the-shelf IS calculation module
    inception_network: nn.Module
        Inception network to use. If None, the default InceptionV3 will be used
        You can reduce time to load inception network by passing the network as an argument
        Last layer of inception network should not contain softmax layer
    resize: tuple
        size to resize the images
    batch_size: int
        batch size for the inception network
    splits: int
        number of splits for the inception score calculation
    normalize: bool
        whether to normalize the images
    device: str
        device to use
    image_file_extension: list e.g. ["pt", "ckpt"]
        image file extensions. Use it if you want to filter out files in the directory
    num_workers: int
        number of workers for the data loader
    """
    # exit condition
    assert images is not None or image_path is not None, "Either images or image_path must be provided"
    assert not (images is not None and image_path is not None), "Only one of images or image_path must be provided"
    # load images
    if image_path is not None:
        images = load_images_from_dir(image_path, image_file_extension)

    if use_off_the_shelf:
        assert version.parse(torchmetrics.__version__) >= version.parse("1.5.0"), "torchmetrics>=1.5.0 is required for off-the-shelf IS calculation"

        images = (images * 255).to(torch.uint8)

        metric = InceptionScore("logits_unbiased", splits=splits)
        IS_mean, IS_var = metric(images)
    else:
        # normalize data
        normalize_transform = []
        if normalize:
            normalize_transform = [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] # imagenet-specific normalization
        transform = Compose([
            Resize(resize),
            *normalize_transform
        ])
        dataset = CustomDataset(images, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # load inception network
        if inception_network is None:
            inception_network = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_network.eval()
        # calculate IS
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                pred = inception_network(batch)
                preds.append(pred)
        preds = torch.cat(preds, dim=0)

        prob = preds.softmax(dim=1)
        log_prob = preds.log_softmax(dim=1)

        prob = prob.chunk(splits, dim=0)
        log_prob = log_prob.chunk(splits, dim=0)

        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        IS_mean, IS_var = kl.mean().item(), kl.var().item()
    
    return IS_mean, IS_var