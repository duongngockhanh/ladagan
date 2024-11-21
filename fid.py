import numpy as np
from scipy import linalg
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import warnings

def get_activations(dataset, inception, batch_size=20):
    n_batches = len(dataset)
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048), 'float32')

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset)):
            start = i * batch_size
            end = start + batch_size
            batch = batch.cuda()
            pred = inception(batch)
            pred_arr[start:end] = pred.cpu().numpy()
            
    return pred_arr

def calculate_activation_statistics(images, model, batch_size=20):
    act = get_activations(images, model, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(real, gen, model, batch_size=20):
    if isinstance(real, list):
        m1, s1 = real
    else:   
        m1, s1 = calculate_activation_statistics(real, model, batch_size)
        
    if isinstance(gen, list):
        m2, s2 = gen
    else:  
        m2, s2 = calculate_activation_statistics(gen, model, batch_size)
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def test_convert(file_path, img_size):
    img = Image.open(file_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img)

def create_fid_ds(img_dir, batch_size, img_size, n_images, seed=42):
    from glob import glob
    import random
    
    img_paths = glob(str(img_dir))
    random.seed(seed)
    random.shuffle(img_paths)
    img_paths = img_paths[:n_images]
    
    dataset = torch.utils.data.Dataset.from_iterable(
        [test_convert(p, img_size) for p in img_paths]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )
    
    print(f'FID dataset size: {len(img_paths)} FID batches: {len(dataloader)}')
    return dataloader

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.inception = inception_v3(pretrained=True, transform_input=True)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        for param in self.inception.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Resize if needed
        if x.shape[-1] != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.inception(x)
