import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download


def deprocess(img):
    return img * 127.5 + 127.5

def train_convert(file_path, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(file_path).convert('RGB')
    return transform(img)

def create_train_iter_ds(train_dir, batch_size, img_size):
    img_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    buffer_size = len(img_paths)

    dataset = torch.utils.data.Dataset.from_iterable(
        (train_convert(p, img_size) for p in img_paths)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    print(f'Train dataset size: {buffer_size}')
    print(f'Train batches: {len(dataloader)}')
    return iter(dataloader)

def get_loss(loss):
    if loss == 'nsl':
        def discriminator_loss(real_img, fake_img):
            real_loss = torch.mean(F.softplus(-real_img))
            fake_loss = torch.mean(F.softplus(fake_img))
            return real_loss + fake_loss

        def generator_loss(fake_img):
            return torch.mean(F.softplus(-fake_img))

        return generator_loss, discriminator_loss

    elif loss == 'hinge':
        def d_real_loss(logits):
            return torch.mean(F.relu(1.0 - logits))

        def d_fake_loss(logits):
            return torch.mean(F.relu(1.0 + logits))

        def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

        def generator_loss(fake_img):
            return -torch.mean(fake_img)

        return generator_loss, discriminator_loss

class Config(object):
    def __init__(self, save_dir, input_dict=None):
        if input_dict is not None:
            for key, value in input_dict.items():
                setattr(self, key, value)
        file_path = os.path.join(save_dir, "config.json")

        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            self.save_config(file_path, save_dir)

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        print(f'Config {file_path} loaded')
        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            print(f'{key}: {value}')
            setattr(self, key, value)


class Loader(object):
    def __init__(self):
        pass
        
    def download(self, ckpt_dir):
        repo_id = 'milmor/LadaGAN'
        if ckpt_dir == 'ffhq_128':
            n_images = 24064000
        elif ckpt_dir == 'bedroom_128':
            n_images = 10624000
        elif ckpt_dir == 'celeba_64':
            n_images = 72192000
        elif ckpt_dir == 'cifar10':
            n_images = 68096000

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-{n_images}.data-00000-of-00001",
            local_dir='./'
        )

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-{n_images}.index",
            local_dir='./'
        )

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/checkpoint",
            local_dir='./')

        config_file = hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/config.json",
            local_dir='./'
        )

        with open(config_file) as f:
            self.config = json.load(f)