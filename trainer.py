"""LadaGAN model for PyTorch.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from PIL import Image
import time
import numpy as np
from diffaug import DiffAugment
from utils import deprocess
from fid import *


def l2_loss(y_true, y_pred):
    return torch.mean(torch.nn.MSELoss()(y_true, y_pred))

def reset_metrics(metrics):
    for _, metric in metrics.items():
        metric.reset()

def update_metrics(metrics, **kwargs):
    for metric_name, metric_value in kwargs.items():
        metrics[metric_name].update(metric_value)

        
class LadaGAN(object):
    def __init__(self, generator, discriminator, conf):
        super(LadaGAN, self).__init__()
        self.generator = generator
        self.ema_generator = generator
        self.discriminator = discriminator
        self.noise_dim = conf.noise_dim
        self.gp_weight = conf.gp_weight
        self.policy = conf.policy
        self.batch_size = conf.batch_size
        self.ema_decay = conf.ema_decay
        self.ema_generator = type(generator)(**generator.__dict__)
        self.ema_generator.load_state_dict(generator.state_dict())
        self.bcr = conf.bcr
        self.cr_weight = conf.cr_weight
        # init ema
        noise = torch.randn(1, conf.noise_dim)
        gen_batch = self.ema_generator(noise)

        # metrics
        self.train_metrics = {}
        self.fid_avg = AverageMeter()

    def build(self, g_optimizer, d_optimizer, g_loss, d_loss):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss = g_loss
        self.d_loss = d_loss
        self._build_metrics()

    def _build_metrics(self):
        metric_names = [
        'g_loss',
        'd_loss',
        'gp',
        'd_total',
        'real_acc',
        'fake_acc',
        'cr'
        ]
        for metric_name in metric_names:
            self.train_metrics[metric_name] = AverageMeter()

    def gradient_penalty(self, real_samples):
        batch_size = real_samples.size(0)
        real_samples.requires_grad_(True)
        logits = self.discriminator(real_samples)[0]

        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=real_samples,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        r1_penalty = torch.sum(gradients.pow(2), dim=1).mean() * self.gp_weight
        return logits, r1_penalty

    def train_step(self, real_images):
        noise = torch.randn(self.batch_size, self.noise_dim)
        
        # train discriminator
        self.d_optimizer.zero_grad()
        if self.bcr:
            fake_images = self.generator(noise)[0]
            fake_logits = self.discriminator(fake_images)[0]
            real_logits, gp = self.gradient_penalty(real_images)
            real_augmented_images = DiffAugment(real_images, policy=self.policy)
            fake_augmented_images = DiffAugment(fake_images.detach(), policy=self.policy)
            real_augmented_images = real_augmented_images.detach()
            fake_augmented_images = fake_augmented_images.detach()
            
            augmented_images = torch.cat((real_augmented_images, fake_augmented_images), dim=0)
            augmented_logits = self.discriminator(augmented_images)[0]
            real_augmented_logits, fake_augmented_logits = torch.split(augmented_logits, self.batch_size)
            consistency_loss = self.cr_weight * (
                l2_loss(real_logits, real_augmented_logits) +
                l2_loss(fake_logits, fake_augmented_logits))
            
            d_loss = self.d_loss(real_logits, fake_logits)
            d_total = d_loss + gp + consistency_loss
            
        else:
            fake_images = self.generator(noise)[0]
            fake_augmented_images = DiffAugment(fake_images.detach(), policy=self.policy)
            real_augmented_images = DiffAugment(real_images, policy=self.policy)
            fake_logits = self.discriminator(fake_augmented_images)[0]
            real_logits, gp = self.gradient_penalty(real_augmented_images)
            
            d_loss = self.d_loss(real_logits, fake_logits)
            d_total = d_loss + gp
            consistency_loss = torch.tensor(0.0)

        d_total.backward()
        self.d_optimizer.step()

        # train generator
        self.g_optimizer.zero_grad()
        noise = torch.randn(self.batch_size, self.noise_dim)
        
        if self.bcr:
            fake_images = self.generator(noise)[0]
            fake_logits = self.discriminator(fake_images)[0]
            g_loss = self.g_loss(fake_logits)
        else:
            fake_images = self.generator(noise)[0]
            fake_augmented_images = DiffAugment(fake_images, policy=self.policy)
            fake_logits = self.discriminator(fake_augmented_images)[0]
            g_loss = self.g_loss(fake_logits)
            
        g_loss.backward()
        self.g_optimizer.step()

        # update EMA generator weights
        with torch.no_grad():
            for ema_param, param in zip(self.ema_generator.parameters(), self.generator.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                
        update_metrics(
            self.train_metrics,
            g_loss=g_loss.item(),
            d_loss=d_loss.item(), 
            gp=gp.item(),
            d_total=d_total.item(),
            real_acc=real_logits.mean().item(),
            fake_acc=fake_logits.mean().item(),
            cr=consistency_loss.item()
        )

    def create_ckpt(self, model_dir, max_ckpt_to_keep, restore_best=True):
        # log dir
        self.model_dir = model_dir
        log_dir = os.path.join(model_dir, 'log-dir')
        os.makedirs(log_dir, exist_ok=True)
        
        # checkpoint dir
        checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
        best_checkpoint_dir = os.path.join(model_dir, 'best-training-checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(best_checkpoint_dir, exist_ok=True)

        self.n_images = 0
        self.fid = 10000.0  # initialize with big value
        self.best_fid = 10000.0  # initialize with big value
        
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint_dir = best_checkpoint_dir
        self.max_ckpt_to_keep = max_ckpt_to_keep

        if restore_best and os.path.exists(os.path.join(best_checkpoint_dir, 'latest.pt')):
            self.load_checkpoint(best=True)
            print(f'Best checkpoint restored from {best_checkpoint_dir}')
        elif not restore_best and os.path.exists(os.path.join(checkpoint_dir, 'latest.pt')):
            self.load_checkpoint(best=False)
            print(f'Checkpoint restored from {checkpoint_dir}')
        else:
            print(f'Checkpoint created at {model_dir} dir')

    def save_checkpoint(self, n_images, is_best=False):
        save_dir = self.best_checkpoint_dir if is_best else self.checkpoint_dir
        
        checkpoint = {
            'generator_state': self.generator.state_dict(),
            'ema_generator_state': self.ema_generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'n_images': n_images,
            'fid': self.fid,
            'best_fid': self.best_fid
        }

        torch.save(checkpoint, os.path.join(save_dir, f'ckpt-{n_images}.pt'))
        torch.save(checkpoint, os.path.join(save_dir, 'latest.pt'))

        # Remove old checkpoints if exceeding max_to_keep
        checkpoints = sorted([f for f in os.listdir(save_dir) if f.startswith('ckpt-')])
        if len(checkpoints) > self.max_ckpt_to_keep:
            os.remove(os.path.join(save_dir, checkpoints[0]))

    def load_checkpoint(self, best=False):
        load_dir = self.best_checkpoint_dir if best else self.checkpoint_dir
        checkpoint = torch.load(os.path.join(load_dir, 'latest.pt'))
        
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.ema_generator.load_state_dict(checkpoint['ema_generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
        self.n_images = checkpoint['n_images']
        self.fid = checkpoint['fid']
        self.best_fid = checkpoint['best_fid']

    def restore_generator(self, model_dir):
        self.model_dir = model_dir
        best_checkpoint_dir = os.path.join(model_dir, 'best-training-checkpoints')
        
        checkpoint = torch.load(os.path.join(best_checkpoint_dir, 'latest.pt'))
        self.ema_generator.load_state_dict(checkpoint['ema_generator_state'])
        self.n_images = checkpoint['n_images']
        print(f'Best checkpoint restored from {best_checkpoint_dir}')

    def save_metrics(self, n_images):
        # Print metrics
        for name, metric in self.train_metrics.items():
            print(f'{name}: {metric.avg:.4f} -', end=" ")
            
        # Reset metrics
        reset_metrics(self.train_metrics)

    def save_ckpt(self, n_images, n_fid_images, fid_batch_size, test_dir, img_size):
        # Calculate FID
        fid = self.fid_score(n_fid_images, fid_batch_size, test_dir, img_size)
        self.fid_avg.update(fid)
        self.fid = fid

        start = time.time()
        if fid < self.best_fid:
            self.best_fid = fid
            self.save_checkpoint(n_images, is_best=True)
            self.save_checkpoint(n_images, is_best=False)
            print(f'FID improved. Best checkpoint saved at {n_images} images')
        else:
            self.save_checkpoint(n_images, is_best=False)
            print(f'Checkpoint saved at {n_images} images')
        print(f'Time for ckpt is {time.time()-start:.4f} sec')

        # Reset metrics
        self.fid_avg.reset()

    def gen_batches(self, n_images, batch_size, dir_path):
        n_batches = n_images // batch_size
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            noise = torch.randn(batch_size, self.noise_dim)
            with torch.no_grad():
                gen_batch = self.ema_generator(noise)[0]
            gen_batch = np.clip(deprocess(gen_batch.cpu().numpy()), 0.0, 255)

            img_index = start
            for img in gen_batch:
                img = Image.fromarray(img.astype('uint8'))
                file_name = os.path.join(dir_path, f'{str(img_index)}.png')
                img.save(file_name, "PNG")
                img_index += 1

    def fid_score(self, n_fid_images, batch_size, test_dir, img_size):
        inception = Inception()
        fid_dir = os.path.join(self.model_dir, 'fid')
        os.makedirs(fid_dir, exist_ok=True)

        start = time.time()
        print('\nGenerating FID images...')
        self.gen_batches(n_fid_images, batch_size, fid_dir)
        gen_fid_ds = create_fid_ds(
            fid_dir + '/*.png', batch_size, img_size, n_fid_images
        )
        real_fid_ds = create_fid_ds(
            test_dir, batch_size, img_size, n_fid_images
        )
        m_gen, s_gen = calculate_activation_statistics(
            gen_fid_ds, inception, batch_size
        )
        m_real, s_real = calculate_activation_statistics(
            real_fid_ds, inception, batch_size
        )
        fid = calculate_frechet_distance(m_real, s_real, m_gen, s_gen)
        print(f'FID: {fid:.4f} - Time for FID score is {time.time()-start:.4f} sec')
        return fid


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count