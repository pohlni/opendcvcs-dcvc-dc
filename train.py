"""
DCVC-DC Transform Training Script

This script trains the DCVC-DC video compression model using a 4-stage process:
  Stage 1: Warm up MV generation (motion estimation network)
  Stage 2: Train other modules with MV frozen
  Stage 3: Train with bit cost, MV still frozen
  Stage 4: End-to-end training (all modules unfrozen)

Training uses Vimeo-90k dataset with precomputed I-frame references.
Evaluation uses UVG dataset.
"""

###############################################################################
#                                 IMPORTS                                      #
###############################################################################

import os
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from timm.utils import unwrap_model
from collections import OrderedDict

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.models.video_model_transform import DMC
from src.models.image_model import IntraNoAR


###############################################################################
#                                CONSTANTS                                     #
###############################################################################

LAMBDA_SET = [85, 170, 380, 840]
STAGE_DESCRIPTIONS = {
    1: "Warm up MV generation part",
    2: "Train other modules",
    3: "Train with bit cost",
    4: "End-to-end training",
    5: "Finetuning (4-frame GOP)",
}

# Distortion weights for stage 5 finetuning (applied per frame position 0-3)
FINETUNE_DISTORTION_WEIGHTS_STAGE5 = [0.5, 1.2, 0.5, 0.9]

# MV-related modules and parameters for freezing/unfreezing
MV_MODULES = [
    'optic_flow', 'mv_encoder', 'mv_decoder',
    'mv_hyper_prior_encoder', 'mv_hyper_prior_decoder',
    'bit_estimator_z_mv', 'mv_y_spatial_prior',
    'mv_y_spatial_prior_adaptor_1', 'mv_y_spatial_prior_adaptor_2',
    'mv_y_spatial_prior_adaptor_3', 'mv_y_prior_fusion_adaptor_0',
    'mv_y_prior_fusion_adaptor_1', 'mv_y_prior_fusion'
]
MV_PARAMS = ['mv_y_q_basic_enc', 'mv_y_q_scale_enc', 'mv_y_q_basic_dec', 'mv_y_q_scale_dec']

# Deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###############################################################################
#                       DISTRIBUTED TRAINING UTILITIES                         #
###############################################################################

def setup_distributed():
    """
    Initialize distributed training if running with torchrun.

    Detects distributed environment via RANK/LOCAL_RANK environment variables
    set by torchrun. Initializes NCCL backend for GPU communication.

    Returns:
        bool: True if distributed training is initialized, False otherwise.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set the device for this process BEFORE init_process_group
        torch.cuda.set_device(local_rank)

        # Initialize the process group with device_id to avoid warnings
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f'cuda:{local_rank}')
        )

        if rank == 0:
            print(f"Distributed training initialized: world_size={world_size}")

        return True
    return False


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    """Check if distributed training is active."""
    return dist.is_initialized()


def get_rank():
    """Get the rank of the current process (0 if not distributed)."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """Get the local rank of the current process (0 if not distributed)."""
    if not dist.is_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size():
    """Get the total number of processes (1 if not distributed)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def check_model_for_nan(model, model_name="model", verbose=False):
    """Check model parameters for NaN or Inf values.

    Returns list of corrupted parameter names.
    """
    corrupted = []
    for name, param in model.named_parameters():
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        if has_nan or has_inf:
            corrupted.append(name)
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()
            total = param.numel()
            print(f"[CORRUPTED] {model_name}.{name}: "
                  f"nan={nan_count}/{total} ({100*nan_count/total:.1f}%), "
                  f"inf={inf_count}/{total} ({100*inf_count/total:.1f}%)", flush=True)

    if corrupted:
        print(f"\n[WARNING] {model_name} has {len(corrupted)} corrupted parameter(s)!", flush=True)
    elif verbose and is_main_process():
        print(f"[OK] {model_name} has no NaN/Inf in weights", flush=True)

    return corrupted


def reduce_dict(input_dict, average=True):
    """
    Reduce a dictionary of tensors/values across all processes.

    Args:
        input_dict: Dictionary with numeric values to reduce.
        average: If True, average the values; otherwise sum them.

    Returns:
        Dictionary with reduced values (on all processes).
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        # Determine device from input tensors or use local rank's GPU
        device = None
        for v in input_dict.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
        if device is None:
            # Fallback to local rank's GPU in distributed mode
            device = torch.device(f'cuda:{get_local_rank()}')

        names = []
        values = []
        for k, v in sorted(input_dict.items()):
            names.append(k)
            # Convert to tensor if needed
            if isinstance(v, torch.Tensor):
                values.append(v.detach().to(device))
            else:
                values.append(torch.tensor(v, device=device, dtype=torch.float32))

        # Stack all values and reduce
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        # Convert back to dictionary
        reduced_dict = {k: v.item() for k, v in zip(names, values)}

    return reduced_dict


def barrier():
    """Synchronization barrier across all processes."""
    if is_distributed():
        dist.barrier()


###############################################################################
#                             HELPER FUNCTIONS                                 #
###############################################################################

def lmb2qindex(lmbda):
    """Convert lambda value to quality index. lmbda [85, 170, 380, 840] --> [0, 1, 2, 3]"""
    if lmbda not in LAMBDA_SET:
        raise ValueError(f"lmbda should be in {LAMBDA_SET}")
    return LAMBDA_SET.index(lmbda)


def mse_to_psnr(mse):
    """Convert MSE to PSNR in dB."""
    return -10 * math.log10(max(mse, 1e-10))


def set_mv_requires_grad(model, requires_grad):
    """Set requires_grad for all MV-related modules and parameters."""
    for name in MV_MODULES:
        for param in getattr(model, name).parameters():
            param.requires_grad = requires_grad
    for name in MV_PARAMS:
        getattr(model, name).requires_grad = requires_grad


def make_dpb(ref_frame, ref_feature=None, ref_y=None, ref_mv_feature=None, ref_mv_y=None):
    """Create a decoded picture buffer (dpb) dictionary for temporal prediction."""
    return {
        "ref_frame": ref_frame,
        "ref_feature": ref_feature,
        "ref_y": ref_y,
        "ref_mv_feature": ref_mv_feature,
        "ref_mv_y": ref_mv_y,
    }


###############################################################################
#                                 DATASETS                                     #
###############################################################################

class Vimeo90kGOPDataset(torch.utils.data.Dataset):
    """
    Dataset for Vimeo-90k septuplets with precomputed I-frame references.

    Returns 6-frame sequences (frames 2-7) with corresponding precomputed
    reference frames for each quality level. Used for training.
    """
    def __init__(self, root_dir, precomputed_dir, septuplet_list, transform=None, crop_size=256, gop_size=7, shuffle_frames=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            septuplet_list (string): Path to the file with list of septuplets.
            transform (callable, optional): Optional transform to be applied on a sample.
            crop_size (int): Size of the random crop.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.shuffle_frames = shuffle_frames
        with open(septuplet_list, 'r') as f:
            self.septuplet_list = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.septuplet_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        septuplet_name = self.septuplet_list[idx]
        frames = []
        precomputed_frames = []

        # Load frames
        for i in range(2, 8):  # Vimeo-90k septuplet has 7 frames Frames 2-7 (to be compressed as P-frames)
            img_name = os.path.join(self.root_dir, septuplet_name, f'im{i}.png')
            image = Image.open(img_name).convert('RGB')
            frames.append(image)

        # Load precomputed frames
        for i in range(1, 7):  # Vimeo-90k septuplet has 7 frames Reference frames 1-6
            precomputed_images = {
                str(j): Image.open(os.path.join(self.precomputed_dir, str(j), septuplet_name, f'ref{i}.png')).convert('RGB')
                for j in range(4)
            }
            precomputed_frames.append(precomputed_images)

        # Apply random crop to the same location for all frames
        if self.crop_size:
            width, height = frames[0].size
            if width >= self.crop_size and height >= self.crop_size:
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in frames]
                precomputed_frames = [
                    {k: img.crop((x, y, x + self.crop_size, y + self.crop_size)) for k, img in v.items()}
                    for v in precomputed_frames
                ]

        if random.random() < 0.5:  # 50% chance to flip horizontally
            frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]
            precomputed_frames = [{k: img.transpose(Image.FLIP_LEFT_RIGHT) for k, img in v.items()} for v in precomputed_frames]

        if random.random() < 0.5:  # 50% chance to flip vertically
            frames = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in frames]
            precomputed_frames = [{k: img.transpose(Image.FLIP_TOP_BOTTOM) for k, img in v.items()} for v in precomputed_frames]


        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]
            precomputed_frames = [
                {k: self.transform(img) for k, img in v.items()} for v in precomputed_frames
            ]

        # Random shuffle frame order if enabled
        if self.shuffle_frames:
            # Create a list of indices and shuffle it
            frame_indices = list(range(len(frames)))
            random.shuffle(frame_indices)
            
            # Reorder frames according to shuffled indices
            frames = [frames[i] for i in frame_indices]
            precomputed_frames = [precomputed_frames[i] for i in frame_indices]
        
        # stack precomputed_frames quality elements
        precomputed_frames = [
            torch.stack([v[k] for k in sorted(v.keys())]) for v in precomputed_frames
        ]


        return torch.stack(frames), torch.stack(precomputed_frames)  # Return frames as a single tensor [S, C, H, W], [S, Q, C, H, W]

class UVGGOPDataset(torch.utils.data.Dataset):
    """
    Dataset for UVG test videos split into GOP sequences.

    Returns frame sequences of specified GOP size for evaluation.
    Supports 7 standard UVG 1080p videos.
    """

    def __init__(self, root_dir, transform=None, gop_size=12):
        self.root_dir = root_dir
        self.transform = transform
        self.gop_size = gop_size
        self.video_sequences = []

        # UVG videos
        video_names = [
            'Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV', 
            'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV', 
            'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV', 
            'YachtRide_1920x1024_120fps_420_8bit_YUV'
        ]

        # Get sequences of frames for each video
        for video_name in video_names:
            video_dir = os.path.join(root_dir, video_name)
            if os.path.isdir(video_dir):
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])

                # Divide frames into sequences of length gop_size (or maximum available)
                for i in range(0, len(frames), gop_size):
                    seq_frames = frames[i:min(i + gop_size, len(frames))]
                    if len(seq_frames) >= 2:  # Need at least 2 frames for P-frame training
                        self.video_sequences.append({
                            'video': video_name,
                            'frames': seq_frames
                        })

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.video_sequences[idx]
        video_name = sequence['video']
        frame_names = sequence['frames']

        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(self.root_dir, video_name, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        return torch.stack(frames)  # Return frames as a single tensor [S, C, H, W]


###############################################################################
#                            TRAINING FUNCTIONS                                #
###############################################################################

def train_one_epoch_fully_batched(model, i_frame_model, train_loader, optimizer, device, stage, epoch, grad_clip_max_norm=None, warmup=False, train_sampler=None):
    """
    Train for one epoch with fully batched GOP sequence processing.

    - Warmup mode: I->P only (single P-frame)
    - Normal mode: I->P->P (two P-frames)

    Args:
        model: The DCVC video compression model
        i_frame_model: Intra-frame model for I-frame encoding (unused in training, precomputed refs used)
        train_loader: DataLoader for Vimeo90k training data
        optimizer: PyTorch optimizer
        device: CUDA device
        stage: Training stage (1-4), controls which modules are frozen
        epoch: Current epoch number for logging
        grad_clip_max_norm: Maximum gradient norm for clipping
        warmup: If True, train I->P only; if False, train I->P->P
        train_sampler: DistributedSampler for distributed training (optional)

    Returns:
        dict: Training metrics (loss, mse, psnr, bpp)
    """
    # Set epoch on distributed sampler for proper shuffling
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0

    # Control parameter freezing based on stage - freeze MV in stages 2,3, unfreeze in 1,4
    # Use unwrap_model to handle both DDP and non-DDP cases
    set_mv_requires_grad(unwrap_model(model), stage not in [2, 3])

    # Process batches of GOP sequences (progress bar only on main process)
    if is_main_process():
        progress_bar = tqdm(train_loader)
    else:
        progress_bar = train_loader
    
    for batch_idx, (batch_frames, batch_precomputed_frames) in enumerate(progress_bar):
        batch_size = batch_frames.size(0)
        seq_length = batch_frames.size(1)
        batch_loss = 0
        num_positions = seq_length - 1

        # Zero gradients once per batch (accumulate across frame positions)
        optimizer.zero_grad()

        # Process each frame position in the sequence (gradients accumulate)
        for frame_pos in range(num_positions):
            current_frames = batch_frames[:, frame_pos, :, :, :].to(device)
            next_frames = batch_frames[:, frame_pos + 1, :, :, :].to(device)

            # Random lambda per frame position
            lmbda = random.choice(LAMBDA_SET)
            q_index = lmb2qindex(lmbda)
            reference_frames = batch_precomputed_frames[:, frame_pos, q_index, :, :, :].to(device)
            dpb = make_dpb(reference_frames)

            result = model(current_frames, dpb, q_in_ckpt=True, q_index=q_index,
                          frame_idx=1, stage=stage, lmbda=lmbda)
            dpb = result["dpb"]

            if warmup:
                loss = result["loss"]
                mse_loss = result["mse_loss"]
            else:
                if stage == 1:
                    dpb["ref_frame"] = result["pixel_rec"]
                    dpb["ref_feature"] = None
                    dpb["ref_y"] = None
                frame_idx = random.choice([0, 1, 2, 3])
                result_next = model(next_frames, dpb, q_in_ckpt=True, q_index=q_index,
                                   frame_idx=frame_idx, stage=stage, lmbda=lmbda)
                loss = (result["loss"] + result_next["loss"]) / 2
                mse_loss = (result["mse_loss"] + result_next["mse_loss"]) / 2

            # Normalize loss by number of frame positions so accumulated gradient
            # is equivalent to the average across positions
            (loss / num_positions).backward()

            # Collect statistics (inside loop, just logging)
            if warmup:
                mse_val = result["mse_loss"].detach().item()
                loss_val = result["loss"].detach().item()
                bpp_val = result["bpp_train"].detach().item()
                batch_loss += loss_val * batch_size
                total_mse += mse_val * batch_size
                total_bpp += bpp_val * batch_size
                total_psnr += mse_to_psnr(mse_val) * batch_size
                n_frames += batch_size
            else:
                mse1 = result["mse_loss"].detach().item()
                mse2 = result_next["mse_loss"].detach().item()
                loss_val = (result["loss"].detach().item() + result_next["loss"].detach().item()) / 2
                mse_val = (mse1 + mse2) / 2
                bpp_val = (result["bpp_train"].detach().item() + result_next["bpp_train"].detach().item()) / 2
                avg_psnr = (mse_to_psnr(mse1) + mse_to_psnr(mse2)) / 2

                batch_loss += loss_val * batch_size
                total_mse += mse_val * batch_size
                total_bpp += bpp_val * batch_size
                total_psnr += avg_psnr * batch_size
                n_frames += batch_size

        # Monitor accumulated gradient norm BEFORE clipping
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # Log if gradient norm is high
        grad_spike = total_grad_norm > 10.0
        if grad_spike:
            print(f"\n[GRAD SPIKE] Batch {batch_idx}: "
                  f"grad_norm={total_grad_norm:.2f}", flush=True)

        # Apply gradient clipping if specified
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

        # Single optimizer step per batch (gradients accumulated across frame positions)
        optimizer.step()

        # Check weights for NaN: every 10 batches OR immediately after gradient spike
        if batch_idx % 10 == 0 or grad_spike:
            corrupted = check_model_for_nan(unwrap_model(model), f"batch_{batch_idx}")
            if corrupted:
                print(f"\n[FATAL] Weights corrupted at batch {batch_idx} (after grad_norm={total_grad_norm:.2f})!", flush=True)
                raise RuntimeError(f"Weights corrupted at batch {batch_idx}")

        # Update total loss
        total_loss += batch_loss

        # Update progress bar with both batch and rolling average stats
        if n_frames > 0 and is_main_process():
            # Current batch stats
            batch_psnr = mse_to_psnr(mse_val)
            progress_bar.set_description(
                f"E{epoch} S{stage} | "
                f"Batch[L:{loss_val:.4f} MSE:{mse_val:.6f} BPP:{bpp_val:.4f} PSNR:{batch_psnr:.2f}] "
                f"Avg[L:{total_loss / n_frames:.4f} MSE:{total_mse / n_frames:.6f} BPP:{total_bpp / n_frames:.4f} PSNR:{total_psnr / n_frames:.2f}]"
            )
        
    # Calculate epoch statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        # FIX Issue #7: Use accumulated per-frame PSNR average (standard in video compression)
        # instead of recalculating from avg_mse (mathematically different)
        avg_psnr = total_psnr / n_frames
        avg_bpp = total_bpp / n_frames

    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0

    # Synchronize metrics across all processes in distributed training
    metrics = {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp,
    }
    metrics = reduce_dict(metrics)

    return metrics


def train_one_epoch_finetune(model, train_loader, optimizer, device, epoch,
                              grad_clip_max_norm=None, train_sampler=None):
    """
    Stage 5 finetuning: Multi-frame GOP with distortion weighting.

    4 frames, random start (0-2), weights [0.5, 1.2, 0.5, 0.9].

    Per batch:
    1. Pick start frame (random in 0-2)
    2. Pick random lambda
    3. First frame: use precomputed I-frame reference
    4. Process frames sequentially, accumulate weighted loss
    5. Backward and update
    """
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    set_mv_requires_grad(unwrap_model(model), True)  # Unfreezes MV (like stage 4)

    num_frames = 4
    distortion_weights = FINETUNE_DISTORTION_WEIGHTS_STAGE5

    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0

    if is_main_process():
        progress_bar = tqdm(train_loader)
    else:
        progress_bar = train_loader

    for batch_idx, (batch_frames, batch_precomputed_frames) in enumerate(progress_bar):
        batch_size = batch_frames.size(0)
        seq_length = batch_frames.size(1)  # Should be 6

        optimizer.zero_grad()
        batch_loss = 0
        batch_mse = 0
        batch_bpp = 0
        batch_psnr = 0

        # Start frame: random in 0-2
        start_frame = random.choice([0, 1, 2])

        # Random lambda for this batch
        lmbda = random.choice(LAMBDA_SET)
        q_index = lmb2qindex(lmbda)

        # Get I-frame reference from precomputed
        reference_frames = batch_precomputed_frames[:, start_frame, q_index, :, :, :].to(device)
        dpb = make_dpb(reference_frames)

        # Process P-frames
        for i in range(num_frames):
            frame_pos = start_frame + i
            current_frames = batch_frames[:, frame_pos, :, :, :].to(device)

            frame_idx = (i + 1) % 4  # frame_idx cycles 1,2,3,0,1,2,...
            result = model(current_frames, dpb, q_in_ckpt=True, q_index=q_index,
                          frame_idx=frame_idx, stage=4, lmbda=lmbda)

            # Update dpb for next frame
            dpb = result["dpb"]

            # Weighted loss: bpp + distortion * weight
            weight = distortion_weights[i]
            frame_loss = result["bpp_train"] + result["distortion"] * weight
            batch_loss += frame_loss

            # Metrics
            mse = result["mse_loss"].detach().item()
            batch_mse += mse
            batch_bpp += result["bpp_train"].detach().item()
            batch_psnr += mse_to_psnr(mse)

        # Average over frames and backward
        batch_loss = batch_loss / num_frames
        batch_loss.backward()

        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

        optimizer.step()

        # Accumulate metrics
        total_loss += batch_loss.item() * batch_size
        total_mse += (batch_mse / num_frames) * batch_size
        total_bpp += (batch_bpp / num_frames) * batch_size
        total_psnr += (batch_psnr / num_frames) * batch_size
        n_frames += batch_size

        if is_main_process() and n_frames > 0:
            progress_bar.set_description(
                f"E{epoch} S{stage} | "
                f"L:{total_loss/n_frames:.4f} MSE:{total_mse/n_frames:.6f} "
                f"BPP:{total_bpp/n_frames:.4f} PSNR:{total_psnr/n_frames:.2f}"
            )

    metrics = {"loss": total_loss/n_frames if n_frames > 0 else 0,
               "mse": total_mse/n_frames if n_frames > 0 else 0,
               "psnr": total_psnr/n_frames if n_frames > 0 else 0,
               "bpp": total_bpp/n_frames if n_frames > 0 else 0}
    return reduce_dict(metrics)


###############################################################################
#                           EVALUATION FUNCTIONS                               #
###############################################################################

def evaluate(model, i_frame_model, test_loader, device, stage, warmup=False):
    """
    Evaluate model matching training behavior.

    - Warmup mode: Evaluate on I->P (single P-frame)
    - Normal mode: Evaluate on I->P->P (two P-frames)

    Args:
        model: The DCVC video compression model
        i_frame_model: Intra-frame model for I-frame encoding
        test_loader: DataLoader for UVG test data
        device: CUDA device
        stage: Training stage (affects dpb handling)
        warmup: If True, evaluate I->P only; if False, evaluate I->P->P

    Returns:
        OrderedDict: Results per quality level and overall average loss
    """
    model.eval()
    results = OrderedDict()

    for quality_index in range(len(LAMBDA_SET)):
        if is_main_process():
            print(f"Evaluating quality index {quality_index} with lmbda {LAMBDA_SET[quality_index]}")
        total_loss = 0
        total_mse = 0
        total_bpp = 0
        total_psnr = 0
        n_frames = 0

        with torch.no_grad():
            for batch_frames in test_loader:
                batch_size = batch_frames.size(0)
                seq_length = batch_frames.size(1)

                # Determine frame range based on mode
                if warmup:
                    # I->P only: skip first frame (need previous as I-frame)
                    frame_range = range(1, seq_length)
                else:
                    # I->P->P: skip first and last frames
                    frame_range = range(1, seq_length - 1)

                for frame_pos in frame_range:
                    previous_frames = batch_frames[:, frame_pos - 1, :, :, :].to(device)
                    current_frames = batch_frames[:, frame_pos, :, :, :].to(device)

                    # Process previous frame as I-frame
                    i_frame_results = i_frame_model.encode_decode(
                        previous_frames, q_in_ckpt=True, q_index=quality_index
                    )
                    reference_frames = i_frame_results["x_hat"]
                    dpb = make_dpb(reference_frames)

                    # First P-frame (I->P)
                    IP_result = model(current_frames, dpb, q_in_ckpt=True, q_index=quality_index,
                                     frame_idx=1, stage=stage, lmbda=LAMBDA_SET[quality_index])

                    if warmup:
                        # Warmup: only I->P, collect stats for single P-frame
                        mse = IP_result["mse_loss"].item()
                        total_loss += IP_result["loss"].item() * batch_size
                        total_mse += mse * batch_size
                        total_bpp += IP_result["bpp_train"].item() * batch_size
                        total_psnr += mse_to_psnr(mse) * batch_size
                        n_frames += batch_size
                    else:
                        # Normal: I->P->P
                        dpb = IP_result["dpb"]
                        if stage == 1:
                            dpb["ref_frame"] = IP_result["pixel_rec"]
                            dpb["ref_feature"] = None
                            dpb["ref_y"] = None

                        next_frames = batch_frames[:, frame_pos + 1, :, :, :].to(device)
                        PP_result = model(next_frames, dpb, q_in_ckpt=True, q_index=quality_index,
                                         frame_idx=2, stage=stage, lmbda=LAMBDA_SET[quality_index])

                        # Average both P-frames
                        mse_IP = IP_result["mse_loss"].item()
                        mse_PP = PP_result["mse_loss"].item()
                        total_loss += (IP_result["loss"].item() + PP_result["loss"].item()) / 2 * batch_size
                        total_mse += (mse_IP + mse_PP) / 2 * batch_size
                        total_bpp += (IP_result["bpp_train"].item() + PP_result["bpp_train"].item()) / 2 * batch_size
                        total_psnr += (mse_to_psnr(mse_IP) + mse_to_psnr(mse_PP)) / 2 * batch_size
                        n_frames += batch_size

        # Calculate average statistics
        if n_frames > 0:
            avg_loss = total_loss / n_frames
            avg_mse = total_mse / n_frames
            avg_psnr = total_psnr / n_frames
            avg_bpp = total_bpp / n_frames
        else:
            avg_loss = avg_mse = avg_psnr = avg_bpp = 0

        # Synchronize metrics across all processes
        quality_metrics = reduce_dict({
            "loss": avg_loss,
            "mse": avg_mse,
            "psnr": avg_psnr,
            "bpp": avg_bpp
        })

        results[quality_index] = quality_metrics

    # Average loss across all quality levels
    results["loss"] = sum(results[i]["loss"] for i in range(len(LAMBDA_SET))) / len(LAMBDA_SET)
    return results


def evaluate_finetune(model, i_frame_model, test_loader, device):
    """
    Stage 5 evaluation: Full GOP (I-frame + all P-frames in sequence).
    """
    model.eval()
    results = OrderedDict()

    for quality_index in range(len(LAMBDA_SET)):
        if is_main_process():
            print(f"Evaluating quality {quality_index}, lambda={LAMBDA_SET[quality_index]}")

        total_loss = 0
        total_mse = 0
        total_bpp = 0
        total_psnr = 0
        n_frames = 0

        with torch.no_grad():
            for batch_frames in test_loader:
                batch_size = batch_frames.size(0)
                seq_length = batch_frames.size(1)

                # I-frame (position 0)
                i_frames = batch_frames[:, 0, :, :, :].to(device)
                i_result = i_frame_model.encode_decode(i_frames, q_in_ckpt=True, q_index=quality_index)
                dpb = make_dpb(i_result["x_hat"])

                # P-frames (positions 1 onwards)
                for frame_pos in range(1, seq_length):
                    current_frames = batch_frames[:, frame_pos, :, :, :].to(device)

                    result = model(current_frames, dpb, q_in_ckpt=True, q_index=quality_index,
                                  frame_idx=frame_pos % 4, stage=4, lmbda=LAMBDA_SET[quality_index])
                    dpb = result["dpb"]

                    mse = result["mse_loss"].item()
                    total_loss += result["loss"].item() * batch_size
                    total_mse += mse * batch_size
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += mse_to_psnr(mse) * batch_size
                    n_frames += batch_size

        if n_frames > 0:
            quality_metrics = reduce_dict({
                "loss": total_loss / n_frames,
                "mse": total_mse / n_frames,
                "psnr": total_psnr / n_frames,
                "bpp": total_bpp / n_frames
            })
        else:
            quality_metrics = {"loss": 0, "mse": 0, "psnr": 0, "bpp": 0}

        results[quality_index] = quality_metrics

    results["loss"] = sum(results[i]["loss"] for i in range(len(LAMBDA_SET))) / len(LAMBDA_SET)
    return results


###############################################################################
#                                   MAIN                                       #
###############################################################################

def main():
    """
    Main training entry point for DCVC-DC transform model.

    Handles 4-stage training process:
    - Stage 1: Warm up MV generation (motion estimation)
    - Stage 2: Train other modules (MV frozen)
    - Stage 3: Train with bit cost (MV frozen)
    - Stage 4: End-to-end training (all modules)

    Supports resuming from checkpoints, learning rate scheduling,
    and gradient clipping.
    """
    parser = argparse.ArgumentParser(description='DCVC Training with Full Batch Processing')
    parser.add_argument('--vimeo_dir', type=str, required=True, help='Path to Vimeo-90k dataset')
    parser.add_argument('--precomputed_dir', type=str, required=True, help='Path to precomputed directory')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_transform', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs_transform', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4, 5], help='Training stage (1-5)')
    parser.add_argument('--finetune_lr', type=float, default=4e-5,
                        help='Learning rate for stage 5 finetuning')
    parser.add_argument('--eval_before_train', action='store_true',
                        help='Run evaluation before training starts (useful to check loaded checkpoint)')
    parser.add_argument('--no_shuffle_frames', action='store_true',
                        help='Disable frame shuffling in training dataset (use temporal order)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='1', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='psnr', choices=['psnr', 'ms-ssim'],
                        help='Model type: psnr or ms-ssim')
    parser.add_argument('--seed', type=int, default=35, help='Random seed')
    parser.add_argument('--previous_stage_checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous stage to resume from')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    
    # Add arguments for resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of epochs to run in warmup mode (I->P only). After this, switches to normal (I->P->P).')

    # Add learning rate scheduler arguments
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Multiplicative factor for learning rate reduction')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='Patience for ReduceLROnPlateau scheduler')

    # torch.compile
    parser.add_argument('--compile', action='store_true', help='Compile the model')

    # Add gradient clipping parameter
    parser.add_argument('--grad_clip_max_norm', type=float, default=2.0,
                        help='Maximum norm for gradient clipping (None to disable clipping)')

    # Distributed training arguments
    parser.add_argument('--scale_lr', action='store_true',
                        help='Scale learning rate linearly by world_size for distributed training')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Convert BatchNorm layers to SyncBatchNorm for distributed training')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='Enable find_unused_parameters in DDP (use when some params are unused)')

    args = parser.parse_args()

    # Initialize distributed training if running with torchrun
    distributed = setup_distributed()

    # Set random seed for full reproducibility (add rank offset for distributed training)
    seed = args.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set CUDA devices
    if distributed:
        # In distributed mode, use local rank for device
        device = torch.device(f'cuda:{get_local_rank()}')
    elif args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create checkpoint and log directories (only on rank 0)
    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
    barrier()  # Ensure directories are created before other ranks proceed

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create training dataset with GOP structure
    # Disable shuffling for stage 5 by default (temporal finetuning)
    shuffle_frames = not args.no_shuffle_frames
    if args.stage == 5 and not args.no_shuffle_frames:
        shuffle_frames = False
        if is_main_process():
            print(f"Stage {args.stage}: Disabling frame shuffling for temporal finetuning")

    train_dataset = Vimeo90kGOPDataset(
        root_dir=args.vimeo_dir,
        precomputed_dir=args.precomputed_dir,
        septuplet_list=args.septuplet_list,
        transform=transform,
        crop_size=args.crop_size,
        gop_size=7,  # Vimeo90k has 7 frames per sequence
        shuffle_frames=shuffle_frames
    )

    # Create DistributedSampler for distributed training
    train_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_distributed()  # Drop last incomplete batch for even distribution
    )

    # Create test dataset (UVG) with GOP structure
    test_dataset = UVGGOPDataset(
        root_dir=args.uvg_dir,
        transform=transform,
        gop_size=32
    )

    # Create DistributedSampler for test dataset (no shuffle)
    test_sampler = None
    if is_distributed():
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Make script print info about UVG dataset at start (only on main process)
    if is_main_process():
        print(f"UVG dataset loaded with {len(test_dataset)} sequences.")

    # Load I-frame model (NO DDP - eval only, no gradients)
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    if "state_dict" in i_frame_load_checkpoint:
        i_frame_load_checkpoint = i_frame_load_checkpoint['state_dict']
    i_frame_model = IntraNoAR()
    i_frame_model.load_state_dict(i_frame_load_checkpoint, strict=True)
    i_frame_model = i_frame_model.to(device)
    i_frame_model.eval()

    #compiling i_frame_model
    if args.compile:
        if is_main_process():
            print("Compiling the I-frame model...")
        i_frame_model = torch.compile(i_frame_model)

    if is_main_process():
        print(f"Training model, stage = {args.stage}")

    # Initialize DCVC model
    model = DMC()
    model = model.to(device)

    # Convert BatchNorm to SyncBatchNorm for distributed training if requested
    if is_distributed() and args.sync_bn:
        if is_main_process():
            print("Converting BatchNorm layers to SyncBatchNorm...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap model with DDP BEFORE torch.compile
    if is_distributed():
        model = DDP(
            model,
            device_ids=[get_local_rank()],
            find_unused_parameters=args.find_unused_parameters
        )
        if is_main_process():
            print(f"Model wrapped with DistributedDataParallel (world_size={get_world_size()})")

    # Compile the model if specified (after DDP wrapping)
    if args.compile:
        if is_main_process():
            print("Compiling the model...")
        model = torch.compile(model)

    # Initialize optimizer with optional LR scaling for distributed training
    # Use finetune_lr for stage 5
    if args.stage == 5:
        base_lr = args.finetune_lr
        if is_main_process():
            print(f"Stage {args.stage} finetuning: using learning rate {base_lr}")
    else:
        base_lr = args.learning_rate
        if is_distributed() and args.scale_lr:
            # Scale learning rate linearly by world size
            scaled_lr = base_lr * get_world_size()
            if is_main_process():
                print(f"Scaling learning rate: {base_lr} -> {scaled_lr} (world_size={get_world_size()})")
            base_lr = scaled_lr

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # Initialize learning rate scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience
    )

    # Initialize starting epoch and best loss
    start_epoch = 0
    best_loss = float('inf')

    log_file = os.path.join(args.log_dir,
                            f'train_log_stage_{args.stage}_{args.model_type}.txt')

    # Resume training from checkpoint if specified
    if args.resume:
        if is_main_process():
            print(f"Resuming training from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)

            # Check if this is a state_dict only or a complete checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_loss = checkpoint['best_loss']

                # Load scheduler state if it exists and scheduler is initialized
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                if is_main_process():
                    print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.6f}")
            else:
                # State dict only
                unwrap_model(model).load_state_dict(checkpoint)
                if is_main_process():
                    print("Loaded model weights only (no training state)")
        except Exception as e:
            if is_main_process():
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch")
    # Load from previous stage checkpoint if no resume but previous_stage_checkpoint is specified
    elif args.previous_stage_checkpoint:
        if is_main_process():
            print(f"Loading model from previous stage checkpoint: {args.previous_stage_checkpoint}")
        try:
            checkpoint = torch.load(args.previous_stage_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                if is_main_process():
                    print("Loaded model weights only (no training state)")
            else:
                unwrap_model(model).load_state_dict(checkpoint)  # Use load_state_dict method as defined in DCVC_net
            if is_main_process():
                print("Successfully loaded model from previous stage")
        except Exception as e:
            if is_main_process():
                print(f"Error loading previous stage checkpoint: {e}")
                print("Starting training from scratch")

    # Check for NaN/Inf in model weights after loading
    if is_main_process():
        print("\n" + "="*60)
        print("Checking model weights for NaN/Inf...")
        print("="*60)
    corrupted_params = check_model_for_nan(unwrap_model(model), "video_model", verbose=True)
    if corrupted_params:
        if is_main_process():
            print(f"\n[FATAL] Model has {len(corrupted_params)} corrupted parameters!")
            print("Training cannot continue with NaN weights.")
            print("Options:")
            print("  1. Load an earlier checkpoint (before corruption)")
            print("  2. Re-initialize the model from scratch")
            print("  3. Load only non-corrupted weights from checkpoint")
            print("="*60 + "\n")
        # Exit to prevent training with corrupted weights
        raise RuntimeError(f"Model has {len(corrupted_params)} corrupted parameters with NaN/Inf values")

    if is_main_process():
        with open(log_file, 'a') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model type: {args.model_type}\n")
            f.write(f"Stage: {args.stage} ({STAGE_DESCRIPTIONS[args.stage]})\n")
            f.write(f"I-frame model: {args.i_frame_model_path}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"LR Scheduler: ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_gamma})\n")
            f.write(f"Batch size per GPU: {args.batch_size}\n")
            if is_distributed():
                f.write(f"Global batch size: {args.batch_size * get_world_size()}\n")
                f.write(f"Distributed training: world_size={get_world_size()}\n")
                f.write(f"Scale LR: {args.scale_lr}\n")
            if args.grad_clip_max_norm is not None:
                f.write(f"Gradient clipping max norm: {args.grad_clip_max_norm}\n")
            if args.previous_stage_checkpoint:
                f.write(f"Previous stage checkpoint: {args.previous_stage_checkpoint}\n")
            if args.resume:
                f.write(f"Resuming from checkpoint: {args.resume}\n")
                f.write(f"Starting from epoch: {start_epoch}\n")
            f.write(f"Training dataset: {len(train_dataset)} sequences\n")
            f.write(f"UVG dataset: {len(test_dataset)} sequences\n")
            f.write(f"Using fully batched GOP processing with parallel P-frame batch processing\n")
            f.write("=" * 80 + "\n")

    # Run evaluation before training if requested
    if args.eval_before_train:
        if is_main_process():
            print("\n" + "="*60)
            print("Running evaluation before training...")
            print("="*60)

        if args.stage == 5:
            pre_train_stats = evaluate_finetune(model, i_frame_model, test_loader, device)
        else:
            is_warmup = (0 < args.warmup_epochs)  # Check if warmup mode at start
            pre_train_stats = evaluate(model, i_frame_model, test_loader, device, args.stage, warmup=is_warmup)

        if is_main_process():
            print(f"Pre-training evaluation - Overall Loss: {pre_train_stats['loss']:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"Pre-training evaluation results:\n")
                f.write(f"  Overall Loss: {pre_train_stats['loss']:.6f}\n")
                for q_index in range(len(LAMBDA_SET)):
                    f.write(f"  Quality {q_index} - Loss: {pre_train_stats[q_index]['loss']:.6f}, "
                           f"PSNR: {pre_train_stats[q_index]['psnr']:.4f}, "
                           f"BPP: {pre_train_stats[q_index]['bpp']:.6f}\n")
                f.write("=" * 80 + "\n")
        barrier()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Record epoch start time
        epoch_start_time = time.time()
        epoch_start_str = time.strftime('%Y-%m-%d %H:%M:%S')

        # Determine training mode
        if args.stage == 5:
            is_warmup = False
            mode_str = "finetune (4-frame GOP)"
        else:
            is_warmup = (epoch < args.warmup_epochs)
            mode_str = "warmup (I->P)" if is_warmup else "normal (I->P->P)"

        # Log current learning rate and mode (only on main process)
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process():
            print(f"Epoch {epoch+1}/{args.epochs} - LR: {current_lr:.6f} - Mode: {mode_str}")
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}\n")
                f.write(f"Training mode: {mode_str}\n")
                f.write(f"Epoch start time: {epoch_start_str}\n")

        # Train one epoch
        if args.stage == 5:
            train_stats = train_one_epoch_finetune(
                model, train_loader, optimizer, device, epoch + 1,
                args.grad_clip_max_norm, train_sampler
            )
            test_stats = evaluate_finetune(model, i_frame_model, test_loader, device)
        else:
            train_stats = train_one_epoch_fully_batched(
                model, i_frame_model, train_loader, optimizer, device,
                args.stage, epoch + 1,
                args.grad_clip_max_norm, warmup=is_warmup,
                train_sampler=train_sampler
            )
            test_stats = evaluate(
                model, i_frame_model, test_loader, device, args.stage, warmup=is_warmup
            )

        # Step scheduler after training
        scheduler.step(test_stats['loss'])

        # Record epoch end time and calculate duration
        epoch_end_time = time.time()
        epoch_end_str = time.strftime('%Y-%m-%d %H:%M:%S')
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Format duration as hours:minutes:seconds
        hours, remainder = divmod(epoch_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Log results (only on main process)
        if is_main_process():
            with open(log_file, 'a') as f:
                f.write(f"Stage {args.stage}, Epoch {epoch + 1}/{args.epochs}:\n")
                f.write(f"  Train Loss: {train_stats['loss']:.6f}\n")
                f.write(f"  Train MSE: {train_stats['mse']:.6f}\n")
                f.write(f"  Train PSNR: {train_stats['psnr']:.4f}\n")
                f.write(f"  Train BPP: {train_stats['bpp']:.6f}\n")
                if 'bpp_y' in train_stats and train_stats['bpp_y'] > 0:
                    f.write(f"  Train BPP_y: {train_stats['bpp_y']:.6f}\n")
                if 'bpp_z' in train_stats and train_stats['bpp_z'] > 0:
                    f.write(f"  Train BPP_z: {train_stats['bpp_z']:.6f}\n")
                if 'bpp_mv_y' in train_stats and train_stats['bpp_mv_y'] > 0:
                    f.write(f"  Train BPP_mv_y: {train_stats['bpp_mv_y']:.6f}\n")
                if 'bpp_mv_z' in train_stats and train_stats['bpp_mv_z'] > 0:
                    f.write(f"  Train BPP_mv_z: {train_stats['bpp_mv_z']:.6f}\n")

                # Log evaluation metrics
                f.write(f"  Test Loss Overall: {test_stats['loss']:.6f}\n")
                for q_index in range(len(unwrap_model(model).mv_y_q_scale_enc)):
                    f.write(f"  Test MSE Quality {q_index}: {test_stats[q_index]['mse']:.6f}\n")
                    f.write(f"  Test PSNR Quality {q_index}: {test_stats[q_index]['psnr']:.4f}\n")
                    f.write(f"  Test BPP Quality {q_index}: {test_stats[q_index]['bpp']:.6f}\n")
                    f.write(f"  Test Loss Quality {q_index}: {test_stats[q_index]['loss']:.6f}\n")

                f.write(f"  Epoch end time: {epoch_end_str}\n")
                f.write(f"  Epoch duration: {duration_str} ({epoch_duration:.2f} seconds)\n")
                f.write("=" * 80 + "\n")

        # Get current test loss
        current_test_loss = test_stats['loss']

        # Save checkpoints (only on main process)
        if is_main_process():
            # Create save dictionary with all training state
            save_dict = {
                'epoch': epoch,
                'model_state_dict': unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_stats['loss'],
                'best_loss': best_loss,
                'stage': args.stage,
            }

            # Add scheduler state if present
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()

            # Check if we have a new best model
            if current_test_loss < best_loss:
                best_loss = current_test_loss
                best_checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_stage_{args.stage}_best.pth'
                )
                # Update best loss in save_dict
                save_dict['best_loss'] = best_loss
                # Save full training state for the best model
                torch.save(save_dict, best_checkpoint_path)

                # Also save just the state dict for easy loading
                best_state_dict_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_stage_{args.stage}_best_state_dict.pth'
                )
                torch.save(unwrap_model(model).state_dict(), best_state_dict_path)

                print(f"New best model saved with test loss: {best_loss:.6f}")
                with open(log_file, 'a') as f:
                    f.write(f"New best model saved with test loss: {best_loss:.6f}\n")

            # Save latest checkpoint with training state for resuming
            latest_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_stage_{args.stage}_latest.pth'
            )
            torch.save(save_dict, latest_checkpoint_path)

            print(f"Epoch {epoch + 1}/{args.epochs} completed. Latest checkpoint saved.")

        # Update best_loss on all ranks (for consistent behavior)
        if current_test_loss < best_loss:
            best_loss = current_test_loss

        # Synchronize all ranks after checkpoint saving
        barrier()

    # Save final model for this stage (only on main process)
    if is_main_process():
        # Save final model (state_dict only for compatibility with original code)
        final_model_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_stage_{args.stage}.pth'
        )
        torch.save(unwrap_model(model).state_dict(), final_model_path)
        print(f"Final model for stage {args.stage} saved to {final_model_path}")

        # If this is the final stage (4), also save with the standard naming convention
        if args.stage == 4:
            standard_model_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_{args.model_type}.pth'
            )
            torch.save(unwrap_model(model).state_dict(), standard_model_path)
            print(f"Final model (standard name) saved to {standard_model_path}")

        with open(log_file, 'a') as f:
            f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        print(f"Training completed for stage {args.stage}!")

    # Clean up distributed training resources
    cleanup_distributed()


if __name__ == '__main__':
    main()
