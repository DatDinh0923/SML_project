"""
Taylor-FO-BN Pruning Implementation
Based on: "Importance Estimation for Neural Network Pruning"
Paper: Molchanov et al., CVPR 2019 (arXiv:1906.10771)

Key improvements over the 2017 Taylor pruning paper:
1. Squared importance: I_m = (g_m * w_m)^2 instead of |g_m * w_m|
2. Gate-based computation after BatchNorm layers:
   I_m = (gamma_m * dE/dgamma_m + beta_m * dE/dbeta_m)^2
3. Exponential moving average (momentum=0.9) for importance accumulation
4. Batch pruning: removes ~2% of filters per iteration
5. Globally consistent scale across layers without per-layer normalization

Algorithm (Section 3.1):
1. Take a trained network as input
2. During each epoch:
   a. For each minibatch, compute gradients, update weights, compute importance
   b. After N minibatches, average importance and remove least important filters
3. Continue until target pruning ratio reached
4. Reset momentum buffer and fine-tune
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class TaylorFOBNPruner:
    """Taylor-FO-BN: First-Order Taylor expansion with BatchNorm gates for pruning."""

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5,
                 prune_percent_per_iter: float = 0.02,
                 minibatches_between_pruning: int = 10,
                 ema_momentum: float = 0.9,
                 prune_skip_connections: bool = False):
        """
        Initialize Taylor-FO-BN Pruner.

        Args:
            model: CNN model to prune
            pruning_ratio: Target ratio of filters to prune (default 0.5)
            prune_percent_per_iter: Fraction of initial filters to prune per
                iteration (default 0.02 = 2%, as recommended in the paper)
            minibatches_between_pruning: Number of minibatches between pruning
                iterations for accumulating importance (default 10 for CIFAR-10)
            ema_momentum: Exponential moving average momentum for importance
                accumulation across pruning iterations (default 0.9)
            prune_skip_connections: Whether to also prune skip connections
                (default False; set True for ResNets as in the paper)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.prune_percent_per_iter = prune_percent_per_iter
        self.minibatches_between_pruning = minibatches_between_pruning
        self.ema_momentum = ema_momentum
        self.prune_skip_connections = prune_skip_connections

        # Prunable BN layers and their associated conv layers
        self.prunable_bns = OrderedDict()  # bn_name -> info dict
        self._hooks = []

        # Importance scores
        self._batch_importance = {}      # Current minibatch accumulation
        self._batch_count = 0
        self._ema_importance = {}        # EMA across pruning iterations

        self._init_prunable_layers()

        # Compute initial filter counts
        self.total_initial_filters = sum(
            info['num_features'] for info in self.prunable_bns.values()
        )
        self.filters_per_prune_iter = max(
            1, int(self.total_initial_filters * self.prune_percent_per_iter)
        )

        logger.info(f"Taylor-FO-BN Pruner initialized:")
        logger.info(f"  Pruning Ratio: {pruning_ratio:.2%}")
        logger.info(f"  Prune {prune_percent_per_iter:.1%} of filters per iter "
                     f"({self.filters_per_prune_iter} filters)")
        logger.info(f"  Minibatches between pruning: {minibatches_between_pruning}")
        logger.info(f"  EMA momentum: {ema_momentum}")
        logger.info(f"  Prune skip connections: {prune_skip_connections}")
        logger.info(f"  Prunable BN layers: {len(self.prunable_bns)}")
        logger.info(f"  Total prunable filters: {self.total_initial_filters}")

    def _init_prunable_layers(self):
        """
        Identify prunable BatchNorm layers and their associated conv layers.

        Paper: "gates are placed immediately after a batch normalization layer
        to capture contributions from scaling and shifting parameters
        simultaneously."

        For ResNet:
        - BN layers after conv layers in residual blocks are prunable
        - Skip the stem BN and downsample BNs (unless prune_skip_connections)
        - For BasicBlock: bn1 after conv1 is prunable (conv1 output can change)
          bn2 after conv2 is NOT prunable (must match skip connection dim)
        - For Bottleneck: bn1 after conv1, bn2 after conv2 are prunable
          bn3 after conv3 is NOT prunable
        """
        # Identify skip layers (same logic as SPP/Taylor pruner)
        skip_conv_ids = set()
        skip_bn_names = set()

        for name, module in self.model.named_modules():
            if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                if hasattr(module, 'conv3'):
                    # Bottleneck: skip conv3/bn3
                    for n, m in module.named_modules():
                        if n == 'conv3':
                            skip_conv_ids.add(id(m))
                        if n == 'bn3':
                            skip_bn_names.add(name + '.bn3')
                else:
                    # BasicBlock: skip conv2/bn2
                    for n, m in module.named_modules():
                        if n == 'conv2':
                            skip_conv_ids.add(id(m))
                        if n == 'bn2':
                            skip_bn_names.add(name + '.bn2')

            # Skip downsample layers
            if 'downsample' in name:
                if isinstance(module, nn.Conv2d):
                    skip_conv_ids.add(id(module))
                if isinstance(module, nn.BatchNorm2d):
                    skip_bn_names.add(name)

        # Skip first stem conv/bn
        first_conv_found = False
        first_bn_found = False
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and not first_conv_found:
                skip_conv_ids.add(id(module))
                first_conv_found = True
            if isinstance(module, nn.BatchNorm2d) and not first_bn_found:
                skip_bn_names.add(name)
                first_bn_found = True

        # Build mapping: find BN layers and their preceding conv layers
        all_modules = list(self.model.named_modules())

        for i, (bn_name, bn_module) in enumerate(all_modules):
            if not isinstance(bn_module, nn.BatchNorm2d):
                continue
            if bn_name in skip_bn_names:
                logger.info(f"  BN {bn_name}: features={bn_module.num_features} [SKIP]")
                continue

            # Find the conv layer that feeds into this BN
            conv_name, conv_module = self._find_preceding_conv(bn_name)
            if conv_name is None:
                logger.info(f"  BN {bn_name}: features={bn_module.num_features} [SKIP - no conv]")
                continue

            # Find the next conv layer that takes this BN's output as input
            next_conv_name = self._find_next_conv_for_bn(bn_name, conv_name)

            self.prunable_bns[bn_name] = {
                'bn_module': bn_module,
                'bn_name': bn_name,
                'conv_name': conv_name,
                'conv_module': conv_module,
                'next_conv_name': next_conv_name,
                'num_features': bn_module.num_features,
            }

            logger.info(
                f"  BN {bn_name}: features={bn_module.num_features} "
                f"[PRUNABLE] conv={conv_name}"
            )

    def _find_preceding_conv(self, bn_name: str) -> Tuple[Optional[str], Optional[nn.Conv2d]]:
        """Find the conv layer preceding a BN layer."""
        # In ResNet: conv1 -> bn1, layer1.0.conv1 -> layer1.0.bn1
        conv_name = bn_name.replace('bn', 'conv')
        for name, module in self.model.named_modules():
            if name == conv_name and isinstance(module, nn.Conv2d):
                return conv_name, module
        return None, None

    def _find_next_conv_for_bn(self, bn_name: str, conv_name: str) -> Optional[str]:
        """Find the next conv that receives this BN's output."""
        parts = conv_name.rsplit('.', 1)
        if len(parts) == 2:
            prefix, name = parts
            if name == 'conv1':
                next_name = prefix + '.conv2'
                for n, m in self.model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
            elif name == 'conv2':
                # Check for Bottleneck conv3
                next_name = prefix + '.conv3'
                for n, m in self.model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
        return None

    def compute_importance_from_bn(self):
        """
        Compute Taylor-FO-BN importance for all prunable BN layers.

        Paper Eq. (gate after BN):
        I_m = (gamma_m * dE/dgamma_m + beta_m * dE/dbeta_m)^2

        This captures the contribution of the BN scaling and shifting
        parameters to the loss, which implicitly includes the contribution
        of the preceding conv filter.
        """
        for bn_name, info in self.prunable_bns.items():
            bn = info['bn_module']

            if bn.weight.grad is None or bn.bias.grad is None:
                continue

            # Gate importance: I_m = (gamma * dE/dgamma + beta * dE/dbeta)^2
            gamma = bn.weight.data          # (C,)
            dgamma = bn.weight.grad.data    # (C,)
            beta = bn.bias.data             # (C,)
            dbeta = bn.bias.grad.data       # (C,)

            importance = (gamma * dgamma + beta * dbeta).pow(2)  # (C,)

            importance = importance.detach().cpu()

            # Accumulate over minibatches
            if bn_name not in self._batch_importance:
                self._batch_importance[bn_name] = torch.zeros_like(importance)
            self._batch_importance[bn_name] += importance

        self._batch_count += 1

    def finalize_importance(self):
        """
        Average accumulated importance over minibatches and apply EMA.

        Paper: "we average importance scores between pruning iterations
        using an exponential moving average filter (momentum) with
        coefficient 0.9"
        """
        for bn_name, accum in self._batch_importance.items():
            avg_importance = accum / max(self._batch_count, 1)

            if bn_name not in self._ema_importance:
                self._ema_importance[bn_name] = avg_importance
            else:
                self._ema_importance[bn_name] = (
                    self.ema_momentum * self._ema_importance[bn_name]
                    + (1 - self.ema_momentum) * avg_importance
                )

        # Reset batch accumulation
        self._batch_importance.clear()
        self._batch_count = 0

    def select_filters_to_prune(self) -> List[Tuple[str, List[int]]]:
        """
        Select the least important filters globally across all layers.

        Returns:
            List of (bn_name, [filter_indices]) tuples to prune
        """
        # Build global list of (bn_name, filter_idx, importance_score)
        all_scores = []
        for bn_name, importance in self._ema_importance.items():
            info = self.prunable_bns[bn_name]
            num_features = info['bn_module'].num_features
            for idx in range(num_features):
                all_scores.append((bn_name, idx, importance[idx].item()))

        # Sort by importance ascending (least important first)
        all_scores.sort(key=lambda x: x[2])

        # Select filters to prune (up to filters_per_prune_iter)
        # But don't prune a layer below 1 filter
        layer_remaining = {
            bn_name: info['bn_module'].num_features
            for bn_name, info in self.prunable_bns.items()
        }

        to_prune = {}  # bn_name -> list of indices
        count = 0

        for bn_name, idx, score in all_scores:
            if count >= self.filters_per_prune_iter:
                break
            if layer_remaining[bn_name] <= 1:
                continue

            if bn_name not in to_prune:
                to_prune[bn_name] = []
            to_prune[bn_name].append(idx)
            layer_remaining[bn_name] -= 1
            count += 1

        return [(bn_name, indices) for bn_name, indices in to_prune.items()]

    def prune_filters(self, filters_to_prune: List[Tuple[str, List[int]]]):
        """
        Structurally prune the selected filters.
        For each BN layer's pruned filters, removes from:
        1. The preceding conv layer (output channels)
        2. The BN layer itself
        3. The next conv layer (input channels)
        """
        for bn_name, filter_indices in filters_to_prune:
            info = self.prunable_bns[bn_name]
            bn_module = info['bn_module']
            conv_name = info['conv_name']
            conv_module = info['conv_module']
            next_conv_name = info['next_conv_name']
            device = bn_module.weight.device

            num_current = bn_module.num_features
            all_indices = set(range(num_current))
            keep_indices = sorted(all_indices - set(filter_indices))
            keep_tensor = torch.tensor(keep_indices, device=device)
            num_kept = len(keep_indices)

            # 1. Prune preceding conv (output channels)
            new_weight = conv_module.weight.data[keep_tensor]
            new_bias = conv_module.bias.data[keep_tensor] if conv_module.bias is not None else None

            new_conv = nn.Conv2d(
                in_channels=conv_module.in_channels,
                out_channels=num_kept,
                kernel_size=conv_module.kernel_size,
                stride=conv_module.stride,
                padding=conv_module.padding,
                dilation=conv_module.dilation,
                groups=conv_module.groups,
                bias=conv_module.bias is not None,
            ).to(device)
            new_conv.weight.data = new_weight
            if new_bias is not None:
                new_conv.bias.data = new_bias
            self._set_module(self.model, conv_name, new_conv)

            # 2. Prune BN layer
            new_bn = nn.BatchNorm2d(num_kept, eps=bn_module.eps,
                                     momentum=bn_module.momentum,
                                     affine=bn_module.affine,
                                     track_running_stats=bn_module.track_running_stats).to(device)
            new_bn.weight.data = bn_module.weight.data[keep_tensor]
            new_bn.bias.data = bn_module.bias.data[keep_tensor]
            if bn_module.track_running_stats:
                new_bn.running_mean = bn_module.running_mean[keep_tensor]
                new_bn.running_var = bn_module.running_var[keep_tensor]
                new_bn.num_batches_tracked = bn_module.num_batches_tracked
            self._set_module(self.model, bn_name, new_bn)

            # 3. Prune next conv (input channels)
            if next_conv_name:
                next_conv = dict(self.model.named_modules())[next_conv_name]
                if next_conv.groups <= 1:
                    new_next_weight = next_conv.weight.data[:, keep_tensor, :, :]
                    new_next_conv = nn.Conv2d(
                        in_channels=num_kept,
                        out_channels=next_conv.out_channels,
                        kernel_size=next_conv.kernel_size,
                        stride=next_conv.stride,
                        padding=next_conv.padding,
                        dilation=next_conv.dilation,
                        groups=next_conv.groups,
                        bias=next_conv.bias is not None,
                    ).to(device)
                    new_next_conv.weight.data = new_next_weight
                    if next_conv.bias is not None:
                        new_next_conv.bias.data = next_conv.bias.data.clone()
                    self._set_module(self.model, next_conv_name, new_next_conv)

            # Update EMA importance (remap to kept indices)
            if bn_name in self._ema_importance:
                self._ema_importance[bn_name] = self._ema_importance[bn_name][keep_indices]

            logger.info(
                f"  Pruned {len(filter_indices)} filter(s) from {bn_name} "
                f"({conv_name}): {num_current} -> {num_kept}"
            )

        # Refresh module references
        self._refresh_module_refs()

    def _refresh_module_refs(self):
        """Update module references after structural pruning."""
        module_dict = dict(self.model.named_modules())
        for bn_name, info in self.prunable_bns.items():
            if bn_name in module_dict:
                info['bn_module'] = module_dict[bn_name]
                info['num_features'] = module_dict[bn_name].num_features
            if info['conv_name'] in module_dict:
                info['conv_module'] = module_dict[info['conv_name']]

    def get_pruning_ratio(self) -> float:
        """Get current pruning ratio."""
        current_filters = sum(
            info['bn_module'].num_features
            for info in self.prunable_bns.values()
        )
        return 1.0 - (current_filters / self.total_initial_filters)

    def should_stop_pruning(self) -> bool:
        """Check if target pruning ratio has been reached."""
        ratio = self.get_pruning_ratio()
        should_stop = ratio >= self.pruning_ratio
        if should_stop:
            logger.info(f"Target pruning ratio reached: {ratio:.2%} >= {self.pruning_ratio:.2%}")
        return should_stop

    def get_stats(self) -> Dict:
        """Get pruning statistics."""
        current_filters = sum(
            info['bn_module'].num_features
            for info in self.prunable_bns.values()
        )
        return {
            'total_initial_filters': self.total_initial_filters,
            'current_filters': current_filters,
            'num_pruned': self.total_initial_filters - current_filters,
            'pruning_ratio': self.get_pruning_ratio(),
            'filters_per_prune_iter': self.filters_per_prune_iter,
        }

    def export_pruned_model(self) -> nn.Module:
        """
        Return the pruned model (already structurally pruned in-place).
        """
        logger.info("Exporting pruned model...")

        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(device)
            output = self.model(dummy)
            logger.info(f"  Export verified: output shape = {output.shape}")

        new_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Parameters: {new_params:,}")
        logger.info(f"  Pruning ratio: {self.get_pruning_ratio():.2%}")

        for bn_name, info in self.prunable_bns.items():
            current = info['bn_module'].num_features
            logger.info(f"  {bn_name}: {current} features")

        return self.model

    def _set_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Set a module in the model by its dot-separated name."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        if parts[-1].isdigit():
            parent[int(parts[-1])] = new_module
        else:
            setattr(parent, parts[-1], new_module)

    def save_checkpoint(self, path: str):
        """Save pruning state."""
        checkpoint = {
            'pruning_ratio': self.pruning_ratio,
            'prune_percent_per_iter': self.prune_percent_per_iter,
            'minibatches_between_pruning': self.minibatches_between_pruning,
            'ema_momentum': self.ema_momentum,
            'total_initial_filters': self.total_initial_filters,
            'current_pruning_ratio': self.get_pruning_ratio(),
            'ema_importance': {k: v.clone() for k, v in self._ema_importance.items()},
        }
        torch.save(checkpoint, path)
        logger.info(f"Pruning checkpoint saved to {path}")
