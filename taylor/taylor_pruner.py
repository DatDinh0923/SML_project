"""
Taylor Expansion-Based Pruning Implementation
Based on: "Pruning Convolutional Neural Networks for Resource Efficient Inference"
Paper: Molchanov et al., ICLR 2017 (arXiv:1611.06440)

Algorithm:
1. Fine-tune the network until convergence on the target task
2. Compute importance of each feature map using Taylor criterion:
   Theta_TE(z_l^(k)) = |1/M * sum_m (dC/dz_{l,m}^(k)) * z_{l,m}^(k)|
3. Normalize importance scores across layers (L2 normalization)
4. Optionally apply FLOPs regularization
5. Remove the least important feature map(s)
6. Fine-tune for N minibatch updates
7. Repeat steps 2-6 until target pruning ratio is reached
8. Final fine-tuning of the pruned model
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class TaylorPruner:
    """Taylor expansion-based filter pruning for CNN acceleration."""

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5,
                 filters_per_iter: int = 1, finetune_updates: int = 100,
                 flops_reg_lambda: float = 0.0, normalize: bool = True):
        """
        Initialize Taylor Pruner.

        Args:
            model: CNN model to prune
            pruning_ratio: Target ratio of filters to prune (default 0.5)
            filters_per_iter: Number of filters to prune per iteration (default 1)
            finetune_updates: Number of minibatch SGD updates between pruning (default 100)
            flops_reg_lambda: FLOPs regularization coefficient (default 0.0, paper uses 1e-3)
            normalize: Whether to apply layer-wise L2 normalization (default True)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.filters_per_iter = filters_per_iter
        self.finetune_updates = finetune_updates
        self.flops_reg_lambda = flops_reg_lambda
        self.normalize = normalize

        # Prunable layers and their info
        self.prunable_layers = OrderedDict()
        self._hooks = []
        self._activations = {}
        self._gradients = {}

        # Accumulated importance scores (averaged over batches)
        self._importance_accum = {}
        self._importance_count = 0

        # Track which filters have been pruned (layer_name -> set of pruned indices)
        self.pruned_filters = {}

        self._init_prunable_layers()
        self._register_hooks()

        # Stats
        self.total_filters = sum(info['out_channels'] for info in self.prunable_layers.values())
        self.num_pruned = 0

        logger.info(f"Taylor Pruner initialized:")
        logger.info(f"  Pruning Ratio: {pruning_ratio:.2%}")
        logger.info(f"  Filters per iteration: {filters_per_iter}")
        logger.info(f"  Fine-tune updates between pruning: {finetune_updates}")
        logger.info(f"  FLOPs regularization lambda: {flops_reg_lambda}")
        logger.info(f"  Layer-wise L2 normalization: {normalize}")
        logger.info(f"  Prunable layers: {len(self.prunable_layers)}")
        logger.info(f"  Total prunable filters: {self.total_filters}")

    def _init_prunable_layers(self):
        """
        Identify prunable convolutional layers.
        Same strategy as SPP: skip stem conv, skip-connection convs, downsample convs.
        Only prune intermediate conv layers within residual blocks.
        """
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))

        # Identify layers to skip
        skip_layers = set()
        for name, module in self.model.named_modules():
            if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                if hasattr(module, 'conv3'):
                    # Bottleneck: skip conv3
                    for n, m in module.named_modules():
                        if n == 'conv3':
                            skip_layers.add(id(m))
                else:
                    # BasicBlock: skip conv2
                    for n, m in module.named_modules():
                        if n == 'conv2':
                            skip_layers.add(id(m))

            if 'downsample' in name and isinstance(module, nn.Conv2d):
                skip_layers.add(id(module))

        # Skip first stem conv
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                skip_layers.add(id(module))
                break

        for name, module in conv_layers:
            if id(module) in skip_layers:
                logger.info(f"  Layer {name}: Conv2d out_ch={module.weight.shape[0]} [SKIP]")
                continue

            out_channels = module.weight.shape[0]
            in_channels = module.weight.shape[1]
            kernel_size = module.kernel_size

            self.prunable_layers[name] = {
                'module': module,
                'out_channels': out_channels,
                'in_channels': in_channels,
                'kernel_size': kernel_size,
            }
            self.pruned_filters[name] = set()

            logger.info(f"  Layer {name}: Conv2d out_ch={out_channels} [PRUNABLE]")

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        for layer_name, info in self.prunable_layers.items():
            module = info['module']

            def make_forward_hook(name):
                def hook(module, input, output):
                    self._activations[name] = output
                return hook

            def make_backward_hook(name):
                def hook(module, grad_input, grad_output):
                    self._gradients[name] = grad_output[0]
                return hook

            self._hooks.append(module.register_forward_hook(make_forward_hook(layer_name)))
            self._hooks.append(module.register_full_backward_hook(make_backward_hook(layer_name)))

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_importance(self):
        """
        Compute Taylor importance for all prunable filters using captured
        activations and gradients from the current batch.

        Theta_TE(z_l^(k)) = |1/M * sum_m (dC/dz_{l,m}^(k)) * z_{l,m}^(k)|

        For a minibatch with T>1 examples, computed per example and averaged over T.
        """
        batch_importance = {}

        for layer_name in self.prunable_layers:
            if layer_name not in self._activations or layer_name not in self._gradients:
                continue

            activation = self._activations[layer_name]  # (B, C, H, W)
            gradient = self._gradients[layer_name]       # (B, C, H, W)

            # Taylor criterion: |activation * gradient|, averaged over spatial dims and batch
            # Per filter: sum over spatial dims (H, W), then average over batch
            # Shape: (B, C, H, W) -> (B, C) -> (C,)
            taylor = (activation * gradient).abs().mean(dim=(2, 3))  # (B, C)
            taylor = taylor.mean(dim=0)  # (C,) average over batch

            batch_importance[layer_name] = taylor.detach().cpu()

        # Accumulate importance across batches
        for layer_name, importance in batch_importance.items():
            if layer_name not in self._importance_accum:
                self._importance_accum[layer_name] = torch.zeros_like(importance)
            self._importance_accum[layer_name] += importance

        self._importance_count += 1

        # Clear activations/gradients to save memory
        self._activations.clear()
        self._gradients.clear()

    def get_averaged_importance(self) -> Dict[str, torch.Tensor]:
        """Get importance scores averaged over accumulated batches."""
        averaged = {}
        for layer_name, accum in self._importance_accum.items():
            averaged[layer_name] = accum / max(self._importance_count, 1)
        return averaged

    def reset_importance(self):
        """Reset accumulated importance scores."""
        self._importance_accum.clear()
        self._importance_count = 0

    def _normalize_importance(self, importance: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply layer-wise L2 normalization to make importance scores comparable across layers.

        hat{Theta}(z_l^(k)) = Theta(z_l^(k)) / sqrt(sum_j (Theta(z_l^(j)))^2)
        """
        normalized = {}
        for layer_name, scores in importance.items():
            l2_norm = torch.sqrt((scores ** 2).sum())
            if l2_norm > 0:
                normalized[layer_name] = scores / l2_norm
            else:
                normalized[layer_name] = scores
        return normalized

    def _compute_flops_cost(self, layer_name: str) -> float:
        """
        Compute the FLOPs cost of a single filter in a given layer.
        FLOPs for one output filter = 2 * C_in * K^2 * H_out * W_out
        (multiply-accumulate counted as 2 operations)
        """
        info = self.prunable_layers[layer_name]
        module = info['module']

        if layer_name in self._activations:
            h_out = self._activations[layer_name].shape[2]
            w_out = self._activations[layer_name].shape[3]
        else:
            # Estimate from input size (CIFAR-10 = 32x32)
            h_out = w_out = 32
            for name, mod in self.model.named_modules():
                if isinstance(mod, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                    if isinstance(mod, nn.Conv2d):
                        stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
                    else:
                        stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
                    h_out = h_out // stride
                    w_out = w_out // stride
                if name == layer_name:
                    break

        c_in = module.in_channels
        k = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        flops = 2 * c_in * k * k * h_out * w_out
        return flops

    def _apply_flops_regularization(self, importance: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply FLOPs regularization to importance scores.
        Theta(z_l^(k)) = Theta(z_l^(k)) - lambda * Theta^flops_l

        This encourages pruning filters from layers with high FLOPs cost.
        """
        regularized = {}
        for layer_name, scores in importance.items():
            flops_cost = self._compute_flops_cost(layer_name)
            regularized[layer_name] = scores - self.flops_reg_lambda * flops_cost
        return regularized

    def select_filters_to_prune(self) -> List[Tuple[str, int]]:
        """
        Select the least important filter(s) globally across all layers.

        Returns:
            List of (layer_name, filter_index) tuples to prune
        """
        importance = self.get_averaged_importance()

        if self.normalize:
            importance = self._normalize_importance(importance)

        if self.flops_reg_lambda > 0:
            importance = self._apply_flops_regularization(importance)

        # Build global ranking: collect all (layer, filter_idx, score) triples
        all_scores = []
        for layer_name, scores in importance.items():
            for idx in range(len(scores)):
                if idx not in self.pruned_filters[layer_name]:
                    all_scores.append((layer_name, idx, scores[idx].item()))

        # Sort by importance (ascending - least important first)
        all_scores.sort(key=lambda x: x[2])

        # Select top-k least important filters
        to_prune = []
        for layer_name, idx, score in all_scores[:self.filters_per_iter]:
            # Safety: don't prune a layer below 1 remaining filter
            remaining = self.prunable_layers[layer_name]['out_channels'] - len(self.pruned_filters[layer_name])
            if remaining > 1:
                to_prune.append((layer_name, idx))

        return to_prune

    def prune_filters(self, filters_to_prune: List[Tuple[str, int]]):
        """
        Structurally prune the selected filters from the model.
        For each filter, removes it from:
        1. The conv layer (output channels)
        2. The following BatchNorm layer
        3. The next conv layer (input channels)
        """
        # Group by layer for efficient processing
        layers_to_prune = {}
        for layer_name, filter_idx in filters_to_prune:
            if layer_name not in layers_to_prune:
                layers_to_prune[layer_name] = []
            layers_to_prune[layer_name].append(filter_idx)
            self.pruned_filters[layer_name].add(filter_idx)
            self.num_pruned += 1

        for layer_name, filter_indices in layers_to_prune.items():
            module = self.prunable_layers[layer_name]['module']
            device = module.weight.device

            # Compute which filters to KEEP
            all_indices = set(range(module.out_channels))
            keep_indices = sorted(all_indices - set(filter_indices))
            keep_tensor = torch.tensor(keep_indices, device=device)

            # 1. Prune conv output filters
            new_weight = module.weight.data[keep_tensor]
            new_bias = module.bias.data[keep_tensor] if module.bias is not None else None

            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=len(keep_indices),
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            ).to(device)

            new_conv.weight.data = new_weight
            if new_bias is not None:
                new_conv.bias.data = new_bias

            self._set_module(self.model, layer_name, new_conv)

            # 2. Prune corresponding BatchNorm
            bn_name = self._find_next_bn(layer_name)
            if bn_name:
                bn_module = dict(self.model.named_modules())[bn_name]
                new_bn = nn.BatchNorm2d(len(keep_indices)).to(device)
                new_bn.weight.data = bn_module.weight.data[keep_tensor]
                new_bn.bias.data = bn_module.bias.data[keep_tensor]
                new_bn.running_mean = bn_module.running_mean[keep_tensor]
                new_bn.running_var = bn_module.running_var[keep_tensor]
                new_bn.num_batches_tracked = bn_module.num_batches_tracked
                self._set_module(self.model, bn_name, new_bn)

            # 3. Prune next conv's input channels
            next_conv_name = self._find_next_conv(layer_name)
            if next_conv_name:
                next_module = dict(self.model.named_modules())[next_conv_name]
                if next_module.groups <= 1:
                    new_next_weight = next_module.weight.data[:, keep_tensor, :, :]

                    new_next_conv = nn.Conv2d(
                        in_channels=len(keep_indices),
                        out_channels=next_module.out_channels,
                        kernel_size=next_module.kernel_size,
                        stride=next_module.stride,
                        padding=next_module.padding,
                        dilation=next_module.dilation,
                        groups=next_module.groups,
                        bias=next_module.bias is not None,
                    ).to(device)

                    new_next_conv.weight.data = new_next_weight
                    if next_module.bias is not None:
                        new_next_conv.bias.data = next_module.bias.data.clone()

                    self._set_module(self.model, next_conv_name, new_next_conv)

            # Update prunable_layers info
            self.prunable_layers[layer_name]['module'] = dict(self.model.named_modules())[layer_name]
            self.prunable_layers[layer_name]['out_channels'] = len(keep_indices)

            # Remap pruned_filters indices since the layer has been physically pruned
            self.pruned_filters[layer_name] = set()

            logger.info(
                f"  Pruned {len(filter_indices)} filter(s) from {layer_name}: "
                f"{module.out_channels} -> {len(keep_indices)}"
            )

        # Re-register hooks since modules have been replaced
        self.remove_hooks()
        self._hooks.clear()
        # Update module references
        for layer_name in self.prunable_layers:
            self.prunable_layers[layer_name]['module'] = dict(self.model.named_modules())[layer_name]
        self._register_hooks()

    def get_pruning_ratio(self) -> float:
        """Get current pruning ratio."""
        current_filters = sum(
            info['module'].out_channels
            for info in self.prunable_layers.values()
        )
        return 1.0 - (current_filters / self.total_filters)

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
            info['module'].out_channels
            for info in self.prunable_layers.values()
        )
        per_layer = {}
        for name, info in self.prunable_layers.items():
            original = info['out_channels']  # This was updated during pruning
            current = info['module'].out_channels
            per_layer[name] = {
                'current': current,
                'original': self.total_filters,  # approximate
            }

        return {
            'total_original_filters': self.total_filters,
            'current_filters': current_filters,
            'num_pruned': self.total_filters - current_filters,
            'pruning_ratio': self.get_pruning_ratio(),
            'per_layer': per_layer,
        }

    def export_pruned_model(self) -> nn.Module:
        """
        Return the pruned model. Since Taylor pruning physically removes filters
        during the pruning process, the model is already structurally pruned.
        """
        logger.info("Exporting pruned model...")

        # Verify model works
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(device)
            output = self.model(dummy)
            logger.info(f"  Export verified: output shape = {output.shape}")

        old_params = self.total_filters  # approximate
        new_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Parameters: {new_params:,}")
        logger.info(f"  Pruning ratio: {self.get_pruning_ratio():.2%}")

        # Log per-layer stats
        for name, info in self.prunable_layers.items():
            current = info['module'].out_channels
            logger.info(f"  {name}: {current} filters")

        return self.model

    def _find_next_bn(self, conv_name: str) -> Optional[str]:
        """Find the BatchNorm layer immediately after a conv layer."""
        bn_name = conv_name.replace('conv', 'bn')
        for name, module in self.model.named_modules():
            if name == bn_name and isinstance(module, nn.BatchNorm2d):
                return bn_name
        return None

    def _find_next_conv(self, conv_name: str) -> Optional[str]:
        """Find the next conv layer that takes this conv's output as input."""
        parts = conv_name.rsplit('.', 1)
        if len(parts) == 2:
            prefix, name = parts
            if name == 'conv1':
                next_name = prefix + '.conv2'
                for n, m in self.model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
            elif name == 'conv2' and self._has_conv3(prefix):
                next_name = prefix + '.conv3'
                for n, m in self.model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
        return None

    def _has_conv3(self, prefix: str) -> bool:
        """Check if a block has conv3 (Bottleneck)."""
        for name, module in self.model.named_modules():
            if name == prefix + '.conv3' and isinstance(module, nn.Conv2d):
                return True
        return False

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
            'filters_per_iter': self.filters_per_iter,
            'finetune_updates': self.finetune_updates,
            'flops_reg_lambda': self.flops_reg_lambda,
            'normalize': self.normalize,
            'total_filters': self.total_filters,
            'num_pruned': self.num_pruned,
            'current_pruning_ratio': self.get_pruning_ratio(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Pruning checkpoint saved to {path}")
