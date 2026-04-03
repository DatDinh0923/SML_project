"""
Structured Probabilistic Pruning (SPP) Implementation
Based on: "Structured Probabilistic Pruning for Convolutional Neural Network Acceleration"
Paper: Wang et al., 2018

Algorithm 1 - SPP:
1. Input training set D, pre-trained CNN model Ω, target pruning ratio R
2. Set hyperparameters A, u, t (default: A=0.05, u=0.25, t=180)
3. Set update number k=0
4. For each weight group in all conv layers, init pruning probability p^k=0
5. Init iteration number i=0
6. repeat:
   - If mod(i,t)=0, update p^k by Eqn(3)&(4), set k=k+1
   - For each weight group, obtain g_i by Monte Carlo sampling based on p^k
   - Prune network by Eqn(2)
   - Train pruned network, updating weights by backprop
   - i=i+1
7. until weight group ratio of p^k=1 equals R
8. Retrain pruned CNN for several iterations
9. Output pruned CNN model Ω'
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class SPPPruner:
    """Structured Probabilistic Pruning for CNN acceleration"""

    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5,
                 A: float = 0.05, u: float = 0.25, update_interval: int = 180):
        """
        Initialize SPP Pruner

        Args:
            model: CNN model to prune
            pruning_ratio: Target pruning ratio R (default 0.5)
            A: Hyperparameter A for exponential function (default 0.05)
            u: Hyperparameter u for controlling flatness (default 0.25, range 0-1)
            update_interval: Update interval t in iterations (default 180)
        """
        self.model = model
        self.pruning_ratio = pruning_ratio  # R in Algorithm 1
        self.A = A
        self.u = u
        self.update_interval = update_interval  # t in Algorithm 1

        # Algorithm 1: Initialize pruning probability p^k = 0 for each weight group
        self.pruning_probs = {}  # p^k for each weight group
        self.masks = {}  # g masks for each weight group
        self.weight_groups = {}  # Store weight group info
        self.frozen = {}  # Track permanently pruned groups (p=1)
        self._saved_weights = {}  # Saved weights before masking (for non-destructive pruning)
        self._saved_biases = {}  # Saved biases before masking

        # Initialize weight groups and pruning probabilities
        self._init_weight_groups()

        # Tracking variables
        self.update_count = 0  # k in Algorithm 1
        self.iter_count = 0  # i in Algorithm 1
        self.training_iteration = 0  # Actual training iteration

        logger.info(f"SPP Pruner initialized:")
        logger.info(f"  Pruning Ratio R={pruning_ratio:.2%}")
        logger.info(f"  Hyperparameters: A={A}, u={u}, update_interval={update_interval}")
        logger.info(f"  Prunable layers: {len(self.weight_groups)}")

    def _init_weight_groups(self):
        """
        Initialize weight groups for prunable convolutional layers.
        Prunes OUTPUT filters (structured pruning).

        For ResNet: only prune intermediate conv layers within blocks,
        NOT the last conv before skip connection addition.
        """
        # Build list of all conv layers with their names
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))

        # Identify which conv layers are safe to prune
        # For ResNet blocks: conv1 in BasicBlock, conv1/conv2 in Bottleneck are safe
        # The last conv in each block (conv2 in Basic, conv3 in Bottleneck) is NOT safe
        # because its output must match the skip connection dimension
        skip_layers = set()
        for name, module in self.model.named_modules():
            # Detect last conv in residual blocks by checking for downsample or
            # being the last conv before the block output
            if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                if hasattr(module, 'conv3'):
                    # Bottleneck: conv3 is the last conv, don't prune it
                    for n, m in module.named_modules():
                        if n == 'conv3':
                            skip_layers.add(id(m))
                else:
                    # BasicBlock: conv2 is the last conv, don't prune it
                    for n, m in module.named_modules():
                        if n == 'conv2':
                            skip_layers.add(id(m))

            # Also skip downsample conv layers
            if 'downsample' in name and isinstance(module, nn.Conv2d):
                skip_layers.add(id(module))

        # Also skip the first stem conv (changing its output affects everything)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                skip_layers.add(id(module))
                break  # Only the first one

        for name, module in conv_layers:
            if id(module) in skip_layers:
                logger.info(f"  Layer {name}: Conv2d out_ch={module.weight.shape[0]} [SKIP - residual path]")
                continue

            out_channels = module.weight.shape[0]
            num_groups = out_channels  # Prune output filters

            self.weight_groups[name] = {
                'module': module,
                'num_groups': num_groups,
                'type': 'conv2d',
            }

            # Algorithm 1, Step 4: Initialize p^k = 0 for each weight group
            self.pruning_probs[name] = torch.zeros(num_groups)

            logger.info(f"  Layer {name}: Conv2d out_ch={out_channels} [PRUNABLE]")

    def _calculate_l1_norm(self, name: str) -> torch.Tensor:
        """
        Calculate L1 norm for each output filter.
        Sum across (in_channels, kernel_h, kernel_w) dimensions.

        Returns:
            L1 norms shape: (out_channels,)
        """
        module = self.weight_groups[name]['module']
        weight = module.weight  # Shape: (out_channels, in_channels, kh, kw)

        # L1 norm per output filter
        l1_norm = weight.detach().abs().sum(dim=(1, 2, 3)).cpu()  # Shape: (out_channels,)

        return l1_norm

    def _compute_delta_increment(self, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Δ(r) increment for pruning probability update
        Based on Eqn(3): Center-symmetric exponential function

        Δ(r) = {
            A*e^(-α*r),           if r ≤ N
            2uA - A*e^(-α(2N-r)), if r > N
        }

        Where:
        - r is the rank of L1 norm (sorted in ascending order)
        - N = -log(u)/α (computed from constraints)
        - α = (log(2)-log(u))/RNc (computed from constraints)
        - Nc = number of ACTIVE (non-frozen) weight groups in layer
        - R = target pruning ratio adjusted for active groups

        Frozen groups (p=1) are excluded from ranking so they don't
        distort the importance ordering of active groups.
        """
        l1_norm = self._calculate_l1_norm(layer_name)
        total = len(l1_norm)
        delta = torch.zeros(total)

        # Determine which groups are frozen (permanently pruned)
        frozen_mask = self.frozen.get(layer_name, torch.zeros(total, dtype=torch.bool))
        active_mask = ~frozen_mask
        num_active = active_mask.sum().item()

        if num_active == 0:
            return delta, torch.zeros(total, dtype=torch.long)

        # Only rank ACTIVE (non-frozen) groups
        active_indices = torch.where(active_mask)[0]
        active_l1 = l1_norm[active_indices]

        # Sort active groups by L1 norm in ascending order to get ranks
        _, sorted_idx = torch.sort(active_l1)
        # rank_of_active[j] = rank of the j-th active group (0 = smallest L1)
        rank_of_active = torch.zeros(len(active_indices), dtype=torch.long)
        rank_of_active[sorted_idx] = torch.arange(len(active_indices))

        # Nc = number of active groups (frozen ones are already pruned)
        Nc = num_active
        # How many MORE active groups need to reach p=1
        num_already_frozen = frozen_mask.sum().item()
        target_total_pruned = int(self.pruning_ratio * total)
        remaining_to_prune = max(target_total_pruned - num_already_frozen, 0)
        # Effective pruning ratio among active groups
        R_eff = remaining_to_prune / Nc if Nc > 0 else 0.0

        if R_eff <= 0 or R_eff >= 1:
            # No more pruning needed, or would prune everything
            return delta, torch.zeros(total, dtype=torch.long)

        # Compute α and N from constraints
        alpha = (np.log(2) - np.log(self.u)) / (R_eff * Nc) if R_eff * Nc > 0 else 1.0
        N = -np.log(self.u) / alpha if alpha > 0 else Nc / 2

        # Compute delta for each active group based on its rank
        for j, orig_idx in enumerate(active_indices):
            r_val = rank_of_active[j].item()

            if r_val <= N:
                exponent = -alpha * r_val
                delta[orig_idx] = self.A * np.exp(exponent)
            else:
                exponent = -alpha * (2 * N - r_val)
                # Clamp exponent to avoid overflow when r >> 2N
                # np.exp(88) ≈ 1.6e38 ≈ float32 max
                exponent = np.clip(exponent, -80, 80)
                delta[orig_idx] = 2 * self.u * self.A - self.A * np.exp(exponent)

        # Build full rank_positions for compatibility (frozen get rank=-1)
        rank_positions = torch.full((total,), -1, dtype=torch.long)
        for j, orig_idx in enumerate(active_indices):
            rank_positions[orig_idx] = rank_of_active[j]

        return delta, rank_positions

    def update_pruning_probabilities(self, layer_name: str):
        """
        Update pruning probabilities based on importance
        Eqn(4): p^{k+1} = clamp(p^k + Δ(r^k), 0, 1)

        Frozen groups (p=1) are permanently pruned and never updated.
        When p newly reaches 1.0, the group is frozen and its weights are
        permanently zeroed.
        """
        delta, ranks = self._compute_delta_increment(layer_name)

        # Initialize frozen mask if needed
        if layer_name not in self.frozen:
            self.frozen[layer_name] = torch.zeros(len(self.pruning_probs[layer_name]), dtype=torch.bool)

        # Only update non-frozen groups
        frozen_mask = self.frozen[layer_name]
        new_probs = torch.clamp(
            self.pruning_probs[layer_name] + delta, 0.0, 1.0
        )
        # Keep frozen groups at p=1
        new_probs[frozen_mask] = 1.0
        self.pruning_probs[layer_name] = new_probs

        # Freeze newly reached p=1 groups and permanently zero their weights
        newly_frozen = (new_probs == 1.0) & (~frozen_mask)
        if newly_frozen.any():
            self.frozen[layer_name] = frozen_mask | newly_frozen
            module = self.weight_groups[layer_name]['module']
            for idx in torch.where(newly_frozen)[0]:
                module.weight.data[idx].zero_()
                if module.bias is not None:
                    module.bias.data[idx].zero_()
                # Also zero saved weights so they stay zero
                if layer_name in self._saved_weights:
                    self._saved_weights[layer_name][idx].zero_()
                    if layer_name in self._saved_biases and self._saved_biases[layer_name] is not None:
                        self._saved_biases[layer_name][idx].zero_()

    def step(self, training_iteration: int):
        """
        Called at each training iteration to update pruning probabilities.
        Algorithm 1: If mod(i,t)=0, update p^k
        """
        self.training_iteration = training_iteration

        if training_iteration % self.update_interval == 0:
            logger.info(f"Iteration {training_iteration}: Updating pruning probabilities (k={self.update_count})")

            for layer_name in self.weight_groups.keys():
                self.update_pruning_probabilities(layer_name)

            self.update_count += 1
            self._log_pruning_status()

    def apply_pruning_masks(self):
        """
        Monte Carlo sampling: generate binary masks from pruning probabilities.
        P(g=0) = p (prune), P(g=1) = 1-p (keep)

        Frozen groups (p=1) always get mask=0.
        """
        for layer_name, probs in self.pruning_probs.items():
            # Sample: mask=1 means keep, mask=0 means prune
            sampled_masks = torch.bernoulli(1 - probs)  # Shape: (out_channels,)

            # Force frozen groups to mask=0
            if layer_name in self.frozen:
                sampled_masks[self.frozen[layer_name]] = 0.0

            module = self.weight_groups[layer_name]['module']
            weight = module.weight  # Shape: (out_channels, in_channels, kh, kw)

            # Expand mask for output channel dimension
            mask = sampled_masks.view(-1, 1, 1, 1)  # Shape: (out_channels, 1, 1, 1)

            self.masks[layer_name] = mask.to(weight.device)

    def prune_forward(self):
        """
        Apply masks to weights: W_pruned = g ⊙ W (Eqn 2).
        NON-DESTRUCTIVE: saves original weights before masking so they can
        be restored after the backward pass (enabling the recovery mechanism).
        """
        for layer_name, mask in self.masks.items():
            module = self.weight_groups[layer_name]['module']

            # Save original weights before masking
            self._saved_weights[layer_name] = module.weight.data.clone()
            if module.bias is not None:
                self._saved_biases[layer_name] = module.bias.data.clone()

            # Apply mask (non-permanently)
            module.weight.data = module.weight.data * mask
            if module.bias is not None:
                bias_mask = mask.view(-1)
                module.bias.data = module.bias.data * bias_mask

    def restore_weights(self):
        """
        Restore original weights after backward pass.
        This preserves weights for probabilistic recovery — only frozen (p=1)
        weights stay zeroed.
        """
        for layer_name in self._saved_weights:
            module = self.weight_groups[layer_name]['module']
            module.weight.data = self._saved_weights[layer_name]
            if layer_name in self._saved_biases and self._saved_biases[layer_name] is not None:
                module.bias.data = self._saved_biases[layer_name]
        self._saved_weights.clear()
        self._saved_biases.clear()

    def zero_masked_gradients(self):
        """
        Zero gradients of masked weights so they are not updated during backprop.
        Paper: "the masked weights are also not updated during back propagations."
        """
        for layer_name, mask in self.masks.items():
            module = self.weight_groups[layer_name]['module']
            if module.weight.grad is not None:
                module.weight.grad.data *= mask
            if module.bias is not None and module.bias.grad is not None:
                bias_mask = mask.view(-1)
                module.bias.grad.data *= bias_mask

    def apply_final_masks(self):
        """
        Apply deterministic masks during retraining phase.
        Filters with p=1 (frozen) are permanently masked, others are kept.
        Non-destructive: saves and restores weights.
        """
        for layer_name, probs in self.pruning_probs.items():
            module = self.weight_groups[layer_name]['module']

            # Deterministic: mask=0 for p=1 (frozen), mask=1 otherwise
            final_mask = (probs < 1.0).float()
            mask = final_mask.view(-1, 1, 1, 1).to(module.weight.device)
            self.masks[layer_name] = mask

            # Save and apply
            self._saved_weights[layer_name] = module.weight.data.clone()
            if module.bias is not None:
                self._saved_biases[layer_name] = module.bias.data.clone()

            module.weight.data = module.weight.data * mask
            if module.bias is not None:
                bias_mask = mask.view(-1)
                module.bias.data = module.bias.data * bias_mask

    def apply_eval_scaling(self):
        """
        Scale weights for evaluation — analogous to dropout scaling at test time.

        During training, filter i is active with probability (1-p_i).
        BN running stats reflect this scaled distribution.
        At eval time, we scale weight and bias by (1-p_i) to match:

            E[output_i] = (1 - p_i) * (W_i * x + b_i)

        This ensures feature distribution at eval matches BN running stats.
        """
        for layer_name, probs in self.pruning_probs.items():
            module = self.weight_groups[layer_name]['module']

            scale = (1.0 - probs).view(-1, 1, 1, 1).to(module.weight.device)

            # Save original
            self._saved_weights[layer_name] = module.weight.data.clone()
            if module.bias is not None:
                self._saved_biases[layer_name] = module.bias.data.clone()

            # Scale weight and bias by (1 - p)
            module.weight.data = module.weight.data * scale
            if module.bias is not None:
                bias_scale = (1.0 - probs).to(module.bias.device)
                module.bias.data = module.bias.data * bias_scale

            # Store scale as mask for zero_masked_gradients compatibility
            self.masks[layer_name] = scale

    def should_stop_pruning(self) -> bool:
        """Check if weight group ratio of p^k=1 equals R"""
        total_groups = 0
        pruned_groups = 0

        for layer_name, probs in self.pruning_probs.items():
            total_groups += len(probs)
            pruned_groups += (probs == 1.0).sum().item()

        current_ratio = pruned_groups / total_groups if total_groups > 0 else 0
        should_stop = current_ratio >= self.pruning_ratio

        if should_stop:
            logger.info(f"Target pruning ratio reached: {current_ratio:.2%} >= {self.pruning_ratio:.2%}")

        return should_stop

    def get_final_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get deterministic final masks based on frozen state.
        Paper: "Only when p is increased to 1 will the weights be
        permanently removed from the network."

        Only p=1 (frozen) groups are pruned. Groups with 0 < p < 1
        are kept — they were never permanently eliminated.

        Returns:
            Dict mapping layer_name -> binary mask (1=keep, 0=prune)
        """
        final_masks = {}
        for layer_name, probs in self.pruning_probs.items():
            # mask=0 only for p==1 (frozen), everything else is kept
            final_masks[layer_name] = (probs < 1.0).float()
        return final_masks

    def export_pruned_model(self) -> nn.Module:
        """
        Export a physically smaller model by removing pruned filters.
        This creates a NEW model with fewer channels — actually reduces
        model size, parameters, and FLOPs.

        Returns:
            New pruned nn.Module with reduced channels
        """
        logger.info("Exporting pruned model...")

        final_masks = self.get_final_masks()

        # Log what we're pruning
        for layer_name, mask in final_masks.items():
            kept = mask.sum().int().item()
            total = len(mask)
            logger.info(f"  {layer_name}: keeping {kept}/{total} filters ({kept/total:.1%})")

        # Build mapping: conv_name -> indices of kept output channels
        kept_indices = {}
        for layer_name, mask in final_masks.items():
            kept_indices[layer_name] = torch.where(mask > 0)[0]

        # Deep copy the model and modify in place
        new_model = copy.deepcopy(self.model)

        # For each prunable conv layer, we need to:
        # 1. Slice its output filters (and bias)
        # 2. Slice the corresponding BatchNorm
        # 3. Slice the NEXT layer's input channels
        self._apply_structural_pruning(new_model, kept_indices)

        # Verify the model works
        device = next(new_model.parameters()).device
        new_model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 32, 32).to(device)
            output = new_model(dummy)
            logger.info(f"  Export verified: output shape = {output.shape}")

        # Log size comparison
        old_params = sum(p.numel() for p in self.model.parameters())
        new_params = sum(p.numel() for p in new_model.parameters())
        logger.info(f"  Parameters: {old_params:,} -> {new_params:,} ({new_params/old_params:.1%})")
        logger.info(f"  Size: {old_params*4/1024**2:.2f} MB -> {new_params*4/1024**2:.2f} MB")

        return new_model

    def _apply_structural_pruning(self, model: nn.Module, kept_indices: Dict[str, torch.Tensor]):
        """
        Physically remove pruned filters from the model.

        For each prunable conv:
        1. Keep only the selected output filters
        2. Update the following BatchNorm
        3. Update the next conv layer's input channels
        """
        # Build ordered list of (name, module) for all conv and bn layers
        all_modules = dict(model.named_modules())

        # For each prunable layer, find the associated BN and next conv
        for layer_name, indices in kept_indices.items():
            if len(indices) == 0:
                logger.warning(f"  {layer_name}: all filters pruned, skipping (would break model)")
                continue

            module = dict(model.named_modules())[layer_name]
            kept = indices.to(module.weight.device)
            num_kept = len(kept)

            # 1. Prune output filters of this conv
            new_weight = module.weight.data[kept]  # (num_kept, in_ch, kh, kw)
            new_bias = module.bias.data[kept] if module.bias is not None else None

            new_conv = nn.Conv2d(
                in_channels=module.in_channels,
                out_channels=num_kept,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            ).to(module.weight.device)

            new_conv.weight.data = new_weight
            if new_bias is not None:
                new_conv.bias.data = new_bias

            self._set_module(model, layer_name, new_conv)

            # 2. Find and prune the corresponding BatchNorm
            bn_name = self._find_next_bn(model, layer_name)
            if bn_name:
                bn_module = dict(model.named_modules())[bn_name]
                new_bn = nn.BatchNorm2d(num_kept).to(bn_module.weight.device)
                new_bn.weight.data = bn_module.weight.data[kept]
                new_bn.bias.data = bn_module.bias.data[kept]
                new_bn.running_mean = bn_module.running_mean[kept]
                new_bn.running_var = bn_module.running_var[kept]
                new_bn.num_batches_tracked = bn_module.num_batches_tracked
                self._set_module(model, bn_name, new_bn)

            # 3. Find and adjust the next conv's input channels
            next_conv_name = self._find_next_conv(model, layer_name)
            if next_conv_name:
                next_module = dict(model.named_modules())[next_conv_name]

                if next_module.groups > 1:
                    # Depthwise conv: adjust groups too
                    continue

                new_next_weight = next_module.weight.data[:, kept, :, :]

                new_next_conv = nn.Conv2d(
                    in_channels=num_kept,
                    out_channels=next_module.out_channels,
                    kernel_size=next_module.kernel_size,
                    stride=next_module.stride,
                    padding=next_module.padding,
                    dilation=next_module.dilation,
                    groups=next_module.groups,
                    bias=next_module.bias is not None,
                ).to(next_module.weight.device)

                new_next_conv.weight.data = new_next_weight
                if next_module.bias is not None:
                    new_next_conv.bias.data = next_module.bias.data.clone()

                self._set_module(model, next_conv_name, new_next_conv)

    def _find_next_bn(self, model: nn.Module, conv_name: str) -> Optional[str]:
        """Find the BatchNorm layer immediately after a conv layer."""
        # In ResNet, BN follows conv with matching name pattern
        # e.g., conv1 -> bn1, layer1.0.conv1 -> layer1.0.bn1
        bn_name = conv_name.replace('conv', 'bn')
        for name, module in model.named_modules():
            if name == bn_name and isinstance(module, nn.BatchNorm2d):
                return bn_name
        return None

    def _find_next_conv(self, model: nn.Module, conv_name: str) -> Optional[str]:
        """Find the next conv layer that takes this conv's output as input."""
        # For ResNet BasicBlock: conv1 -> bn1 -> relu -> conv2
        # So next conv after conv1 is conv2 in the same block
        parts = conv_name.rsplit('.', 1)
        if len(parts) == 2:
            prefix, name = parts
            if name == 'conv1':
                next_name = prefix + '.conv2'
                for n, m in model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
            elif name == 'conv2' and self._has_conv3(model, prefix):
                # Bottleneck: conv2 -> conv3
                next_name = prefix + '.conv3'
                for n, m in model.named_modules():
                    if n == next_name and isinstance(m, nn.Conv2d):
                        return next_name
        return None

    def _has_conv3(self, model: nn.Module, prefix: str) -> bool:
        """Check if a block has conv3 (Bottleneck)."""
        for name, module in model.named_modules():
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

    def _log_pruning_status(self):
        """Log current pruning status"""
        logger.info(f"Pruning status at iteration {self.training_iteration}:")

        for layer_name, probs in self.pruning_probs.items():
            num_total = len(probs)
            num_pruned = (probs == 1.0).sum().item()
            num_intermediate = ((probs > 0) & (probs < 1.0)).sum().item()
            ratio_pruned = num_pruned / num_total if num_total > 0 else 0
            ratio_intermediate = num_intermediate / num_total if num_total > 0 else 0

            logger.info(
                f"  {layer_name}: "
                f"pruned={num_pruned}/{num_total} ({ratio_pruned:.1%}), "
                f"intermediate={num_intermediate}/{num_total} ({ratio_intermediate:.1%})"
            )

    def get_pruning_ratio(self) -> float:
        """Get current pruning ratio (number of p=1 groups / total groups)"""
        total_groups = 0
        pruned_groups = 0

        for layer_name, probs in self.pruning_probs.items():
            total_groups += len(probs)
            pruned_groups += (probs == 1.0).sum().item()

        return pruned_groups / total_groups if total_groups > 0 else 0

    def get_stats(self) -> Dict:
        """Get pruning statistics"""
        total_groups = 0
        pruned_groups = 0
        intermediate_groups = 0

        for layer_name, probs in self.pruning_probs.items():
            total_groups += len(probs)
            pruned_groups += (probs == 1.0).sum().item()
            intermediate_groups += ((probs > 0) & (probs < 1.0)).sum().item()

        return {
            'total_groups': total_groups,
            'pruned_groups': pruned_groups,
            'intermediate_groups': intermediate_groups,
            'pruning_ratio': pruned_groups / total_groups if total_groups > 0 else 0,
            'update_count': self.update_count,
            'training_iteration': self.training_iteration,
        }

    def save_checkpoint(self, path: str):
        """Save pruning state"""
        checkpoint = {
            'pruning_probs': self.pruning_probs,
            'masks': self.masks,
            'frozen': self.frozen,
            'update_count': self.update_count,
            'training_iteration': self.training_iteration,
            'pruning_ratio': self.pruning_ratio,
            'A': self.A,
            'u': self.u,
            'update_interval': self.update_interval,
        }
        torch.save(checkpoint, path)
        logger.info(f"Pruning checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load pruning state"""
        checkpoint = torch.load(path)
        self.pruning_probs = checkpoint['pruning_probs']
        self.masks = checkpoint['masks']
        self.frozen = checkpoint.get('frozen', {})
        self.update_count = checkpoint['update_count']
        self.training_iteration = checkpoint['training_iteration']
        logger.info(f"Pruning checkpoint loaded from {path}")
