"""
GAN upsampler for complex CSM super-resolution.

This module implements a GAN-inspired upsampler adapted from the HRTF GAN
architecture of Hogg et al. to the CSM setting used in this repository.

Adaptations for this codebase:
- Uses standard 2D convolutions on real/imaginary CSM channels.
- Keeps generator inference compatible with existing upsampler interfaces.
- Owns adversarial and content optimisation inside ``training_step``.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import cast

import torch
import torch.nn.functional as torch_f
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from upsampler.base import TrainableUpsampler

OptimiserInput = torch.optim.Optimizer | dict[str, torch.optim.Optimizer]


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with two conv layers and a skip connection.

    As we are working with complex CSMs represented as 2-channel real tensors, we use standard 2D
    convolutions with 2 input/output channels and appropriate activations, rather than the
    CubeSphereConv used in the original HRTF GAN.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize the residual convolutional block.

        Parameters
        ----------
        channels : int
            Number of input and output channels (should be 2 for real/imaginary CSM representation).
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float32
            ),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float32
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W], where C should match the
            number of channels specified in the constructor.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input, after applying the residual block.
        """
        return x + cast(torch.Tensor, self.block(x))


class UpsampleBlock(nn.Module):
    """
    Pixel-shuffle upsampling block.

    As we are working with complex CSMs represented as 2-channel real tensors, we use standard 2D
    convolutions with 2 input/output channels and appropriate activations, rather than the
    CubeSphereConv used in the original HRTF GAN.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialise the upsampling block.

        Parameters
        ----------
        channels : int
            Number of input channels (should be 2 for real/imaginary CSM representation).
        """
        super().__init__()
        self.upsample_block_1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1,
                bias=True,
                dtype=torch.float32,
            ),
        )
        self.upsample_block_2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply upsampling transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W], where C should match the
            number of channels specified in the constructor.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, C, 2*H, 2*W] after applying pixel-shuffle upsampling.
        """
        out1 = self.upsample_block_1(x)
        return cast(torch.Tensor, self.upsample_block_2(out1))


class GANGenerator(nn.Module):
    """
    Generator network for CSM super-resolution.

    This generator is adapted from the HRTF GAN architecture of Hogg et al. to the CSM setting used
    in this repository.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels: int,
        n_residual_blocks: int = 8,
    ) -> None:
        """
        Initialise the generator.

        Parameters
        ----------
        in_channels : int
            Number of input channels (should be 2 for real/imaginary CSM representation).
        out_channels : int
            Number of output channels (should be 2 for real/imaginary CSM representation).
        feature_channels : int
            Number of feature channels in the convolutional layers.
        n_residual_blocks : int
            Number of residual blocks in the features trunk.
            Default is 8, as in the original HRTF GAN architecture.
        """
        super().__init__()
        if out_channels % in_channels != 0:
            raise ValueError(
                "out_channels must be an integer multiple of in_channels. "
                f"Got in_channels={in_channels}, out_channels={out_channels}."
            )

        upscale_factor = out_channels // in_channels
        if upscale_factor <= 0:
            raise ValueError("upscale_factor must be > 0.")
        if upscale_factor & (upscale_factor - 1) != 0:
            raise ValueError(
                "GANUpsampler requires power-of-two upscale factor for PixelShuffle. "
                f"Got upscale_factor={upscale_factor}."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.upscale_factor = int(upscale_factor)

        # First convultional layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                2, feature_channels, kernel_size=3, padding=1, bias=True, dtype=torch.float32
            ),
            nn.PReLU(),
        )

        # Features trunk blocks
        trunk: list[nn.Module] = []
        for _ in range(n_residual_blocks):
            trunk.append(ResidualConvBlock(feature_channels))
        self.trunk = nn.Sequential(*trunk)

        # Second convolutional layer
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                dtype=torch.float32,
            ),
            nn.BatchNorm2d(feature_channels),
        )

        # Upscale block
        upsampling_layers: list[nn.Module] = []
        num_upsampling_blocks = self.upscale_factor.bit_length() - 1
        for _ in range(num_upsampling_blocks):
            upsampling_layers.append(UpsampleBlock(feature_channels))
        self.upsampling = nn.Sequential(*upsampling_layers)

        # Output layer
        self.conv_block3 = nn.Conv2d(
            feature_channels,
            2,  # Output 2 channels for real and imaginary parts of the CSM
            kernel_size=3,
            padding=1,
            bias=True,
            dtype=torch.float32,
        )

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Initialise generator weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward generator pass on 2-channel real-valued tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 2, H, W], where the 2 channels represent the real and
            imaginary parts of the CSM.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, 2, H', W'], where H' and W' are determined by the
            upscale factor (out_channels // in_channels) and the architecture of the generator.
        """
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)
        return cast(torch.Tensor, out)


class GANDiscriminator(nn.Module):
    """Discriminator network for adversarial CSM training."""

    def __init__(self, out_channels: int) -> None:
        """
        Initialise the discriminator.

        Parameters
        ----------
        out_channels : int
            Number of output channels from the generator
            (should be 2 for real/imaginary CSM representation).
        """
        super().__init__()
        # Number of hidden channels in each layer, adapted from Hogg et al.'s HRTF GAN architecture.
        hidden = [64, 64, 128, 128, 256, 256, 512, 512]
        layers: list[nn.Module] = []
        in_ch = 2  # Input has 2 channels for real and imaginary parts of the CSM

        # Loop to build the convolutional layers
        # It is the same, but abbreviated, as in the orginal implementation.
        for index, out_ch in enumerate(hidden):
            # Match original discriminator schedule:
            # [64, 64, 128, 128, 256, 256, 512, 512] with strides
            # [1, 1, 1, 2, 1, 2, 1, 2]
            stride = 1 if index == 1 or index % 2 == 0 else 2
            use_bias = index == 0
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=use_bias,
                    dtype=torch.float32,
                )
            )
            if index != 0:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden[-1] * 2 * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        self._initialise_weights()

        self.expected_out_channels = int(out_channels)

    def _initialise_weights(self) -> None:
        """Initialise discriminator weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify real/fake upsampled samples.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 2, H, W], where the 2 channels represent the real and
            imaginary parts of the CSM.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, 1], where each value represents the discriminator's
            confidence that the corresponding input sample is real (close to 1) or fake
            (close to 0).
        """
        if x.shape[-2:] != (self.expected_out_channels, self.expected_out_channels):
            expected_shape = (self.expected_out_channels, self.expected_out_channels)
            actual_shape = tuple(x.shape[-2:])
            raise ValueError(
                "Discriminator input must match expected shape "
                f"{expected_shape}, got {actual_shape}."
            )

        out = self.features(x)
        # Global average pooling to reduce spatial dimensions to 2x2 before the classifier
        out = torch_f.adaptive_avg_pool2d(out, (2, 2))
        out = torch.flatten(out, 1)
        return cast(torch.Tensor, self.classifier(out))


class GANUpsampler(TrainableUpsampler):  # type: ignore[no-any-unimported]
    """GAN-based upsampler for complex CSM tensors."""

    EXPECTED_INPUT_NDIM = 4
    COMPLEX_CHANNELS = 2
    BETAS_LEN = 2

    def __init__(  # noqa: C901, PLR0913
        self,
        in_channels: int = 4,
        out_channels: int = 32,
        feature_channels: int = 128,
        n_residual_blocks: int = 8,
        loss_name: str = "l1",
        adversarial_weight: float = 0.01,
        content_weight: float = 1.0,
        critic_iters: int = 1,
        discriminator_lr_scale: float = 1.0,
        beta1_adam: float = 0.9,
        beta2_adam: float = 0.999,
    ) -> None:
        """
        Initialise the GAN upsampler.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels (default: 4, representing real and imaginary parts of 2 CSM
            channels).
        out_channels : int, optional
            Number of output channels (default: 32).
        feature_channels : int, optional
            Number of feature channels in the generator (default: 128).
        n_residual_blocks : int, optional
            Number of residual blocks in the generator trunk.
            Defaults to 8.
        loss_name : str, optional
            Name of the reconstruction loss to use in the content loss. Options are "l1" and "mse"
            (default: "l1").
        adversarial_weight : float, optional
            Weight for the adversarial loss component in the generator loss (default: 0.01).
        content_weight : float, optional
            Weight for the content loss component in the generator loss (default: 1.0).
        critic_iters : int, optional
            Number of iterations to train the critic per generator iteration (default: 1).
        discriminator_lr_scale : float, optional
            Scale factor for the discriminator learning rate (default: 1.0).
        beta1_adam : float, optional
            Beta1 for the Adam optimiser (default: 0.9).
        beta2_adam : float, optional
            Beta2 for the Adam optimiser (default: 0.999).
        """
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_channels = int(feature_channels)
        self.n_residual_blocks = int(n_residual_blocks)
        self.loss_name = str(loss_name).strip().lower()
        self.adversarial_weight = float(adversarial_weight)
        self.content_weight = float(content_weight)
        self.critic_iters = int(critic_iters)
        self.discriminator_lr_scale = float(discriminator_lr_scale)
        self.beta1 = float(beta1_adam)
        self.beta2 = float(beta2_adam)

        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be > 0.")
        if self.n_residual_blocks <= 0:
            raise ValueError("n_residual_blocks must be > 0.")
        if self.adversarial_weight < 0.0:
            raise ValueError("adversarial_weight must be >= 0.")
        if self.content_weight <= 0.0:
            raise ValueError("content_weight must be > 0.")
        if self.critic_iters <= 0:
            raise ValueError("critic_iters must be > 0.")
        if self.discriminator_lr_scale <= 0.0:
            raise ValueError("discriminator_lr_scale must be > 0.")
        if not 0.0 < self.beta1 < 1.0:
            raise ValueError("beta1_adam must be in (0, 1).")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError("beta2_adam must be in (0, 1).")

        self.generator = GANGenerator(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_channels=self.feature_channels,
            n_residual_blocks=self.n_residual_blocks,
        )
        self.discriminator = GANDiscriminator(out_channels=self.out_channels)

        self.reconstruction_loss = self._build_reconstruction_loss(self.loss_name)
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self._step_counter = 0

    def _build_reconstruction_loss(self, loss_name: str) -> nn.Module:
        """
        Build reconstruction criterion used in content loss.

        Parameters
        ----------
        loss_name : str
            Name of the reconstruction loss to use. Options are "l1" and "mse".

        Returns
        -------
        nn.Module
            The reconstruction loss criterion module.
        """
        if loss_name == "l1":
            return nn.L1Loss()
        if loss_name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unsupported loss_name '{loss_name}'. Use one of: l1, mse.")

    def _extract_optimisers(
        self, optimiser: OptimiserInput
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Extract generator and discriminator optimisers from training input.

        Parameters
        ----------
        optimiser : StepOptimiser
            Optimiser input for the training step. This can be a single optimiser
            (for non-adversarial models) or a dictionary of optimisers for models
            with multiple parameter groups (e.g., GAN generator/discriminator).

        Returns
        -------
        tuple[torch.optim.Optimizer, torch.optim.Optimizer]
            A tuple containing the generator optimiser and discriminator optimiser.
        """
        if isinstance(optimiser, dict):
            optim_g = optimiser.get("generator")
            optim_d = optimiser.get("discriminator")
            if optim_g is None or optim_d is None:
                raise ValueError(
                    "GANUpsampler requires optimiser dict keys 'generator' and 'discriminator'."
                )
            return optim_g, optim_d

        raise ValueError(
            "GANUpsampler requires a dict of optimisers with keys "
            "'generator' and 'discriminator'."
        )

    def _reshape_csm_to_ri(self, S: torch.Tensor, channels: int) -> torch.Tensor:
        """
        Convert complex CSM tensor [B,F,C,C] to real-imag tensor [B*F,2,C,C].

        Parameters
        ----------
        S : torch.Tensor
            Input complex CSM tensor of shape (batch_size, num_bands, channels, channels).
        channels : int
            Number of CSM channels (should match the last two dimensions of S).

        Returns
        -------
        torch.Tensor
            Output real-imag tensor of shape (batch_size * num_bands, 2, channels, channels), where
            the 2 channels represent the real and imaginary parts of the CSM.
        """
        batch_size, num_bands, _, _ = S.shape
        real = S.real.reshape(batch_size * num_bands, 1, channels, channels)
        imag = S.imag.reshape(batch_size * num_bands, 1, channels, channels)
        return torch.cat((real, imag), dim=1)

    def _reshape_ri_to_csm(self, x: torch.Tensor, batch_size: int, num_bands: int) -> torch.Tensor:
        """
        Convert real-imag tensor [B*F,2,C,C] back to complex CSM [B,F,C,C].

        Parameters
        ----------
        x : torch.Tensor
            Input real-imag tensor of shape (batch_size * num_bands, 2,
            out_channels, out_channels), where the 2 channels represent the real and imaginary parts
            of the CSM.
        batch_size : int
            Original batch size before reshaping.
        num_bands : int
            Original number of frequency bands before reshaping.

        Returns
        -------
        torch.Tensor
            Output complex CSM tensor of shape (batch_size, num_bands, out_channels, out_channels).
        """
        S_up = torch.complex(x[:, 0], x[:, 1]).reshape(
            batch_size, num_bands, self.out_channels, self.out_channels
        )
        return 0.5 * (S_up + S_up.transpose(-1, -2).conj())

    def _forward_no_metrics(self, S_low: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without timing/FLOPs/memory collection.

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex CSM tensor of shape [B, F, in_channels, in_channels].

        Returns
        -------
        torch.Tensor
            Output complex CSM tensor of shape [B, F, out_channels, out_channels].
        """
        batch_size, num_bands, _, _ = S_low.shape
        gen_input = self._reshape_csm_to_ri(S_low, channels=self.in_channels)
        gen_output = self.generator(gen_input.to(torch.float32)).to(gen_input.dtype)

        if gen_output.shape[1] != self.COMPLEX_CHANNELS:
            raise RuntimeError(
                "Generator output channel mismatch. "
                f"Expected {self.COMPLEX_CHANNELS}, got {gen_output.shape[1]}."
            )
        if gen_output.shape[-2:] != (self.out_channels, self.out_channels):
            raise RuntimeError(
                "Generator output spatial shape mismatch. Expected "
                f"({self.out_channels}, {self.out_channels}), got {tuple(gen_output.shape[-2:])}."
            )

        return self._reshape_ri_to_csm(gen_output, batch_size=batch_size, num_bands=num_bands)

    def forward(
        self, S_low: torch.Tensor, collect_metrics: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Upsample complex CSM tensors from [B,F,in_channels,in_channels].

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex CSM tensor of shape [B, F, in_channels, in_channels].
        collect_metrics : bool, optional
            Whether to collect timing, FLOPs, and memory metrics during the forward pass.
            Default is False.

        Returns
        -------
        torch.Tensor
            If collect_metrics is False, returns the upsampled complex CSM tensor of shape
            [B, F, out_channels, out_channels].
        tuple[torch.Tensor, dict[str, float]]
            If collect_metrics is True, returns a tuple containing the upsampled complex CSM tensor
            and a dictionary of collected metrics, including:
                - "upsampler_time_ms": Time taken for the forward pass in milliseconds.
                - "upsampler_flops": Total FLOPs for the forward pass.
                - "upsampler_flops_per_frame": FLOPs per frame (total FLOPs divided by batch size).
                - "upsampler_memory_mb": Peak memory usage during the forward pass in megabytes.
                - "num_frames": Number of frames in the input batch.
        """
        if S_low.ndim != self.EXPECTED_INPUT_NDIM:
            raise ValueError(f"S_low must be 4D [B, F, C, C], got shape={tuple(S_low.shape)}")
        if not S_low.is_complex():
            raise ValueError("S_low must be a complex tensor.")
        if S_low.shape[-2:] != (self.in_channels, self.in_channels):
            raise ValueError(
                "S_low must have shape [B, F, "
                f"{self.in_channels}, {self.in_channels}] on the last two dimensions, "
                f"got {tuple(S_low.shape)}"
            )

        use_cuda = S_low.device.type == "cuda"
        start = 0.0
        if collect_metrics:
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(S_low.device)
                torch.cuda.synchronize(S_low.device)
            else:
                tracemalloc.start()
            start = time.perf_counter()

        batch_size = int(S_low.shape[0])
        S_up = self._forward_no_metrics(S_low)

        if not collect_metrics:
            return S_up

        if use_cuda:
            torch.cuda.synchronize(S_low.device)
        end = time.perf_counter()

        if use_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(S_low.device) / (1024.0 * 1024.0)
        else:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory / (1024.0 * 1024.0)

        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad():
            with flop_counter:
                self._forward_no_metrics(S_low)
        total_flops = float(flop_counter.get_total_flops())
        flops_per_frame = total_flops / batch_size if batch_size > 0 else 0.0

        metrics: dict[str, float] = {
            "upsampler_time_ms": (end - start) * 1000.0,
            "upsampler_flops": total_flops,
            "upsampler_flops_per_frame": flops_per_frame,
            "upsampler_memory_mb": float(peak_memory_mb),
            "num_frames": float(batch_size),
        }
        return S_up, metrics

    def _compute_reconstruction_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute complex-domain reconstruction loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted complex CSM tensor of shape [B, F, out_channels, out_channels].
        target : torch.Tensor
            Target complex CSM tensor of shape [B, F, out_channels, out_channels].

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            A tuple containing:
            - The computed reconstruction loss as a scalar tensor.
            - A dictionary of loss statistics, which includes:
                - "loss_l1": The L1 loss value if L1 loss is used.
                - "loss_mse": The MSE loss value if MSE loss is used.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

        loss_real = self.reconstruction_loss(pred.real, target.real)
        loss_imag = self.reconstruction_loss(pred.imag, target.imag)
        content = 0.5 * (loss_real + loss_imag)

        content_value = float(content.detach().cpu().item())
        stats: dict[str, float] = {}
        if self.loss_name == "l1":
            stats["loss_l1"] = content_value
        elif self.loss_name == "mse":
            stats["loss_mse"] = content_value
        return content, stats

    def training_step(
        self,
        S_low: torch.Tensor,
        S_high: torch.Tensor,
        optimiser: OptimiserInput,
        grad_clip_norm: float = 0.0,
    ) -> dict[str, float]:
        """
        Run GAN optimisation step for discriminator and generator.

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex CSM tensor of shape [B, F, in_channels, in_channels].
        S_high : torch.Tensor
            Target complex CSM tensor of shape [B, F, out_channels, out_channels].
        optimiser : StepOptimiser
            Dictionary of optimisers with keys "generator" and "discriminator".
        grad_clip_norm : float, optional
            Maximum norm for gradient clipping. If <= 0, no clipping is applied. Default is 0.0
            (no clipping).

        Returns
        -------
        dict[str, float]
            Dictionary of training statistics, including:
            - "loss_total": Total generator loss (content + adversarial).
            - "loss_d": Total discriminator loss.
            - "loss_d_real": Discriminator loss on real samples.
            - "loss_d_fake": Discriminator loss on fake samples.
            - "loss_g": Total generator loss (same as "loss_total").
            - "loss_g_adv": Adversarial component of the generator loss.
            - "loss_g_content": Content component of the generator loss.
            - "generator_updated": 1.0 if the generator was updated in this step, 0.0 otherwise
            (based on critic_iters schedule).
            - "critic_iters": The number of critic iterations per generator iteration as specified
            in the GANUpsampler configuration.
            - "num_frames": Number of frames in the input batch.
            - "num_bands": Number of frequency bands in the input batch.
        """
        optim_g, optim_d = self._extract_optimisers(optimiser)

        # Reshape CSMs to real-imag format for discriminator and generator input, and prepare labels
        # for adversarial loss
        batch_size, num_bands, _, _ = S_low.shape
        real_ri = self._reshape_csm_to_ri(S_high, channels=self.out_channels).to(torch.float32)
        fake_csm = self._forward_no_metrics(S_low)
        fake_ri = self._reshape_csm_to_ri(fake_csm, channels=self.out_channels).to(torch.float32)

        label_real = torch.ones((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)
        label_fake = torch.zeros((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)

        # Update discriminator
        optim_d.zero_grad(set_to_none=True)
        d_real = self.discriminator(real_ri)
        d_fake = self.discriminator(fake_ri.detach())
        loss_d_real = self.adversarial_criterion(d_real, label_real)
        loss_d_fake = self.adversarial_criterion(d_fake, label_fake)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=grad_clip_norm)
        optim_d.step()

        self._step_counter += 1

        optim_g.zero_grad(set_to_none=True)
        fake_csm_for_g = self._forward_no_metrics(S_low)
        content_raw, content_stats = self._compute_reconstruction_loss(
            pred=fake_csm_for_g,
            target=S_high,
        )

        # Reshape fake CSM to real-imag format for generator adversarial loss computation
        fake_ri_for_g = self._reshape_csm_to_ri(fake_csm_for_g, channels=self.out_channels).to(
            torch.float32
        )
        adv_loss = self.adversarial_criterion(self.discriminator(fake_ri_for_g), label_real)

        # Combine content and adversarial losses with respective weights
        weighted_content = self.content_weight * content_raw
        weighted_adv = self.adversarial_weight * adv_loss
        loss_g = weighted_content + weighted_adv

        # Update generator based on critic_iters schedule
        should_step_generator = (self._step_counter % self.critic_iters) == 0
        if should_step_generator:
            loss_g.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=grad_clip_norm)
            optim_g.step()

        loss_total = float(loss_g.detach().cpu().item())
        stats: dict[str, float] = {
            "loss_total": loss_total,
            "loss_d": float(loss_d.detach().cpu().item()),
            "loss_d_real": float(loss_d_real.detach().cpu().item()),
            "loss_d_fake": float(loss_d_fake.detach().cpu().item()),
            "loss_g": loss_total,
            "loss_g_adv": float(weighted_adv.detach().cpu().item()),
            "loss_g_content": float(weighted_content.detach().cpu().item()),
            "generator_updated": 1.0 if should_step_generator else 0.0,
            "critic_iters": float(self.critic_iters),
            "num_frames": float(batch_size),
            "num_bands": float(num_bands),
        }
        stats.update(content_stats)

        return cast(dict[str, float], self.normalise_step_stats(stats))

    @torch.no_grad()
    def validation_step(self, S_low: torch.Tensor, S_high: torch.Tensor) -> dict[str, float]:
        """
        Run validation using generator reconstruction and report GAN diagnostics.

        Parameters
        ----------
        S_low : torch.Tensor
            Input complex CSM tensor of shape [B, F, in_channels, in_channels].
        S_high : torch.Tensor
            Target complex CSM tensor of shape [B, F, out_channels, out_channels].

        Returns
        -------
        dict[str, float]
            Dictionary of validation statistics, including:
            - "loss_total": Total generator loss (content + adversarial).
            - "loss_d": Total discriminator loss.
            - "loss_d_real": Discriminator loss on real samples.
            - "loss_d_fake": Discriminator loss on fake samples.
            - "loss_g": Total generator loss (same as "loss_total").
            - "loss_g_adv": Adversarial component of the generator loss.
            - "loss_g_content": Content component of the generator loss.
            - "generator_updated": 0.0 (generator is not updated during validation).
            - "critic_iters": The number of critic iterations per generator iteration as specified
            in the GANUpsampler configuration.
            - "num_frames": Number of frames in the input batch.
            - "num_bands": Number of frequency bands in the input batch.
        """
        fake_csm = self._forward_no_metrics(S_low)
        content_raw, content_stats = self._compute_reconstruction_loss(pred=fake_csm, target=S_high)

        real_ri = self._reshape_csm_to_ri(S_high, channels=self.out_channels).to(torch.float32)
        fake_ri = self._reshape_csm_to_ri(fake_csm, channels=self.out_channels).to(torch.float32)
        label_real = torch.ones((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)
        label_fake = torch.zeros((real_ri.shape[0], 1), dtype=real_ri.dtype, device=real_ri.device)

        d_real = self.discriminator(real_ri)
        d_fake = self.discriminator(fake_ri)
        loss_d_real = self.adversarial_criterion(d_real, label_real)
        loss_d_fake = self.adversarial_criterion(d_fake, label_fake)
        loss_d = loss_d_real + loss_d_fake

        adv_loss = self.adversarial_criterion(self.discriminator(fake_ri), label_real)
        weighted_content = self.content_weight * content_raw
        weighted_adv = self.adversarial_weight * adv_loss
        loss_total = weighted_content + weighted_adv

        stats: dict[str, float] = {
            "loss_total": float(loss_total.detach().cpu().item()),
            "loss_d": float(loss_d.detach().cpu().item()),
            "loss_d_real": float(loss_d_real.detach().cpu().item()),
            "loss_d_fake": float(loss_d_fake.detach().cpu().item()),
            "loss_g": float(loss_total.detach().cpu().item()),
            "loss_g_adv": float(weighted_adv.detach().cpu().item()),
            "loss_g_content": float(weighted_content.detach().cpu().item()),
            "generator_updated": 0.0,
            "critic_iters": float(self.critic_iters),
        }
        stats.update(content_stats)
        return cast(dict[str, float], self.normalise_step_stats(stats))
