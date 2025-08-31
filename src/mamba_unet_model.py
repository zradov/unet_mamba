import math
import torch
import config
import logging
from torch.optim import Adam
from torch.functional import F
from torch.nn import (
    ReLU,
    SiLU,
    Conv2d,
    MaxPool2d,
    ModuleList,
    Module,
    ModuleList,
    ConvTranspose2d,
    BCEWithLogitsLoss,
    Parameter,
    LayerNorm
)
from torch import Tensor
from typing import Tuple, List
from torchvision.transforms import CenterCrop
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(
    level=logging.DEBUG,
    format=config.DEBUG_MESSAGE_FORMAT
)
logger = logging.getLogger(__name__)


class PatchPartition(Module):

    def __init__(self, patch_size: int):
        super(PatchPartition, self).__init__()
        self.patch_size = patch_size


    def forward(self, x):
        """
        Partitions the input tensor into non-overlapping patches of size 
        patch_size x patch_size and rearranges them for further processing.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, num_patches, patch_area * C), where 
            num_patches = (H // patch_size) * (W // patch_size) and 
            patch_area = patch_size * patch_size.
        """

        B, C, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0

        x = x.view(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // P) * (W // P), P * P * C)

        return x
        

class PatchMerging(Module):

    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor) -> Tensor:
        """ 
        Partitions the input tensor into non-overlapping patches of size 
        patch_size x patch_size and rearranges them for further processing.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, num_patches, patch_area * C), where 
            num_patches = (H // patch_size) * (W // patch_size) and 
            patch_area = patch_size * patch_size.
        """

        B, C, H, W = x.shape

        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, H // 2, W // 2, 4 * C)
        x = x.permute(0, 3, 1, 2)
        
        return x
        
    
class PatchExpanding(Module):

    def __init__(self):
        super().__init__()


    def forward(self, x: Tensor) -> Tensor:
        """
        Expands the input tensor by rearranging its elements to increase spatial dimensions.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, C // 4, H * 2, W * 2), where the spatial dimensions
            are doubled and the number of channels is quartered.
        """

        B, C, H, W = x.shape

        x_out = x.permute(0, 2, 3, 1)
        x_out = x_out.reshape(B, H * 2, W * 2, C // 4)
        x_out = x_out.permute(0, 3, 1, 2)
        
        return x_out
    

class LayerNormalization(Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_dim)


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies layer normalization to the input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of the same shape as input, with layer normalization applied.
        """

        B, C, H, W = x.shape

        x_out = x.reshape(B, C, H * W).permute(0, 2, 1)
        x_out = self.norm(x_out)
        x_out = x_out.permute(0, 2, 1).reshape(B, C, H, W)

        return x_out


class LinearEmbedding(Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a linear embedding to the input tensor using a 1x1 convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, out_channels, H, W), where out_channels is the number
            of output channels specified during initialization.
        """

        return self.projection(x)


class DWCNN(Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int=3, 
                 stride: int=1, 
                 padding: int=1):
        super().__init__()
        self.depthwise = Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.activation = SiLU()


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies depthwise separable convolution followed by an activation function.
        
        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, out_channels, H, W), where out_channels is the number
            of output channels specified during initialization. The spatial dimensions
            (H, W) depend on the kernel and the stride sizes.
        """

        x_out = self.depthwise(x)
        x_out = self.pointwise(x_out)
        x_out = self.activation(x_out)

        return x_out
    

class MambaSSM(Module):

    def __init__(self, 
                 channels: int, seq_len, row_direction=True):
        super().__init__()
        self.row_direction = row_direction
        self.A = Parameter(torch.randn(1, channels, seq_len))
        self.B = Parameter(torch.randn(1, channels, seq_len))
        self.C = Parameter(torch.randn(1, channels, seq_len))


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a state space model (SSM) operation along either the rows or columns of the input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.  
        
        Returns:
            Tensor of the same shape as input, with the SSM operation applied.
        """

        B, C, H, W = x.shape

        x_temp = x.permute(0, 2, 1, 3).reshape(B * H, C, W) if self.row_direction else \
            x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        cum_sum = torch.cumsum(x_temp, dim=-1) 
        y = self.A * x_temp + self.B * cum_sum + self.C
        y = y.reshape(B, H, C, W).permute(0, 2, 1, 3)

        return y
    

class SS2D(Module):

    def __init__(self, 
                 channels: int, 
                 seq_len: int):
        super().__init__()
        self.ssm_rows = MambaSSM(channels=channels, seq_len=seq_len)
        self.ssm_cols = MambaSSM(channels=channels, seq_len=seq_len, row_direction=False)


    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a two-dimensional state space model (SSM) operation by sequentially applying
        SSM along the rows and then along the columns of the input tensor.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of the same shape as input, with the 2D SSM operation applied.
        """

        x_out = self.ssm_rows(x)
        x_out = self.ssm_cols(x_out) 

        return x_out 


class VSSBlock(Module):

    def __init__(self, 
                 in_channels: int,
                 seq_len: int, 
                 enable_patch_merging: bool=False, 
                 hidden_dim: int=None):
        """
        Initializes a Visual Mamba Block (VSSBlock) that combines patch merging,
        
        Args:
            in_channels: Number of input channels.
            seq_len: Length of the sequence for the state space model.
            enable_patch_merging: If True, applies patch merging to reduce spatial dimensions.
            hidden_dim: Dimension of the hidden layer. If None, it defaults to in_channels.
        """

        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim else in_channels
        self.enable_patch_merging = enable_patch_merging
        temp_in_channels = in_channels
        temp_seq_len = seq_len
        if enable_patch_merging:
            self.patch_merging = PatchMerging()
            # Increase the number of channels 4 times
            temp_in_channels = in_channels * 4
            # Reduce the sequence size by double
            temp_seq_len //= 2
            self.hidden_dim = temp_in_channels
        self.in_proj = LinearEmbedding(temp_in_channels, self.hidden_dim)
        self.out_proj = LinearEmbedding(self.hidden_dim, temp_in_channels)
        if enable_patch_merging:
            self.res_conn_downscale = Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=2, padding=1)
        self.dwcnn = DWCNN(temp_in_channels, self.hidden_dim)
        self.ss2d = SS2D(self.hidden_dim, temp_seq_len)
        self.norm = LayerNormalization(self.hidden_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies a Visual Mamba Block operation, which includes optional patch merging,
        linear projections, depthwise separable convolution, and a 2D state space model.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of the same shape as input, with the VSSBlock operation applied.
        """

        x_out = x
        if self.enable_patch_merging:
            x_out = self.patch_merging(x_out)
        x_out = self.norm(x_out)
        x_in_proj1 = self.in_proj(x_out)
        x_in_proj2 = self.in_proj(x_out)
        x_out = self.dwcnn(x_in_proj1)
        x_out = self.ss2d(x_out)
        x_out = self.norm(x_out)
        x_out = x_out * x_in_proj2
        if self.enable_patch_merging:
            x_down = self.res_conn_downscale(x)
            x_out = self.out_proj(x_out) + x_down
        else:
            x_out = self.out_proj(x_out) + x
        
        return x_out


class UNetMambaEncoder(Module):

    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 patch_size: int=4, 
                 embed_output_channels: int=64):
        super().__init__()
        C, H, _ = input_shape
        self.patch_size = patch_size
        self.patch_partition = PatchPartition(patch_size)
        self.final_patch_merging = PatchMerging()
        self.in_proj = LinearEmbedding(C * patch_size ** 2, embed_output_channels)
        self.vss_blocks = ModuleList([
            VSSBlock(embed_output_channels, H // 4),
            VSSBlock(embed_output_channels, H // 4),
            VSSBlock(embed_output_channels, H // 4, enable_patch_merging=True),
            VSSBlock(embed_output_channels * patch_size, H // 8),
            VSSBlock(embed_output_channels * patch_size, H // 8, enable_patch_merging=True),
            VSSBlock(embed_output_channels * (patch_size ** 2), H // 16)
        ])
        self.bottleneck_vss_blocks = ModuleList([
            VSSBlock(embed_output_channels * (patch_size ** 3), H // 32),
            VSSBlock(embed_output_channels * (patch_size ** 3), H // 32)
        ])
        self.residual_conns = []

            
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the UNet Mamba Encoder operation, which includes patch partitioning,
        linear embedding, multiple Visual Mamba Blocks, and patch merging.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is batch size, 
               C is number of channels, H and W are height and width.

        Returns:
            Tensor of shape (B, C_out, H_out, W_out), where C_out, H_out, and W_out
            depend on the operations performed within the encoder.
        """

        self.residual_conns = []
        x_out = self.patch_partition(x)
        B, S, C = x_out.shape
        dim = int(math.sqrt(S))
        x_out = x_out.permute(0, 2, 1).reshape(B, C, dim, dim)
        x_out = self.in_proj(x_out)
        for i, vss_block in enumerate(self.vss_blocks):
            x_out = vss_block(x_out)
            if i % 2 == 0:
                self.residual_conns.append(x_out)
        
        x_out = self.final_patch_merging(x_out)

        for vss_block in self.bottleneck_vss_blocks:
            x_out = vss_block(x_out)

        return x_out


class DecoderBlock(Module):

    def __init__(self, 
                 in_channels: int, 
                 seq_len: int, 
                 res_conn_channels_num: int):
        super().__init__()
        self.patch_expanding = PatchExpanding()
        self.vssBlock1 = VSSBlock(in_channels // 4, seq_len * 2, enable_patch_merging=False)
        self.vssBlock2 = VSSBlock(in_channels // 4, seq_len * 2, enable_patch_merging=False)
        self.transform_features = Conv2d((in_channels + res_conn_channels_num) // 4, in_channels // 4, kernel_size=1)


    def forward(self, 
                x: Tensor, 
                res_conn: Tensor) -> Tensor:
        x_out = self.patch_expanding(x)
        _, C, H, W = x_out.shape
        x_out = self.vssBlock1(x_out)
        x_out = self.vssBlock2(x_out)
        # assert x_out.shape == res_conn.shape, "The shapes of the Visual Mamba Block output and the residual connection should be equal
        # If the following condition is true, there is an error in the code or in the model architecture.
        if (x_out.shape != res_conn.shape):
            res_conn = res_conn.reshape(x_out.shape)
        enc_features = CenterCrop([H, W])(res_conn)
        print(f"DecoderBlock - x_out.shape: {x_out.shape}, res_conn.shape: {res_conn.shape}, enc_features.shape: {enc_features.shape}")
        x_out = torch.cat([x_out, enc_features], dim=1)
        print(f"After concatenation: x_out.shape: {x_out.shape}")
        x_out = self.transform_features(x_out)

        return x_out


class UNetMambaDecoder(Module):
    def __init__(self, 
                 in_channels: int, 
                 seq_len: int, 
                 res_conns_channels: List[int]):
        super().__init__()
        depth = len(res_conns_channels)
        self.vss_blocks = ModuleList([
            DecoderBlock(in_channels[i], seq_len * 2 ** i, res_conns_channels[i])
            for i in range(depth)
        ])
        self.patch_expanding = PatchExpanding()

    def forward(self, 
                x: Tensor, 
                res_conns: List[Tensor]) -> Tensor:
        x_out = x
        for i, vss_block in enumerate(self.vss_blocks):
            x_out = vss_block(x_out, res_conns[i])
            logger.debug(f"(Decoder) x_out.shape: {x_out.shape}")
        x_out = self.patch_expanding(x_out)
        logger.debug(f"(Decoder last patch expanding): {x_out.shape}")

        return x_out
    

class UNetMambaModel(Module):

    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 dec_in_channels: int, 
                 dec_seq_len: int, 
                 res_conns_channels: List[int], 
                 num_classes: int):
        super().__init__()
        self.encoder = UNetMambaEncoder(input_shape)
        self.decoder = UNetMambaDecoder(dec_in_channels, dec_seq_len, res_conns_channels) 
        self.patch_expanding = PatchExpanding()
        # The value for the number of input channels needs to be determined dynamically instead 
        # of setting it to a fixed value.
        self.out_proj = Conv2d(4, num_classes, kernel_size=1)

    
    def forward(self, x):
        x_out = self.encoder(x)
        logger.debug(f"Model encoder output shape: {x_out.shape}")
        x_out = self.decoder(x_out, self.encoder.residual_conns)
        logger.debug(f"Model decoder output shape: {x_out.shape}")
        x_out = self.patch_expanding(x_out)
        logger.debug(f"Model patch expanding layer output shape: {x_out.shape}")
        x_out = self.out_proj(x_out)
        logger.debug(f"Final layer output shape: {x_out.shape}")

        return x_out
    

class LitMambaUnet(LightningModule):

    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 dec_in_channels: int, 
                 dec_seq_len: int, 
                 res_conns_channels: List[int], 
                 num_classes: int=1, 
                 learning_rate: float=1e-3):
        """
        Initializes the LitMambaUnet model for image segmentation tasks.

        Args:
            input_shape: Tuple representing the shape of the input images (C, H, W).
            dec_in_channels: List of integers representing the number of input channels for each decoder block.
            dec_seq_len: Integer representing the sequence length for the state space model in the decoder.
            res_conns_channels: List of integers representing the number of channels in the residual connections from the encoder.
            num_classes: Integer representing the number of output classes for segmentation. Default is 1.  
            learning_rate: Float representing the learning rate for the optimizer. Default is 1e-3.
        """
        
        super().__init__()
        self.save_hyperparameters()
        self.encoder = UNetMambaEncoder(input_shape)
        self.decoder = UNetMambaDecoder(dec_in_channels, dec_seq_len, res_conns_channels) 
        self.patch_expanding = PatchExpanding()
        # The value for the number of input channels needs to be determined dynamically instead 
        # of setting it to a fixed value.
        self.out_proj = Conv2d(4, num_classes, kernel_size=1)
        self.learning_rate = learning_rate
        self.loss_fn = BCEWithLogitsLoss()
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_test_loss = []
        self.train_steps = 0
        self.val_steps = 0
        self.test_steps = 0
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.total_test_loss = 0


    def forward(self, batch):
        x_out = self.encoder(batch)
        #print(f"Model encoder output shape: {x_out.shape}")
        x_out = self.decoder(x_out, self.encoder.residual_conns)
        #print(f"MOdel decoder output shape: {x_out.shape}")
        x_out = self.patch_expanding(x_out)
        #print(f"Model patch expanding layer output shape: {x_out.shape}")
        mask_pred = self.out_proj(x_out)
        #print(f"Final layer output shape: {x_out.shape}")

        return mask_pred
    

    def get_prediction(self, batch):
        img, mask = batch
        mask_pred = self.forward(img)

        return mask_pred, mask
    
    
    def get_prediction_loss(self, batch):
        mask_pred, mask = self.get_prediction(batch)
        loss = self.loss_fn(mask_pred, mask) 

        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self.get_prediction_loss(batch)
        self.log("step_train_loss", loss)
        self.total_train_loss += loss
        self.train_steps += 1

        return loss
    
    
    def on_train_epoch_end(self):
        avg_train_loss = self.total_train_loss / self.train_steps
        self.log("epoch_avg_train_loss", avg_train_loss)
        self.epoch_train_loss.append(avg_train_loss.item())
        self.total_train_loss = 0
        self.train_steps = 0
        current_lr_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
        logger.info(f"Learning rate at the end of the epoch: {current_lr_rate}")


    def validation_step(self, batch, batch_idx):
        loss = self.get_prediction_loss(batch)
        self.log("step_val_loss", loss)
        self.total_val_loss += loss
        self.val_steps += 1


    def on_validation_epoch_end(self):
        avg_val_loss = self.total_val_loss / self.val_steps
        self.log("epoch_avg_val_loss", avg_val_loss)
        self.epoch_val_loss.append(avg_val_loss.item())
        self.total_val_loss = 0
        self.val_steps = 0


    def test_step(self, batch, batch_idx):
        loss = self.get_prediction_loss(batch)
        self.log("step_test_loss", loss)
        self.total_test_loss += loss
        self.test_steps += 1

    
    def on_test_epoch_end(self):
        avg_test_loss = self.total_test_loss / self.test_steps
        self.log("epoch_avg_test_loss", avg_test_loss)
        self.epoch_test_loss.append(avg_test_loss.item())
        self.total_test_loss = 0
        self.test_steps = 0


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "epoch_avg_val_loss",
                "frequency": self.trainer.check_val_every_n_epoch
            }
        }
