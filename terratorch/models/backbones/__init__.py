# Copyright contributors to the Terratorch project

# import so they get registered
import terratorch.models.backbones.prithvi_swin
import terratorch.models.backbones.prithvi_vit
import terratorch.models.backbones.clay_v1
import terratorch.models.backbones.scalemae
# Add this import if it's not already there
from terratorch.models.backbones.dofa_vit import (
    dofav1_base_patch16_224,
    dofav1_large_patch16_224,
    dofav2_large_patch14_224,  # Add the new model
)

# Make sure any other necessary imports or registrations are here
import terratorch.models.backbones.torchgeo_swin_satlas
import terratorch.models.backbones.torchgeo_resnet
import terratorch.models.backbones.multimae_register
from terratorch.models.backbones.unet import UNet
