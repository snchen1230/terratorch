import os
from pathlib import Path

def generate_yaml_files(model_name, dataset_name):
    # Define paths and hyperparameter combinations
    base_dir = Path("/home/xshadow/terratorch/examples/confs")
    target_dir = base_dir / f"od_{model_name}" / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    lrs = [5e-5, 1e-4, 3e-4]
    decays = [0.05, 0.01, 0.005]

    # Template for the YAML content
    template = """# lightning.pytorch==2.1.1
seed_everything: 0
trainer:
  accelerator: auto
  strategy:
    class_path: lightning.pytorch.strategies.DDPStrategy
    init_args:
      find_unused_parameters: true

  devices: auto
  num_nodes: 1
  precision: 16-mixed
  logger:
    class_path: TensorBoardLogger
    init_args:
      save_dir: /home/xshadow/terratorch/logs/od_{dataset_name}_{model_name}_hpo/
      name: od_{dataset_name}_{model_name}_lr_{lr}_wd_{decay}_warmup_2_e40
  callbacks:
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: ModelCheckpoint
      init_args:
          mode: max
          monitor: val_map
          filename: best-{{epoch:02d}}
  max_epochs: 40
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
  enable_checkpointing: true
  default_root_dir: /home/xshadow/terratorch/logs/
data:
  class_path: terratorch.datamodules.mWindTurbineDataModule
  init_args:
    batch_size: 8
    num_workers: 4
    img_size: 512
    root: /home/xshadow/terratorch/data/wind_turbine

model:
  class_path: terratorch.tasks.ObjectDetectionTask
  init_args:
    model_factory: ObjectDetectionModelFactory
    model_args:
      framework: faster-rcnn  # Model name of TorchVision (one of mask-rcnn, faster-rcnn, fcos, or retinanet)
      backbone: prithvi_eo_v2_300
      num_classes: 12  # Number of classes including background
      # trainable_layers: 3  # Number of trainable layers
      backbone_pretrained: true
      backbone_img_size: 512
      framework_min_size: 512
      framework_max_size: 512
      backbone_bands:
        - RED
        - GREEN
        - BLUE
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
        - name: FeaturePyramidNetworkNeck
        
    freeze_backbone: false
    freeze_decoder: false
    class_names:
      - Background
      - WindTurbine

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: {lr}
    weight_decay: {decay}

lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: {lr}
    total_steps: 40
    pct_start: 0.05
    anneal_strategy: "cos"
    div_factor: 10.0
    final_div_factor: 1000.0
"""

    # Generate and write YAML files
    for lr in lrs:
        for decay in decays:
            filename = f"od_{dataset_name}_{model_name}_lr_{lr}_wd_{decay}_warmup_2_e40.yaml"
            filepath = target_dir / filename

            yaml_content = template.format(
                model_name=model_name,
                dataset_name=dataset_name,
                lr=lr,
                decay=decay
            )

            with open(filepath, "w") as f:
                f.write(yaml_content)

    print(f"Generated 9 config files in {target_dir}")

# Example usage:
generate_yaml_files("prithvi2", "windturbine")

