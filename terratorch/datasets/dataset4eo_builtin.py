import dataset4eo as eodata
from typing import List, Union
from huggingface_hub import snapshot_download

def load_dataset4eo_builtin(dataset_name: str, input_dir: str, num_channels: int, split: str = "train",
                             channels_to_select: Union[List, None] = None, **kwargs):
    from huggingface_hub import snapshot_download
    import dataset4eo as eodata

    repo_id = eodata.builtin_datasets[dataset_name]
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=input_dir,
        revision="main"
    )

    return eodata.StreamingDataset(
        input_dir=f"{local_path}/{split}",
        num_channels=num_channels,
        channels_to_select=channels_to_select,
        **kwargs
    )
