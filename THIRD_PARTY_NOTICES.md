# Third-Party Notices

This repository depends on third-party packages. Their licenses are governed by the respective projects.

The list below captures the **direct Python dependencies** used by this repo (as pinned in `requirements.txt`).

## Python dependencies

- `azure-identity==1.15.0`
- `matplotlib==3.9.2`
- `numpy==1.26.4`
- `openai==2.15.0`
- `pandas==2.2.3`
- `PyYAML==6.0.2`
- `rasterio==1.4.4`
- `requests==2.32.3`
- `scikit-learn==1.5.2`
- `seaborn==0.13.2`
- `torch==2.3.1`
- `torchvision==0.18.1`
- `tqdm==4.66.5`

## Models and datasets

- This repository does **not** include proprietary or restricted datasets.
- Foundation model checkpoints are downloaded on-demand from Hugging Face and/or upstream sources via `download_hf_models.py`.
- Any dataset/model usage must comply with the applicable licenses and terms of use.
