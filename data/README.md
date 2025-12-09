# Downloading Datasets

To download the FFHQ dataset (512 x 512) in `lmdb` format, run the following command:
```bash
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download isometricneko/FFHQ512-val-1000 --repo-type dataset --local-dir data/ffhq512first1000
```

### TODO list
- [ ] Add 1024x1024 version of FFHQ
- [ ] Add ImageNet
