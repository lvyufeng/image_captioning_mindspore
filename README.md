# Image Captioning

The mindspore version implementation of the [Show, Attend, and Tell](https://arxiv.org/abs/1502.03044) paper.

## Environment requirements

### Supported hardware

- GPU(with memory more than 8GB)
- Ascend 910

### Dependencies

- MindSpore(>=1.7.0)
- Other Python libs
    
    ```bash
    pip install -r requirements.txt
    ```

## Prepare dataset

### Download dataset

Download COCO dataset and release all images. The dataset split method is following
[Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent)

```bash
bash scripts/download_dataset.sh
```

## Create input files

Store the image and captions data to [MindRecord](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.mindrecord.html)

```
bash scripts/create_input_files.sh
```

## Train

```
python train.py
```

## Prediction

TODO

## Inference on Ascend 310

TODO