# Swin-UNETR-EPA

<hr />

# Installation
Dependencies can be installed using:
```shell
pip install -r requirements.txt
```


<hr />

# DataSet
We use two datasets for [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and [AMOS challenge](https://amos22.grand-challenge.org/)

<hr />

# Training
```shell
python main.py --data_dir=<data-path> --json_list=<json-path> --save_checkpoint --max_epochs=4000 --batch_size=1 --infer_overlap=0.5 --val_every=100 --roi_x=96 --roi_y=96 --roi_z=96
```



# References
[1] Y. Tang et al., “Self-supervised pre-training of swin transformers for 3d medical image analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 20730–20740.

[2]
A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. Roth, and D. Xu, “Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images,” arXiv preprint arXiv:2201.01266, 2022.
