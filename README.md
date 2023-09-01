# Swin-UNETR-EPA

<hr />

# Installation
Dependencies can be installed using:
```shell
pip install -r requirements.txt
```
<hr />

# DataSet
We use two datasets, [BTCV challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/89480) and [AMOS challenge](https://amos22.grand-challenge.org/).
<hr />

# Training
Training from scratch on single GPU:
```shell
python main.py --data_dir=<data-path> --json_list=<json-path> --save_checkpoint --max_epochs=4000 --batch_size=1 --infer_overlap=0.5 --val_every=100 --roi_x=96 --roi_y=96 --roi_z=96
```
<hr />

# Evaluation
To evaluate a model on a single GPU, place the model checkpoint in pretrained_models folder.
```shell
python test.py --data_dir=<data-path> --json_list=<json-path> --infer_overlap=0.5 --pretrained_model_name=<model-name> --roi_x=96 --roi_y=96 --roi_z=96
```
<hr />

# Visualization
By following the commands for evaluating model in the above, ```test.py``` saves the segmentation outputs in the original spacing in a folder based on the name of the ```outputs```


# References
[1] Y. Tang et al., “Self-supervised pre-training of swin transformers for 3d medical image analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 20730–20740.

[2]
A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. Roth, and D. Xu, “Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images,” arXiv preprint arXiv:2201.01266, 2022.
