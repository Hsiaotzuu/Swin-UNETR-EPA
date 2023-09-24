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
[1] Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[2] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R. and Xu, D., 2021, September. Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In International MICCAI Brainlesion Workshop (pp. 272-284). Cham: Springer International Publishing.

[3] Shaker, A., Maaz, M., Rasheed, H., Khan, S., Yang, M.H. and Khan, F.S., 2022. UNETR++: delving into efficient and accurate 3D medical image segmentation. arXiv preprint arXiv:2212.04497.

[4] Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R. and Xu, D., 2022. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (pp. 574-584).
