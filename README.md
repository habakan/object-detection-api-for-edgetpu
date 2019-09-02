# Tensorflow Object Detection API for Edge TPU

## Directory Structure

```
├── Dockerfile
├── README.md
├── config
│   ├── label_map.pbtxt
│   └── ssd_mobilenet_v1.config
├── out # output model file
└── utils
    ├── create_tf_record.py
    ├── create_tf_record_trainandval.py
    └── shell_script
        ├── convert_tfrecord.sh
        └── train_and_convert.sh
```

## Requirements

#### Environment

- Docker
- nvidia-docker

#### File

- Training dataset (Ex: PascalVOC dataset)
- Model config (Ex: config/ssd_mobilenet_v1.config)
- label_map.pbtxt (Ex: config/label_map.pbtxt)

### Build Docker image

```shell
$ sudo dokcer build ./ -t tfodapi
```

## Training pacalVOC dataset

### Download dataset

```bash
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ tar -xvzf VOCtrainval_11-May-2012.tar
```

### Run Docker Conainer

```shell
$ sudo nvidia-docker run -it -v /path-to-dir/pascalVOC/train/VOCdevkit/VOC2012:/data \
		-v ~/objectdetection_train_and_convert_edgetpu/config:/config \
		-v ~/objectdetection_train_and_convert_edgetpu/out:/output \
		--name tfodapi tfodapi /bin/bash
```

### Create tf-record

```shell
$ python object_detection/dataset_utils/create_tf_record.py \
		--annotations_dir=/data/Annotations --images_dir=/data/JPEGImages \
		--output_dir=/data/tfrecord --label_map_path=/config/label_map.pb
```

### Training

```shell
$ cd /models/research
$ mkdir -p /output/train_model /output/tflite_model
```

```shell
$ python object_detection/legacy/train.py --logtostderr --pipeline_config_path \
		/config/ssd_mobilenet_v1.config  --train_dir /output/train_model
```

## Build Edge TPU model

### Convert tflite file

```shell
$ python object_detection/export_tflite_ssd_graph.py \
		--pipeline_config_path /output/train_model/pipeline.config \
		--trained_checkpoint_prefix /output/train_model/model.ckpt-200000 \
		--output_directory /output/tflite_model --add_postprocessing_op=true
```

```shell
$ tflite_convert --output_file /output/tflite_model/output_tflite_graph.tflite \
		--graph_def_file /output/tflite_model/tflite_graph.pb --inference_type=QUANTIZED_UINT8 \
    --input_arrays normalized_input_image_tensor \
    --output_arrays "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" \
    --mean_values 128 --std_dev_values 128 --input_shapes 1,300,300,3 \
    --change_concat_input_ranges false \
    --allow_nudging_weights_to_use_fast_gemm_kernel true --allow_custom_ops
```

### Compile Edge TPU model

```shell
$ edgetpu_compiler /output/tflite_model/output_tflite_graph.tflite -o /output/tflite_model
```

## Acknowledgement

- https://github.com/Jwata/sushi_detector_dataset

  Modification of create_tf_record.py