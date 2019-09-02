mkdir -p /output/train_model /output/tflite_model

python object_detection/legacy/train.py --logtostderr --pipeline_config_path /config/ssd_mobilenet_v1.config  --train_dir /output/train_model 

python object_detection/export_tflite_ssd_graph.py --pipeline_config_path /output/train_model/pipeline.config --trained_checkpoint_prefix /output/train_model/model.ckpt-200000 --output_directory /output/tflite_model --add_postprocessing_op=true

tflite_convert --output_file /output/tflite_model/output_tflite_graph.tflite --graph_def_file /output/tflite_model/tflite_graph.pb --inference_type=QUANTIZED_UINT8  --input_arrays normalized_input_image_tensor --output_arrays "TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3" --mean_values 128 --std_dev_values 128 --input_shapes 1,300,300,3 --change_concat_input_ranges false --allow_nudging_weights_to_use_fast_gemm_kernel true --allow_custom_ops

edgetpu_compiler /output/tflite_model/output_tflite_graph.tflite -o /output/tflite_model
