### Example usage:

`python -m seq2seq_converter.seq2seq_converter  \`  
`--model_name_or_path t5-large \  `  
`--do_train True \`  
`--do_eval False \ `  
`--do_predict True \`  
`--output_dir PATH_TO_SAVED_MODEL \`  
`--task question_convert \`  
`--per_device_train_batch_size=16 \`  
`--per_device_eval_batch_size=16 \`  
`--overwrite_output_dir \`  
`--predict_with_generate \`  
`--overwrite_cache True \ `  
`--max_source_length 256 \`  
`--pad_to_max_length False \`  
`--output_path PATH_TO_OUTPUT_FILE \`  
`--output_format csv \`  
`--prediction_file PATH_TO_THE_PREDICTION_FILE \`  
`--train_file PATH_TO_THE_TRAINING_FILE \`  
`--validation_file PATH_TO_THE_EVAL_FILE \`  
`--data_source esnli \`  
`--num_train_epochs 5`
