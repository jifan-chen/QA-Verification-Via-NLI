#!/usr/bin/env bash

working_dir="./model_data"
data_url='https://docs.google.com/uc?export=download&id=12yj_SQXfFZ5MBTojzcr5A1sLviEP07xx'
model_3b_url="https://docs.google.com/uc?export=download&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz"
dataset_path="${working_dir}/nq-nli-dev.jsonl"
decontextualizer_3b_path="${working_dir}/decontextualizer-t5-3b.tar.gz"
decontextualizer_base_path="${working_dir}/decontextualizer-t5-base.tar.gz"
output_path="${dataset_path}.predictions"
model_3b_extraction_path="${working_dir}/decontextualizer-t5-3b"
model_base_extraction_path="${working_dir}/decontextualizer-t5-base"


if [[ ! -d ${working_dir} ]]; then
    mkdir ${working_dir}
else
    echo "${working_dir} already exist"
fi

if [[ ! -f ${dataset_path} ]]; then
    echo "Downloading data and model ........"
    wget -O ${dataset_path} ${data_url}
fi

if [[ ! -f ${decontextualizer_3b_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz" -O "${decontextualizer_3b_path}" && rm -rf /tmp/cookies.txt
fi

if [[ ! -f ${decontextualizer_base_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz" -O "${decontextualizer_base_path}" && rm -rf /tmp/cookies.txt
fi


if [[ ! -d ${model_3b_extraction_path} ]]; then
    mkdir ${model_3b_extraction_path}
    echo "extracting files ..."
    tar -xzvf ${decontextualizer_3b_path} -C ${model_3b_extraction_path}
fi

if [[ ! -d ${model_base_extraction_path} ]]; then
    mkdir ${model_base_extraction_path}
    echo "extracting files ..."
    tar -xzvf ${decontextualizer_base_path} -C ${model_base_extraction_path}
fi

echo "Running the decontextualizer"
python -m seq2seq_converter.seq2seq_converter \
	--model_name_or_path ${model_3b_extraction_path} \
	--do_train False \
	--do_eval False \
	--do_predict True \
	--output_dir ${model_3b_extraction_path} \
	--task decontext \
	--per_device_train_batch_size=4 \
	--per_device_eval_batch_size=4 \
	--overwrite_output_dir \
	--predict_with_generate \
	--overwrite_cache True \
	--max_source_length 512 \
	--pad_to_max_length False \
	--output_path ${output_path} \
	--output_format json \
	--prediction_file ${dataset_path} \
	--validation_file ${dataset_path} \
	--data_source qa-nli \

