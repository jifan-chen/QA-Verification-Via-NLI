#!/usr/bin/env bash

working_dir="./model_data"
data_url='https://docs.google.com/uc?export=download&id=12yj_SQXfFZ5MBTojzcr5A1sLviEP07xx'
model_url="https://docs.google.com/uc?export=download&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz"
dataset_path="${working_dir}/nq-nli-dev.jsonl"
question_converter_path="${working_dir}/question-converter-t5-3b.tar.gz"
output_path="${dataset_path}.predictions"
converter_extraction_path="${working_dir}/question-converter-t5-3b"

if [[ ! -d ${working_dir} ]]; then
    mkdir ${working_dir}
else
    echo "${working_dir} already exist"
fi

if [[ ! -f ${dataset_path} ]]; then
    echo "Downloading data and model ........"
    wget -O ${dataset_path} ${data_url}
fi

if [[ ! -f ${question_converter_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zQEKSIuJmX08FxE5JYLwnj4v5Y8TbtYz" -O "${question_converter_path}" && rm -rf /tmp/cookies.txt
fi

if [[ ! -d ${converter_extraction_path} ]]; then
    mkdir ${converter_extraction_path}
    echo "extracting files from ${question_converter_path} ..."
    tar -xzvf ${question_converter_path} -C ${converter_extraction_path}
fi

echo "Running the question converter"
python -m seq2seq_converter.seq2seq_converter \
	--model_name_or_path ${converter_extraction_path} \
	--do_train False \
	--do_eval False \
	--do_predict True \
	--output_dir ${converter_extraction_path} \
	--task question_convert \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--overwrite_output_dir \
	--predict_with_generate \
	--overwrite_cache True \
	--max_source_length 128 \
	--pad_to_max_length False \
	--output_path ${output_path} \
	--output_format json \
	--prediction_file ${dataset_path} \
	--validation_file ${dataset_path} \
	--data_source qa-nli \

