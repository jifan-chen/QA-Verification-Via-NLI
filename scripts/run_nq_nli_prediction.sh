#!/usr/bin/env bash

working_dir="./model_data"
qa_nli_model_path="./model_data/nq-nli-model.tar.gz"
dataset_path="${working_dir}/nq-nli-dev.jsonl"
data_url='https://docs.google.com/uc?export=download&id=12yj_SQXfFZ5MBTojzcr5A1sLviEP07xx'

if [[ ! -d ${working_dir} ]]; then
    mkdir ${working_dir}
else
    echo "${working_dir} already exist"
fi

if [[ ! -f ${dataset_path} ]]; then
    echo "Downloading data and model ........"
    wget -O ${dataset_path} ${data_url}
fi

if [[ ! -f ${qa_nli_model_path} ]]; then
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VtgQpm15T1xrudKwQAnfOeBZ16ZCdaSt' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VtgQpm15T1xrudKwQAnfOeBZ16ZCdaSt" -O "${qa_nli_model_path}" && rm -rf /tmp/cookies.txt
fi

output_csv=${dataset_path}-predictions.csv
output_json=${dataset_path}-predictions.json
# predictor_name: [qa_nli, textual_entailment]
python3 -m scripts.run_qa_nli \
--qa_nli_path=${dataset_path} \
--output_csv_path="${output_csv}" \
--output_json_path="${output_json}" \
--entailment_model_path=${qa_nli_model_path} \
--predictor_name="qa_nli" \
--use_full_context=False \
--use_decontext=True \
--f1_threshold=0.8 \
--batch_size=8 \
--use_answer_score=False \
--use_qa_concat=False

#python -m scripts.post_processing_entailment_output \
#--input_path=${output_json} \
#--f1_threshold=0.8 \
#--score_type=combination_learned \
#--write_to_csv=False \
#--linear_combination=False \
#--error_analysis=False \
#--compute_f1_at_coverage=True