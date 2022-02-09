import json
import csv
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from typing import Text, List
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import import_module_and_submodules
from sklearn.metrics import f1_score, precision_score, recall_score

FLAGS = flags.FLAGS

flags.DEFINE_string('qa_nli_path', None,
                    'path to the qed annotation file')

flags.DEFINE_string("output_csv_path", None,
                    "path to the output csv file")

flags.DEFINE_string("output_json_path", None,
                    "path to the output json file")

flags.DEFINE_string("entailment_model_path",
                    None,
                    "path to the pre-trained entailment model")

flags.DEFINE_string("predictor_name",
                    None,
                    "Name of the predictor used for evaluation")

flags.DEFINE_bool("use_full_context",
                  False,
                  "whether to use the whole paragraph or the answering sentence")

flags.DEFINE_bool("use_decontext",
                  True,
                  "whether to use the decontexted answer sent as the premise")

flags.DEFINE_float("f1_threshold",
                   0.9,
                   "an example is correct if its f1 >= f1_threshold")

flags.DEFINE_integer("batch_size",
                     24,
                     "batch size used for batch decoding")


flags.DEFINE_bool("use_answer_score",
                  False,
                  "whether to use answer score as feature"
                  )

flags.DEFINE_bool("use_qa_concat",
                  False,
                  "whether to construct premise as Q [SEP] A")

csv_fileds = ['raw_question', 'converted_question', 'original_ans_sent',
              'decontext_answer_sent', 'full_context', "pred_answer",
              "gold_answers", 'label',
              'is_correct', 'has_gold_answer']


def write_to_csv_file(raw_questions: List[Text], parsed_questions: List[Text],
                      selected_sents: List[Text], predictor):
    csv_file = open(FLAGS.output_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(csv_fileds)
    for rq, pq, sent in zip(
            raw_questions,
            parsed_questions,
            selected_sents):
        result = predictor.predict(
            hypothesis=pq,
            premise=sent
        )
        csv_writer.writerow([rq, pq, sent, result['label']])


def main(_):
    import_module_and_submodules("src")
    file_path = FLAGS.qa_nli_path
    logging.info("reading data from {} ...".format(file_path))
    print(FLAGS.predictor_name)
    predictor = Predictor.from_path(
        FLAGS.entailment_model_path,
        FLAGS.predictor_name)

    csv_file = open(FLAGS.output_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(csv_fileds)
    answer_status = []
    raw_answer_status = []
    unique_ids = set()
    predictions = []
    gold_labels = []
    predicted_examples = []
    csv_data = []
    print("reading file ..................")
    with open(file_path) as fin:
        premises = []
        hypothesises = []
        examples = []
        answer_scores = []
        instance_count = 0
        for line in tqdm(fin):
            example = json.loads(line)
            if example['example_id'] in unique_ids:
                continue
            unique_ids.add(example['example_id'])
            if FLAGS.use_full_context:
                premise = example['paragraph_text']
            else:
                if FLAGS.use_decontext:
                    premise = example['decontext_answer_sent_text']
                else:
                    premise = example['answer_sent_text']

            premises.append(premise)
            if FLAGS.use_qa_concat:
                hypothesises.append("{} </s> {}".format(
                    example['question_text'],
                    example['answer_text']
                ))

            hypothesises.append(example['question_statement_text'])
            examples.append(example)
            answer_scores.append(example['answer_score'])
            instance_count += 1

            if instance_count == FLAGS.batch_size:
                if FLAGS.predictor_name == 'textual_entailment':
                    results = [predictor.predict(
                        premise=premises[0],
                        hypothesis=hypothesises[0]
                    )]
                else:
                    results = predictor.predict_batch(
                        premises=premises,
                        hypothesises=hypothesises,
                        answer_scores=answer_scores if FLAGS.use_answer_score else None
                    )
                for result, example in zip(results, examples):
                    if not result:
                        continue
                    if 'f1' in example.keys():
                        f1 = example['f1'] if example['f1'] else 0
                        is_correct = True if f1 >= FLAGS.f1_threshold else False
                    else:
                        is_correct = example['is_correct']

                    answer_status.append([example['has_gold'],
                                          True,
                                          is_correct,
                                          result['logits'][0]]
                                         )

                    if 'gold_answers' in example.keys():
                        gold_answers = ' | '.join(example['gold_answers'])
                    else:
                        gold_answers = ''

                    csv_data.append(
                        [
                         example['question_text'],
                         example['question_statement_text'],
                         example['answer_sent_text'],
                         example['decontext_answer_sent_text'],
                         example['paragraph_text'],
                         example['answer_text'],
                         gold_answers,
                         result['label'],
                         is_correct,
                         example['has_gold'],
                         result['logits'][0]
                         ]
                    )

                    raw_answer_status.append([example['has_gold'],
                                              True,
                                              is_correct,
                                              example['answer_score']]
                                             )

                    example['label'] = result['label']
                    example['confidence_score'] = result['logits'][0]
                    predicted_examples.append(example)

                    if result['label'] == 'entailment' or result['label'] == 'entail':
                        predictions.append(1)
                    else:
                        predictions.append(0)
                    gold_label = 1 if is_correct else 0
                    gold_labels.append(gold_label)

                premises = []
                hypothesises = []
                examples = []
                instance_count = 0

    # p = precision_score(gold_labels, predictions)
    # r = recall_score(gold_labels, predictions)
    # f1 = f1_score(gold_labels, predictions)
    # print('Precision:', p)
    # print('Recall:', r)
    # print('f1:', f1)

    # csv_data.sort(key=lambda x: x[-1], reverse=True)
    for row in csv_data:
        csv_writer.writerow(row)
    answer_status.sort(key=lambda x: x[-1], reverse=True)
    raw_answer_status.sort(key=lambda x: x[-1], reverse=True)

    if FLAGS.output_json_path is not None:
        with open(FLAGS.output_json_path, 'w') as fout:
            for example in predicted_examples:
                json.dump(example, fout)
                fout.write('\n')


if __name__ == "__main__":
    app.run(main)
