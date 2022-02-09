# Can NLI Models Verify QA Systems' Predictions?

This repository contains the data and code for the following paper:

> [**Can NLI Models Verify QA Systems' Predictions?
**](https://arxiv.org/pdf/2104.08731.pdf)<br/>
> Jifan Chen, Eunsol Choi, Greg Durrett<br/>
> EMNLP 2021 Findings
```
@article{chen2021can,
  title={Can NLI Models Verify QA Systems' Predictions?},
  author={Chen, Jifan and Choi, Eunsol and Durrett, Greg},
  journal={EMNLP Findings},
  year={2021}
}
```

## Datasets
The NLI data converted from QA datasets through our pipeline described in the paper can be found [here](https://drive.google.com/drive/folders/1DW_HvIuUgPYgUJoIMsEOuO0N5uN9k8Hq?usp=sharing)
 
### Data Format
The data files are formatted as jsonlines; each example is described as the following: 

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `example_id`                   | Example ID  |
| `title_text`                | Title of the Wikipedia page of the example, could be NONE    |
| `paragraph_text`             | Paragraph containing the answer |
| `question_text`                   | Question   |
| `answer_text`                  | Answer of the question   |
| `answer_sent_text`      | Sentence containing the answer       |
| `decontext_answer_sent_text`      | Decontextualized sentence containing the answer       |
| `question_statement_text`      | Declarative version of the question by combining the answer|
| `answer_scores`          | Top 5 Answer score computed by the QA(BERT-joint) model         |
| `is_correct`          | Whether the answer is correct |
| `answer_sent_text`      | Sentence containing the answer       |


## Models

### Getting started
`git clone https://github.com/jifan-chen/QA-Verification-Via-NLI.git`

Install the dependencies by running 
`pip install -r requirements.txt`

### Question Converter & Decontextualizer
See [README](https://github.com/jifan-chen/QA-Verification-Via-NLI/tree/master/seq2seq_converter) in seq2seq_converter.

### NQ-NLI
To run the pre-trained NQ-NLI model, simply run `bash scripts/run_nq_nli_prediction.sh`. The pre-trained Roberta-based model and dataset will be automatically downloaded and the predictions will be saved as ${dataset_path}-predictions.csv and ${dataset_path}-predictions.json.

## Contact 

Please contact at `jfchen@cs.utexas.edu` if you have any questions.