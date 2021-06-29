## 后续工作
抽取式摘要过于复杂，为保证效果，涉及到多种复杂NLP技术，不是未来发展方向。
如果后续有时间，我会考虑实现一个基于transformer的端到端生成式摘要模型。
欢迎讨论

## Data Preprocess

* Standford Core NLP Toolkit

https://stanfordnlp.github.io/CoreNLP/

`set classpath C:\Users\VY\Desktop\Project\chinese_summarizer\utils\stanford-corenlp-full-2017-06-09\stanford-corenlp-3.8.0.jar`

`python preprocess.py`

## Train

`python train.py`

## Test
* make data

`python preprocess.py -oov_test True -data_name DATA_NAME -raw_path RAW_PATH`
