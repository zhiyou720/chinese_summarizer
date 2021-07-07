## 后续工作
我又去翻了翻老代码，试了下效果，这个效果说实话挺垃圾的。

之前为了保证效果，还搞了很多句法分析在里面，太复杂了。

之后抽空写一个纯生成的摘要模型。

2021.7.7

抽取式摘要过于复杂，为保证效果，涉及到多种复杂NLP技术，不是未来发展方向。

如果后续有时间，我会考虑实现一个基于transformer的端到端生成式摘要模型。

欢迎一起讨论

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
