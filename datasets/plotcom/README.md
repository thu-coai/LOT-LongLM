# Plot Completion Dataset

### Data Example

```
{
	"story": 
		"在一个金碧辉煌的天国王宫里住着两位公主，她们都非常的美丽善良。<MASK>大王就跟国王说，你要是不把你的女儿交出来我就杀了你，正当国王在想办法的时候，一个白马王子出现了，他直接把山寨的大王杀了，跟国王说，我要娶两位公主，国王也同意让女儿嫁了白马王子。从此以后，两位天国的公主与白马王子过上了幸福的日子。", 
	"plot": 
		"有一次，山寨的大王要来天国王宫里提亲，可是国王并不同意把自己的女儿嫁给山寨的大王做山寨夫人。"
}
```

- "story" (`str`)：input story，`<MASK>` means the position of the removed sentence.
- "plot" (`str`)：the removed sentence.

  

### Evaluation

The prediction result should have the same format with `test.jsonl`

```shell
python eval.py prediction_file test.jsonl
```



We use bleu and distinct as the evaluation metrics. The output of the script `eval.py` is a dictionary as follows:

```python
{'bleu-1': '_', 'bleu-2': '_', 'bleu-3': '_', 'bleu-4': '_', 'distinct-1': '_', 'distinct-2': '_',  'distinct-3': '_', 'distinct-4': '_'}
```

- Dependencies: jieba=0.42.1, nltk=3.6.2, numpy=1.20.3

