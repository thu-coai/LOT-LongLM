# Outline-Conditioned Generation Dataset

### Data Example

```
{
	"story":
		"有个人把神像放在驴子背上，赶着进城。凡是遇见他们的人都对着神像顶礼膜拜。驴子以为人们是向它致敬，便洋洋得意，大喊大叫，再也不肯往前走了。结果挨了驴夫狠狠的一棍。", 
	"outline":
  	["对着神像顶礼膜拜", "再也不肯往前走", "神像放在驴子", "赶着进城", "驴夫狠狠", "洋洋得意", "大喊大叫", "遇见"], 
	"title":
		"运神像的驴子"
}
```

- "title" (`str`)：input story title
- "outline"（`list of str`）：input story outline (an out-of-order list of phrases)
- "story" (`str`)：the target story



### Evaluation

The prediction result should have the same format with `test.jsonl`

```shell
python eval.py prediction_file test.jsonl
```



We use bleu, distinct, coverage and order as the evaluation metrics. The output of the script `eval.py` is a dictionary as follows:

```python
{'bleu-1': '_', 'bleu-2': '_', 'bleu-3': '_', 'bleu-4': '_', 'distinct-1': '_', 'distinct-2': '_',  'distinct-3': '_', 'distinct-4': '_', 'coverage': '_', 'order': '_'}
```

- Dependencies: rouge\=\=1.0.0, jieba=0.42.1, nltk=3.6.2, numpy=1.20.3

