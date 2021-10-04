# Sentence Position Prediction Dataset

### Data Example

```
{
	"story": 
		"为了证明自己看见了这一切，路过银树林时，守望星折了一根小树枝。[MASK]丽娜听到树枝折断的声音时，回头问：“什么声音？”[MASK]“什么声音也没有。”[MASK]她的大姐说，“可能是哪座城堡的塔楼里，猫头鹰在叫唤。”[MASK]她讲话的时候，迈克悄悄地溜到前头，上了楼梯，第一个进了公主们的房间。[MASK]他推开窗户，顺着藤条滑了下去。[MASK]到花园的时候，太阳刚刚开始升起，他要开始工作了。[MASK]这一天，迈克捆扎鲜花的时候，故意把那根银色的树枝扎进了献给小公主的花里。[MASK]不过，她没有告诉姐姐们。", 
	"sentence": 
		"丽娜发现银树枝时，吃惊极了。", 
	"label": 
		8
}
```

- "Story" (`str`)：input story，`<MASK>` means the candidate position
- "sentence" (`str`)：the removed sentence
- "label" (`int`): label=$l$ means the $l$-th position is correct.



### Evaluation

The prediction result should have the same format with `test.jsonl`

```shell
python eval.py prediction_file test.jsonl
```



We use accuracy as the evaluation metric. The output of the script `eval.py` is a dictionary as follows:

```python
{"accuracy": _}
```

