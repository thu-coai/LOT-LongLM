# Cloze Test Dataset

### Data Example

```
{
	"story": 
		"一只猫望着窗外飞翔的鸟儿馋涎欲滴，但自己又捕捉不到。于是它便想了一个法子。它给那些鸟儿们寄去请柬，邀请他们来参加自己的生日宴会。<mask>鸟儿一进来，猫就关上了门。鸟儿们彻底入了虎穴，被猫一只一只抓来吃掉了。", 
	"plot0": 
		"可是没有一只鸟儿愿意来。", 
	"plot1": 
		"有些单纯的鸟儿赴宴来了。", 
	"label": 
		"1"
}
```

- "story" (`str`)：input story，`<mask>` means the removed sentence
- "plot0" (`str`)：candidate #0
- "plot1" (`str`)：candidate #1
- "label" (`str`): 0 means candidate #0 is correct, while 1 means candidate #1 is correct.



### Citation

```
@misc{guan2021lot,
      title={LOT: A Benchmark for Evaluating Chinese Long Text Understanding and Generation}, 
      author={Jian Guan and Zhuoer Feng and Yamei Chen and Ruilin He and Xiaoxi Mao and Changjie Fan and Minlie Huang},
      year={2021},
      eprint={2108.12960},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



### Evaluation

The prediction result should have the same format with `test.jsonl`

```shell
python eval.py prediction_file test.jsonl
```



We use accuracy as the evaluation metric. The output of the script `eval.py` is a dictionary as follows:

```python
{"accuracy": _}
```

