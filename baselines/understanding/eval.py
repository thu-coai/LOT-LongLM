from logging import log
import traceback
import sys
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
import os
import numpy as np
from transformers import (
        T5Tokenizer,
        AutoTokenizer,
        AutoModel,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        BeamSearchScorer,
    )
import random

device = "cuda:0"
print("using %s"%device)
model_name_path = "./model"
print(model_name_path)
name = "data"
task = "test"
with open("./%s/%s.source"%(name, task), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/%s.target"%(name, task), "r") as fin:
    opt = [line.strip() for line in fin]

tokenizer = T5Tokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})
model = T5ForConditionalGeneration.from_pretrained(model_name_path).to(device)
model.eval()

num = 0
batch_size = 10
st, ed = 0, 0
all_loss = []
with torch.no_grad():
    while ed < len(ipt):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
        
        input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512)
        tgt_ids = tokenizer(opt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
        decoder_input_ids = model._shift_right(tgt_ids)

        src_ids = input_ids.input_ids.to(device)
        src_mask = input_ids.attention_mask.to(device)
        outputs = model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True, use_cache=False)

        # [batch_size, length, hidden_size]
        encoder_hidden_states = outputs["decoder_hidden_states"][-1] #outputs["decoder_last_hidden_state"]
        # [batch_size, length]
        mask1 = torch.eq(decoder_input_ids, torch.tensor(tokenizer.mask_token_id).to(decoder_input_ids.device)).float()
        mask2 = 1 - torch.lt(decoder_input_ids, 32000).float() # 32000 250101
        mask_tmp = torch.eq(torch.cumsum(mask1, 1).int(), 0).float()
        mask2 *= mask_tmp

        # [batch_size, length]
        logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
        logits -= (1 - mask2) * (1e20)
        label = torch.sum(tgt_ids * mask1, 1).int()

        for ip, op, truth in zip(decoder_input_ids, logits, label):
            pred = op.to("cpu").numpy().tolist()
            id_ = np.argmax(pred)
            pred_id = int(ip[id_].to("cpu").numpy())

            label_id = truth

            if pred_id == label_id:
                num += 1
print("accuracy:", num / len(opt))
