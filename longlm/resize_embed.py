import torch
from torch import Tensor, device, dtype, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json
import traceback
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except:
    pass
def resize_token_embeddings(model, new_num_tokens: Optional[int] = None) -> nn.Embedding:
    model_embeds = _resize_token_embeddings(model, new_num_tokens)
    if new_num_tokens is None:
        return model_embeds

    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    model.vocab_size = new_num_tokens

    # Tie weights again if needed
    model.tie_weights()

    return model_embeds
def _resize_token_embeddings(model, new_num_tokens):
    old_embeddings = model.get_input_embeddings()
    new_embeddings = _get_resized_embeddings(model, old_embeddings, new_num_tokens)
    model.set_input_embeddings(new_embeddings)

    # if word embeddings are not tied, make sure that lm head is resized as well
    if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
        old_lm_head = model.get_output_embeddings()
        new_lm_head = _get_resized_lm_head(model, old_lm_head, new_num_tokens)
        model.set_output_embeddings(new_lm_head)

    return model.get_input_embeddings()

def resize(new_embeddings, old_embeddings, transposed=False, bias=False):
    with open("./model4/t5-xlarge_train0/map_from_ours_to_mt5.json") as fin:
        map_from_ours_to_mt5 = json.load(fin)
        for k in map_from_ours_to_mt5:
            try:
                if bias:
                    new_embeddings.bias.data[int(k)] = old_embeddings.bias.data[int(map_from_ours_to_mt5[k])]
                else:
                    if transposed:
                        new_embeddings.weight.data[:, int(k)] = old_embeddings.weight.data[:, int(map_from_ours_to_mt5[k])]
                    else:
                        new_embeddings.weight.data[int(k), :] = old_embeddings.weight.data[int(map_from_ours_to_mt5[k]), :]
            except Exception as e:
                # traceback.print_exc()
                print(k, end=",")
                continue

def _get_resized_embeddings(
    model, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
) -> nn.Embedding:
    """
    Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
    initialized vectors at the end. Reducing the size will remove vectors from the end

    Args:
        old_embeddings (:obj:`torch.nn.Embedding`):
            Old embeddings to be resized.
        new_num_tokens (:obj:`int`, `optional`):
            New number of tokens in the embedding matrix.

            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
            :obj:`torch.nn.Embedding`` module of the model without doing anything.

    Return:
        :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
        :obj:`new_num_tokens` is :obj:`None`
    """
    if new_num_tokens is None:
        return old_embeddings

    try:
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    except:
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, nn.Embedding):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
            f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
        )

    # Build new embeddings
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
        model.device, dtype=old_embeddings.weight.dtype
    )

    # initialize all new embeddings (in particular added tokens)
    model._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights

    # numbers of tokens to copy
    n = min(old_num_tokens, new_num_tokens)
    try:
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    resize(new_embeddings, old_embeddings)
                    # new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            resize(new_embeddings, old_embeddings)
            # new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    except:
        resize(new_embeddings, old_embeddings)

    return new_embeddings


def _get_resized_lm_head(
    model, old_lm_head: torch.nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
) -> torch.nn.Linear:
    """
    Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
    vectors at the end. Reducing the size will remove vectors from the end

    Args:
        old_lm_head (:obj:`torch.nn.Linear`):
            Old lm head liner layer to be resized.
        new_num_tokens (:obj:`int`, `optional`):
            New number of tokens in the linear matrix.

            Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
            vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
            :obj:`torch.nn.Linear`` module of the model without doing anything.
        transposed (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether ``old_lm_head`` is transposed or not. If True ``old_lm_head.size()`` is ``lm_head_dim,
            vocab_size`` else ``vocab_size, lm_head_dim``.

    Return:
        :obj:`torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if
        :obj:`new_num_tokens` is :obj:`None`
    """
    if new_num_tokens is None:
        return old_lm_head

    old_num_tokens, old_lm_head_dim = (
        old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
    )

    if old_num_tokens == new_num_tokens:
        return old_lm_head

    if not isinstance(old_lm_head, nn.Linear):
        raise TypeError(
            f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}."
            f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Linear}."
        )

    # Build new lm head
    new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
    has_new_lm_head_bias = old_lm_head.bias is not None
    new_lm_head = nn.Linear(*new_lm_head_shape, bias=has_new_lm_head_bias).to(model.device)

    # initialize new lm head (in particular added tokens)
    model._init_weights(new_lm_head)

    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

    # Copy old lm head weights to new lm head
    if not transposed:
        resize(new_lm_head, old_lm_head)
        # new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
    else:
        resize(new_lm_head, old_lm_head, transposed=True)
        # new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

    # Copy bias weights to new lm head
    if has_new_lm_head_bias:
        resize(new_lm_head, old_lm_head, bias=True)
        # new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    return new_lm_head