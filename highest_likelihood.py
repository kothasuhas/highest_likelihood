import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, Tuple, Union

import random
from time import time
import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast

import argparse

parser = argparse.ArgumentParser(description='Find the highest likelihood sequence of tokens of a given length')
parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-70m", help='The model name to use')
parser.add_argument('--num_tokens', type=int, default=50, help='The number of tokens to consider')
parser.add_argument('--desired_length', type=int, default=1, help='The length of the sequence to find')
parser.add_argument('--greedy', action='store_true', help='Use greedy decoding instead of DFS')
parser.add_argument('--input_file', type=str, default="", help="Path to input file with problems to solve")
args = parser.parse_args()

def get_log_likelihood(token_list_batch, model):
    # token_list_batch = [[0] + token_list for token_list in token_list_batch]
    num_token_lists = len(token_list_batch)
    input_ids = torch.LongTensor(token_list_batch)
    target_ids = input_ids
    outputs = model.forward(input_ids, labels=target_ids, return_dict=True)
    neg_log_likelihood = outputs.loss.reshape(num_token_lists, -1).sum(dim=1)
    return - neg_log_likelihood

token = "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"
MODEL_NAME = args.model_name
# MODEL_NAME = "EleutherAI/pythia-70m"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=token)

# modified loss tracking in forward pass to get per sequence loss in a batch
def forward(
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[int] = None,
) :
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    # if model has gpt_neox, use it, else use gpt2
    if hasattr(model, "gpt_neox"):
        outputs = model.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = model.embed_out(hidden_states)
    elif hasattr(model, "model"):
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        lm_logits = model.lm_head(hidden_states)
    elif hasattr(model, "transformer"):
        outputs = model.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = model.lm_head(hidden_states) 
    else:
        raise ValueError("Model does not have attribute gpt_neox or transformer")

    lm_loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    return CausalLMOutputWithPast(
        loss=lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

model.forward = forward
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)

digits = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

# extracting common tokens and selecting a small random subset
TOKEN_CANDIDATES = []
for digit in digits:
    token = tokenizer.encode(digit)[-1:]
    TOKEN_CANDIDATES += token

TOKEN_CANDIDATES = list(set(TOKEN_CANDIDATES))
random.seed(0)
random.shuffle(TOKEN_CANDIDATES)
NUM_TOKENS = args.num_tokens
TOKEN_CANDIDATES = TOKEN_CANDIDATES[:NUM_TOKENS]
BOS_TOKEN = tokenizer.bos_token_id
print(BOS_TOKEN)

print('Trying', len(TOKEN_CANDIDATES), 'tokens')
print([tokenizer.decode([token]) for token in TOKEN_CANDIDATES])

def compute_highest_likelihood(problem):
    # first_candidate = [BOS_TOKEN]
    first_candidate = tokenizer.encode(problem)
    # for w in problem:
    #     first_candidate += tokenizer.encode(w)[-1:]
    print('Problem:', problem)
    print('Root tokens:', first_candidate)
    # print([tokenizer.decode([token]) for token in first_candidate])
    print('Root decoded:', tokenizer.decode(first_candidate))

    # depth first search while maintaing most likely full length sequence
    candidate_sequences = [first_candidate]

    stack = [(candidate_sequence, 0, None) for candidate_sequence in candidate_sequences]  # (sequence, depth, likelihood)
    best_likelihood = -100000000000
    best_sequence = None
    DESIRED_LENGTH = args.desired_length

    num_evaluations = 0

    filename = f"{'greedy-' if args.greedy else ''}{MODEL_NAME.split('/')[1]}-count{NUM_TOKENS}-len{DESIRED_LENGTH}.log"
    logging.basicConfig(filename=f"logs/{filename}", level=logging.INFO, format='%(message)s', filemode='w')

    start_time = time()

    while stack:
        candidate_sequence, depth, sequence_likelihood = stack.pop()

        if sequence_likelihood is not None and sequence_likelihood <= best_likelihood:
            # sequence has no promise
            continue
        else:
            if depth > 0:
                depth_str = f"{'  '*(depth-1)}{candidate_sequence}, {sequence_likelihood.item()}"
                print(depth_str)
                logging.info(depth_str)
            if depth == DESIRED_LENGTH:
                # best full length sequence so far
                best_likelihood = sequence_likelihood
                best_sequence = candidate_sequence
                print(f"Best sequence so far: {tokenizer.decode(best_sequence)}")
                logging.info(f"Best sequence so far: {tokenizer.decode(best_sequence)}")
                if args.greedy:
                    break
            else:
                # get children of this node
                new_candidate_sequence_batch = [candidate_sequence + [token] for token in TOKEN_CANDIDATES]
                num_evaluations += 1
                # evaluate their likelihoods
                log_likelihoods = get_log_likelihood(new_candidate_sequence_batch, model)
                # DFS heuristic of sort candidates by likelihood
                sorted_batch = sorted(zip(log_likelihoods, new_candidate_sequence_batch), key=lambda pair: pair[0])
                # only keep valid candidates
                for candidate_log_likelihood, new_candidate_sequence in sorted_batch:
                    assert sequence_likelihood is None or candidate_log_likelihood <= sequence_likelihood
                    if candidate_log_likelihood > best_likelihood:
                        stack.append((new_candidate_sequence, depth + 1, candidate_log_likelihood))  # push into the stack

    end_time = time()

    print('Time taken in seconds:', end_time - start_time)
    print('Number of forward passes:', num_evaluations)
    print('Worst case number of forward passes:', sum([len(TOKEN_CANDIDATES)**i for i in range(DESIRED_LENGTH)]))        
    print('Best sequence:', best_sequence, best_likelihood.item())
    print(tokenizer.decode(best_sequence))

    logging.info(f"Time taken in seconds: {end_time - start_time}")
    logging.info(f"Number of forward passes: {num_evaluations}")
    logging.info(f"Worst case number of forward passes: {sum([len(TOKEN_CANDIDATES)**i for i in range(DESIRED_LENGTH)])}")
    logging.info(f"Best sequence: {best_sequence}, {best_likelihood.item()}")
    logging.info(tokenizer.decode(best_sequence))
    if best_sequence:
        return tokenizer.decode(best_sequence)
    else:
        return []

if args.input_file != "":
    lines = open(args.input_file).read().splitlines()
else:
    lines = []

for problem in lines:
    result = compute_highest_likelihood(problem)
    print(result)

