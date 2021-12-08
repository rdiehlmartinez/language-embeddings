# coding=utf-8
__author__ = 'Richard Diehl Martinez'

""" Compute  Embeddings for sequence labeling tasks."""

import pickle
import torch
from transformers import XLMRobertaForMaskedLM

from utils import device, set_seed
from utils.data_util import get_dataloader, move_to_device

# Pre-defined set of languages contained in the TED-talk dataset
languages = ['ar', 'de', 'es', 'fr', 'he', 'it', 'nl', 'pt-br', 'ru']

extraction_layer_names = [f"roberta.encoder.layer.{x}.output.dense.weight" for x in range(12)] + ["lm_head.dense.weight"]

def compute_fisher(model, input_mask):
    ''' Computing the diagonal fisher matrix of the final dense layer of the masked LM head'''

    fisher_emb_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name in extraction_layer_names:
            fisher_emb = torch.diagonal(param.grad) ** 2 
            fisher_emb_dict[name] = fisher_emb.cpu().detach()
    return fisher_emb_dict

def compute_embedding(model, dataloader, loss_fn):

    # TODO: finetune model (parts of model) on data 

    model.eval()
    total_num_examples = 0 
    global_fisher_emb_dict = {layer_name: torch.zeros(768) for layer_name in extraction_layer_names}

    for step_num, batch in enumerate(dataloader):
        # with bz of 32 about 160k datapoints
        if step_num > 5000: 
            break
        
        model.zero_grad()
        batch = move_to_device(batch, device)
        output = model(**batch)

        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']

        # (batch_size, sequence_length, vocab_size)
        logits = output.logits
        probs = torch.softmax(logits, dim=-1)
        
        # squishing together batch_size and sequence_length 

        batch_size, seq_len, vocab_size = logits.shape
        input_mask_flattened = input_mask.flatten()
        probs_flattened = probs.reshape([batch_size*seq_len, vocab_size])

        # NOTE: we need to get the output labels y ~ p_w(y|x), 
        # this is not the same as the y from the training data 
        sampled_output_ids = torch.multinomial(probs_flattened, 1, True).flatten()
        masked_sampled_output_ids = sampled_output_ids * input_mask_flattened + -1 * (input_mask_flattened - 1)

        # Calculating CE with predicted labels
        logits_transposed = torch.transpose(logits, -2, -1)
        output_ids = masked_sampled_output_ids.reshape([batch_size, seq_len])

        loss = loss_fn(logits_transposed, output_ids)
        loss.backward()

        batch_fisher_emb_dict = compute_fisher(model, input_mask)

        for layer_name, fisher_emb in batch_fisher_emb_dict.items():
            global_fisher_emb_dict[layer_name] += fisher_emb

        # Calculating total number of gradients that are 
        num_examples = torch.sum(input_mask).item()
        total_num_examples += num_examples

    for layer_name, fisher_emb in global_fisher_emb_dict.items():
        global_fisher_emb_dict[layer_name] = fisher_emb/total_num_examples

    return global_fisher_emb_dict

def main():
    ''' Main embedding extraction loop '''
    set_seed(42) # For reproducibility
    model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base").to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1, reduction='sum') #NOTE: 1 is the padding index 

    language_embeddings = {}
    for language in languages: 
        print(f"** Extracting Embedding for Language: {language}")

        # Data assumed to be stored under /data/preprocessed_data/
        fp = f"data/preprocessed_data/{language}.txt"

        dataloader = get_dataloader(fp)

        embedding_dict = compute_embedding(model, dataloader, loss_fn)
        
        language_embeddings[language] = embedding_dict
    
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(language_embeddings, f, protocol=4)


if __name__ == '__main__':
    main()