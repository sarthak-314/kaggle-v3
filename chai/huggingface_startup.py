import tensorflow as tf

import transformers 
import datasets 
from transformers import (
    AutoTokenizer, TFAutoModel, TFAutoModel, TFAutoModelForQuestionAnswering, 
    EvalPrediction, 
)
from datasets import (
    concatenate_datasets, list_datasets, 
)

from qa_utils import *
from huggingface_qa import *

from termcolor import colored

class ChaiQAModel(TFAutoModelForQuestionAnswering): 
    def __init__(self, *args, loss_weights, negative_weight, **kwargs): 
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.negative_weight = negative_weight
        print(colored('Negative Weight:', 'red'), negative_weight)
        print('Start weight:', loss_weights['start'], 'End weight:', loss_weights['end'])
        
    
    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        start_logits, end_logits = logits
        start_loss = loss_fn(labels['start_position'], start_logits)
        end_loss = loss_fn(labels['end_position'], end_logits)

        start_weight, end_weight = self.loss_weights['start'], self.loss_weights['end']
        loss = (start_weight * start_loss + end_weight * end_loss) / (start_weight + end_weight)

        # Multiply loss by negative weight where end position is 0
        loss = tf.where(tf.squeeze(labels['end_position']) == 0, loss*self.negative_weight, loss)

        # Debugging when eager execution
        # print('labels: ', labels)
        # print('logits: ', logits)
        # print('loss: ', loss)

        return loss




NUM_WORKERS = 4
def prepare_features(examples, tokenizer, max_seq_len, doc_stride): 
    examples['question'] = [q.lstrip() for q in examples['question']]
    tokenized_examples = tokenizer(
        examples['question'], 
        examples['context'], 
        truncation='only_second',  # Only Context
        max_length=max_seq_len, 
        stride=doc_stride, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length', 
    )
    # Feature to it's corrosponding example
    feature_to_example_idx = tokenized_examples.pop('overflow_to_sample_mapping')

    # Token to char position in the original context
    # Used to compute start_positions, end_positions
    offset_mapping = tokenized_examples.pop('offset_mapping')

    # Start positions, end positions are calculated
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for batch_index, offsets in enumerate(offset_mapping): 
        # offsets contain offset mapping for tokens in the feature
        input_ids = tokenized_examples["input_ids"][batch_index]
        cls_index = input_ids.index(tokenizer.cls_token_id) # mostly 0

        # get the sequence for full context and question
        tokens_to_sentence_id = tokenized_examples.sequence_ids(batch_index)

        # index of example that contains this span
        example_idx = feature_to_example_idx[batch_index]
        answers = examples['answers'][example_idx]

        no_answers = len(answers['text']) == 0 
        if no_answers: 
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        
        else: 
            # Start/end character index of the answer in the text (Only takes first answer)
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while tokens_to_sentence_id[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while tokens_to_sentence_id[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


def prepare_valid_features(examples, tokenizer, max_seq_len, doc_stride):
    examples['question'] = [q.lstrip() for q in examples['question']]
    tokenized_examples = tokenizer(
        examples['question'], 
        examples['context'], 
        truncation='only_second',  # Only Context
        max_length=max_seq_len, 
        stride=doc_stride, 
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length', 
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def tokenize_dataset(huggingface_dataset, prepare_features_fn): 
    huggingface_dataset = huggingface_dataset.map(
        prepare_features_fn, 
        batched=True, 
        num_proc=NUM_WORKERS, 
        remove_columns=huggingface_dataset.column_names, 
    )
    return huggingface_dataset

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)