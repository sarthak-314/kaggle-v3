# https://github.com/huggingface/transformers/blob/master/examples/tensorflow/question-answering/run_qa.py
import tensorflow as tf


WIKI_LANGS = [
    'hi', 'gu', 'mr', 'pa', 'ur', 
    'ta', 'ml', 'te', 'kn', 
    'en',  
]
TYDIQA_LANGS = ['en', 'te']
WIKI_ANN_NER_LANGS = [
    'hi', 'mr', 'ur', 
    'ta', 'te', 
    'en', 
]
MLQA_LANGS = ['hi', 'en']

def convert_dataset_for_tensorflow(dataset, batch_size):
    """
    Converts a Hugging Face dataset to a Tensorflow Dataset. The dataset_mode controls whether we pad all batches
    to the maximum sequence length, or whether we only pad to the maximum length within that batch. The former
    is most useful when training on TPU, as a new graph compilation is required for each sequence length.
    """

    def densify_ragged_batch(features, label=None):
        features = {
            feature: ragged_tensor.to_tensor(shape=batch_shape[feature]) if feature in tensor_keys else ragged_tensor
            for feature, ragged_tensor in features.items()
        }
        if label is None:
            return features
        else:
            return features, label

    tensor_keys = ["attention_mask", "input_ids"]
    label_keys = ["start_positions", "end_positions"]
    data = {key: tf.ragged.constant(dataset[key]) for key in tensor_keys}
    batch_shape = {
        key: tf.concat(([batch_size], ragged_tensor.bounding_shape()[1:]), axis=0)
        for key, ragged_tensor in data.items()
    }
    if all([key in dataset.features for key in label_keys]):
        for key in label_keys:
            data[key] = tf.convert_to_tensor(dataset[key])
        dummy_labels = tf.zeros_like(dataset[key])
        tf_dataset = tf.data.Dataset.from_tensor_slices((data, dummy_labels))
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    tf_dataset = tf_dataset.batch(batch_size=batch_size).map(densify_ragged_batch)
    return tf_dataset.prefetch(tf.data.AUTOTUNE)

# Training preprocessing
def prepare_train_features(examples, tokenizer, max_seq_length, doc_stride, pad_on_right):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples['question'] = [q.lstrip() for q in examples['question']]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples['question' if pad_on_right else 'context'],
        examples['context' if pad_on_right else 'question'],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
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