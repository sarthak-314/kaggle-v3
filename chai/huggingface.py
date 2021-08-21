def tokenize_dataset(huggingface_dataset): 
    prepare_features_fn = partial(
        prepare_train_features, 
        tokenizer=tokenizer, 
        max_seq_length=MAX_SEQ_LEN, 
        doc_stride=DOC_STRIDE, 
        pad_on_right=True, 
    )
    huggingface_dataset = huggingface_dataset.map(
        prepare_features_fn, 
        batched=True, 
        num_proc=NUM_WORKERS, 
        remove_columns=huggingface_dataset.column_names, 
    )
    return huggingface_dataset

def add_answers_column(valid_dataset): 
    def func(example): 
        example['answers'] = {
            'answer_start': [example['answer_start']], 
            'text': [example['answer_text']], 
        }
        return example
    valid_dataset = valid_dataset.map(func)
    return valid_dataset