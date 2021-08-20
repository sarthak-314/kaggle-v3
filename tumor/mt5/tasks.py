import functools
import seqio
import t5
import metrics

from multilingual_t5.evaluation import metrics as mt5_metrics
from multilingual_t5 import preprocessors

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model"

DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)
DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets": t5.data.Feature(
        vocabulary=DEFAULT_VOCAB, add_eos=True)
}
DEFAULT_VOCAB = t5.data.SentencePieceVocabulary(DEFAULT_SPM_PATH)

def add_mc4(languages, weight=1): 
    for lang in languages:
        seqio.TaskRegistry.add(
            "mc4.{}".format(lang.replace("-", "_")),
            source=seqio.TfdsDataSource(
                tfds_name="c4/multilingual:3.0.1",
                splits={
                    "train": lang,
                    "validation": f"{lang}-validation"
                }),
            preprocessors=[
                functools.partial(
                    t5.data.preprocessors.rekey,
                    key_map={
                        "inputs": None,
                        "targets": "text"
                    }),
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                t5.data.preprocessors.span_corruption,
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[]
        )
    mc4 = ["mc4.{}".format(lang.replace("-", "_")) for lang in languages]
    seqio.MixtureRegistry.add("mc4", mc4, default_rate=weight)

def add_tydiqa(languages, weight=1): 
    seqio.TaskRegistry.add(
        "mt5_tydiqa_train_dev",
        source=seqio.TfdsDataSource(
            tfds_name="tydi_qa/goldp:2.1.0", splits=["train", "validation"]),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad])

    for lang in languages:
        seqio.TaskRegistry.add(
            "mt5_tydiqa_dev.{}".format(lang),
            source=seqio.TfdsDataSource(
                tfds_name="tydi_qa/goldp:2.1.0",
                splits={"validation": "validation-{}".format(lang)}),
            preprocessors=[
                preprocessors.xquad,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            postprocess_fn=t5.data.postprocessors.qa,
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[metrics.squad])

    tydiqa = (["mt5_tydiqa_train_dev"] +
            ["mt5_tydiqa_dev.{}".format(lang) for lang in languages])
    seqio.MixtureRegistry.add("mt5_tydiqa", tydiqa, default_rate=weight)
    
    
def add_eng_squad(): 
    seqio.TaskRegistry.add(
        "mt5_squad_train_dev",
        source=seqio.TfdsDataSource(
            tfds_name="squad/v1.1:3.0.0", splits=["train", "validation"]),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[metrics.squad])

def add_xquad(languages): 
    for lang in languages:
        seqio.TaskRegistry.add(
        "mt5_xquad_translate_train_dev.{}".format(lang),
        source=seqio.TfdsDataSource(
            tfds_name="xquad/{}:3.0.0".format(lang),
            splits={
                "train": "translate-train",
                "validation": "translate-dev"
            }),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.squad])
    

    for lang in languages:
        seqio.TaskRegistry.add(
            "mt5_xquad_test.{}".format(lang),
            source=seqio.TfdsDataSource(
                tfds_name="xquad/{}:3.0.0".format(lang), splits=["test"]),
            preprocessors=[
                preprocessors.xquad,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],      
            postprocess_fn=t5.data.postprocessors.qa,
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[metrics.squad])

def add_mlqa(languages):
    for lang in languages:
        seqio.TaskRegistry.add(
        "mt5_mlqa_dev_test.{}".format(lang),
        source=seqio.TfdsDataSource(
            tfds_name="mlqa/{}:1.0.0".format(lang), splits=["validation",
                                                            "test"]),
        preprocessors=[
            preprocessors.xquad,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        postprocess_fn=t5.data.postprocessors.qa,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[functools.partial(mt5_metrics.mlqa, lang=lang)])
        

def add_wiki_ann_ner(languages, weight=1): 
    for lang in languages:
        seqio.TaskRegistry.add(
            "mt5_ner_train.{}".format(lang),
            source=seqio.TfdsDataSource(
                tfds_name="wikiann/{}:1.0.0".format(lang), splits=["train"]),
            preprocessors=[
                preprocessors.wikiann,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[mt5_metrics.span_f1])

        seqio.TaskRegistry.add(
            "mt5_ner_eval.{}".format(lang),
            source=seqio.TfdsDataSource(
                tfds_name="wikiann/{}:1.0.0".format(lang),
                splits=["validation", "test"]),
            preprocessors=[
                preprocessors.wikiann,
                seqio.preprocessors.tokenize,
                seqio.CacheDatasetPlaceholder(),
                seqio.preprocessors.append_eos_after_trim,
            ],
            output_features=DEFAULT_OUTPUT_FEATURES,
            metric_fns=[mt5_metrics.span_f1])
    
    # NER multilingual
    seqio.MixtureRegistry.add(
        "mt5_ner_multilingual",
        ["mt5_ner_train.{}".format(lang) for lang in languages] +
        ["mt5_ner_eval.{}".format(lang) for lang in languages],
        default_rate=weight)