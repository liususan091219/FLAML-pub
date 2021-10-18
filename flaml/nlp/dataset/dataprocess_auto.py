from collections import OrderedDict
from functools import partial

from transformers import AutoTokenizer
from .sentence_keys_auto import get_sentence_keys

import collections
import tqdm
import numpy as np


def inserting_sepp(sent, start, end, this_tokenizer):
    return (
        sent[:start].rstrip()
        + " "
        + this_tokenizer.sep_token
        + " "
        + sent[start:end]
        + " "
        + this_tokenizer.sep_token
        + " "
        + sent[end:].lstrip()
    )


def tokenize_superglue_copa(
    this_example, this_tokenizer, dataset_name, subdataset_name=None, **kwargs
):
    return None


def tokenize_superglue_wic_gpt2(
    this_example, this_tokenizer, dataset_name, subdataset_name=None, **kwargs
):
    return None


def tokenize_superglue_wic(
    this_example, this_tokenizer, dataset_name, subdataset_name=None, **kwargs
):
    """
    tokenize the data from the wic task (word-in-context dataset),
    e.g., sentence 1: "There's a lot of trash on the bed of the river"
    sentence 2: "I keep a glass of water next to my bed when I sleep",
    label = False (different word senses)
    In the superglue data, the position of the word in sentence 1 and 2 are provided
    What this function does is to update the span position after tokenization, based on each LM's own tokenizer,
    The key is to insert an [SEP] before and after the original sentence, then feed it into the LM's tokenizer.
    There are two challenges:
       (1) Each LM's tokenizations are different, e.g., in XLNet's tokenizer, the paddings are on the left'
       (2) Some LM's tokenization would add an underline symbol before the word, e.g., "There's a lot"
       -> [_There, _', _s, _a, _lot]
       When underline meets special char such as '"', "'", the tokenized sequence after adding [SEP] needs to be
       aligned with the sequence tokenized without [SEP]. We use a two pointer algorithm for the alignment
    """
    sent1, sent2 = this_example["sentence1"], this_example["sentence2"]
    start1, end1 = this_example["start1"], this_example["end1"]
    start2, end2 = this_example["start2"], this_example["end2"]
    """
        Add [SEP] to the sentence
    """
    altered_sent1 = inserting_sepp(sent1, start1, end1, this_tokenizer)
    altered_sent2 = inserting_sepp(sent2, start2, end2, this_tokenizer)
    input_ids_sepp = this_tokenizer(
        *(altered_sent1, altered_sent2),
        padding="max_length",
        max_length=1024,
        truncation=True
    )["input_ids"]
    data_pair = (sent1, sent2)
    assert "max_seq_length" in kwargs, "max_seq_length must be provided for glue"
    this_data = this_tokenizer(
        *data_pair,
        padding="max_length",
        max_length=kwargs["max_seq_length"],
        truncation=True
    )
    input_ids = this_data["input_ids"]
    which_sepp = 0

    """
        span_start_end: a 2x2 array:
        * (span_start_end[0][0], span_start_end[0][1]) are the spans of the position of the word in the first sentence
        * (span_start_end[1][0], span_start_end[1][1]) are the spans of the position of the word in the second sentence
    """
    span_start_end = [[-1, -1], [-1, -1]]

    ptr_sepp = ptr_nosepp = 0
    try:
        padding_direction = this_tokenizer.padding_side
        if padding_direction == "left":
            padding_id = input_ids_sepp[0]
            while input_ids_sepp[ptr_sepp] == padding_id:
                ptr_sepp += 1
            while input_ids[ptr_nosepp] == padding_id:
                ptr_nosepp += 1
    except KeyError:
        pass
    sep_id = this_tokenizer.convert_tokens_to_ids([this_tokenizer.sep_token])[0]
    """
        use two pointers to align the tokenized sequence before and after adding [SEP];
        ptr_sepp: the pointer after adding; ptr_nosepp: the pointer without adding
    """
    while (
        ptr_sepp < len(input_ids_sepp)
        and ptr_nosepp < len(input_ids)
        and input_ids_sepp[ptr_sepp] != 0
        and input_ids[ptr_nosepp] != 0
    ):
        if input_ids_sepp[ptr_sepp] == input_ids[ptr_nosepp]:
            ptr_sepp += 1
            ptr_nosepp += 1
        else:
            if not (
                input_ids_sepp[ptr_sepp] == sep_id
                or this_tokenizer.convert_ids_to_tokens([input_ids_sepp[ptr_sepp]])[0]
                in ("▁", "_")
            ):
                break
            if input_ids_sepp[ptr_sepp] == sep_id:
                span_start_end[int(which_sepp / 2)][which_sepp % 2] = ptr_nosepp
                which_sepp += 1
                ptr_sepp += 1
            else:
                ptr_sepp += 1
    """
        max_word_span is the maximum tokens of the word
        It is set to 16 following deberta:
        https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/tasks/superglue_tasks.py#L1054
    """
    max_word_span = 16
    word_indices = []
    for idx1 in range(2):
        if span_start_end[idx1][1] < kwargs["max_seq_length"]:
            first_span = [
                x
                for x in range(span_start_end[idx1][0], span_start_end[idx1][1])
                if x < kwargs["max_seq_length"]
            ] + [0] * (
                max_word_span - span_start_end[idx1][1] + span_start_end[idx1][0]
            )
            word_indices.append(first_span)
    this_data["word_spans"] = word_indices
    return this_data


def tokenize_glue(
    this_example, this_tokenizer, dataset_name, subdataset_name=None, **kwargs
):
    sentence_keys = get_sentence_keys(dataset_name, subdataset_name)

    if len(sentence_keys) > 1:
        sentence1_key, sentence2_key = sentence_keys[0], sentence_keys[1]
    else:
        sentence1_key = sentence_keys[0]
        sentence2_key = None

    data_pair = (
        (this_example[sentence1_key],)
        if sentence2_key is None
        else (this_example[sentence1_key], this_example[sentence2_key])
    )
    assert "max_seq_length" in kwargs, "max_seq_length must be provided for glue"
    return this_tokenizer(
        *data_pair,
        padding="max_length",
        max_length=kwargs["max_seq_length"],
        truncation=True
    )


def tokenize_squad(this_example, this_tokenizer, subdataset_name=None, **kwargs
):
    
    """
    tokenize the data from the squad dataset (question answering dataset),
    e.g., context: "John lives in the greater metropolitan area"
    question: "Where does John live?",
    answers = {'answer_start': 27, 'text': 'metropolitan area'}
    In the squad dataset, we are given a context, a question, and an answer. We assume that the answers a found
    within the context- hence why the answers are given by their start position. I.e. in the above example,
    the answer "metropolitan area" is given by its starting index, 27.
    
    In the case where a dataset contains unanswerable questions (i.e. the context is not found within the answer),
    we must finetune using squad_v2.  
    This preprocessing functions supports both squad and aquad_v2, however; you must specify the which squad version to use
    
    In short this function tokenizes and transforms the raw input into the proper format for QA
    There are several challenges:
       (1) Each LM's tokenizations are different, e.g., in XLNet's tokenizer, the paddings are on the left'
       (2) Contexts can be longer than the maximum sequence length
       (3) A question may be unanswerable 
       (4) We need to define seperate tokenization strategies for the training and testing sets
       
    """
    assert "data_split" in kwargs, "the training split must be provided (i.e. train or test)"
    
    # The maximum length of a feature (question and context)
    max_length = kwargs['max_seq_length'] if 'max_seq_length' in kwargs else 384
    
    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = kwargs['doc_stride'] if 'doc_stride' in kwargs else 128 
    
    #For this function to work with any kind of models, 
    #we need to account for the special case where the model expects padding on the left 
    #(in which case we switch the order of the question and the context)
    pad_on_right = this_tokenizer.padding_side == "right"
    
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    this_example["question"] = [q.lstrip() for q in this_example["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_example = this_tokenizer(
        this_example["question" if pad_on_right else "context"],
        this_example["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
    
    if kwargs['data_split'] == 'train' or kwargs['data_split'] == 'validation':
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_example.pop("offset_mapping")
    
        # Let's label those examples!
        tokenized_example["start_positions"] = []
        tokenized_example["end_positions"] = []
    
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_example["input_ids"][i]
            cls_index = input_ids.index(this_tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = this_example["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_example["start_positions"].append(cls_index)
                tokenized_example["end_positions"].append(cls_index)
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
                    tokenized_example["start_positions"].append(cls_index)
                    tokenized_example["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_example["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_example["end_positions"].append(token_end_index + 1)
    
        return tokenized_example
    
    
    elif kwargs['data_split'] == 'test':
        
        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_example["example_id"] = []
    
        for i in range(len(tokenized_example["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_example["example_id"].append(this_example["id"][sample_index])
    
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_example["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_example["offset_mapping"][i])
            ]
    
        return tokenized_example


def postprocess_qa_predictions(
        test_data, tokenized_test_data, raw_predictions, this_tokenizer, **kwargs
):
    
    assert "squad_v2" in kwargs, "You must specify if you are using squad_v2"
    
    squad_v2 = kwargs['squad_v2']
    n_best_size = kwargs['n_best_size'] if 'n_best_size' in kwargs else 20
    
    # The authorized overlap between two part of the context when splitting it is needed.
    max_answer_length = kwargs['max_answer_length'] if 'max_answer_length' in kwargs else 30 
    
    tokenized_test_data.set_format(type=tokenized_test_data.format["type"], columns=list(tokenized_test_data.features.keys()))
    
    #test_examples = datasets['validation']
    test_examples = test_data
    test_features = tokenized_test_data
    
    example_id_to_index = {k: i for i, k in enumerate(test_examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(test_features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
        
    
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(test_data["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(tokenized_test_data):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(test_data)} example predictions split into {len(tokenized_test_data)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(test_data)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = tokenized_test_data[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = tokenized_test_data[feature_index]["input_ids"].index(this_tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions
    


TOKENIZER_MAPPING = OrderedDict(
    [
        (("hate_speech18", ""), tokenize_glue),
        (("dbpedia_14", ""), tokenize_glue),
        (("sentiment140", ""), tokenize_glue),
        (("imdb", ""), tokenize_glue),
        (("yelp_review_full", ""), tokenize_glue),
        (("amazon_polarity", ""), tokenize_glue),
        (("amazon_reviews_multi", ""), tokenize_glue),
        (("yelp_polarity", ""), tokenize_glue),
        (("glue", "rte"), tokenize_glue),
        (("glue", "mrpc"), tokenize_glue),
        (("glue", "cola"), tokenize_glue),
        (("glue", "wnli"), tokenize_glue),
        (("glue", "stsb"), tokenize_glue),
        (("glue", "sst2"), tokenize_glue),
        (("glue", "mnli"), tokenize_glue),
        (("glue", "qqp"), tokenize_glue),
        (("glue", "qnli"), tokenize_glue),
        (("anli", ""), tokenize_glue),
        (("super_glue", "wic"), tokenize_superglue_wic),
        (("hyperpartisan_news_detection", "bypublisher"), tokenize_glue),
    ]
)


class AutoEncodeText:
    """
    This is a generic input text tokenization class that will be instantiated as one of the
    tokenization classes of the library when created with the
    `~flaml.nlp.dataset.AutoEncodeText.from_model_and_dataset_name` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoEncodeText is designed to be instantiated "
            "using the `AutoEncodeText.from_model_and_dataset_name(cls,"
            "data_raw,model_checkpoint_path,dataset_name,subdataset_name = None,**kwargs)` methods."
        )

    @classmethod
    def from_model_and_dataset_name(
        cls,
        subfold_dataset,
        model_checkpoint_path,
        dataset_name_list: list = None,
        subdataset_name=None,
        **kwargs
    ):
        """
        Instantiate one of the input text tokenization classes from the raw data, model checkpoint path, dataset name
        and sub dataset name. The raw data is used for creating a mapping function from the raw tokens to the
        tokenized token ids.

        Args:
            data_raw:
                The raw data (a datasets.Dataset object)

            model_checkpoint_path:
                A string variable which specifies the model path, e.g., "google/electra-base-discriminator"

            dataset_name_list:
                A list which is the dataset name, e.g., ["glue"]

            subdataset_name:
                A string variable which is the sub dataset name,e.g., "rte"

            kwargs:
                The values in kwargs of any keys will be used for the mapping function

        Examples:
            >>> from datasets import load_dataset
            >>> data_raw = load_dataset("glue", "rte")
            >>> AutoEncodeText.from_model_and_dataset_name(data_raw, "google/electra-base-discriminator", ["glue"], "rte")

        """
        from ..result_analysis.azure_utils import JobID

        dataset_name = JobID.dataset_list_to_str(dataset_name_list)
        if (dataset_name, subdataset_name) in TOKENIZER_MAPPING.keys():
            this_tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint_path, use_fast=True
            )
            token_func = TOKENIZER_MAPPING[(dataset_name, subdataset_name)]
            return subfold_dataset.map(
                partial(
                    token_func,
                    this_tokenizer=this_tokenizer,
                    dataset_name=dataset_name,
                    subdataset_name=subdataset_name,
                    **kwargs
                ),
                batched=False,
            )
        raise ValueError(
            "Unrecognized method {},{} for this kind of AutoGridSearchSpace: {}.\n"
            "Method name should be one of {}.".format(
                dataset_name,
                subdataset_name,
                cls.__name__,
                ", ".join(c[0] for c in TOKENIZER_MAPPING.keys()),
            )
        )
