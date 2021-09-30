"""Require: pip install flaml[nlp,ray]
"""
import argparse
from flaml.nlp.result_analysis.azure_utils import JobID
import numpy as np
import torch


def extract_sorted_config_list(dataset2configscorelist, topk):
    dataset2topkconfigs = {}
    for dataset, configscorelist in dataset2configscorelist.items():
        all_configscorelist = []
        for scorelist in configscorelist:
            for item in scorelist:
                if item[0] not in [x[0] for x in all_configscorelist]:
                    all_configscorelist.append(item)
        sorted_all_configscorelist = sorted(
            all_configscorelist, key=lambda x: x[1], reverse=True
        )
        topk_configs = []

        for each_hp in (
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "warmup_ratio",
            "weight_decay",
            "adam_epsilon",
        ):
            topk_configs.append(
                (
                    each_hp,
                    [sorted_all_configscorelist[x][0][each_hp] for x in range(topk)],
                )
            )
        topk_configs.append(
            ("perf", [sorted_all_configscorelist[x][1] for x in range(topk)])
        )

        dataset2topkconfigs[dataset] = topk_configs
    return dataset2topkconfigs


def dict2tuple(this_dict):
    tuple_list = []
    for key in sorted(this_dict.keys()):
        tuple_list.append(this_dict[key])
    return tuple(tuple_list)


def merge_configscore_list(small_dataset2configscorelist):
    dataset2merged_configscorelist = {}
    for (dataset, each_configscore_list) in small_dataset2configscorelist.items():
        merged_configscore_list = {}
        for rep_id in range(len(each_configscore_list)):
            for each_configscore_entry in each_configscore_list[rep_id]:
                exists = False
                for configscore in merged_configscore_list.keys():
                    if configscore[0] == each_configscore_entry[0]:
                        exists = True
                        break
                if exists is False:
                    merged_configscore_list[
                        dict2tuple(each_configscore_entry[0])
                    ] = each_configscore_entry[1]
        dataset2merged_configscorelist[dataset] = merged_configscore_list
    return dataset2merged_configscorelist


def print_all_configs(console_args, partial_jobid_config):
    from flaml.nlp.result_analysis.azure_utils import AzureUtils, JobID

    azure_utils = AzureUtils(console_args=console_args)
    matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
        console_args.root_log_path, partial_jobid_config
    )

    for each_configscore_list in matched_config_score_lists:
        for (
            config_dict,
            score,
            time_stamp,
        ) in each_configscore_list._config_score_list:
            print_config(config_dict)
            print(score)
            print()


def print_config(config_dict):
    for key in sorted(config_dict.keys()):
        if key in ("attention_probs_dropout_prob", "hidden_dropout_prob", "seed"):
            continue
        if key == "per_device_train_batch_size":
            short_key = "batch_size"
        elif key == "num_train_epochs":
            short_key = "epochs"
        else:
            short_key = key
        print(short_key, config_dict[key])


def compare_small_vs_large(console_args):
    from flaml.nlp.result_analysis.azure_utils import AzureUtils, JobID

    azure_utils = AzureUtils(console_args=console_args)

    partial_jobid_config = JobID()
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "uni"
    partial_jobid_config.presz = "base"

    small_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_jobid(
        console_args.root_log_path, partial_jobid_config
    )

    small_mergedconfiglist = merge_configscore_list(small_dataset2configscorelist)

    partial_jobid_config = JobID()
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "uni"
    partial_jobid_config.presz = "large"

    large_dataset2configscorelist = azure_utils.get_config_and_score_from_partial_jobid(
        console_args.root_log_path, partial_jobid_config
    )

    large_mergedconfiglist = merge_configscore_list(large_dataset2configscorelist)

    for (each_dataset, merged_small_configlist) in small_mergedconfiglist.items():
        merged_large_configlist = large_mergedconfiglist[each_dataset]
        print(each_dataset)
        print()
        for (each_tuple, large_score) in sorted(
            merged_large_configlist.items(), key=lambda x: x[1], reverse=True
        ):
            # small_score = merged_small_configlist[each_tuple]
            is_in_onlysmall = each_tuple in small_mergedconfiglist[each_dataset]
            for each_val in each_tuple:
                print(each_val, end=", ")
            print(large_score, is_in_onlysmall, sep=",")
        print()
        for (each_tuple, small_score) in sorted(
            small_mergedconfiglist[each_dataset].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            is_in_large = each_tuple in large_mergedconfiglist[each_dataset]
            for each_val in each_tuple:
                print(each_val, end=", ")
            print(small_score, is_in_large, sep=",")


def analyze_exhaustive_sweep(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID, AzureUtils, ConfigScoreList

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "grid"
    partial_jobid_config.pre = "funnel"
    partial_jobid_config.presz = "xlarge"

    azure_utils = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )

    for subdat in ["cola"]:
        partial_jobid_config.subdat = subdat
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                console_args.root_log_path, partial_jobid_config
            )
        )
        merged_list = ConfigScoreList(
            [
                x
                for config_score_list in matched_config_score_lists
                for x in config_score_list._config_score_list
            ]
        )
        hp2avg_pearsonr = {}
        hp2avg_pearsonp = {}
        import math

        for rep in range(1):
            # sorted_merged_list = random.sample(merged_list._config_score_list, 36)
            sorted_merged_list = sorted(
                merged_list._config_score_list,
                key=lambda x: x.metric_score["max"],
                reverse=True,
            )[:36]
            print(sorted_merged_list[0].config["learning_rate"])

            metric_scores = [x.metric_score["max"] for x in sorted_merged_list]
            for each_hp in [
                "learning_rate",
                "per_device_train_batch_size",
                "num_train_epochs",
                "warmup_ratio",
                "weight_decay",
                "adam_epsilon",
            ]:
                hp_val = [x.config[each_hp] for x in sorted_merged_list]
                from scipy.stats import pearsonr

                pearsonr, pearsonp = pearsonr(hp_val, metric_scores)
                hp2avg_pearsonr.setdefault(each_hp, [])
                hp2avg_pearsonr[each_hp].append(pearsonr)
                hp2avg_pearsonp.setdefault(each_hp, [])
                hp2avg_pearsonp[each_hp].append(math.log(pearsonp))

        for each_hp in hp2avg_pearsonr.keys():
            import numpy

            print(each_hp)
            print(numpy.mean(hp2avg_pearsonr[each_hp]))
            print(math.exp(numpy.mean(hp2avg_pearsonp[each_hp])))


def get_distribution_from_list(sorted_merged_list, each_hp):
    hp2count = {}
    totalcount = 0
    for each_config in sorted_merged_list:
        hp_val = each_config.config[each_hp]
        hp2count.setdefault(hp_val, 0)
        hp2count[hp_val] += 1
        totalcount += 1

    sorted_hp_vals = []
    for each_hp in sorted(hp2count.keys()):
        sorted_hp_vals.append(float(hp2count[each_hp]) / totalcount)
    return sorted_hp_vals


def compute_kl_div(pk, qk):
    import math

    return sum(pk[i] * math.log2(pk[i] / qk[i]) for i in range(len(pk)))


def output_csv(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID, ConfigScoreList
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "grid"
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.subdat = "mrpc"
    presz_sizes = ["large"]
    all_hps = [
        "learning_rate",
        "per_device_train_batch_size",
        "num_train_epochs",
        "warmup_ratio",
        "weight_decay",
        "adam_epsilon",
    ]

    for presz in presz_sizes:
        partial_jobid_config.presz = presz
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                console_args.root_log_path, partial_jobid_config
            )
        )
        merged_list = ConfigScoreList(
            [
                x
                for config_score_list in matched_config_score_lists
                for x in config_score_list._config_score_list
            ]
        )
        lr2score = {}
        with open(presz + ".csv", "w") as fout:
            for each_hp in all_hps:
                fout.write(each_hp + ",")
            fout.write("performance\n")
            for x in merged_list._config_score_list:
                lr2score.setdefault(x.config["learning_rate"], [])
                lr2score[x.config["learning_rate"]].append(x.metric_score["max"])
                for each_hp in all_hps:
                    fout.write(str(x.config[each_hp]) + ",")
                fout.write(str(x.metric_score["max"]) + "\n")

        for each_lr in lr2score.keys():
            print(each_lr)
            print(np.mean(lr2score[each_lr]))


def analyze_small_large(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID, ConfigScoreList
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "grid"
    partial_jobid_config.pre = "deberta"
    partial_jobid_config.subdat = "cola"

    hp2sz2distribution = {}
    presz_sizes = ["large", "base"]
    all_hps = [
        "learning_rate",
        "per_device_train_batch_size",
        "num_train_epochs",
        "warmup_ratio",
        "weight_decay",
        "adam_epsilon",
    ]

    for presz in presz_sizes:
        partial_jobid_config.presz = presz
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                console_args.root_log_path, partial_jobid_config
            )
        )
        merged_list = ConfigScoreList(
            [
                x
                for config_score_list in matched_config_score_lists
                for x in config_score_list._config_score_list
            ]
        )._config_score_list
        sorted_merged_list = sorted(
            merged_list, key=lambda x: x.metric_score["max"], reverse=True
        )[:36]
        for each_hp in all_hps:
            this_distribution = get_distribution_from_list(sorted_merged_list, each_hp)
            hp2sz2distribution.setdefault(each_hp, {})
            hp2sz2distribution[each_hp][presz] = this_distribution

    for each_hp in all_hps:
        large_distribtion = hp2sz2distribution[each_hp][presz_sizes[0]]
        small_distribution = hp2sz2distribution[each_hp][presz_sizes[1]]
        kl_div_small_large = compute_kl_div(small_distribution, large_distribtion)
        uniform_distribution = [
            1.0 / len(large_distribtion) for x in range(len(large_distribtion))
        ]
        kl_div_uniform_large = compute_kl_div(uniform_distribution, large_distribtion)
        print(each_hp)
        print(kl_div_small_large)
        print(kl_div_uniform_large)


def create_partial_config_bestnn():
    jobid_config = JobID()
    # funnel xlarge
    # jobid_config.mod = "bestnn"
    jobid_config.spa = "uni"
    # jobid_config.arg = "cus"
    # jobid_config.alg = "cfo"
    jobid_config.pre = "funnel"
    jobid_config.presz = "xlarge"
    # funnel small
    # jobid_config.mod = "list"
    # jobid_config.pre = "funnel"
    # jobid_config.presz = "small"
    # jobid_config.rep = 0

    # # deberta large
    # jobid_config.mod = "bestnn"
    # jobid_config.spa = "uni"
    # jobid_config.arg = "cus"
    # jobid_config.alg = "cfo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "large"

    # # deberta base
    # jobid_config.mod = "hpo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "base"
    # jobid_config.rep = 0

    # # deberta large
    # jobid_config.mod = "hpo"
    # jobid_config.pre = "deberta"
    # jobid_config.presz = "large"

    return jobid_config


def get_exhaustive_sweep_result(
    console_args, each_root_log_path, partial_jobid_config, topk
):
    from flaml.nlp.result_analysis.azure_utils import ConfigScoreList
    from flaml.nlp import AzureUtils

    azure_utils = AzureUtils(
        root_log_path=each_root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
        root_log_path=each_root_log_path, partial_jobid=partial_jobid_config
    )
    merged_list = ConfigScoreList(
        [
            x
            for config_score_list in matched_config_score_lists
            for x in config_score_list._config_score_list
        ]
    )._config_score_list
    return get_top1_score_and_config(merged_list, topk)


def get_top1_score_and_config(merged_list, k):
    id_and_score_list = [
        (x, merged_list[x].metric_score["max"])
        for x in range(len(merged_list))
        if isinstance(merged_list[x].metric_score, dict)
    ]
    sorted_id_and_score_list = sorted(
        id_and_score_list, key=lambda x: x[1], reverse=True
    )
    topk_scores = [
        sorted_id_and_score_list[x][1]
        for x in range(k)
        if x < len(sorted_id_and_score_list)
    ]
    topk_idxs = [
        sorted_id_and_score_list[x][0]
        for x in range(k)
        if x < len(sorted_id_and_score_list)
    ]
    topk_configs = [merged_list[x].config for x in topk_idxs]
    return topk_scores, topk_configs


def load_modelinfo():
    from pandas import read_csv

    modelinfo = read_csv("modelinfo.csv", delimiter=",")
    model2size = {}

    for idx in range(len(modelinfo)):
        model_name = modelinfo["model name"][idx].replace("/", "-")
        model_size = modelinfo["Size number"][idx]
        model_unit = modelinfo["Size unit"][idx]
        if model_unit == "GB":
            model_size *= 1000
        model2size[model_name] = model_size
    return model2size


def plot_boxplot(console_args):

    # dat=imdb_subdat=_
    # mod=hpo_spa=gnr_arg=dft_alg=rs
    # _pru=None_pre_full=facebook-muppet-roberta-large
    # _presz=large_spt=rspt_rep=0_sddt=43_sdhf=42_var1=_var2=.json
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils
    import json

    partial_jobid_config = JobID()
    partial_jobid_config.dat = ["hate_speech18"]
    partial_jobid_config.subdat = ""
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.spa = "gnr"
    partial_jobid_config.arg = "dft"
    partial_jobid_config.alg = "bs"
    partial_jobid_config.pre_full = "facebook/muppet-roberta-large"
    partial_jobid_config.spt = "rspt"

    azure_utils = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_blob_list = azure_utils.get_configblob_from_partial_jobid(
        console_args.root_log_path,
        partial_jobid_config,
    )
    assert len(matched_blob_list) == 1

    for (each_jobconfig, each_blob) in matched_blob_list:
        azure_utils.download_azure_blob(each_blob.name)
        data_json = json.load(open(each_blob.name, "r"))
        plot(
            [x["metric_score"]["max"] for x in data_json["val_log"]],
            "".join(partial_jobid_config.dat)
            + "_"
            + partial_jobid_config.subdat
            + "_"
            + partial_jobid_config.pre_full,
        )


def plot(spread, legend):
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    plt.scatter([1] * len(spread), spread)
    ax1.set_title("Basic Plot")
    ax1.boxplot(spread)
    plt.xlabel(legend)
    plt.show()


def print_modelhub_result(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils
    import json
    import re

    model2size = load_modelinfo()
    model2config2score = {}

    for each_dat in ["glue", "yelp-polarity", "imdb", "amazon-polarity"]:
        for model_config in ["hp1", "hp2", "hp1_trainsize", "hp2_trainsize"]:
            partial_jobid_config = JobID()
            partial_jobid_config.dat = [each_dat]
            # partial_jobid_config.presz = presz
            if each_dat == "glue":
                partial_jobid_config.subdat = "sst2"
            each_root_log_path = "logs_modelhub/" + model_config + "/"
            azure_utils = AzureUtils(
                root_log_path=each_root_log_path,
                azure_key_path=console_args.key_path,
                jobid_config=partial_jobid_config,
            )
            matched_blob_list = azure_utils.get_configblob_from_partial_jobid(
                each_root_log_path,
                partial_jobid_config,
            )
            for (each_jobconfig, each_blob) in matched_blob_list:
                azure_utils.download_azure_blob(each_blob.name)
                data_json = json.load(open(each_blob.name, "r"))
                valid_acc = data_json["valid_metric"]["eval_accuracy"]
                # test_acc = data_json["valid_metric"]["test"]["accuracy"]
                match_result = re.search(
                    ".*pre_full=(?P<pre_full>[^_]+)_.*sdhf=(?P<seed>[^_]+)_.*",
                    each_blob.name,
                )
                this_model_name = match_result.group("pre_full")
                this_config = each_dat + "_" + model_config
                model2config2score.setdefault(this_model_name, {})
                model2config2score[this_model_name][this_config] = valid_acc

    sorted_models = sorted(
        [x for x in model2config2score.keys() if x in model2size]
    )  # and model2size[x] > 100])
    config2best = {}

    for each_dat in ["glue", "yelp-polarity", "imdb", "amazon-polarity"]:
        model_configs = ["hp1", "hp2", "hp1_trainsize", "hp2_trainsize"]
        if each_dat not in ["glue", "yelp-polarity"]:
            model_configs = ["hp1", "hp2"]
        for model_config in model_configs:
            each_config = each_dat + "_" + model_config
            # print(each_config)
            for each_model in sorted_models:
                if each_model == "lordtt13-COVID-SciBERT":
                    continue
                try:
                    this_score = model2config2score[each_model][each_config]
                    if model2size[each_model] < 45:
                        config2best.setdefault(each_config, -1)
                        config2best[each_config] = max(
                            config2best[each_config], this_score
                        )
                except KeyError:
                    this_score = ""
                print(this_score, end=",")
            print()

    model2regret = {}

    for each_model in model2config2score.keys():
        if each_model in model2size and model2size[each_model] < 315:
            for each_config in model2config2score[each_model]:
                try:
                    this_score = model2config2score[each_model][each_config]
                    model2regret.setdefault(each_model, [])
                    model2regret[each_model].append(
                        config2best[each_config] - this_score
                    )
                except KeyError:
                    pass
    # sorted_model2regret = sorted(
    #     model2regret.items(), key=lambda x: np.mean(x[1]), reverse=False
    # )

    dominated_model_list = set([])
    for idx1 in range(len(sorted_models) - 1):
        for idx2 in range(idx1 + 1, len(sorted_models)):
            model1 = sorted_models[idx1].replace("/", "-")
            model2 = sorted_models[idx2].replace("/", "-")
            model1_size = model2size[model1]
            model2_size = model2size[model2]
            is_dominating1 = is_dominating(
                model2config2score[model1], model2config2score[model2]
            )
            is_dominating2 = is_dominating(
                model2config2score[model2], model2config2score[model1]
            )
            if is_dominating1 is True and model1_size <= model2_size:
                dominated_model_list.add(model2)
            if is_dominating2 is True and model2_size <= model1_size:
                dominated_model_list.add(model1)
    dominating_list = [x for x in sorted_models if x not in dominated_model_list]
    for idx1 in range(len(dominating_list)):
        for idx2 in range(len(dominating_list)):
            model1 = dominating_list[idx1]
            model2 = dominating_list[idx2]
            is_dominating1 = is_dominating(
                model2config2score[model1], model2config2score[model2]
            )
            if idx1 == idx2:
                is_dominating1 = 0
            print(int(is_dominating1), end=",")
        print()


def compare_muppet(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID, ConfigScoreList
    from flaml.nlp import AzureUtils

    dats = ["yelp-polarity", "glue", "amazon-polarity", "imdb"]
    subdats = [None, "sst2", None, None]
    for idx in range(2, 3):
        partial_jobid_config = JobID()
        partial_jobid_config.dat = [dats[idx]]
        partial_jobid_config.subdat = subdats[idx]
        partial_jobid_config.spa = "grid"
        partial_jobid_config.alg = "grid"
        partial_jobid_config.arg = "dft"
        partial_jobid_config.presz = "large"
        partial_jobid_config.pre_full = "facebook-muppet-roberta-large"

        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                console_args.root_log_path, partial_jobid_config
            )
        )
        best_config = matched_config_score_lists[0]
        merged_list = ConfigScoreList(
            [
                x
                for config_score_list in matched_config_score_lists
                for x in config_score_list._config_score_list
            ]
        )._config_score_list
        top1_merged_list = sorted(
            [x for x in merged_list if isinstance(x.metric_score, dict)],
            key=lambda x: x.metric_score["max"],
            reverse=True,
        )[:1]
        print(len(matched_config_score_lists))
        print(len(merged_list))
        print(partial_jobid_config.dat)
        print(top1_merged_list[0].metric_score["max"])
        print(best_config._test_metric)


def is_dominating(config2score1, config2score2):
    is_larger = True
    for key in config2score1.keys():
        try:
            if config2score1[key] < config2score2[key] / 1.01:
                is_larger = False
        except KeyError:
            pass
    return is_larger


def print_crossvalidation_result(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils
    import json

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.pre = "funnel"
    partial_jobid_config.subdat = "cola"
    partial_jobid_config.presz = "small"

    each_root_log_path = "logs_cv/"
    azure_utils = AzureUtils(
        root_log_path=each_root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_blob_list = azure_utils.get_configblob_from_partial_jobid(
        each_root_log_path,
        partial_jobid_config,
    )
    for (each_jobconfig, each_blob) in matched_blob_list:
        azure_utils.download_azure_blob(each_blob.name)
        data_json = json.load(open(each_blob.name, "r"))
        avg_acc = np.mean(
            [x["eval_matthews_correlation"] for x in data_json["valid_metric"]]
        )
        print(each_blob.name, avg_acc)


def compare_learningrate(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "grid"
    partial_jobid_config.pre = "funnel"
    partial_jobid_config.subdat = "mrpc"
    partial_jobid_config.presz = "small"

    each_root_log_path = "logs_seed/"
    azure_utils = AzureUtils(
        root_log_path=each_root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
        root_log_path=each_root_log_path, partial_jobid=partial_jobid_config
    )
    # split_configscorelist(matched_config_score_lists)
    get_val_test_scores(
        matched_config_score_lists,
        "accuracy" if partial_jobid_config.subdat != "cola" else "matthews_correlation",
    )


def get_seed_configs(matched_config_score_lists):
    seed_configs = []
    for configscore_list in matched_config_score_lists:
        if configscore_list._test_metric:
            this_seed_config = str(configscore_list._jobid_config.sdbs)
            seed_configs.append(this_seed_config)
    return set(seed_configs)


def get_val_test_scores(matched_config_score_lists, metric_name, all_seed_configs):
    val_scores = []
    test_scores = []
    trial_nums = []
    all_this_configs = []

    for configscore_list in matched_config_score_lists:
        if configscore_list._test_metric:
            this_seed_config = str(configscore_list._jobid_config.sdbs)
            best_config = configscore_list.get_best_config()
            if this_seed_config in all_seed_configs:
                trial_nums.append(len(configscore_list._config_score_list))
                val_scores.append(best_config.metric_score["max"])
                test_scores.append(configscore_list._test_metric[metric_name])
                all_this_configs.append(this_seed_config)
    print(
        "{}\t{}\t{}".format(
            "%.4f" % np.mean(val_scores),
            "%.4f" % np.mean(test_scores),
            "%.4f" % np.mean(trial_nums),
        )
    )
    print([float("%.4f" % x) for x in sorted(val_scores)])
    print([float("%.4f" % x) for x in sorted(test_scores)])


def create_partial_config_list():
    jobid_config = JobID()
    jobid_config.mod = "list"
    jobid_config.spa = "uni"
    jobid_config.presz = "xlarge"
    return jobid_config


def randomly_sample_gridunion():
    space = [
        ("learning_rate", [2e-05, 4e-05, 0.00015, 1e-05, 0.0001, 3e-05, 5e-05]),
        ("per_device_train_batch_size", [32, 8, 16]),
        ("num_train_epochs", [10, 3]),
        ("warmup_ratio", [0.06, 0.0, 0.1]),
        ("weight_decay", [0.1, 0.0]),
        ("adam_epsilon", [1e-08, 1e-06]),
    ]
    import random

    for (each_hp, each_space) in space:
        print(each_hp, random.choice(each_space))


def rename_azure_file(console_args):
    import copy
    from flaml.nlp import AzureUtils

    dat_name = ["amazon_polarity"]
    subdat_name = ""

    # logs_azure/glue_sst2/dat=glue_subdat=sst2_mod=
    # hpo_spa=grid_arg=cus_alg=grid_pru=None_pre_full=facebook-muppet-roberta-large_
    # presz=large_spt=cv_rep=0_sddt=43_sdhf=42_var1=_var2=.json

    old_jobid_configs = JobID()
    old_jobid_configs.dat = dat_name
    old_jobid_configs.subdat = subdat_name
    old_jobid_configs.mod = "grid"
    old_jobid_configs.spa = "grid"
    old_jobid_configs.arg = "cus"
    old_jobid_configs.alg = "grid"
    old_jobid_configs.pru = "None"
    old_jobid_configs.pre_full = "facebook-muppet-roberta-large"
    old_jobid_configs.presz = "large"
    old_jobid_configs.spt = "rspt"
    old_jobid_configs.rep = 0
    old_jobid_configs.sddt = 43
    old_jobid_configs.sdhf = 42
    old_jobid_configs.var1 = []
    old_jobid_configs.var2 = []

    new_jobid_configs = copy.deepcopy(old_jobid_configs)
    new_jobid_configs.mod = "gridcv"

    azure_utils = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path="../../",
        jobid_config=old_jobid_configs,
    )
    azure_utils.rename_one_file(
        root_log_path=console_args.root_log_path,
        old_jobid=old_jobid_configs,
        new_jobid=new_jobid_configs,
    )


def print_benchmark(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "qnli"
    partial_jobid_config.sddt = set(["101", "102", "103"])
    partial_jobid_config.alg = "bs"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"
    overlap_seed_configs = None
    config2matchedbloblists = {}

    # azure_utils = AzureUtils(root_log_path=console_args.root_log_path,
    #                          azure_key_path=console_args.key_path,
    #                          jobid_config=partial_jobid_config)
    # matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
    #     root_log_path=console_args.root_log_path,
    #     partial_jobid=partial_jobid_config)

    for partial_jobid_config.pru in ["asha", "None"]:
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                root_log_path=console_args.root_log_path,
                partial_jobid=partial_jobid_config,
            )
        )
        this_seed_configs = get_seed_configs(matched_config_score_lists)
        if overlap_seed_configs is None:
            overlap_seed_configs = this_seed_configs
        else:
            overlap_seed_configs = overlap_seed_configs.intersection(this_seed_configs)

    for partial_jobid_config.pru in ["asha", "None"]:
        print("pruner", partial_jobid_config.pru)
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                root_log_path=console_args.root_log_path,
                partial_jobid=partial_jobid_config,
            )
        )
        # get_val_test_scores(matched_config_score_lists, "accuracy", overlap_seed_configs)
        config2matchedbloblists[partial_jobid_config.pru] = matched_config_score_lists

    azure_utils.plot_hpo_curves(
        config2matchedbloblists,
        config2color={"asha": "red", "None": "blue"},
        plot_title=partial_jobid_config.subdat
        + "_"
        + partial_jobid_config.alg
        + "_"
        + partial_jobid_config.pre_full,
    )


def print_by_method(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "bypublisher"
    partial_jobid_config.sddt = set(["101"])
    partial_jobid_config.sdbs = set(["20", "30", "40"])
    partial_jobid_config.pru = "asha"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"
    overlap_seed_configs = None
    config2matchedbloblists = {}

    # azure_utils = AzureUtils(root_log_path=console_args.root_log_path,
    #                          azure_key_path=console_args.key_path,
    #                          jobid_config=partial_jobid_config)
    # matched_config_score_lists = azure_utils.get_config_and_score_from_partial_jobid(
    #     root_log_path=console_args.root_log_path,
    #     partial_jobid=partial_jobid_config)

    for partial_jobid_config.alg in ["bs", "rs", "optuna"]:
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                root_log_path=console_args.root_log_path,
                partial_jobid=partial_jobid_config,
            )
        )
        this_seed_configs = get_seed_configs(matched_config_score_lists)
        if overlap_seed_configs is None:
            overlap_seed_configs = this_seed_configs
        else:
            overlap_seed_configs = overlap_seed_configs.intersection(this_seed_configs)

    for partial_jobid_config.alg in ["bs", "rs", "optuna"]:
        print("pruner", partial_jobid_config.pru)
        azure_utils = AzureUtils(
            root_log_path=console_args.root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                root_log_path=console_args.root_log_path,
                partial_jobid=partial_jobid_config,
            )
        )
        # get_val_test_scores(matched_config_score_lists, "accuracy", overlap_seed_configs)
        config2matchedbloblists[partial_jobid_config.alg] = matched_config_score_lists

    azure_utils.plot_hpo_curves(
        config2matchedbloblists,
        config2color={"optuna": "red", "rs": "blue", "bs": "green"},
        plot_title=partial_jobid_config.subdat
        + "_"
        + partial_jobid_config.pru
        + "_"
        + partial_jobid_config.pre_full,
    )


def print_asha(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "qnli"
    partial_jobid_config.sddt = set(["101"])
    partial_jobid_config.sdbs = set(["20", "30", "40"])
    partial_jobid_config.alg = "bs"
    partial_jobid_config.pru = "asha"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"
    # overlap_seed_configs = None
    config2matchedbloblists = {}

    for root_log_path in ["logs_benchmark/latest/", "logs_benchmark/latest/ckpt_5/"]:
        azure_utils = AzureUtils(
            root_log_path=root_log_path,
            azure_key_path=console_args.key_path,
            jobid_config=partial_jobid_config,
        )
        matched_config_score_lists = (
            azure_utils.get_config_and_score_from_partial_jobid(
                root_log_path=root_log_path,
                partial_jobid=partial_jobid_config,
            )
        )
        # get_val_test_scores(matched_config_score_lists, "accuracy", overlap_seed_configs)
        config2matchedbloblists[root_log_path] = matched_config_score_lists

    azure_utils.plot_hpo_curves(
        config2matchedbloblists,
        config2color={
            "logs_benchmark/latest/": "red",
            "logs_benchmark/latest/ckpt_5/": "blue",
        },
        plot_title=partial_jobid_config.subdat
        + "_"
        + partial_jobid_config.pru
        + "_"
        + partial_jobid_config.pre_full,
    )


def analyze_asha(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "qnli"
    partial_jobid_config.alg = "optuna"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"
    # overlap_seed_configs = None
    # config2matchedbloblists = {}

    # sdbs must be a set
    for partial_jobid_config.sdbs in [{"20"}, {"30"}, {"40"}]:
        for partial_jobid_config.pru in ["None", "asha"]:
            acc2count = {}
            azure_utils = AzureUtils(
                root_log_path=console_args.root_log_path,
                azure_key_path=console_args.key_path,
                jobid_config=partial_jobid_config,
            )
            matched_config_score_lists = (
                azure_utils.get_config_and_score_from_partial_jobid(
                    root_log_path=console_args.root_log_path,
                    partial_jobid=partial_jobid_config,
                )
            )
            assert len(matched_config_score_lists) == 1
            all_scores = [
                float(int(10 * x.metric_score["max"])) / 10
                for x in matched_config_score_lists[0]._config_score_list
                if isinstance(x.metric_score, dict)
            ]
            for each_score in all_scores:
                acc2count.setdefault(each_score, 0)
                acc2count[round_val(each_score)] += 1
            total_count = sum(acc2count.values())
            acc2count = {
                key: round_val(float(val) / total_count)
                for (key, val) in acc2count.items()
            }
            print(sorted(acc2count.items(), key=lambda x: x[0]))
        # stop = 0


def count_pruned_trials(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "rte"
    partial_jobid_config.sddt = set(["101"])
    partial_jobid_config.sdbs = set(["40"])
    partial_jobid_config.alg = "bs"
    partial_jobid_config.pru = "asha"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"

    root_log_path = "logs_benchmark/latest/glue_rte/"

    azure_utils = AzureUtils(
        root_log_path=root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_config_score_list = (
        azure_utils.get_config_and_score_from_partial_jobid(
            root_log_path=root_log_path,
            partial_jobid=partial_jobid_config,
        )
    )[0]

    early_stop_count = all_count = 0
    for each_config_score in matched_config_score_list._config_score_list:
        if len(each_config_score.all_ckpts["all_ckpt_results"]) == 0:
            continue
        max_epoch = each_config_score.all_ckpts["max_epoch"]
        min_epoch = each_config_score.all_ckpts["all_ckpt_results"][0]["epoch"]
        is_early_stop = AzureUtils.is_early_stop(
            max_epoch, min_epoch, each_config_score.config
        )
        early_stop_count += int(is_early_stop)
        all_count += 1

    print(early_stop_count, all_count)


def plot_trials_with_plt(matched_config_score_list):
    import matplotlib.pyplot as plt

    for each_trial_entry in matched_config_score_list._config_score_list:
        xs = [x["epoch"] for x in each_trial_entry.all_ckpts["all_ckpt_results"]]
        ys = [x["score"] for x in each_trial_entry.all_ckpts["all_ckpt_results"]]
        plt.plot(xs, ys)

    plt.show()


def plot_hpo_trials(console_args):
    from flaml.nlp.result_analysis.azure_utils import JobID
    from flaml.nlp import AzureUtils

    partial_jobid_config = JobID()
    partial_jobid_config.mod = "hpo"
    partial_jobid_config.subdat = "qnli"
    partial_jobid_config.sddt = set(["101"])
    partial_jobid_config.sdbs = set(["40"])
    partial_jobid_config.alg = "bs"
    partial_jobid_config.pru = "asha"
    partial_jobid_config.pre_full = "facebook-muppet-roberta-base"

    root_log_path = "logs_benchmark/latest/glue_qnli/"

    azure_utils = AzureUtils(
        root_log_path=root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config=partial_jobid_config,
    )
    matched_config_score_list = (
        azure_utils.get_config_and_score_from_partial_jobid(
            root_log_path=root_log_path,
            partial_jobid=partial_jobid_config,
        )
    )[0]

    plot_trials_with_plt(matched_config_score_list)


def round_val(val):
    return float(int(100 * val)) / 100


if __name__ == "__main__":
    from flaml.nlp.utils import load_dft_args

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--key_path", type=str, help="key path", required=False, default="../../../"
    )
    arg_parser.add_argument(
        "--root_log_path",
        type=str,
        help="root log path of blob storage",
        required=True,
        default="logs_azure/",
    )
    console_args = load_dft_args()

    plot_hpo_trials(console_args=console_args)
    # analyze_small_large(console_args=args)
    # compare_learningrate(console_args=args)
    # print_crossvalidation_result(console_args=args)
    # print_modelhub_result(console_args=args)
    # randomly_sample_gridunion()
    # compare_muppet(console_args=args)
    # rename_azure_file(console_args=args)
    # plot_boxplot(console_args=args)
