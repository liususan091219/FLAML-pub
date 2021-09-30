"""Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
"""
import os
import shutil

import flaml
from flaml.nlp import AutoTransformers
from flaml.nlp import AzureUtils, JobID
from flaml.nlp.result_analysis.wandb_utils import WandbUtils
from flaml.nlp.utils import load_dft_args

global azure_log_path
global azure_key


def get_resplit_portion(jobid_config, console_args):
    assert len(console_args.split_portion) == 6
    train_split = [float(x) for x in console_args.split_portion[0:2]]
    validation_split = [float(x) for x in console_args.split_portion[2:4]]
    test_split = [float(x) for x in console_args.split_portion[4:6]]
    return {
        "source": console_args.source_fold,
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }
    # if jobid_config.dat == ["glue"] and jobid_config.subdat in {"mnli", "qqp"}:
    #     return {"source": ["train", "validation"], "train": [0, 0.25], "validation": [0.25, 0.275],
    #             "test": [0.275, 0.3]}
    # elif jobid_config.dat[0] in {"imdb", "dbpedia_14", "yelp_review_full", "amazon_reviews_multi"}:
    #     return {"source": ["train", "test"], "train": [0.5, 1.0], "validation": [0.05, 0.055], "test": [0.055, 0.06]}
    # elif jobid_config.dat[0] in {"hate_speech18"}:
    #     return {"source": ["train"], "train": [0, 0.8], "validation": [0.8, 0.9], "test": [0.9, 1.0]}
    # elif jobid_config.dat[0] in {"yelp_polarity"}:
    #     return {"source": ["train"], "train": [0, 0.05], "validation": [0.05, 0.055], "test": [0.055, 0.06]}
    # elif jobid_config.dat[0] in {"amazon_polarity"}:
    #     return {"source": ["train"], "train": [0.1, 0.2], "validation": [0.01, 0.011], "test": [0.011, 0.012]}
    # else:
    #     return {"source": ["train", "validation"], "train": [0, 0.8], "validation": [0.8, 0.9], "test": [0.9, 1.0]}


def get_preparedata_setting(
    console_args, jobid_config, wandb_utils=None, **custom_args
):
    preparedata_setting = {
        "server_name": console_args.server_name,
        "data_root_path": console_args.data_root_dir,
        "max_seq_length": 128,
        "jobid_config": jobid_config,
        "wandb_utils": wandb_utils,
    }
    if jobid_config.spt in ("rspt", "cv"):
        preparedata_setting["resplit_portion"] = get_resplit_portion(
            jobid_config, console_args
        )
    if ("albert" == jobid_config.pre and jobid_config.dat == ["squad"]) or (
        "funnel" in jobid_config.pre
        and jobid_config.dat[0]
        in {
            "imdb",
            "yelp_reviews_full",
            "yelp_polarity",
            "amazon_polarity",
            "amazon_review_multi",
        }
    ):
        preparedata_setting["max_seq_length"] = 512
    if jobid_config.dat[0] == "glue" and jobid_config.subdat == "mnli":
        preparedata_setting["fold_name"] = [
            "train",
            "validation_matched",
            "test_matched",
        ]
    elif jobid_config.dat[0] == "anli":
        preparedata_setting["fold_name"] = ["train_r3", "dev_r3", "test_r3"]
    preparedata_setting.update(custom_args)
    return preparedata_setting


def get_autohf_settings(console_args, **custom_args):
    if console_args.algo_name == "optuna":
        seed_bs = console_args.seed_bs - 10
    else:
        seed_bs = console_args.seed_bs

    autohf_settings = {
        "resources_per_trial": {"gpu": 1, "cpu": 1},
        "num_samples": console_args.sample_num,
        "time_budget": console_args.time_budget,
        "ckpt_per_epoch": 5,
        "seed_bs": seed_bs,
        "keep_checkpoints_num": 10000000,
    }
    for other_attr in ["ds_config", "rep_id"]:
        if hasattr(console_args, other_attr):
            autohf_settings[other_attr] = getattr(console_args, other_attr)
        else:
            autohf_settings[other_attr] = None
    if len(custom_args) > 0:
        autohf_settings.update(custom_args)
    return autohf_settings


def rm_home_result():
    from os.path import expanduser

    home = expanduser("~")
    if os.path.exists(home + "/ray_results/"):
        shutil.rmtree(home + "/ray_results/")


def get_best_base_config(console_args, jobid_config, autohf, wandb_utils):
    import copy
    import re

    args_small = copy.deepcopy(console_args)
    args_small.algo_name = "optuna"
    args_small.search_alg_args_mode = "dft"
    args_small.algo_mode = "hpo"
    args_small.space_mode = "uni"
    args_small.pruner = "None"

    if "funnel" not in args_small.pretrained_model_size:
        args_small.algo_mode = "hpo"
    else:
        args_small.algo_mode = "list"
    args_small.sample_num = 10000
    args_small.time_budget = 3600
    args_small.rep_id = 0
    jobid_config_small = JobID(args_small)
    if jobid_config_small.pre == "deberta":
        jobid_config_small.presz = "base"
    else:
        jobid_config_small.presz = "small"
    jobid_config_small.pre_full = re.sub(
        "(xlarge|large|intermediate)",
        jobid_config_small.presz,
        jobid_config_small.pre_full,
    )
    azure_utils_small = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        autohf=autohf,
    )
    preparedata_setting = get_preparedata_setting(
        console_args, jobid_config, wandb_utils
    )
    autohf.prepare_data(**preparedata_setting)
    autohf.set_metric()

    best_config = azure_utils_small.get_config_and_score_from_partial_jobid(
        args_small.root_log_path, jobid_config_small
    )[0].get_best_config()
    return best_config


def search_base_and_search_around_best(console_args, jobid_config, autohf, wandb_utils):
    console_args.algo_name = "bs"
    console_args.search_alg_args_mode = "dft"
    console_args.spa = "uni"
    console_args.pru = "None"
    best_config = get_best_base_config(console_args, jobid_config, autohf, wandb_utils)

    import copy

    args_large = copy.deepcopy(console_args)
    args_large.time_budget = console_args.time_budget - 3600
    args_large.sample_num = 100000
    args_large.algo_name = "cfo"
    args_large.search_alg_args_mode = "cus"
    args_large.space_mode = "uni"
    jobid_config_large = JobID(args_large)
    jobid_config_large.presz = jobid_config.presz
    jobid_config_large.pre_full = jobid_config.pre_full
    azure_utils_large = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        autohf=autohf,
    )

    _test_hpo(
        args_large,
        jobid_config_large,
        autohf,
        wandb_utils,
        azure_utils_large,
        autohf_settings=get_autohf_settings(
            args_large, **{"points_to_evaluate": [best_config]}
        ),
    )


def evaluate_configs(autohf, console_args, points_to_evaluate, wandb_utils=None):
    jobid_config = JobID(console_args)
    if wandb_utils is None:
        wandb_utils = WandbUtils(
            is_wandb_on=False,
            wandb_key_path=console_args.key_path,
            jobid_config=jobid_config,
        )
        wandb_utils.set_wandb_per_run()

    autohf.jobid_config = jobid_config
    azure_utils_large = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        autohf=autohf,
    )

    _test_hpo(
        console_args,
        jobid_config,
        autohf,
        wandb_utils,
        azure_utils_large,
        autohf_settings=get_autohf_settings(
            console_args, **{"points_to_evaluate": points_to_evaluate}
        ),
    )


def evaluate_configs_cv(autohf, console_args, cv_k, wandb_utils):
    # cv_first_step(console_args, autohf, wandb_utils)
    topk_score, topk_config = cv_second_step(console_args, cv_k)
    configscore_lists = cv_third_step(console_args, autohf, topk_config)
    cv_fourth_step(
        console_args,
        configscore_lists,
        other_results={
            "metric_score": [
                [
                    each_configscore.metric_score
                    for each_configscore in each_configscore_list
                ]
                for each_configscore_list in configscore_lists
            ],
            "topk_config": topk_config,
        },
    )


def cv_first_step(console_args, autohf, wandb_utils):
    # the first step of cv: running hpo
    jobid_config = JobID(console_args)
    jobid_config.mod = console_args.algo_mode[:-2]
    _test_hpo(console_args, jobid_config, autohf, wandb_utils)


def cv_second_step(console_args, k):
    # the second step of cv: load the topk configs from the saved output from hpo
    from run_analysis import get_exhaustive_sweep_result

    sweep_jobid_config = JobID(console_args)
    sweep_jobid_config.mod = console_args.algo_mode[:-2]
    sweep_jobid_config.pre = None
    sweep_jobid_config.pre_full = sweep_jobid_config.pre_full.replace("/", "-")
    topk_score, topk_config = get_exhaustive_sweep_result(
        console_args, console_args.root_log_path, sweep_jobid_config, k
    )
    return topk_score, topk_config


def cv_third_step(console_args, autohf, topk_config):
    # the third step of cv: evaluate the topk configs from the previous step, return the topk x #trials matrix
    cv_jobid_config = JobID(console_args)
    cv_jobid_config.mod = "hpo"
    cv_jobid_config.spa = "gnr"
    cv_jobid_config.arg = "cus"
    cv_jobid_config.spt = "cv"
    cv_jobid_config.alg = "bs"
    autohf.jobid_config = cv_jobid_config
    azure_utils = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        autohf=autohf,
    )

    custom_args = {"foldnum": 3}

    preparedata_setting = get_preparedata_setting(
        console_args, cv_jobid_config, wandb_utils, **custom_args
    )
    autohf.prepare_data(**preparedata_setting)
    console_args.sample_num = len(topk_config)
    console_args.time_budget = 100000
    autohf_settings = get_autohf_settings(
        console_args, **{"points_to_evaluate": topk_config}
    )
    import copy

    cv_k = len(autohf.train_datasets)
    validation_metrics = []
    configscore_lists = []
    autohf_settings_copies = []
    for idx in range(cv_k):
        autohf_settings_copies.append(copy.deepcopy(autohf_settings))
    for idx in range(0, cv_k):
        idx, validation_metric, analysis = train_cv(
            idx,
            train_dataset=autohf.train_datasets[idx],
            eval_dataset=autohf.eval_datasets[idx],
            autohf_settings=autohf_settings_copies[idx],
        )
        if analysis is not None:
            configscore_list = azure_utils.extract_configscore_list_from_analysis(
                analysis
            )
        else:
            configscore_list = None
        validation_metrics.append(validation_metric)
        configscore_lists.append(configscore_list)
    return configscore_lists


def cv_fourth_step(console_args, configscore_lists, other_results=None):
    # the fourth step of cv: rerun evaluation for the top1 config found in the previous step
    import copy

    top1_config, top1_score = load_and_select_top1_config(configscore_lists)

    jobid_config = JobID(console_args)
    jobid_config_origin = copy.deepcopy(jobid_config)

    jobid_config.mod = "hpo"
    jobid_config.spa = "gnr"
    jobid_config.arg = "cus"
    jobid_config.spt = "rspt"
    jobid_config.alg = "bs"

    autohf.jobid_config = jobid_config
    azure_utils = AzureUtils(
        root_log_path=console_args.root_log_path,
        azure_key_path=console_args.key_path,
        jobid_config_rename=jobid_config_origin,
        autohf=autohf,
    )
    console_args.split_portion[1] = str(
        max(
            [float(console_args.split_portion[1]), float(console_args.split_portion[3])]
        )
    )

    console_args.sample_num = 1
    console_args.time_budget = 100000
    other_results["avg_cv_score"] = top1_score
    _test_hpo(
        console_args,
        jobid_config,
        autohf,
        wandb_utils,
        azure_utils,
        autohf_settings=get_autohf_settings(
            console_args, **{"points_to_evaluate": [top1_config]}
        ),
        other_results=other_results,
    )

    rm_home_result()


def convert_config_to_different_size(origin_config, mode):
    import re
    import copy

    if mode == "small":
        new_config = copy.deepcopy(origin_config)
        if new_config.pre == "funnel":
            new_config.mod = "list"
        else:
            new_config.mod = "hpo"
        if new_config.pre == "funnel":
            new_config.presz = "small"
        else:
            new_config.presz = "base"
        new_config.pre_full = re.sub(
            "(xlarge|large|intermediate)", new_config.presz, origin_config.pre_full
        )
    elif mode == "large":
        new_config = copy.deepcopy(origin_config)
        new_config.mod = "hpo"
        if new_config.pre == "funnel":
            new_config.presz = "xlarge"
            new_config.pre_full = re.sub("(small)", "xlarge", origin_config.pre_full)
        else:
            new_config.presz = "large"
            new_config.pre_full = re.sub("(small)", "large", origin_config.pre_full)

    return new_config


def add_dict_item_to_list(this_list, this_dict):
    is_exist = len([x for x in this_list if x == this_dict]) > 0
    if not is_exist:
        this_list.append(this_dict)
    return this_list


def train_cv(idx, train_dataset, eval_dataset, autohf_settings):
    # azure_utils = batch_dict["azure_utils"]
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 4)
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    autohf.train_dataset = train_dataset
    autohf.eval_dataset = eval_dataset
    validation_metric, analysis = autohf.fit(**autohf_settings)
    # json.dump(validation_metric, open("tmp_" + str(idx) + ".json", "w"))
    # azure_utils.write_autohf_output(valid_metric=validation_metric,
    #                                 local_file_path=)
    return idx, validation_metric, analysis


def load_and_select_top1_config(configscore_lists):
    import numpy as np

    trialid2config = {}
    trialid2scores = {}

    for fold_idx in range(len(configscore_lists)):
        for trial_idx in range(len(configscore_lists[0])):
            metric_score = configscore_lists[fold_idx][trial_idx].metric_score[
                autohf.metric_mode_name
            ]
            this_config = configscore_lists[fold_idx][trial_idx].config
            trialid2scores.setdefault(trial_idx, [])
            trialid2scores[trial_idx].append(metric_score)
            trialid2config[trial_idx] = this_config

    sorted_trialid2scores = sorted(
        trialid2scores.items(), key=lambda x: np.mean(x[1]), reverse=True
    )
    return trialid2config[sorted_trialid2scores[0][0]], np.mean(
        sorted_trialid2scores[0][1]
    )


def _test_hpo(
    console_args,
    jobid_config,
    autohf,
    wandb_utils,
    azure_utils=None,
    autohf_settings=None,
    jobid_config_rename=None,
    other_results=None,
    **custom_args
):
    import subprocess
    import re

    preparedata_setting = get_preparedata_setting(
        console_args, jobid_config, wandb_utils, **custom_args
    )
    autohf.prepare_data(**preparedata_setting)

    analysis = validation_metric = None
    if not autohf_settings:
        autohf_settings = get_autohf_settings(console_args, **custom_args)

    if console_args.algo_mode != "hfhpo":
        validation_metric, analysis = autohf.fit(**autohf_settings)
    else:
        autohf.fit_hf(**autohf_settings)
    predictions, test_metric = autohf.predict()
    if test_metric:
        validation_metric.update({"test": test_metric})

    if not azure_utils:
        if jobid_config_rename:
            azure_utils = AzureUtils(
                root_log_path=console_args.root_log_path,
                azure_key_path=console_args.key_path,
                jobid_config=jobid_config_rename,
            )
        else:
            azure_utils = AzureUtils(
                root_log_path=console_args.root_log_path,
                azure_key_path=console_args.key_path,
                autohf=autohf,
            )

    if analysis is not None:
        configscore_list = azure_utils.extract_configscore_list_from_analysis(analysis)
    else:
        configscore_list = None

    repo_url = "git://github.com/liususan091219/FLAML.git"
    process = subprocess.Popen(
        ["git", "ls-remote", repo_url, "--heads", "exp"], stdout=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    sha = re.split(r"\t+", stdout.decode("ascii"))[0]
    print(sha)

    azure_utils.write_autohf_output(
        configscore_list=configscore_list,
        valid_metric=validation_metric,
        predictions=predictions,
        duration=autohf.last_run_duration,
        other_results=other_results,
        gitsha=sha,
        console_args=console_args.__dict__,
    )
    rm_home_result()


def _exhaustive_sweep(
    console_args,
    jobid_config,
    autohf,
    wandb_utils,
    azure_utils=None,
    autohf_settings=None,
):
    from flaml.nlp.hpo.hpo_searchspace import AutoHPOSearchSpace

    console_args.space_mode = jobid_config.spa = "cus"
    console_args.algo_mode = jobid_config.mod = "grid"
    console_args.algo_name = jobid_config.alg = "grid"

    gridunion_space = AutoHPOSearchSpace.from_model_and_dataset_name(
        "uni",
        jobid_config.pre,
        jobid_config.presz,
        jobid_config.dat,
        jobid_config.subdat,
    )

    gridunion_space["learning_rate"] = [float(x) for x in console_args.learning_rate]
    gridunion_space["weight_decay"] = [float(x) for x in console_args.weight_decay]
    _test_hpo(
        console_args,
        jobid_config,
        autohf,
        wandb_utils,
        azure_utils,
        autohf_settings,
        root_log_path=console_args.root_log_path,
        **{"hpo_space": gridunion_space}
    )


def modelcard_exp(console_args):
    if "hp1" in console_args.root_log_path:
        console_args.seed_transformers = 42
        evaluate_configs(
            autohf,
            console_args,
            [
                {
                    "learning_rate": 3e-05,
                    "per_device_train_batch_size": 16,
                    "num_train_epochs": 10,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.1,
                    "adam_epsilon": 1e-6,
                }
            ],
        )
    else:
        console_args.seed_transformers = 41
        evaluate_configs(
            autohf,
            console_args,
            [
                {
                    "learning_rate": 1e-05,
                    "per_device_train_batch_size": 32,
                    "num_train_epochs": 3,
                    "warmup_ratio": 0.0,
                    "weight_decay": 0.0,
                    "adam_epsilon": 1e-8,
                }
            ],
        )


if __name__ == "__main__":
    # from flaml.nlp.hpo.hpo_searchspace import AutoHPOSearchSpace
    import itertools

    console_args = load_dft_args()

    jobid_config = JobID(console_args)
    autohf = AutoTransformers()
    wandb_utils = WandbUtils(
        is_wandb_on=False,
        wandb_key_path=console_args.key_path,
        jobid_config=jobid_config,
    )
    wandb_utils.set_wandb_per_run()

    # _test_hpo(console_args, jobid_config, autohf, wandb_utils)

    # search_base_and_search_lower_lr(console_args, jobid_config, autohf, wandb_utils)

    # evaluate_small_best_configs_on_large(console_args, autohf)

    # evaluate_large_best_configs_on_small(console_args, autohf)

    # _exhaustive_sweep(console_args, jobid_config, autohf, wandb_utils)

    # evaluate_configs(autohf, console_args)

    if console_args.algo_mode.endswith("cv"):
        evaluate_configs_cv(autohf, console_args, 10, wandb_utils)
    elif console_args.algo_mode.endswith("hpo"):
        _test_hpo(console_args, jobid_config, autohf, wandb_utils)
    else:
        modelcard_exp(console_args)
