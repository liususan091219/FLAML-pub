'''Require: pip install torch transformers datasets wandb flaml[blendsearch,ray]
'''
#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
import argparse,os
import subprocess
from flaml.nlp.result_analysis.azure_utils import JobID

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

def create_partial_config_list():
    jobid_config = JobID()
    jobid_config.mod = "list"
    jobid_config.spa = "uni"
    jobid_config.presz = "xlarge"
    return jobid_config

def create_partial_config_hpo():
    jobid_config = JobID()
    jobid_config.mod = "hpo"
    jobid_config.spa = "uni"
    return jobid_config

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', type=str, help='analysis mode', required=True, choices=["summary", "analysis", "plot", "extract"])
    arg_parser.add_argument('--key_path', type=str, help='key path', required=False, default = "../../")
    args = arg_parser.parse_args()

    partial_config_large = create_partial_config_bestnn()
    from flaml.nlp.result_analysis.generate_result_summary import compare_small_vs_large, get_result, check_conflict, \
    print_cfo

    #get_result(args, partial_config_large)
    #check_conflict(args, [partial_config_large])
    print_cfo(args)