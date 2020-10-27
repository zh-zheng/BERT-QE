# Path of the original collection
robust04_collection_path = ""
gov2_collection_path = ""

# Path of the processed collection
robust04_output_path = ""
gov2_output_path = ""

# DOCNO in the initial ranking
robust04_docno_list = ""
gov2_docno_list = ""

# Path of the trained checkpoints and config files of BERT
checkpoint_dict = {
    "medium": "Your_Path",
    "small": "Your_Path",
    "tiny": "Your_Path",
    "base": "Your_Path",
    "large": "Your_Path"}

config_dict = {"small": "Your_Path/bert_config.json",
               "tiny": "Your_Path/bert_config.json",
               "medium": "Your_Path/bert_config.json",
               "base": "Your_Path/bert_config.json",
               "large": "Your_Path/bert_config.json"}

# Path of cross-validation partitions.
cv_folder_path = {
    "robust04": "Your_Path/robust04/cv",
    "gov2": "Your_Path/gov2/cv"
}

# Path of the trec_eval script used for evaluation
trec_eval_script_path = 'Your_Path/trec_eval'
