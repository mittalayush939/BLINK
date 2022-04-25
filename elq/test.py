# -*- coding: utf-8 -*-
"""
Testing ELQ on the SMART 2021 dataset using custom "main_dense1" python code

"""

import json
import argparse 
import elq.main_dense1 as main_dense

f = open("./smart-2021-rl-wikidata-train.json") # the path of the dataset to be tested

#loading the dataset
data = json.load(f)
a_key = "question"
values_of_key = [a_dict[a_key] for a_dict in data] 
count = 0
list_test = []

#preparing the laoded dataset for the ELQ prediction
for item in values_of_key:
  dict_test = {
      "id": count,
      "text": item.lower(),
  }
  list_test.append(dict_test)
  count = count+1

data_to_link = list_test
models_path = "./models/" # the path where you stored the ELQ models

config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "./results_RL", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
    "save_preds_dir": "./results_RL/",
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)


predictions = main_dense.run(args, None, *models, test_data=data_to_link)
