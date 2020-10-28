# BERT-QE

This repo contains the code and resources for the paper:

 [BERT-QE: Contextualized Query Expansion for Document Re-ranking](https://arxiv.org/abs/2009.07258).
 In *Findings of ACL: EMNLP 2020*.

## Introduction
BERT-QE leverages the strength of BERT to select relevant document chunks for query expansion. The BERT-QE model consists of three phases, in which BERT models of different sizes can be used to balance effectiveness and efficiency. Some experimental results on Robust04 are listed below:


  Model    | FLOPs |  P@20 | NDCG@20 | MAP | 
:----------: | :------: | :------: | :------: | :----: |
BERT-Large    | 1.00x | 0.4769 | 0.5397 | 0.3743 |
BERT-QE-LLL   | 11.19x| 0.4888 | 0.5533 | 0.3865 |
BERT-QE-LMT   | 1.03x | 0.4839 | 0.5483 | 0.3765 |
BERT-QE-LLS   | 1.30x | 0.4869 | 0.5501 | 0.3798 |


## Requirements

We recommend to install [Anaconda](https://www.anaconda.com/). Then install the packages using Anaconda:
```
conda install --yes --file requirements.txt
```

NOTE: in the paper, we run the experiments using a TPU. Alternatively, you can use GPUs and install `tensorflow-gpu` (see `requirements.txt`).

## Getting Started

In this repo, we provide instructions on how to run BERT-QE on Robust04 and GOV2 datasets.

### Data preparation

You need to obtain [Robust04](https://trec.nist.gov/data_disks.html) (i.e. TREC Disks 4&5) and [GOV2](http://ir.dcs.gla.ac.uk/test_collections/gov2-summary.htm) collections.

The (useful) directory structure of Robust04:
```
disk4/
├── FR94
└── FT

disk5/
├── FBIS
└── LATIMES
```

The directory structure of GOV2:
```
gov2/
├── GX000
├── GX001
├── ...
└── GX272
```

### Preprocess 

To preprocess the datasets, in `config.py`, you need to specify the root path to each dataset and
the output path in which the processed data will be placed, e.g. `robust04_collection_path` and `robust04_output_path` for Robust04. 
As the collection is huge, you can choose to only process documents in the initial ranking.

For example, given an initial ranking `Robust04_DPH_KL.res`, extract all unique document ids by:
```
cut -d ' ' -f3 Robust04_DPH_KL.res | sort | uniq > robust04_docno_list.txt
```
And assign its path to `robust04_docno_list` in `config.py`.

You can now preprocess Robust04 and GOV2 using `robust04_preprocess.py` and `gov2_preprocess.py`, respectively.
Finally, you need to merge all the processed text files into a single file,
which will be used as `dataset_file` in `run.sh`.

For Robust04:
```
cat ${robust04_output_path}/* > robust04_collection.txt
```

As titles are available in Robust04, in the output file, the first column is the document id, the second column is the title, and the third column is the document text.


For GOV2:
```
cat ${gov2_output_path}/*/*.txt > gov2_collection.txt
```

In the output file, the first column is the document id and the second column is the document text.


### Training and evaluation

We first need to fine-tune the BERT models of different sizes from the [BERT repo](https://github.com/google-research/bert) on the MS MARCO collection. 
For details, please refer to [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert). 
After fine-tuning the models, you should specify the paths of the fine-tuned BERT checkpoints and the config files ([`bert_config.json`](#resources)) in `config.py`. 
If you want to skip this step, you can refer to [PARADE](https://github.com/canjiali/PARADE) and [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert) (for BERT-Large) to download the trained checkpoints.

Then we continue to fine-tune BERT models on the target dataset, i.e. Robust04 or GOV2, and select chunks to perform query expansion.
You can download our partitions of cross-validation from [here](#cv) and the TREC evaluation script from [here](https://trec.nist.gov/trec_eval/), and set the `cv_folder_path` and `trec_eval_script_path` in `config.py`. 
The last step is to fill in the configurations in `run.sh` (see comments for instructions) and run
```
bash run.sh
```

The training and evaluation of BERT-QE will be conducted automatically!

**NOTE**: if you plan to use BERT models of different sizes in three phases (e.g. BERT-QE-LMT), you need to first fine-tune those models on the target dataset.
Specifically, you should specify the `first_model_size` and run the code before line 69 (i.e. before "phase2") in `run.sh` for each model.

## Resources

We release the run files, BERT models fine-tuned on two collections, and the partitions of cross-validation to help the community reproduce our results.

* Fine-tuned BERT models (incl. `bert_config.json`)

| Model        | Robust04  | GOV2 | 
|--------------|----------|--------------------|
| BERT-Large | [Download](https://drive.google.com/drive/folders/1rdquOffuns-oRFWlFV6W7C3wn2w1PZip?usp=sharing)|  [Download](https://drive.google.com/drive/folders/18HVhdxlrg5rIcLXgiGwMaHa06NUYFeFY?usp=sharing)   |  
| BERT-Base    | [Download](https://drive.google.com/drive/folders/1KKsllpvPxwbnJDVJl23MbQy4KFQGtGTj?usp=sharing) |  [Download](https://drive.google.com/drive/folders/1KqcjHbnDvdHQFw-wkl9tYfDEUtB7Isg8?usp=sharing)   | 
| BERT-Medium   | [Download](https://drive.google.com/drive/folders/1rakEIVBQv3mZ_v9D6mVUAovWL_q-uYzm?usp=sharing) |   [Download](https://drive.google.com/drive/folders/1ucLyg_5eEFS7NTbAu-YpbRJY_Ers0F5Z?usp=sharing)   | 
| BERT-Small*    | [Download](https://drive.google.com/drive/folders/1FlhTwEiMNS0YOxKzORG7EGAETX5GpvF2?usp=sharing) |  [Download](https://drive.google.com/drive/folders/1HcjVMsiQYQpR8iI4QRxL0wo0d6RAR5uz?usp=sharing)   | 
| BERT-Tiny    | [Download](https://drive.google.com/drive/folders/1L7u86kECHDhmsGvdt6bfimT82BQLgZ7v?usp=sharing)  |  [Download](https://drive.google.com/drive/folders/1C__wzyeL95SmXjG7gNgfDdv9_Ygfp0RP?usp=sharing)   | 

\* Note that the BERT-Small corresponds to BERT-Mini in the [BERT repo](https://github.com/google-research/bert), for the sake of convenient descriptions in the paper.

**Usage**: take BERT-Large fine-tuned on Robust04 for example, you need to first unzip all the `fold-*.zip` files, then rename the root folder from `BERT-Large-Robust04` to `large`, and put the folder in the directory `${main_path}/robust04/model/`. Note that `main_path` is specified in `run.sh`.

* [Run files of BERT-QE](https://drive.google.com/file/d/1ZaaDYXGHNoLa7YvHlIHPt22m0uZGxxVj/view?usp=sharing)
* [Initial rankings](https://drive.google.com/file/d/12p4FnLKW90qbdJLQaoZfjFTQXg_B-shq/view?usp=sharing)
* <a name="cv"></a>[Partitions of cross-validation](https://drive.google.com/file/d/1g2_S4xe34A8C1CWNZrEg6T8Q8mEgf1rl/view?usp=sharing)
* [Queries (JSON format)](https://drive.google.com/file/d/15jK4Rxqtl1Hrga8AQNAId-Ryl8cVyKgU/view?usp=sharing)
## Citation

If you use our code or resources, please cite this paper:
```
@inproceedings{zheng2020bertqe,
    title={BERT-QE: Contextualized Query Expansion for Document Re-ranking},
    author={Zheng, Zhi and Hui, Kai and He, Ben and Han, Xianpei and Sun, Le and Yates, Andrew},
    booktitle = "Findings of ACL: EMNLP",
    year = "2020",
    publisher = "Association for Computational Linguistics"
}
``` 
## Acknowledgement

Some snippets of the code are borrowed from [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert) and [NPRF](https://github.com/ucasir/NPRF).