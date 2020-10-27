# !/bin/bash

# You need to specify these configurations
dataset="robust04"                                           # robust04 or gov2
passage_size=100                                             # length of passages
passage_overlap=50                                           # length of overlaps between adjacent passages
chunk_size=10                                                # length of chunks
chunk_overlap=5                                              # length of overlaps between adjacent chunks
kc=10                                                        # number of chunks used for expansion, i.e. kc in the paper
rerank=1000                                                  # number of documents that you want to re-rank
first_model_size="large"                                     # size of BERT used in phase 1
second_model_size="large"                                    # size of BERT used in phase 2
third_model_size="large"                                     # size of BERT used in phase 3

main_path=Your_Path                                          # root path
run_file=${main_path}/${dataset}/Robust04_DPH_KL.res         # initial ranking
dataset_file=${main_path}/${dataset}/robust04_collection.txt # collection file after preprocessing
query_file=${main_path}/${dataset}/queries.json              # query file, JSON format (released in resources on GitHub)
qrels_file=${main_path}/${dataset}/qrels                     # qrels file
vocab_file=./bert/vocab.txt                                  # vocabulary used in BERT
# End here

passage_path=${main_path}/${dataset}/passage/${first_model_size}
output_path=${main_path}/${dataset}/output/m-${chunk_size}_${first_model_size}_${second_model_size}_${third_model_size}

# Select the most relevant passage from a document
python3 convert_tfrecord.py \
  --output_path ${passage_path} \
  --collection_file ${dataset_file} \
  --vocab ${vocab_file} \
  --queries ${query_file} \
  --qrels ${qrels_file} \
  --run_file ${run_file} \
  --dataset ${dataset} \
  --doc_depth 1000 \
  --task passage \
  --window_size ${passage_size} \
  --stride ${passage_overlap}

python3 select_pieces.py \
  --output_path ${passage_path} \
  --dataset ${dataset} \
  --model_size ${first_model_size} \
  --task passage \
  --batch_size 32

# Run 5-fold cross-validation
for idx in {1..5}; do
  first_model_path=${main_path}/${dataset}/model/${first_model_size}/fold-${idx}/

  # Fine-tune the BERT model and perform the first-round re-ranking (phase 1)
  python3 maxp_tfrecord.py \
    --passage_path ${passage_path} \
    --qrels ${qrels_file} \
    --vocab ${vocab_file} \
    --queries ${query_file} \
    --output_path ${output_path} \
    --fold ${idx} \
    --dataset ${dataset}

  python3 maxp_train.py \
    --passage_path ${passage_path}/fold-${idx} \
    --data_path ${output_path}/fold-${idx}/ \
    --output_path ${first_model_path} \
    --dataset ${dataset} \
    --model_size ${first_model_size} \
    --batch_size 32
done

for idx in {1..5}; do
  first_model_path=${main_path}/${dataset}/model/${first_model_size}/fold-${idx}/
  third_model_path=${main_path}/${dataset}/model/${third_model_size}/fold-${idx}/

  # Select chunks from PRF documents (phase 2)
  python3 convert_tfrecord.py \
    --output_path ${output_path} \
    --collection_file ${passage_path}/passage_id_text.txt \
    --vocab ${vocab_file} \
    --queries ${query_file} \
    --qrels ${qrels_file} \
    --first_model_path ${first_model_path} \
    --dataset ${dataset} \
    --window_size ${chunk_size} \
    --stride ${chunk_overlap} \
    --fold ${idx} \
    --task chunk \
    --doc_depth 10

  python3 select_pieces.py \
    --output_path ${output_path} \
    --dataset ${dataset} \
    --fold ${idx} \
    --model_size ${second_model_size} \
    --task chunk \
    --batch_size 32

  # Use chunks for expansion (phase 3)
  python3 expansion_tfrecord.py \
    --vocab ${vocab_file} \
    --qrels ${qrels_file} \
    --output_path ${output_path} \
    --queries ${query_file} \
    --fold ${idx} \
    --rerank_num ${rerank} \
    --kc ${kc} \
    --dataset ${dataset} \
    --passage_path ${passage_path} \
    --first_model_path ${first_model_path}

  python3 expansion_inference.py \
    --output_path ${output_path}/fold-${idx}/ \
    --kc ${kc} \
    --third_model_path ${third_model_path} \
    --model_size ${third_model_size} \
    --batch_size 32 \
    --dataset ${dataset} \
    --rerank_num ${rerank} \
    --first_model_path ${first_model_path}

  # interpolation according to Equation (5) in the paper
  python3 interpolate.py \
    --expansion_path ${output_path}/fold-${idx}/rerank-${rerank}_kc-${kc}/result/ \
    --maxp_path ${first_model_path} \
    --dataset ${dataset} \
    --rerank_num ${rerank}
done

# Get the final result according to validation sets
python3 get_final_result.py \
  --qrels ${qrels_file} \
  --run_file ${run_file} \
  --dataset ${dataset} \
  --rerank_num ${rerank} \
  --kc ${kc} \
  --main_path ${output_path}

echo "Done."
