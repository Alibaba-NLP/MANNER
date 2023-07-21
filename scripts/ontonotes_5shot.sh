
seed=171 
N=4 # ontonotes: 4, wnut: 3, gum: 2, conll: 1 
K=5 # k-shot 
device=0 
experiment_name=ontonotes_5shot 

# pretraining on source domain 
python cross_dataset/main.py \
    --data_path=/nfs/manner/data/ \
    --types_path=/nfs/manner/data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --max_seq_len=128 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --memory_size=15 \
    --sample_size=5 \
    --gpu_device=${device} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=16 \
    --lr=1e-4 \
    --max_train_steps=500 \
    --eval_every_train_steps=100 \
    --lr_finetune=1e-4 \
    --max_finetune_steps=50 \
    --ignore_eval_test 


# finetuning & evaluate on target domain 
python cross_dataset/main.py \
    --data_path=data/ \
    --types_path=data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --max_seq_len=128 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --memory_size=15 \
    --sample_size=5 \
    --gpu_device=${device} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=16 \
    --lr=1e-4 \
    --max_train_steps=500 \
    --eval_every_train_steps=100 \
    --lr_finetune=1e-4 \
    --max_finetune_steps=50 \
    --ignore_eval_test \
    --test_only 