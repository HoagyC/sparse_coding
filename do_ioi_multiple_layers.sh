python generate_test_data.py --model="EleutherAI/pythia-410m-deduped" --n_chunks=30 --layers 3

python basic_l1_sweep.py --dataset_dir="activation_data/layer_3" --output_dir="dicts_l3" --ratio=4
python ioi_feature_ident.py 3