
python main_61_26_GPU_ClustV2.py --epochs 100 --batch-size 12048 --test-batch-size 12048 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss
python main_61_26_GPU_ClustV3.py --epochs 100 --batch-size 1024 --test-batch-size 1024 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss

python main_61_26_GPU_ClustV2.py --epochs 100 --batch-size 12048 --test-batch-size 12048

python main_61_26_GPU_Clust_Dice_ROI_FE.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss
python main_61_26_GPU_Clust_Dice_ROI_FE.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 1 --use_clustering_loss
python train_final.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 1 --use_clustering_loss --quick_test --use_feature_extractor
python train_final.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 1 --use_clustering_loss --quick_test --use_concat --device '1'
python train_final.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 10 --use_clustering_loss --use_embedding --device '0'

python train_final.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 1 --use_clustering_loss --use_feature_extractor --device '1'
python train_final.py --epochs 100 --batch-size 4096 --test-batch-size 4096 --clustering_weight 10 --use_clustering_loss --use_feature_extractor --device '0'
