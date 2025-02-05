
python main_61_26_GPU_ClustV2.py --epochs 100 --batch-size 12048 --test-batch-size 12048 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss
python main_61_26_GPU_ClustV3.py --epochs 100 --batch-size 1024 --test-batch-size 1024 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss

python main_61_26_GPU_ClustV2.py --epochs 100 --batch-size 12048 --test-batch-size 12048

python main_61_26_GPU_Clust_Dice.py --epochs 100 --batch-size 1024 --test-batch-size 1024 --clustering_weight 10 --use_clustering_loss --use_dice_a_loss

