########################## RUN ###################################

# ##### train MCTformer V1 ##########
python -W ignore main.py --model deit_small_MCTformerV1_patch16_224 \
                --batch-size 8 \
                --data-set VOC12 \
                --img-list voc12 \
                --data-path voc12/data \
                --layer-index 12 \
                --output_dir MCTformer_results/MCTformer_v1 \
                --finetune deit_small_patch16_224-cd65a155.pth \
                --clr_loss \
                --mIOU_metric \
                --cls_weight 1 \
                --clr_weight 1 \

# ######## train MCTformer V2 ##########
# python main.py --model deit_small_MCTformerV2_patch16_224 \
#                 --batch-size 64 \
#                 --data-set VOC12 \
#                 --img-list voc12 \
#                 --data-path voc12/data \
#                 --layer-index 12 \
#                 --output_dir MCTformer_results/MCTformer_v2 \
#                 --finetune deit_small_patch16_224-cd65a155.pth

# ##### Generating class-specific localization maps ##########
# python main.py --model deit_small_MCTformerV1_patch16_224 \
#                 --data-set VOC12MS \
#                 --scales 1.0 \
#                 --img-list voc12 \
#                 --data-path voc12/data \
#                 --output_dir MCTformer_results/MCTformer_v1 \
#                 --resume MCTformer_results/MCTformer_v1/checkpoint_best.pth \
#                 --gen_attention_maps \
#                 --attention-type fused \
#                 --layer-index 3 \
#                 --visualize-cls-attn \
#                 --patch-attn-refine \
#                 --attention-dir MCTformer_results/MCTformer_v1/attn-patchrefine \
#                 --cam-npy-dir MCTformer_results/MCTformer_v1/attn-patchrefine-npy \

# ######## Evaluating the generated class-specific localization maps ##########
# python evaluation.py --list voc12/train_aug_id.txt \
#                      --gt_dir voc12/data/SegmentationClass \
#                      --logfile MCTformer_results/MCTformer_v1/evallog.txt \
#                      --type npy \
#                      --curve True \
#                      --predict_dir MCTformer_results/MCTformer_v1/attn-patchrefine-npy \
#                      --comment "train100"

######### Generating class-specific localization maps ##########
# python main.py --model deit_small_MCTformerV2_patch16_224 \
#                 --data-set VOC12MS \
#                 --scales 1.0 \
#                 --img-list voc12 \
#                 --data-path voc12/data \
#                 --resume MCTformerV2.pth \
#                 --gen_attention_maps \
#                 --attention-type fused \
#                 --layer-index 3 \
#                 --visualize-cls-attn \
#                 --patch-attn-refine \
#                 --attention-dir MCTformer_results/MCTformer_v2/attn-patchrefine \
#                 --cam-npy-dir MCTformer_results/MCTformer_v2/attn-patchrefine-npy \
#                 --out-crf MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf \

######### Evaluating the generated class-specific localization maps ##########
# python evaluation.py --list voc12/train_id.txt \
#                      --gt_dir voc12/data/SegmentationClass \
#                      --logfile MCTformer_results/MCTformer_v2/evallog_crf=0.txt \
#                      --type npy \
#                      --curve True \
#                      --predict_dir MCTformer_results/MCTformer_v2/attn-patchrefine-npy \
#                      --comment "train 1464"

########################## AFFINITY ###################################

# python psa/train_aff.py --weights res38_cls.pth \
#                         --voc12_root voc12/data \
#                         --la_crf_dir MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf_1 \
#                         --ha_crf_dir MCTformer_results/MCTformer_v2/attn-patchrefine-npy-crf_12 \

# python psa/infer_aff.py --weights resnet38_aff.pth \
#                     --infer_list psa/voc12/train_aug.txt \
#                     --cam_dir MCTformer_results/MCTformer_v2/attn-patchrefine-npy \
#                     --voc12_root voc12/data \
#                     --out_rw MCTformer_results/MCTformer_v2/pgt-psa-rw \

# python evaluation.py --list voc12/train_id.txt \
#                      --gt_dir voc12/data/SegmentationClass \
#                      --logfile MCTformer_results/MCTformer_v2/evallog_rw.txt \
#                      --type png \
#                      --predict_dir MCTformer_results/MCTformer_v2/pgt-psa-rw \
#                      --comment "train 1464"

########################## SEGMENTATION ###################################

# python seg/train_seg.py --network resnet38_seg \
#                     --num_epochs 30 \
#                     --seg_pgt_path MCTformer_results/MCTformer_v2/pgt-psa-rw \
#                     --init_weights res38_cls.pth \
#                     --save_path MCTformer_results/MCTformer_v2/model \
#                     --list_path voc12/train_aug_id.txt \
#                     --img_path voc12/data/JPEGImages \
#                     --num_classes 21 \
#                     --batch_size 4

# python seg/infer_seg.py --weights MCTformer_results/MCTformer_v2/model/model_29.pth \
#                       --network resnet38_seg \
#                       --list_path voc12/val_id.txt \
#                       --gt_path voc12/data/SegmentationClass \
#                       --img_path voc12/data/JPEGImages \
#                       --save_path MCTformer_results/MCTformer_v2/val_ms_crf \
#                       --save_path_c MCTformer_results/MCTformer_v2/val_ms_crf_c \
#                       --scales 0.5 0.75 1.0 1.25 1.5 \
#                       --use_crf True

# python evaluation.py --list voc12/val_id.txt \
#                      --gt_dir voc12/data/SegmentationClass \
#                      --logfile MCTformer_results/MCTformer_v2/evallog_valseg.txt \
#                      --type png \
#                      --predict_dir MCTformer_results/MCTformer_v2/val_ms_crf \
#                      --comment "val 1449"

# python seg/infer_seg.py --weights MCTformer_results/MCTformer_v2/model/model_29.pth \
#                       --network resnet38_seg \
#                       --list_path voc12/train_id.txt \
#                       --gt_path voc12/data/SegmentationClass \
#                       --img_path voc12/data/JPEGImages \
#                       --save_path MCTformer_results/MCTformer_v2/train_ms_crf \
#                       --save_path_c MCTformer_results/MCTformer_v2/train_ms_crf_c \
#                       --scales 0.5 0.75 1.0 1.25 1.5 \
#                       --use_crf True

# python evaluation.py --list voc12/train_id.txt \
#                      --gt_dir voc12/data/SegmentationClass \
#                      --logfile MCTformer_results/MCTformer_v2/evallog_trainseg.txt \
#                      --type png \
#                      --predict_dir MCTformer_results/MCTformer_v2/train_ms_crf \
#                      --comment "train 1464"