# export CUDA_VISIBLE_DEVICES=1
# fairseq-hydra-train --config-dir conf/mlavt_finetune --config-name large_vox_433h_a.yaml \
#    task.data=/root/autodl-tmp/data/mlavt_tedx/es/200h_data \
#    task.label_dir=/root/autodl-tmp/data/mlavt_tedx/es/200h_data \
#    task.tokenizer_bpe_model=/root/autodl-tmp/data/mlavt_tedx/es/spm1000/spm_unigram1000.model \
#    model.w2v_path=/root/autodl-tmp/model/av_hubert/avhubert/checkpoints/large_vox_iter5.pt \
#    hydra.run.dir=/root/autodl-tmp/model/av_hubert/avhubert/model/finetune_large_vox_433h_a \
#    common.user_dir=`pwd`

## av_tmux3 a_tmux4
 export CUDA_VISIBLE_DEVICES=2
 fairseq-hydra-train --config-dir conf/mlavt_finetune --config-name large_vox_433h_av.yaml \
    task.data=/root/autodl-tmp/data/mlavt_tedx/es/200h_data_en \
    task.label_dir=/root/autodl-tmp/data/mlavt_tedx/es/200h_data_en \
    task.tokenizer_bpe_model=/root/autodl-tmp/data/mlavt_tedx/es/spm1000_en/spm_unigram1000.model \
    model.w2v_path=/root/autodl-tmp/model/av_hubert/avhubert/checkpoints/large_vox_iter5.pt \
    hydra.run.dir=/root/autodl-tmp/model/av_hubert/avhubert/model/avst_finetune_large_vox_433h_av \
    common.user_dir=`pwd`

# python ${PATH-TO-FAIRSEQ_ROOT}/fairseq_cli/train.py ${args}