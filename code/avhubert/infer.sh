# export CUDA_VISIBLE_DEVICES=1
# python -B infer_s2s.py --config-dir ./conf/ \
#   --config-name s2s_decode.yaml \
#   dataset.gen_subset=test \
#   common_eval.path=/root/autodl-tmp/model/av_hubert/avhubert/model/finetune_large_vox_433h_av/checkpoints/checkpoint_best.pt \
#   common_eval.results_path=/root/autodl-tmp/model/av_hubert/avhubert/model/decode/s2s/test_es_avsr_new \
#   override.modalities=['audio','video'] \
#   common.user_dir=`pwd`

export CUDA_VISIBLE_DEVICES=1
python -B infer_s2s.py --config-dir ./conf/ \
  --config-name s2s_decode.yaml \
  dataset.gen_subset=test \
  common_eval.path=/root/autodl-tmp/model/av_hubert/avhubert/model/finetune_large_vox_433h_av/checkpoints/checkpoint_best.pt \
  common_eval.results_path=/root/autodl-tmp/model/av_hubert/avhubert/model/decode/s2s/test_es_asr_av_new_new \
  override.modalities=['audio','video'] \
  common.user_dir=`pwd`
