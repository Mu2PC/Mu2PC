nshard = 1
lrs3 = "/root/autodl-tmp/data/mlavt_tedx/es"
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.audio.${rank}; done > ${lrs3}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.video.${rank}; done > ${lrs3}/nframes.video