# coding:utf-8
#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

seed=('1')
approach=('DSAD')
batch_size=('128')
n_epochs=('500')
lr=('0.0001')
lr_milestone=('150')
rep_dim=('16' '32')
pretrain=('False')

for i in ${approach[@]}
do
  for j in ${batch_size[@]}
  do
    for k in ${lr[@]}
    do
      for n in ${rep_dim[@]}
      do
        for m in ${seed[@]}
        do
          for p in ${n_epochs[@]}
          do
            for r in ${pretrain[@]}
            do
              python main.py yamaha /home/taki/yamaha/DeepSAD/log/0802_resnet_SAD_test /dataset/dataset/taki/data/yamaha/ad_dataset --height 256 --width 256 --approach $i --rep_dim $n --nu 1.0 --seed $m --n_epochs $p --lr $k --lr_milestone 150 --batch_size $j --pretrain $r
            done
          done
        done
      done
    done
  done
done
