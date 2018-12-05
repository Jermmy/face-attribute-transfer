# train emotionnet
lr=1e-3
image_size=160
batch_size=32
epochs=30
continue_train=0
pooling=avg
loss_type=l2

model=lr_${lr}_pooling_${pooling}_${loss_type}
ckpt_path=ckpt/${model}
result_path=result/${model}

python train_emotion.py --image_size ${image_size} --lr ${lr}
          --batch_size ${batch_size} --epochs ${epochs}
          --continue_train ${continue_train}
          --pooling ${pooling} --loss_type ${loss_type}
          --ckpt_path ${ckpt_path} --result_path ${result_path}