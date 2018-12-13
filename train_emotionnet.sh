# train emotionnet or glemotionnet
use_model=glemotionnet
lr=1e-4
image_size=160
batch_size=32
epochs=20
continue_train=0
pooling=avg
loss_type=smoothl1
l_landmark=1.0

model=${use_model}/lr_${lr}_pooling_${pooling}_${loss_type}
ckpt_path=ckpt/${model}
result_path=result_emotion/${model}

load_model=ckpt/${model}/epoch-40.pkl

python train_emotion.py --image_size ${image_size} --lr ${lr} \
          --batch_size ${batch_size} \
          --epochs ${epochs} \
          --continue_train ${continue_train} \
          --pooling ${pooling} \
          --l_landmark ${l_landmark} \
          --loss_type ${loss_type} \
          --ckpt_path ${ckpt_path} \
          --result_path ${result_path} \
          --device_ids 0,1 \
          --use_model ${use_model} \
