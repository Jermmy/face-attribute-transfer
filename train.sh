content_img=data/w2.jpg
style_img=data/ju3.jpg

device_id=1
init=noise
network=vggface
lr=1e-3
epochs=20000
lc=10.
ls=10.
pooling=avg
emotion_loss=cxloss

if [ $network == 'vgg' ]; then
     echo 'vgg'
     model=${network}
     result_path=result/${model}/${init}_${emotion_loss}
     ckpt_path=pretrain/vgg16-397923af.pth
     content_layers=r43,r42,r41
     style_layers=r33,r32,r31
elif [ $network == 'vggface' ]; then
     echo 'vggface'
     model=${network}
     result_path=result/${model}/${init}_${emotion_loss}
     ckpt_path=pretrain/VGG_FACE.pth
     content_layers=r10,r9,r8
     style_layers=r6,r5,r4
else
     echo 'emotionnet'
     model=${network}/lr_1e-4_pooling_avg_smoothl1
     result_path=result/${model}_${init}_${emotion_loss}
     ckpt_path=ckpt/${model}/epoch-20.pkl

     content_layers=r33,r32,r31
     # style_layers=fc8,fc7,fc6,r53,r52,r51,r43,r42,r41
     style_layers=r53,r52,r51,r43,r42,r41
fi
# VGG


python train.py --ckpt_path ${ckpt_path} \
          --content_img ${content_img} \
          --style_img ${style_img} \
          --lr ${lr} \
          --epochs ${epochs} \
          --lc ${lc} \
          --ls ${ls} \
          --pooling ${pooling} \
          --content_layers ${content_layers} \
          --style_layers ${style_layers} \
          --result_path ${result_path} \
          --network ${network} \
          --init ${init} \
          --emotion_loss ${emotion_loss} \
          --device_id ${device_id}
