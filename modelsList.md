Список всех моделей и датасетов из статьи
Этот список не полный. Вы можете помочь, пополнив его самостоятельно

ViT? on ImageNet, on JFT
(vision-language contrastive learning fine-tunning on ImageNet)?
diffusion models(FID?) https://github.com/facebookresearch/FiD
(autoregression,masked language modeling, fine-tunning)?
benchmarks: v2,a,r,sketch,objectNet(dataset)
Transformer?,MLP,ResNet,U-Net,Hybrid
ViT-B/16 on imageNet (settings: image res 224 batch size 4096)

таблица(model\dataset):
ResNet-50 ImageNet,ReaL,V2,A,R,ImageNet-21K,MSCOCO,Flickr30K
Mixer-S/16
Mixer-B/16
ViT-S/16
ViT-H/14
CoAtNet-1
CoAtNet-3
--Vit-(-)/(-)
LiT

(page 8)
vit,mixer,lit models https://github.com/google-research/vision_transformer#available-vit-models
coatNet https://github.com/chinhsuanwu/coatnet-pytorch

CLIP(pre-trained on JFT-5B)? https://github.com/openai/CLIP
BASIC-L on ImageNet
U-Net on ImageNet (image synthesis) https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py
Imagen on internal dataset (text to image) https://github.com/lucidrains/imagen-pytorch
Transformer on page 11 ??
table on page 12 LLM fine tuned on CoLA,SST-2,MRPC,STS-B,QQP,MNLI-m,MNLI-mm,QNLI,RTE(in glue)
BERT on C4 dataset https://github.com/google-research/bert
T5 fine-tuned on GLUE https://huggingface.co/docs/transformers/model_doc/t5
https://github.com/md-experiments/glue_benchmark

open-source? datasets https://paperswithcode.com/datasets

image models
| model\dataset                                                                                           | [imagenet](https://image-net.org/index.php) | imagenet21 | [v2](https://github.com/modestyachts/ImageNetV2) | [a](https://github.com/hendrycks/natural-adv-examples) | [objectnet](https://objectnet.dev/) | real(benchmark) | [mscoco](https://cocodataset.org/#download) | flickr (in progress) |
|---------------------------------------------------------------------------------------------------------|---------------------------------------------|------------|--------------------------------------------------|--------------------------------------------------------|-------------------------------------|-----------------|---------------------------------------------|----------------------|
| image classification                                                                                    |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [resnet](https://huggingface.co/facebook/detr-resnet-50)                                                |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [mixer](https://github.com/google-research/vision_transformer)                                          |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [vit](https://github.com/google-research/vision_transformer)                                            |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [coatnet](https://huggingface.co/timm/coatnet_1_rw_224.sw_in1k)                                         |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| basic-l                                                                                                 |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| image segmentation                                                                                      |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [u-net](https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py) |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| image generation                                                                                        |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| text to image                                                                                           |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |
| [imagen](https://github.com/lucidrains/imagen-pytorch)                                                  |                                             |            |                                                  |                                                        |                                     |                 |                                             |                      |


text models
| model\dataset                                    | [glue(benchmark)](https://github.com/md-experiments/glue_benchmark) | [c4](https://www.tensorflow.org/datasets/catalog/c4) |
|--------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------|
| text encoders                                    |                                                                     |                                                      |
| [bert](https://huggingface.co/bert-base-uncased) |                                                                     |                                                      |
| text decoders                                    |                                                                     |                                                      |
| [t5](https://huggingface.co/t5-base)             |                                                                     |                                                      |
