FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN pip install faiss-cpu==1.7.2 scipy==1.11.1 scikit-learn==1.3.0
RUN pip install torchmetrics==1.2.0 timm==0.9.10 tensorboard==2.15.1
RUN python -c "import torchvision.models as models; models.wide_resnet50_2(pretrained=True)"