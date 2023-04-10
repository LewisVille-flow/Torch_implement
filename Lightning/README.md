# Pytorch Lightning
### 전체적인 시퀀스
### 1. Preprocessing
- **Load를 위한 Dataset**
    - init, len, getitem 구현은 동일

- **DataModule**
    - init
    - prepare_data: 전체 시퀀스에서 1번만 호출됨
    - setup: gpu 마다 호출된다(배치마다 호출된다는 얘기인지 다시확인)
    self.train_dataset, self.valid_dataset, self.test_dataset 을 세팅해둔다.
    이따 dataloader에서 사용함

    **기타 함수**
    - tokenizing
    - train_dataloader
    - valid_dataloader
    - test_dataloader: loader 함수들은 단순히 wrapper함수이며, return DataLoader 함수를 호출한다.

### 2. Model
- init
    - Pretrained/직접 구축한 모델 세팅. 직접 구현이 가능하다.
    - pl.LightningModule은 nn.Module을 상속하기 때문에, nn.Linear()라는 것으로 사용 가능하다.
    - Optimizer, Criterion 세팅(torchmetrics로 간단히 Accuracy ... 사용가능)
- forward
    - 기존 모델들의 forward와 동일.
    - forward 함수이므로 "self(input)" 라는 것으로 추후 호출 가능한 함수...
- configure_optimizers
    - optimizer 세팅함수로 보임.
- training_step
- validation_step
- test_step
    - 각 스텝별 반복되는 행위들이 간략화되어 구현되어있는 부분이다.
    - self.log라는 것으로 자동 logging이 가능함에 유의.
    - BERT based model은 label을 직접 모델에 Input으로 넣어 loss를 같이 받아옴(output.loss)


### 주의할 점
시퀀스가 상당히 간단해진다. 학습 및 추론을 위한 시퀀스는,  
- Model() 호출 -> DataModule() 호출 -> device 세팅 간략히 -> pl.Trainer(... 세팅 ...) -> .train() & .fit()  

- pl.Trainer(devices=torch.cuda.device_count(), max_epochs=1, accelerator='gpu')