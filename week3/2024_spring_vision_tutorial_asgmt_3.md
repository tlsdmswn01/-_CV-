**Week3**

**2024\_spring\_vision\_tutorial**

**Goal : Wandb,DDP 구현 및 Timm을 활용해 Resnet50으로 82% 이상 성능 도달**

**[주요 실험 및 결과]**

<table><tr><th colspan="1" valign="top"></th><th colspan="1" valign="top"></th><th colspan="1" valign="top"><b>Default Setting</b></th><th colspan="1" valign="top"><b>Experiment&Results1</b></th><th colspan="1" valign="top"><b>Experiment&Results2</b></th></tr>
<tr><td colspan="2" valign="top"><b>Batch_size</b></td><td colspan="1" valign="top">512</td><td colspan="1" valign="top">512</td><td colspan="1" valign="top">512</td></tr>
<tr><td colspan="1" rowspan="3" valign="top"><p></p><p><b>Optimizer</b></p></td><td colspan="1" valign="top">Opt.</td><td colspan="1" valign="top">SGD</td><td colspan="1" valign="top">SGD</td><td colspan="1" valign="top">SGD</td></tr>
<tr><td colspan="1" valign="top">Momentum</td><td colspan="1" valign="top">0\.4</td><td colspan="1" valign="top">0\.9</td><td colspan="1" valign="top">0\.9</td></tr>
<tr><td colspan="1" valign="top">Weight-decay</td><td colspan="1" valign="top">1e-9</td><td colspan="1" valign="top">1e-3</td><td colspan="1" valign="top">1e-3</td></tr>
<tr><td colspan="1" rowspan="5" valign="top"><p></p><p></p><p><b>Scheduler</b></p></td><td colspan="1" valign="top">Sched</td><td colspan="1" valign="top">CosineAnnealingLR</td><td colspan="1" valign="top">CosineAnnealingLR</td><td colspan="1" valign="top">CosineAnnealingLR</td></tr>
<tr><td colspan="1" valign="top">Learning rate</td><td colspan="1" valign="top">0\.1</td><td colspan="1" valign="top">0\.25</td><td colspan="1" valign="top">0\.25</td></tr>
<tr><td colspan="1" valign="top">Warmup-lr</td><td colspan="1" valign="top">0\.01</td><td colspan="1" valign="top">0\.01</td><td colspan="1" valign="top">0\.01</td></tr>
<tr><td colspan="1" valign="top">Min-lr</td><td colspan="1" valign="top">1e-6</td><td colspan="1" valign="top">1e-6</td><td colspan="1" valign="top">1e-6</td></tr>
<tr><td colspan="1" valign="top">Warmup-epochs</td><td colspan="1" valign="top">5</td><td colspan="1" valign="top">5</td><td colspan="1" valign="top">5</td></tr>
<tr><td colspan="1" rowspan="5" valign="top"><p></p><p></p><p><b>Augmentation</b></p></td><td colspan="1" valign="top">Hflip</td><td colspan="1" valign="top">0\.7</td><td colspan="1" valign="top">0\.7</td><td colspan="1" valign="top">0\.7</td></tr>
<tr><td colspan="1" valign="top">Color-jitter</td><td colspan="1" valign="top">0\.2</td><td colspan="1" valign="top">0\.2</td><td colspan="1" valign="top">X</td></tr>
<tr><td colspan="1" valign="top">Auto Augment</td><td colspan="1" valign="top">None</td><td colspan="1" valign="top">None</td><td colspan="1" valign="top">original</td></tr>
<tr><td colspan="1" valign="top">Cut mix</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0</td></tr>
<tr><td colspan="1" valign="top">Mix-up</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0</td></tr>
<tr><td colspan="2" valign="top"><b>Best Accuracy</b></td><td colspan="1" valign="top">70\.72</td><td colspan="1" valign="top">78\.19</td><td colspan="1" valign="top">80\.31</td></tr>
<tr><td colspan="2" valign="top"><b>Time</b></td><td colspan="1" valign="top">48m (epoch:200)</td><td colspan="1" valign="top">49m 45s (epoch:200)</td><td colspan="1" valign="top">47m 25s (epoch:200)</td></tr>
<tr><td colspan="2" valign="top"><p></p><p></p><p></p><p></p><p></p><p><b>Loss History plot</b></p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.001.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.002.png)</p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.003.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.004.png)</p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.005.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.006.png)</p></td></tr>
<tr><td colspan="2" valign="top"><p></p><p></p><p></p><p></p><p></p><p><b>Analyze</b></p></td><td colspan="1" valign="top"><p>Learning rate, weight decay, Momemtum값이 작아서 local minimum에 빠져 특정 정확도 값에 수렴하는 것으로 의심</p><p>- Learning rate,Momemtum, weight_deacy 값을 키워보자</p></td><td colspan="1" valign="top"><p>정확도가 오르긴 했지만, train_loss는  epoch이 진행됨에 따라 계속 감소하는 것과는 반대로 eval_loss는 일정 epoch이상에서는 더 이상 감소를 보이지 않았다. </p><p>-> Overfitting이 의심되어, 데이터 셋에 적절한 Augment를 찾아 적용해 데이터의 복잡도를 늘려보자</p></td><td colspan="1" valign="top"><p>진한 초록색 부분</p><p>이전보다 train_loss의 감소에 따라 eval_loss가 감소하는 것을 볼 수 있었지만, 여진히 Overfitting이 의심되어, Cutmix를 통해 데이터 복잡도를 늘려보고자 함</p></td></tr>
</table>

**[사용한 기법]**

- **Momemtum: 이전 단계에서의 파라미터 업데이트 방향으로 힘을 받아 현재 단계에서 더 나은 방향으로 파라미터를 조정  (수렴 속도 향상 및 이상치에 대한 저항성)**
- **Weight\_decay: 학습과정에서 모델의 파라미터에 대한 규제를 가하는 방법. 파라미터에 대한 규제를 가하면 모델이 좀 더 일반화 되어 훈련 데이터에 대해 과적합을 예방하는데 도움이 된다.** 
- **Color Jitter: Data augmentation의 기법 중 하나로, 이미지의 Lightness, Hue 그리고 saturation등이 임의로 변하는 것을 의미**
- **AutoAugment: 모델의 일반화 성능을 향상시키기 위해 데이터셋에 자동으로 증강을 적용시키는 알고리즘. 다양한 변형을 진행하면서 어떤 변형을 얼마나 적용할지를 강화학습을 통해 최적화 한다. 이번 실험에서는 timm의 autoaugment를 사용했기에, Policy가 ‘v0’와 ‘original’이 있었는데 Imagenet 데이터셋 기반 policy인 original을 적용해 실험을 진행하였다.(cifar10 또는 cifar100 policy를 찾아보았는데 timm에서는 지원해주지 않는 것 같았다..)**

<table><tr><th colspan="1" valign="top"></th><th colspan="1" valign="top"></th><th colspan="1" valign="top"><b>Experiment&Results3</b></th><th colspan="1" valign="top"><b>Experiment&Results4</b></th><th colspan="1" valign="top"><b>Experiment&Results5</b></th></tr>
<tr><td colspan="2" valign="top"><b>Batch_size</b></td><td colspan="1" valign="top">512</td><td colspan="1" valign="top">512</td><td colspan="1" valign="top">512</td></tr>
<tr><td colspan="1" rowspan="3" valign="top"><p></p><p><b>Optimizer</b></p></td><td colspan="1" valign="top">Opt.</td><td colspan="1" valign="top">SGD</td><td colspan="1" valign="top">SGD</td><td colspan="1" valign="top">SGD</td></tr>
<tr><td colspan="1" valign="top">Momentum</td><td colspan="1" valign="top">0\.9</td><td colspan="1" valign="top">0\.9</td><td colspan="1" valign="top">0\.9</td></tr>
<tr><td colspan="1" valign="top">Weight-decay</td><td colspan="1" valign="top">1e-3</td><td colspan="1" valign="top">1e-3</td><td colspan="1" valign="top">1e-3</td></tr>
<tr><td colspan="1" rowspan="5" valign="top"><p></p><p></p><p><b>Scheduler</b></p></td><td colspan="1" valign="top">Sched</td><td colspan="1" valign="top">CosineAnnealingLR</td><td colspan="1" valign="top">CosineAnnealingLR</td><td colspan="1" valign="top">CosineAnnealingLR</td></tr>
<tr><td colspan="1" valign="top">Learning rate</td><td colspan="1" valign="top">0\.25</td><td colspan="1" valign="top">0\.25</td><td colspan="1" valign="top">0\.25</td></tr>
<tr><td colspan="1" valign="top">Warmup-lr</td><td colspan="1" valign="top">0\.01</td><td colspan="1" valign="top">0\.01</td><td colspan="1" valign="top">0\.01</td></tr>
<tr><td colspan="1" valign="top">Min-lr</td><td colspan="1" valign="top">1e-6</td><td colspan="1" valign="top">1e-6</td><td colspan="1" valign="top">1e-6</td></tr>
<tr><td colspan="1" valign="top">Warmup-epochs</td><td colspan="1" valign="top">5</td><td colspan="1" valign="top">5</td><td colspan="1" valign="top">5</td></tr>
<tr><td colspan="1" rowspan="5" valign="top"><p></p><p></p><p><b>Augmentation</b></p></td><td colspan="1" valign="top">Hflip</td><td colspan="1" valign="top">0\.7</td><td colspan="1" valign="top">0\.7</td><td colspan="1" valign="top">0\.7</td></tr>
<tr><td colspan="1" valign="top">Color-jitter</td><td colspan="1" valign="top">X</td><td colspan="1" valign="top">X</td><td colspan="1" valign="top">X</td></tr>
<tr><td colspan="1" valign="top">Auto Augment</td><td colspan="1" valign="top">Original</td><td colspan="1" valign="top">original</td><td colspan="1" valign="top">Original</td></tr>
<tr><td colspan="1" valign="top">Cut mix</td><td colspan="1" valign="top">1\.0</td><td colspan="1" valign="top">1\.0</td><td colspan="1" valign="top">0\.5</td></tr>
<tr><td colspan="1" valign="top">Mix-up</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0</td><td colspan="1" valign="top">0\.5</td></tr>
<tr><td colspan="2" valign="top"><b>Best Accuracy</b></td><td colspan="1" valign="top">80\.91</td><td colspan="1" valign="top">82\.13</td><td colspan="1" valign="top">81\.28</td></tr>
<tr><td colspan="2" valign="top"><b>Time</b></td><td colspan="1" valign="top">49m 31s (epoch: 200)</td><td colspan="1" valign="top">1h 1m (epoch : 250)</td><td colspan="1" valign="top">1h 1m (epoch: 250)</td></tr>
<tr><td colspan="2" valign="top"><p></p><p></p><p></p><p></p><p></p><p></p><p><b>Loss History plot</b></p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.007.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.008.png)</p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.009.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.010.png)</p></td><td colspan="1" valign="top"><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.011.png)</p><p>![](Aspose.Words.3ad6c5be-c582-47b7-b002-6c260278bb31.012.png)</p></td></tr>
<tr><td colspan="2" valign="top"><p></p><p></p><p></p><p></p><p></p><p></p><p><b>Analyze</b></p></td><td colspan="1" valign="top">Cutmix를 1.0으로 설정하니 Train_loss가 이전 실험과 다르게 값이 높아 학습자체를 어려워 하는 것 같았음. Eval_loss를 확인했을 때 계속해서 감소하는 추세를 보여 epoch를 더 늘려서 특정 값에 수렴할 수 있도록 확인해보고자 함.</td><td colspan="1" valign="top"><p>갈색 부분</p><p>Epoch값을 증가해서 확인한 결과 eval_loss값이 이전 실험값들에 비해 더 적은 값으로 수렴하는 것을 확인</p></td><td colspan="1" valign="top">Mixup을 함께 적용해보았지만, Train과정에서 더욱 불안정해지는 모습과 이전 실험들과 eval_loss를 비교했을 때 큰 차이가 없었다. 결과적으로 성능 향상에 있어 유의미한 결과를 보기는 어려웠다.</td></tr>
</table>


