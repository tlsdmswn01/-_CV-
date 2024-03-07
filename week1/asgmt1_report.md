**Week1**

**2024\_spring\_vision\_tutorial**

**Goal : Cifar 100데이터셋으로 성능 80%이상 올리기**

**[주요 실험 및 결과]**

||**Experiment&Results1**|**Experiment&Results2**|**Experiment&Results3**|
| :-: | :-: | :-: | :-: |
|**Batch\_size**|128|512|512|
|**Transform**|<p>RandomCrop(32,padding=4)</p><p>RandomHorizontalFlip(p=0.5)</p>|<p>RandomCrop(32,padding=4)</p><p>RandomHorizontalFlip(p=0.5)</p><p>AutoAugment</p>|<p>RandomCrop(32,padding=4)</p><p>RandomHorizontalFlip(p=0.8)</p><p>AutoAugment </p>|
|**Data Normalization**|0\.5|0\.5|<p>[0.50707513,0.48654887,0.44091785]</p><p>[0.50707513,0.48654887,0.44091785]</p>|
|**Cutmix**|X|X|Cutmix ratio=1|
|**Model**|Resnet50|Resnet50|Resnet50|
|**Optimizer**|SGD|Adam|NAdam|
|**Scheduler**|CosineAnnealingLR|CosineAnnealingLR|CosineAnnealingLR|
|**Best Accuracy**|0\.68|0\.7389|0\.78|
|**Time**|2h 15m(epoch:)|2h 35m|2h|
|<p>**Loss**</p><p>**History plot**</p>|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 001](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/4d604caf-ade8-4d2b-8d95-244b7fbfd034)|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 002](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/03ebe1e5-c27a-4eee-9689-19b5a63ab417)|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 003](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/7dba2566-7d44-4a14-ac16-792df4525855)|
|<p>**Accuracy**</p><p>**History plot**</p>|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 004](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/27697712-5d68-4fa2-96f1-ac909f2d2791)|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 005](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/71010eb3-de59-4948-8c7a-2b9cd53d85d1)|![Aspose Words eddfb506-b2f9-4887-898b-4bc7e4494e8c 007](https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/e1426af4-7f12-40c7-9cd8-b1a14748fa98)|
|**Analyze**|Overfiting이 발생해 특정 지점부터 학습이 잘 되지 않음 확인|<p>Overfiting 방지를 위해 데이터 복잡도를 올리고자 AutoAugment를 추가하였고, optimizer도 Adam으로 변경</p><p>결과적으로 Overfiting이 완화되었고, 성성능 오른 것을 확인</p><p>하지만 여전히 70% 초반대에서 Overfiting이 발생해 학습이 더 되지 않음</p>|<p>데이터 정규화를 0.5의 값이 아닌 각 RGB별로 평균과 표준편차를 구해서 진행해주었다.</p><p>데이터 복잡도를 더 주고자 Cutmix 비율을 1로 하여 적용하고, optimizer을 NAdam으로 바꿈으로써 최종적으로 77%성능에 도달하였다.</p>|

**[위 표 이외에 다른 실험결과]**

- 데이터 전처리: 데이터 복잡도를 높이고자 Colojittering과 RandomVertificalfilp을 사용은, 성능개선에 있어 유의미하지 않았다. 반면 AutoAugment를 통해 해당 데이터에 가장 적합한 전처리 기법 plolicy를 찾고 적용하는 것이 성능개선에 있어 유의미한 결과를 보였다.
- Batch 사이즈를 높이고, Optimizer을 Adam, NAdam으로 수정하였을 때 성능이 좋아지는 것을 확인할 수 있었다.
- Scheduler : StepLR(임의로 설정한 step에서 lr을 수정하는), ReduceLR(임의로 설정한 metirc값을 고려해 몇 patience를 넘어가게 되면 lr을 감소시키는), CosineAnnealingLR(처음 몇 step에서는 lr값을 크게 높였다가, 점점 줄이는 방향으로)을 각각 사용해보았을 때, CosineAnnealingLR이 가장 적합한 lr을 찾는데 용이하였다.



<div align="center">
  <table>
    <tr>
      <th>Best</th>
      <td>Experiment&Results</td>
    </tr>
    <tr>
      <td align="center">Batch_size</td>
      <td>512</td>
    </tr>
    <tr>
      <td align="center">Transform</td>
      <td>RandomCrop(32,padding=4)<br>RandomHorizontalFlip(p=0.5)<br>AutoAugment</td>
    </tr>
    <tr>
      <td align="center">Data Normalization</td>
      <td>[0.50707513,0.48654887,0.44091785]<br>[0.50707513,0.48654887,0.44091785]</td>
    </tr>
    <tr>
      <td align="center">Cutmix</td>
      <td>Cutmix ratio=1</td>
    </tr>
    <tr>
      <td align="center">Model</td>
      <td>SE-Resnet50, ReLU()->Mish()</td>
    </tr>
    <tr>
      <td align="center">Optimizer</td>
      <td>SGD, lr=0.25, momentum=0.9</td>
    </tr>
    <tr>
      <td align="center">Scheduler</td>
      <td>CosineAnnealingLR<br>Warmup: 1e-4 -> 0.2</td>
    </tr>
    <tr>
      <td align="center">Best Accuracy</td>
      <td>0.8145</td>
    </tr>
    <tr>
      <td align="center">Time</td>
      <td>4h 20m (epoch:200)</td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="https://github.com/tlsdmswn01/2024_spring_vision_tutorial/assets/135305102/b24013e7-f21b-488a-a798-d58ac1d5827f" width="100%">
</p>
