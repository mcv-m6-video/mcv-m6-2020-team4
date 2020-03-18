# mcv-m6-2020-team4
## Team 4
## Contributors 👫👫
- [Sara Lumbreras Navarro](https://github.com/lunasara) - jfslumbreras@gmail.com
- [Maria Vila](https://github.com/mariavila) - mariava.1213@gmail.com
- [Yael Tudela](https://github.com/yaeltudela) - yaeltudelabarroso@gmail.com
- [Diego Alejandro Velázquez](https://github.com/dvd42) - diegovd0296@gmail.com


## Usage 💻
```
python3 week1.py
```

```
python3 week2.py[-color (str) -channels (index of channels)]
```

```
python3 week3.py
```


## Link to slides

| Week | Link |
| ------------- | ------------ |
|**Week 1**|[Slides week1](https://docs.google.com/presentation/d/16PFxQ5oOF8AiYmNZvJJbCN2tBN9ZxJ5noHVAtzE7whU/edit?usp=sharing)|
|**Week 2**|[Slides week2](https://docs.google.com/presentation/d/1-UHPhtYsmF_734AwNUiY4mrO-wnJpJK_MNgYcoq-Xbs/edit?usp=sharing)|
|**Week 3**|[Slides week3](https://docs.google.com/presentation/d/1Adx_ArI-yE8k_10-vO0ZZc20_3yzGV9KDUf5H1HKdTk/edit?usp=sharing)|




### Week1 
* [x] Task 1: Detection metrics.

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/frame_guai.png)

| Mask RCNN     | SSD512        | Yolo3 |
| ------------- |:-------------:| -----:|
| 0.447         | 0.382         | 0.435 |


* [x] Task 2: Detection metrics. Temporal analysis.

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week1/iou_noisy.gif)

* [x] Task 3: Optical flow evaluation metrics.

**MSEN**
| Seq 45        | Seq 157      | 
| ------------- | ------------ |
| 10.627        | 2.750        | 

**PEPN**
|Seq 45        | Seq 157 |
|:------------:| :------:|
|78.560        | 34.047  |

* [x] Task 4: Visual representation optical flow.

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week1/OF.PNG)


### Week2

* [x] Task 1.1: Gaussian. Implementation

Mean image
![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/mu.png)

Variance image
![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/simga.png)

* [x] Task 1.2: Gaussian. Discussion

Denoising method
![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/denoise.gif)

* [x] Task 2.1: Adaptive modeling 

Grid search and finetuning of parameters

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/2_1.PNG)

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/2_1b.PNG)

* [x] Task 2.2: Adaptive vs non-adaptive models

mAP

|Non-Adaptive        | Adaptive |
|:------------------:| :-------:|
|0.286               | 0.564    |


* [x] Task 3: Comparison with the state of the art

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week2/graph.png)

* [x] Task 4: Color sequences

|Color Space|Channels|mAP|
|:-:|:-:|:-:|
|RGB|R & B|0.51|
|HSV|H & V|0.20|
|HSV|H|0.12|
|YCrCb|Cr & Cb|0.02|
|CIELAB|A & B|0.03|


### Week3
* [x] Task 1.1: Object Detection: Off-the-Shelf
AP 0.5 car
|Model|AP|
|:-:|:-:|
|RetinaNet (R50-FPN)|0.436|
|Faster R-CNN (R50-FPN)|0.595|
|Mask R-CNN|0.447|
|SSD 512|0.382|
|Yolo v3|0.435|

* [x] Task 1.2: Object Detection: Fine-Tuned
 For Faster R-CNN (R50-FPN) Fine-tuned we've got a AP 0.5 for car of **0.902**

* [x] Task 2.1: Tracking by Maximum Overlap
* [x] Task 2.2: Tracking with a Kalman Filter
* [x] Task 2.3: IDF1 for Multiple Object Tracking

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week3/apcar.png)

![](https://github.com/mcv-m6-video/mcv-m6-2020-team4/blob/master/Results/Week3/idf1.png)





