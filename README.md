# Universal Prototype Transport
This repository contains example code for the IJCV 2023 paper: "Universal Prototype Transport for Zero-Shot Action Recognition and Localization". This code will be able to perform zero-shot action recognition on UCF-101 using both action- and object-based models.
<br>
The paper is available here: https://link.springer.com/article/10.1007/s11263-023-01846-2

## Downloading pre-computed features and meta-data for UCF-101
Ready-to-use data for UCF-101 has been compiled in a public folder, available here: https://isis-data.science.uva.nl/mettes/universal-prototype-transport/
<br>
If you want to use the code, copy the 'data/' folder in this repo.

## Computing action target prototypes
The 'data/' folder already contains the target prototypes for all actions in UCF-101 for various settings used in the paper, see the 'ot/' subfolder.
<br>
To compute your own target prototypes, run the following code:
```
python optimal_destinations.py -n 1000
```
Where -n denotes the number of clusters, 1000 in the example above.

## Performing zero-shot recognition
For zero-shot action recognition on UCF-101, you can run the following command to get the baseline scores for the action-based model as follows:
```
python zeroshot_classification_ucf101.py -l 1
```
Which should get you a top 1 accuracy of 39.2%. To get the zero-shot recognition with the proposed Universal Prototype Transport, run the following command:
```
python zeroshot_classification_ucf101.py -l 0.5
```
This should get you an accuracy of 42.4%, as reported in the paper.
<br><br>
To additionally get the final accuracy by combining action- and object-based models with Universal Prototype Transport, run:
```
python zeroshot_classification_ucf101.py -l 0.5 -f 0.5 -t 10
```
If you also want to re-produce HMDB51, let me know and I'll prepare the meta-data.

## Citing the paper
Please cite the paper accordingly:
```
@article{mettes2023universal,
  title={Universal prototype transport for zero-shot action recognition and localization},
  author={Mettes, Pascal},
  journal={International Journal of Computer Vision},
  pages={1--14},
  year={2023},
  publisher={Springer}
}
```
Or in text: Mettes, P. (2023). Universal prototype transport for zero-shot action recognition and localization. International Journal of Computer Vision, 1-14.
