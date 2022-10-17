
# [Concurrent Subsidiary Supervision for Source-Free DA (ECCV22)](https://sites.google.com/view/sticker-sfda)

Code for our **ECCV** 2022 paper 'Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation'. 

[[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900177.pdf) [[Project Page]](https://sites.google.com/view/sticker-sfda)

## Dataset preparation

Download the [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) (use our provided image list files) dataset. Put the dataset in data folder

## Office-Home experiments
Code for Single Source Domain Adaptation (SSDA) is in the 'SSDA_OH' folder. 

```
sh SSDA_OH/run.sh
```

Code for Multi Source Domain Adaptation (SSDA) is in the 'MSDA_OH' folder. 

```
sh MSDA_OH/run.sh
```

## Pre-trained checkpoints (coming soon)

## Citation
If you find our work useful in your research, please cite the following paper:
```
@InProceedings{kundu2022concurrent,
  title={Concurrent Subsidiary Supervision for Unsupervised Source-Free Domain Adaptation},
  author={Kundu, Jogendra Nath and Bhambri, Suvaansh and Kulkarni, Akshay and Sarkar, Hiran and Jampani, Varun and Babu, R. Venkatesh},
  booktitle={European Conference on Computer Vision},
  year={2022},
}
```
