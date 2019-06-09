# [Progressive Learning for Person Re-Identification with One Example](https://ieeexplore.ieee.org/document/8607049)

Pytorch implementation for our paper [[Link]](https://yu-wu.net/pdf/TIP2019_One-Example-reID.pdf).
This code is based on the [Open-ReID](https://github.com/Cysu/open-reid) library and [Exploit-Unknown-Gradually](https://github.com/Yu-Wu/Exploit-Unknown-Gradually).

## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.4.1)
- scikit-learn, metric-learn, tqdm

### Download datasets 
- DukeMTMC-VideoReID: This [page](https://github.com/Yu-Wu/DukeMTMC-VideoReID) contains more details and baseline code.
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Market-1501: [[Direct Link]](http://108.61.70.170/share/market1501.tar) [[Google Drive]](https://drive.google.com/file/d/1kbDAPetylhb350LX3EINoEtFsXeXB0uW/view)
- DukeMTMC-reID: [[Direct Link]](http://108.61.70.170/share/duke.tar) [[Google Drive]](https://drive.google.com/file/d/17mHIip2x5DXWqDUT97aiqKsrTQvSI830/view)
- Move the downloaded zip files to `./data/` and unzip here.


## Train

```shell
sh run.sh
```

Please set the `max_frames` smaller if your GPU memory is less than 11G.

## Performances

The performances varies according to random splits for initial labeled data. To reproduce the performances in our paper, please use the one-shot splits at `./examples/`


## Citation

Please cite the following papers in your publications if it helps your research:

    @article{wu2019progressive,
      title  = {Progressive Learning for Person Re-Identification with One Example},
      author = {Wu, Yu and Lin, Yutian and Dong, Xuanyi and Yan, Yan and Bian, Wei and Yang, Yi},
      journal= {IEEE Transactions on Image Processing},
      year   = {2019}, 
      volume = {28}, 
      number = {6}, 
      pages  = {2872-2881}, 
      doi    = {10.1109/TIP.2019.2891895}, 
      ISSN   = {1057-7149}, 
      month  = {June},
    }
    
    @inproceedings{wu2018cvpr_oneshot,
        title = {Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning},
        author = {Wu, Yu and Lin, Yutian and Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2018}
    }

    
## Contact

To report issues for this code, please open an issue on the [issues tracker](https://github.com/Yu-Wu/One-Example-Person-ReID/issues).

If you have further questions about this paper, please do not hesitate to contact me. 

[Yu Wu's Homepage](https://yu-wu.net)

