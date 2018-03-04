# RefineNet 

This repository is an attempt at replicating some results presented in Guosheng Lin et al.'s paper : ["RefineNet: Multi-Path Refinement Network for High-Resolution Semantic Segmentation"](https://arxiv.org/pdf/1611.06612.pdf).

## Requirements :

### ["MIT Sccene Parsing Benchmark"](http://sceneparsing.csail.mit.edu/) :

The dataset can be downloaded here : 

* ["Train/Val"](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
* ["Test"](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip)

## Experiments :

Here are some early results :

* Left : input picture
* Middle : ground truth semantic segmentation
* Right : predicted semantic segmentation 

Epoch | Samples  
------|------------
15 | ![Dreconst1-1](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/reconst_images/1.png) 
 ![Dgen1-1](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/gen_images/1.png)
25 | ![Dreconst1-10](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/reconst_images/10.png) 
 ![Dgen1-10](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/gen_images/10.png)
45 | ![Dreconst1-30](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/reconst_images/30.png) 
 ![Dgen1-30](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/gen_images/30.png)
65 | ![Dreconst1-50](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/reconst_images/70.png) 
 ![Dgen1-50](/doc/dSprite/dSprite--beta4.0-layers3-z10-conv32/gen_images/70.png)

## References :

Some of the code present in this repository for handling the dataset, for instance, are (heavily) inspired by :

* ["CSAILVision's SceneParsing repository"](https://github.com/CSAILVision/sceneparsing)
* ["hangzhaomit's semantic-segmentation-pytorch repository"](https://github.com/hangzhaomit/semantic-segmentation-pytorch)

