
# VBHMM x-vectors Diarization (aka VBx)

Diarization recipe for CALLHOME, AMI and DIHARD II by Brno University of Technology. \
The recipe consists of 
- computing x-vectors
- doing agglomerative hierarchical clustering on x-vectors as a first step to produce an initialization
- apply variational Bayes HMM over x-vectors to produce the diarization output
- score the diarization output

More details about the full recipe in\
F. Landini, J. Profant, M. Diez, L. Burget: [Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks](https://arxiv.org/abs/2012.14952)

### Updates from the JHU HLTCOE Team
 1. Integrated 2-pass Leave-One-Out Gaussian PLDA (2-pass LGP) diarization algorithm into the VBx code infrastructure
 2. Updated feature extraction pipeline to support using kaldi features (using torchaudio as the backend)
 3. Updated xvector extraction and alignment to support custom xVector models.
 4. Added top-level scripts to run CALLHOME, DIHARD2, and AMI datasets using the 2-pass LGP algorithm.  (note that these scripts require you to prepare the data directories a-priori)

 K. Karra, A. McCree:[Speaker Diarization using Two-pass Leave-One-Out Gaussian PLDA Clustering of DNN Embeddings
  Submitted to Interspeech 2021](https://arxiv.org/pdf/2104.02469.pdf)


## Usage
To run the recipe, execute the run scripts for the different datasets with the corresponding parameters. Please refer to the scripts for more details. The CALLHOME and DIHARD II recipes require the corresponding datasets and the paths need to be provided. For AMI, the recordings need to be downloaded (for free) but the VAD segments and reference rttms are obtained from [our proposed setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup).

This repository has x-vector extractors already trained to function as a standalone recipe. However, the recipes for training the extractors can be found [here](https://github.com/phonexiaresearch/VBx-training-recipe).

### Updates from the JHU HLTCOE Team
To reproduce the results shown below, we used our own xvector model.  Details, and pretrained models can be found [here](https://github.com/hltcoe/xvectors).


## Getting started
We recommend to create [anaconda](https://www.anaconda.com/) environment
```bash
conda create -n VBx python=3.6
conda activate VBx
```
Clone the repository
```bash
git clone https://github.com/BUTSpeechFIT/VBx.git
```
Install the package
```bash
pip install -e .
```
Initialize submodule `dscore`:
```bash
git submodule init
git submodule update
```
Run the example
```bash
./run_example.sh
```
The output (last few lines) should look like this
```
File               DER    JER    B3-Precision    B3-Recall    B3-F1    GKT(ref, sys)    GKT(sys, ref)    H(ref|sys)    H(sys|ref)    MI    NMI
---------------  -----  -----  --------------  -----------  -------  ---------------  ---------------  ------------  ------------  ----  -----
ES2005a           7.06  29.99            0.65         0.78     0.71             0.71             0.56          1.14          0.59  1.72   0.67
*** OVERALL ***   7.06  29.99            0.65         0.78     0.71             0.71             0.56          1.14          0.59  1.72   0.67
```


## Citations
In case of using the software please cite:\
F. Landini, J. Profant, M. Diez, L. Burget: [Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks](https://arxiv.org/abs/2012.14952)

K. Karra, A. McCree: [Speaker Diarization using Two-pass Leave-One-Out Gaussian PLDA Clustering of DNN Embeddings
  Submitted to Interspeech 2021](https://arxiv.org/pdf/2104.02469.pdf)

## Results
We present here the Diarization Error Rates (DER) for our systems for the different datasets and different evaluation protocols. A more thorough discussion on the protocols and results can be found in the paper.


###CALLHOME
|           	| VBx   	| 2-pass LGP 	|
|-----------	|-------	|------------	|
| **Forgiving** 	| 4.42  	| 3.92       	|
| **Fair**      	| 14.21 	| 13.82      	|
| **Full**      	| 21.77 	| 21.16      	|

###DIHARD II (Dev)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Fair 	| 12.23 	| 13.39           	| 11.73           	|
| Full 	| 18.19 	| 19.21           	| 17.8            	|

###DIHARD II (Eval)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Fair 	| 12.29 	| 15.03           	| 12.66           	|
| Full 	| 18.55 	| 20.83           	| 18.76            	|

###AMI Beamformed (Dev)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Forgiving 	| 2.80 	|  3.18      	|  4.33          	|
| Fair 	        | 10.81 	|  11.75    |  12.21          	|
| Full 	        | 17.66 	|  18.56    |  18.94           	|


###AMI Beamformed (Eval)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Forgiving 	| 3.90 	    |   3.91         |   3.88       |
| Fair 	        | 14.23 	|   13.52        |  13.38       |
| Full 	        | 20.84 	|   19.95        |  19.84       |


###AMI Mix-Headset (Dev)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Forgiving 	| 1.56 	    |  2.1       |  2.00          	|
| Fair 	        | 9.68 	    |  10.23     |     10.14       	|
| Full 	        | 16.33 	|  16.93     | 16.74            |


###AMI Mix-Headset (Eval)
|      	| VBx   	| 2-pass LGP (NB) 	| 2-pass LGP (WB) 	|
|------	|-------	|-----------------	|-----------------	|
| Forgiving 	| 2.10 	    | 1.76         	 |   2.13       |
| Fair 	        | 12.53 	| 11.73          |  11.57       |
| Full 	        | 18.99 	| 17.94          |  17.79       |


## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



## Contact
If you have any comment or question, please contact landini@fit.vutbr.cz or mireia@fit.vutbr.cz

For questions regarding the 2-pass LGP algorithm, please contact: kiran.karra@jhuapl.edu 
