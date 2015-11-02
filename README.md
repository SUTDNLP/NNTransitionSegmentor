NNTransitionSegmentor
======
NNTransitionSegmentor is a package for Word Segmentation using neural networks based on package [LibN3L](https://github.com/SUTDNLP/LibN3L). 
The current version is a re-implementation of segmentor in ZPar

Performance
======
Take averaged perceptron as an example (CTB6.0, please refer to [LibN3L: A lightweight Package for Neural NLP](https://github.com/SUTDNLP/LibN3L/blob/master/description\(expect%20for%20lrec2016\).pdf) for details):  
Both ZPar and this package obtain performance about 95.10%

Compile
======
* Download [LibN3L](https://github.com/SUTDNLP/LibN3L) library and compile it. 
* Open [CMakeLists.txt](CMakeLists.txt) and change "../LibN3L/" into the directory of your [LibN3L](https://github.com/SUTDNLP/LibN3L) package.  

`cmake .`  
`make`  

Input data format 
======
one line one sentene, with words seperated by spaces  

