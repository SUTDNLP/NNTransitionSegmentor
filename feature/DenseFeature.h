/*
 * DenseFeature.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATURE_H_
#define FEATURE_DENSEFEATURE_H_

#include "N3L.h"

template<typename xpu>
class DenseFeature {
public:
	//all state inter dependent features
	vector<vector<int> > _words, _actions;
	vector<Tensor<xpu, 3, dtype> > _wordPrime, _wordPrimeLoss, _wordPrimeMask;
	vector<Tensor<xpu, 3, dtype> > _actionPrime, _actionPrimeLoss, _actionPrimeMask;
	vector<Tensor<xpu, 2, dtype> > _wordRep, _wordRepLoss;
	vector<Tensor<xpu, 2, dtype> > _actionRep, _actionRepLoss;
	vector<Tensor<xpu, 2, dtype> > _wordHidden, _wordHiddenLoss;
	vector<Tensor<xpu, 2, dtype> > _actionHidden, _actionHiddenLoss;
	vector<Tensor<xpu, 2, dtype> > _wordRNNHidden, _wordRNNHiddenLoss;  //lstm
	vector<Tensor<xpu, 2, dtype> > _actionRNNHidden, _actionRNNHiddenLoss;  //lstm

	int _steps;
	int _wordnum;

public:
	DenseFeature() {
		_steps = 0;
	}

	~DenseFeature() {
		clear();
	}

public:
	inline void init(int wordnum, int steps, int wordDim, int wordNgram, int wordHiddenDim, int wordRNNDim,
			int actionDim, int actionNgram, int actionHiddenDim, int actionRNNDim) {
		_steps = steps;
		_wordnum = wordnum;

		if(wordnum > 0){
			_words.resize(wordnum);
			for (int idx = 0; idx < wordnum; idx++) {
				_words[idx].resize(wordNgram);
				_wordPrime[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_wordRep[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_wordHidden[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				_wordRNNHidden[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);


				_wordPrimeLoss[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_wordPrimeMask[idx] = NewTensor<xpu>(Shape3(wordNgram, 1, wordDim), d_zero);
				_wordRepLoss[idx] = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
				_wordHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
				_wordRNNHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);
			}
		}

		if(steps > 0) {
			_actions.resize(steps);
			for (int idx = 0; idx < steps; idx++) {
				_actions[idx].resize(actionNgram);
				_actionPrime[idx] = NewTensor<xpu>(Shape3(actionNgram, 1, actionDim), d_zero);
				_actionRep[idx] = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
				_actionHidden[idx] = NewTensor<xpu>(Shape2(1, actionHiddenDim), d_zero);
				_actionRNNHidden[idx] = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);

				_actionPrimeLoss[idx] = NewTensor<xpu>(Shape3(actionNgram, 1, actionDim), d_zero);
				_actionPrimeMask[idx] = NewTensor<xpu>(Shape3(actionNgram, 1, actionDim), d_zero);
				_actionRepLoss[idx] = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
				_actionHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, actionHiddenDim), d_zero);
				_actionRNNHiddenLoss[idx] = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);
			}
		}

	}

	inline void clear() {
		for (int idx = 0; idx < _wordnum; idx++) {
			FreeSpace(&(_wordPrime[idx]));
			FreeSpace(&(_wordRep[idx]));
			FreeSpace(&(_wordHidden[idx]));
			FreeSpace(&(_wordRNNHidden[idx]));

			FreeSpace(&(_wordPrimeLoss[idx]));
			FreeSpace(&(_wordPrimeMask[idx]));
			FreeSpace(&(_wordRepLoss[idx]));
			FreeSpace(&(_wordHiddenLoss[idx]));
			FreeSpace(&(_wordRNNHiddenLoss[idx]));
		}

		for (int idx = 0; idx < _steps; idx++) {
			FreeSpace(&(_actionPrime[idx]));
			FreeSpace(&(_actionRep[idx]));
			FreeSpace(&(_actionHidden[idx]));
			FreeSpace(&(_actionRNNHidden[idx]));

			FreeSpace(&(_actionPrimeLoss[idx]));
			FreeSpace(&(_actionPrimeMask[idx]));
			FreeSpace(&(_actionRepLoss[idx]));
			FreeSpace(&(_actionHiddenLoss[idx]));
			FreeSpace(&(_actionRNNHiddenLoss[idx]));
		}
	}


};

#endif /* FEATURE_DENSEFEATURE_H_ */
