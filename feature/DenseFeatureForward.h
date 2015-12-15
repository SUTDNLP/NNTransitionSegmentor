/*
 * DenseFeatureForward.h
 *
 *  Created on: Dec 11, 2015
 *      Author: mason
 */

#ifndef FEATURE_DENSEFEATUREFORWARD_H_
#define FEATURE_DENSEFEATUREFORWARD_H_

#include "N3L.h"
template<typename xpu>
class DenseFeatureForward {
public:
	//state inter dependent features
	Tensor<xpu, 3, dtype> _wordPrime, _wordPrimeMask;
	Tensor<xpu, 3, dtype> _actionPrime, _actionPrimeMask;
	Tensor<xpu, 2, dtype> _wordRep;
	Tensor<xpu, 2, dtype> _actionRep;
	Tensor<xpu, 2, dtype> _wordHidden;
	Tensor<xpu, 2, dtype> _actionHidden;
	Tensor<xpu, 2, dtype> _wordRNNHidden;  //lstm
	Tensor<xpu, 2, dtype> _actionRNNHidden;  //lstm
	//state inter independent features
	Tensor<xpu, 2, dtype> _sepInHidden, _sepInHiddenLoss;  //sep in
	Tensor<xpu, 2, dtype> _sepOutHidden, _sepOutHiddenLoss;  //sep out
	Tensor<xpu, 2, dtype> _appInHidden, _appInHiddenLoss;  //app in
	Tensor<xpu, 2, dtype> _appOutHidden, _appOutHiddenLoss;  //app out

	bool _bAllocated;
	bool _bTrain;

public:
	DenseFeatureForward() {
		_bAllocated = false;
		_bTrain = false;
	}

	~DenseFeatureForward() {
		clear();
	}

public:
	inline void init(int wordDim, int wordNgram, int wordHiddenDim, int wordRNNDim,
			int actionDim,int actionNgram,  int actionHiddenDim, int actionRNNDim,
			int sepInHiddenDim, int appInHiddenDim, int sepOutHiddenDim, int appOutHiddenDim,
			bool bTrain = false) {
		clear();
		_wordPrime = NewTensor<xpu>(Shape3(wordNgram, 1,  wordDim), d_zero);
		_wordRep = NewTensor<xpu>(Shape2(1, wordNgram * wordDim), d_zero);
		_wordHidden = NewTensor<xpu>(Shape2(1, wordHiddenDim), d_zero);
		_wordRNNHidden = NewTensor<xpu>(Shape2(1, wordRNNDim), d_zero);


		_actionPrime = NewTensor<xpu>(Shape3(actionNgram, 1,  actionDim), d_zero);
		_actionRep = NewTensor<xpu>(Shape2(1, actionNgram * actionDim), d_zero);
		_actionHidden = NewTensor<xpu>(Shape2(1, actionHiddenDim), d_zero);
		_actionRNNHidden = NewTensor<xpu>(Shape2(1, actionRNNDim), d_zero);


		_sepInHidden = NewTensor<xpu>(Shape2(1,  sepInHiddenDim), d_zero);
		_appInHidden = NewTensor<xpu>(Shape2(1, appInHiddenDim), d_zero);
		_sepOutHidden = NewTensor<xpu>(Shape2(1, sepOutHiddenDim), d_zero);
		_appOutHidden = NewTensor<xpu>(Shape2(1, appOutHiddenDim), d_zero);

		if(bTrain){
			_bTrain = bTrain;

			_wordPrimeMask = NewTensor<xpu>(Shape3(wordNgram, 1,  wordDim), d_zero);
			_actionPrimeMask = NewTensor<xpu>(Shape3(actionNgram, 1,  actionDim), d_zero);


			_sepInHiddenLoss = NewTensor<xpu>(Shape2(1,  sepInHiddenDim), d_zero);
			_appInHiddenLoss = NewTensor<xpu>(Shape2(1, appInHiddenDim), d_zero);
			_sepOutHiddenLoss = NewTensor<xpu>(Shape2(1, sepOutHiddenDim), d_zero);
			_appOutHiddenLoss = NewTensor<xpu>(Shape2(1, appOutHiddenDim), d_zero);

		}


		_bAllocated = true;
	}

	inline void clear() {
		if (_bAllocated) {
			FreeSpace(&_wordPrime);
			FreeSpace(&_wordRep);
			FreeSpace(&_wordHidden);
			FreeSpace(&_wordRNNHidden);

			FreeSpace(&_actionPrime);
			FreeSpace(&_actionRep);
			FreeSpace(&_actionHidden);
			FreeSpace(&_actionRNNHidden);

			FreeSpace(&_sepInHidden);
			FreeSpace(&_appInHidden);
			FreeSpace(&_sepOutHidden);
			FreeSpace(&_appOutHidden);

			if(_bTrain){
				FreeSpace(&_wordPrimeMask);
				FreeSpace(&_actionPrimeMask);

				FreeSpace(&_sepInHiddenLoss);
				FreeSpace(&_appInHiddenLoss);
				FreeSpace(&_sepOutHiddenLoss);
				FreeSpace(&_appOutHiddenLoss);
			}

			_bAllocated = false;
			_bTrain = false;
		}
	}

	inline void copy(const DenseFeatureForward<xpu>& other) {
		if (other._bAllocated) {
			tcopy(other._wordPrime, _wordPrime, _bAllocated);
			tcopy(other._wordRep, _wordRep, _bAllocated);
			tcopy(other._wordHidden, _wordHidden, _bAllocated);
			tcopy(other._wordRNNHidden, _wordRNNHidden, _bAllocated);

			tcopy(other._actionPrime, _actionPrime, _bAllocated);
			tcopy(other._actionRep, _actionRep, _bAllocated);
			tcopy(other._actionHidden, _actionHidden, _bAllocated);
			tcopy(other._actionRNNHidden, _actionRNNHidden, _bAllocated);

			tcopy(other._sepInHidden, _sepInHidden, _bAllocated);
			tcopy(other._appInHidden, _appInHidden, _bAllocated);
			tcopy(other._sepOutHidden, _sepOutHidden, _bAllocated);
			tcopy(other._appOutHidden, _appOutHidden, _bAllocated);

			if(_bTrain){

				tcopy(other._wordPrimeMask, _wordPrimeMask, _bAllocated);
				tcopy(other._actionPrimeMask, _actionPrimeMask, _bAllocated);

				tcopy(other._sepInHiddenLoss, _sepInHiddenLoss, _bAllocated);
				tcopy(other._appInHiddenLoss, _appInHiddenLoss, _bAllocated);
				tcopy(other._sepOutHiddenLoss, _sepOutHiddenLoss, _bAllocated);
				tcopy(other._appOutHiddenLoss, _appOutHiddenLoss, _bAllocated);
			}
		}

		_bAllocated = other._bAllocated;
	}

	inline DenseFeatureForward<xpu>& operator=(const DenseFeatureForward<xpu> &rhs) {
		// Check for self-assignment!
		if (this == &rhs)      // Same object?
			return *this;        // Yes, so skip assignment, and just return *this.
		copy(rhs);
		return *this;
	}

};

#endif /* FEATURE_DENSEFEATUREFORWARD_H_ */
