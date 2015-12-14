/*
 * StackLSTMBeamSearcher.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_StackLSTMBeamSearcher_H_
#define SRC_StackLSTMBeamSearcher_H_

#include <hash_set>
#include <iostream>

#include <assert.h>
#include "Feature.h"
#include "DenseFeatureExtraction.h"
#include "DenseFeature.h"
#include "N3L.h"
#include "NeuralState.h"
#include "Action.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
template<typename xpu>
class StackLSTMBeamSearcher {
public:
	StackLSTMBeamSearcher() {
		_dropOut = 0.5;
		_delta = 0.2;
	}
	~StackLSTMBeamSearcher() {
	}

public:
	SparseUniLayer1O<xpu> _splayer_output;
	UniLayer1O<xpu> _nnlayer_sep_output;
	UniLayer1O<xpu> _nnlayer_app_output;
	LookupTable<xpu> _words;
	LookupTable<xpu> _chars;
	LookupTable<xpu> _bichars;
	LookupTable<xpu> _actions;

	RNN<xpu> _char_left_rnn;
	RNN<xpu> _char_right_rnn;
	RNN<xpu> _word_increased_rnn;
	RNN<xpu> _action_increased_rnn;
	UniLayer<xpu> _nnlayer_sep_hidden;
	UniLayer<xpu> _nnlayer_app_hidden;
	UniLayer<xpu> _nnlayer_word_hidden;
	UniLayer<xpu> _nnlayer_char_hidden;
	UniLayer<xpu> _nnlayer_action_hidden;

	int _wordSize;
	int _wordDim;
	int _wordNgram;
	int _wordRepresentDim;
	int _charSize, _biCharSize;
	int _charDim, _biCharDim;
	int _charcontext, _charwindow;
	int _charRepresentDim;
	int _actionSize;
	int _actionDim;
	int _actionNgram;
	int _actionRepresentDim;

	int _wordRNNHiddenSize;
	int _charRNNHiddenSize;
	int _actionRNNHiddenSize;
	int _wordHiddenSize;
	int _charHiddenSize;
	int _actionHiddenSize;
	int _sep_hiddenOutSize;
	int _sep_hiddenInSize;
	int _app_hiddenOutSize;
	int _app_hiddenInSize;

	FeatureExtraction<xpu> fe;

	int _linearfeatSize;

	Metric _eval;

	dtype _dropOut;

	dtype _delta;

	enum {
		BEAM_SIZE = 16, MAX_SENTENCE_SIZE = 512
	};

public:

	inline void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
		fe.addToFeatureAlphabet(feat_stat, featCutOff);
	}

	inline void addToWordAlphabet(hash_map<string, int> word_stat, int wordCutOff = 0) {
		fe.addToWordAlphabet(word_stat, wordCutOff);
	}

	inline void addToCharAlphabet(hash_map<string, int> char_stat, int charCutOff = 0) {
		fe.addToCharAlphabet(char_stat, charCutOff);
	}

	inline void addToBiCharAlphabet(hash_map<string, int> bichar_stat, int bicharCutOff = 0) {
		fe.addToBiCharAlphabet(bichar_stat, bicharCutOff);
	}

	inline void addToActionAlphabet(hash_map<string, int> action_stat) {
		fe.addToActionAlphabet(action_stat);
	}

	inline void setAlphaIncreasing(bool bAlphaIncreasing) {
		fe.setAlphaIncreasing(bAlphaIncreasing);
	}

	inline void initAlphabet() {
		fe.initAlphabet();
	}

	inline void loadAlphabet() {
		fe.loadAlphabet();
	}

	inline void extractFeature(const CStateItem<xpu>* curState, const CAction& nextAC, Feature& feat) {
		fe.extractFeature(curState, nextAC, feat);
	}

public:

	inline void init(const NRMat<dtype>& wordEmb, int wordNgram, int wordHiddenSize, int wordRNNHiddenSize, const NRMat<dtype>& actionEmb, int actionNgram,
			int actionHiddenSize, int actionRNNHiddenSize, const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize,
			int charRNNHiddenSize, int sep_hidden_out_size, int app_hidden_out_size) {
		_linearfeatSize = 3 * fe._featAlphabet.size();
		_splayer_output.initial(_linearfeatSize, 10);

		_wordSize = fe._wordAlphabet.size();
		if (_wordSize != wordEmb.nrows())
			std::cout << "word number does not match for initialization of word emb table" << std::endl;
		_wordDim = wordEmb.ncols();
		;
		_wordNgram = wordNgram;
		_wordRepresentDim = _wordNgram * _wordDim;
		_charSize = fe._charAlphabet.size();
		if (_charSize != charEmb.nrows())
			std::cout << "char number does not match for initialization of char emb table" << std::endl;
		_biCharSize = fe._bicharAlphabet.size();
		if (_biCharSize != bicharEmb.nrows())
			std::cout << "bichar number does not match for initialization of bichar emb table" << std::endl;
		_charDim = charEmb.ncols();
		_biCharDim = bicharEmb.ncols();
		_charcontext = charcontext;
		_charwindow = 2 * charcontext + 1;
		_charRepresentDim = (_charDim + _biCharDim) * _charwindow;
		_actionSize = fe._actionAlphabet.size();
		if (_actionSize != actionEmb.nrows())
			std::cout << "action number does not match for initialization of action emb table" << std::endl;
		_actionDim = actionEmb.ncols();
		_actionNgram = actionNgram;
		_actionRepresentDim = _actionNgram * _actionDim;

		_wordRNNHiddenSize = wordRNNHiddenSize;
		_charRNNHiddenSize = charRNNHiddenSize;
		_actionRNNHiddenSize = actionRNNHiddenSize;
		_wordHiddenSize = wordHiddenSize;
		_charHiddenSize = charHiddenSize;
		_actionHiddenSize = actionHiddenSize;
		_sep_hiddenOutSize = sep_hidden_out_size;
		_app_hiddenOutSize = app_hidden_out_size;
		_sep_hiddenInSize = _wordRNNHiddenSize + actionRNNHiddenSize + 2 * _charRNNHiddenSize;
		_app_hiddenInSize = actionRNNHiddenSize + 2 * _charRNNHiddenSize;

		_nnlayer_sep_output.initial(_sep_hiddenOutSize, 10);
		_nnlayer_app_output.initial(_app_hiddenOutSize, 20);

		_words.initial(wordEmb);
		_chars.initial(charEmb);
		_bichars.initial(bicharEmb);
		_actions.initial(actionEmb);

		_char_left_rnn.initial(_charRNNHiddenSize, _charHiddenSize, true, 30);
		_char_right_rnn.initial(_charRNNHiddenSize, _charHiddenSize, false, 40);
		_word_increased_rnn.initial(_wordRNNHiddenSize, _wordHiddenSize, false, 50);
		_action_increased_rnn.initial(_actionRNNHiddenSize, _actionHiddenSize, false, 60);
		_nnlayer_sep_hidden.initial(_sep_hiddenOutSize, _sep_hiddenInSize, true, 70, 0);
		_nnlayer_app_hidden.initial(_app_hiddenOutSize, _app_hiddenInSize, true, 80, 0);
		_nnlayer_word_hidden.initial(_wordHiddenSize, _wordRepresentDim, true, 90, 0);
		_nnlayer_char_hidden.initial(_charHiddenSize, _charRepresentDim, true, 100, 0);
		_nnlayer_action_hidden.initial(_actionHiddenSize, _actionRepresentDim, true, 110, 0);

	}

	inline void release() {
		_splayer_output.release();

		_nnlayer_sep_output.release();
		_nnlayer_app_output.release();

		_words.release();
		_chars.release();
		_bichars.release();
		_actions.release();

		_char_left_rnn.release();
		_char_right_rnn.release();
		_word_increased_rnn.release();
		_action_increased_rnn.release();
		_nnlayer_sep_hidden.release();
		_nnlayer_app_hidden.release();
		_nnlayer_word_hidden.release();
		_nnlayer_char_hidden.release();
		_nnlayer_action_hidden.release();

	}

	dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
		fe.setFeatureFormat(false);
		//setAlphaIncreasing(true);
		fe.setFeatAlphaIncreasing(true);
		_eval.reset();
		dtype cost = 0.0;
		for (int idx = 0; idx < sentences.size(); idx++) {
			cost += trainOneExample(sentences[idx], goldACs[idx]);
		}

		return cost;
	}

	// scores do not accumulate together...., big bug, refine it tomorrow or at thursday.
	dtype trainOneExample(const std::vector<std::string>& sentence, const vector<CAction>& goldAC) {
		if (sentence.size() >= MAX_SENTENCE_SIZE)
			return 0.0;
		static CStateItem<xpu> lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem<xpu> *lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.size();
		dtype cost = 0.0;
		dtype score = 0.0;

		const static CStateItem<xpu> *pGenerator;
		const static CStateItem<xpu> *pBestGen;
		static CStateItem<xpu> *correctState;

		bool bCorrect;  // used in learning for early update
		int index, tmp_i, tmp_j;
		CAction correct_action;
		bool correct_action_scored;
		std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction<xpu>, CScoredStateAction_Compare<xpu> > beam(BEAM_SIZE);
		static CScoredStateAction<xpu> scored_action; // used rank actions
		static CScoredStateAction<xpu> scored_correct_action;
		static DenseFeature<xpu> pBestGenFeat, pGoldFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		vector<int> charIds(length), biCharIds(length);
		Tensor<xpu, 3, dtype> charprime = NewTensor<xpu>(Shape3(length, 1, _charDim), d_zero);
		Tensor<xpu, 3, dtype> bicharprime = NewTensor<xpu>(Shape3(length, 1, _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charpre = NewTensor<xpu>(Shape3(length, 1, _charDim + _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charpreMask = NewTensor<xpu>(Shape3(length, 1, _charDim + _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charInput = NewTensor<xpu>(Shape3(length, 1, _charRepresentDim), d_zero);
		Tensor<xpu, 3, dtype> charHidden = NewTensor<xpu>(Shape3(length, 1, _charHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charLeftRNNHidden = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charRightRNNHidden = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 2, dtype> charRNNHiddenDummy = NewTensor<xpu>(Shape2(1, _charRNNHiddenSize), d_zero);

		Tensor<xpu, 3, dtype> charprime_Loss = NewTensor<xpu>(Shape3(length, 1, _charDim), d_zero);
		Tensor<xpu, 3, dtype> bicharprime_Loss = NewTensor<xpu>(Shape3(length, 1, _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charpre_Loss = NewTensor<xpu>(Shape3(length, 1, _charDim + _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charInput_Loss = NewTensor<xpu>(Shape3(length, 1, _charRepresentDim), d_zero);
		Tensor<xpu, 3, dtype> charHidden_Loss = NewTensor<xpu>(Shape3(length, 1, _charHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charLeftRNNHidden_Loss = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charRightRNNHidden_Loss = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 2, dtype> charRNNHiddenDummy_Loss = NewTensor<xpu>(Shape2(1, _charRNNHiddenSize), d_zero);

		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charIds[idx] < 0)
				charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			biCharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (biCharIds[idx] < 0)
				biCharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charIds[idx], charprime[idx]);
			_bichars.GetEmb(biCharIds[idx], bicharprime[idx]);
			concat(charprime[idx], bicharprime[idx], charpre[idx]);
			dropoutcol(charpreMask[idx], _dropOut);
			charpre[idx] = charpre[idx] * charpreMask[idx];
		}

		windowlized(charpre, charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charInput, charHidden);
		_char_left_rnn.ComputeForwardScore(charHidden, charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charHidden, charRightRNNHidden);

		correctState = lattice_index[0];

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;
			correct_action = goldAC[index - 1];
			bCorrect = false;
			correct_action_scored = false;

			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				//std::cout << "new" << std::endl;
				//std::cout << pGenerator->str() << std::endl;
				pGenerator->getCandidateActions(actions);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram, _actionNgram);
					_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score += pGenerator->_score;

					scored_action.nnfeat.init(_wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
							_actionRNNHiddenSize, _sep_hiddenInSize, _app_hiddenInSize, _sep_hiddenOutSize, _app_hiddenOutSize, true);

					//neural
					//action list
					for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
						_actions.GetEmb(scored_action.feat._nActionFeat[tmp_k], scored_action.nnfeat._actionPrime[tmp_k]);
						dropoutcol(scored_action.nnfeat._actionPrimeMask[tmp_k], _dropOut);
						scored_action.nnfeat._actionPrime[tmp_k] = scored_action.nnfeat._actionPrime[tmp_k] * scored_action.nnfeat._actionPrimeMask[tmp_k];
					}
					concat(scored_action.nnfeat._actionPrime, scored_action.nnfeat._actionRep);
					_nnlayer_action_hidden.ComputeForwardScore(scored_action.nnfeat._actionRep, scored_action.nnfeat._actionHidden);

					if (pGenerator->_nextPosition == 0) {
						_action_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._actionHidden, scored_action.nnfeat._actionRNNHidden);
					} else {
						_action_increased_rnn.ComputeForwardScoreIncremental(pGenerator->_nnfeat._actionRNNHidden, scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHidden);
					}

					//read word
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k]);
							dropoutcol(scored_action.nnfeat._wordPrimeMask[tmp_k], _dropOut);
							scored_action.nnfeat._wordPrime[tmp_k] = scored_action.nnfeat._wordPrime[tmp_k] * scored_action.nnfeat._wordPrimeMask[tmp_k];
						}
						concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
						_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordRep, scored_action.nnfeat._wordHidden);
						const CStateItem<xpu> * preSepState = pGenerator->_prevSepState;
						if (preSepState == 0) {
							_word_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._wordHidden, scored_action.nnfeat._wordRNNHidden);
						} else {
							_word_increased_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._wordRNNHidden, scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHidden);
						}

						//
						if (pGenerator->_nextPosition < length) {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charLeftRNNHidden[pGenerator->_nextPosition],
									charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._sepInHidden);
						} else {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charRNNHiddenDummy, charRNNHiddenDummy,
									scored_action.nnfeat._sepInHidden);
						}
						_nnlayer_sep_hidden.ComputeForwardScore(scored_action.nnfeat._sepInHidden, scored_action.nnfeat._sepOutHidden);
						_nnlayer_sep_output.ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
					} else {
						concat(scored_action.nnfeat._actionRNNHidden, charLeftRNNHidden[pGenerator->_nextPosition], charRightRNNHidden[pGenerator->_nextPosition],
								scored_action.nnfeat._appInHidden);
						_nnlayer_app_hidden.ComputeForwardScore(scored_action.nnfeat._appInHidden, scored_action.nnfeat._appOutHidden);
						_nnlayer_app_output.ComputeForwardScore(scored_action.nnfeat._appOutHidden, score);
					}

					scored_action.score += score;
					//std::cout << "add start, action = " << actions[tmp_j] << ", cur ac score = " << scored_action.score << ", orgin score = " << pGenerator->_score << std::endl;;

					if (actions[tmp_j] != correct_action) {
						scored_action.score += _delta;
					}

					beam.add_elem(scored_action);

					//std::cout << "new scored_action : " << scored_action.score << ", action = " << scored_action.action << ", state = " << scored_action.item->str() << std::endl;
					//for (int tmp_k = 0; tmp_k < beam.elemsize(); ++tmp_k) {
					//  std::cout << tmp_k << ": " << beam[tmp_k].score << ", action = " << beam[tmp_k].action << ", state = " << beam[tmp_k].item->str() << std::endl;
					//}

					if (pGenerator == correctState && actions[tmp_j] == correct_action) {
						scored_correct_action = scored_action;
						correct_action_scored = true;
						//std::cout << "add gold finish" << std::endl;
					} else {
						//std::cout << "add finish" << std::endl;
					}

				}
			}

			//std::cout << "check beam finish" << std::endl;

			if (beam.elemsize() == 0) {
				std::cout << "error" << std::endl;
				for (int idx = 0; idx < sentence.size(); idx++) {
					std::cout << sentence[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				return -1;
			}

			//std::cout << "check beam start" << std::endl;
			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action);
				lattice_index[index + 1]->_score = beam[tmp_j].score;
				lattice_index[index + 1]->_curFeat.copy(beam[tmp_j].feat);

				//std::cout << tmp_j << ": " << beam[tmp_j].score << std::endl;

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}
				if (pGenerator == correctState && beam[tmp_j].action == correct_action) {
					correctState = lattice_index[index + 1];
					bCorrect = true;
				}

				++lattice_index[index + 1];
			}
			//std::cout << "check beam finish" << std::endl;

			if (pBestGen->IsTerminated())
				break; // while

			// update items if correct item jump out of the agenda

			if (!bCorrect) {
				// note that if bCorrect == true then the correct state has
				// already been updated, and the new value is one of the new states
				// among the newly produced from lattice[index+1].
				correctState->move(lattice_index[index + 1], correct_action);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);

				++lattice_index[index + 1];
				assert(correct_action_scored); // scored_correct_act valid
				//TRACE(index << " updated");
				//std::cout << index << " updated" << std::endl;
				pBestGenFeat.init(pBestGen->_wordnum, index, _wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
						_actionRNNHiddenSize);
				pGoldFeat.init(correctState->_wordnum, index, _wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
						_actionRNNHiddenSize);
				backPropagationStates(pBestGen, correctState, 1.0, -1.0, charLeftRNNHidden_Loss, charRightRNNHidden_Loss, charRNNHiddenDummy_Loss, pBestGenFeat,
						pGoldFeat);
				pBestGenFeat.clear();
				pGoldFeat.clear();

				_eval.correct_label_count += index;
				_eval.overall_label_count += length + 1;
				return cost;
			}

		}

		// make sure that the correct item is stack top finally
		if (pBestGen != correctState) {
			if (!bCorrect) {
				correctState->move(lattice_index[index + 1], correct_action);
				correctState = lattice_index[index + 1];
				lattice_index[index + 1]->_score = scored_correct_action.score;
				lattice_index[index + 1]->_curFeat.copy(scored_correct_action.feat);
				lattice_index[index + 1]->_nnfeat.copy(scored_correct_action.nnfeat);
				assert(correct_action_scored); // scored_correct_act valid
			}

			//std::cout << "best:" << pBestGen->str() << std::endl;
			//std::cout << "gold:" << correctState->str() << std::endl;
			pBestGenFeat.init(pBestGen->_wordnum, index, _wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize);
			pGoldFeat.init(correctState->_wordnum, index, _wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize);
			backPropagationStates(pBestGen, correctState, 1.0, -1.0, charLeftRNNHidden_Loss, charRightRNNHidden_Loss, charRNNHiddenDummy_Loss, pBestGenFeat,
					pGoldFeat);
			pBestGenFeat.clear();
			pGoldFeat.clear();

			_eval.correct_label_count += length;
			_eval.overall_label_count += length + 1;
		} else {
			_eval.correct_label_count += length + 1;
			_eval.overall_label_count += length + 1;
		}

		return cost;
	}

	void backPropagationStates(const CStateItem<xpu> *pPredState, const CStateItem<xpu> *pGoldState, dtype predLoss, dtype goldLoss,
			Tensor<xpu, 3, dtype> charLeftRNNHidden_Loss, Tensor<xpu, 3, dtype> charRightRNNHidden_Loss, Tensor<xpu, 2, dtype> charRNNHiddenDummy_Loss,
			DenseFeature<xpu>& predDenseFeature, DenseFeature<xpu>& goldDenseFeature) {

		if (pPredState->_nextPosition != pGoldState->_nextPosition) {
			std::cout << "state align error" << std::endl;
		}

		static int position, word_position;

		if (pPredState->_nextPosition == 0) {

			return;
		}

		if (pPredState != pGoldState) {
			//sparse
			_splayer_output.ComputeBackwardLoss(pPredState->_curFeat._nSparseFeat, predLoss);
			_splayer_output.ComputeBackwardLoss(pGoldState->_curFeat._nSparseFeat, goldLoss);

			int length = charLeftRNNHidden_Loss.size(0);

			//predState
			position = pPredState->_nextPosition - 1;

			if (pPredState->_lastAction._code == CAction::SEP || pPredState->_lastAction._code == CAction::FIN) {
				_nnlayer_sep_output.ComputeBackwardLoss(pPredState->_nnfeat._sepOutHidden, predLoss, pPredState->_nnfeat._sepOutHiddenLoss);
				_nnlayer_sep_hidden.ComputeBackwardLoss(pPredState->_nnfeat._sepInHidden, pPredState->_nnfeat._sepOutHidden, pPredState->_nnfeat._sepOutHiddenLoss,
						pPredState->_nnfeat._sepInHiddenLoss);

				word_position = pPredState->_wordnum - 1;
				if (position < length) {
					unconcat(predDenseFeature._wordRNNHiddenLoss[word_position], predDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position],
							charLeftRNNHidden_Loss[position], pPredState->_nnfeat._sepInHiddenLoss);
				} else {
					unconcat(predDenseFeature._wordRNNHiddenLoss[word_position], predDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss,
							charRNNHiddenDummy_Loss, pPredState->_nnfeat._sepInHiddenLoss);
				}

			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pPredState->_nnfeat._appOutHidden, predLoss, pPredState->_nnfeat._appOutHiddenLoss);
				_nnlayer_app_hidden.ComputeBackwardLoss(pPredState->_nnfeat._appInHidden, pPredState->_nnfeat._appOutHidden, pPredState->_nnfeat._appOutHiddenLoss,
						pPredState->_nnfeat._appInHiddenLoss);

				if (position < length) {
					unconcat(predDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position], charLeftRNNHidden_Loss[position],
							pPredState->_nnfeat._appInHiddenLoss);
				} else {
					unconcat(predDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss, charRNNHiddenDummy_Loss, pPredState->_nnfeat._appInHiddenLoss);
				}

			}

			//goldState
			position = pGoldState->_nextPosition - 1;

			if (pGoldState->_lastAction._code == CAction::SEP || pGoldState->_lastAction._code == CAction::FIN) {
				_nnlayer_sep_output.ComputeBackwardLoss(pGoldState->_nnfeat._sepOutHidden, goldLoss, pGoldState->_nnfeat._sepOutHiddenLoss);
				_nnlayer_sep_hidden.ComputeBackwardLoss(pGoldState->_nnfeat._sepInHidden, pGoldState->_nnfeat._sepOutHidden, pGoldState->_nnfeat._sepOutHiddenLoss,
						pGoldState->_nnfeat._sepInHiddenLoss);

				word_position = pGoldState->_wordnum - 1;
				if (position < length) {
					unconcat(goldDenseFeature._wordRNNHiddenLoss[word_position], goldDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position],
							charLeftRNNHidden_Loss[position], pGoldState->_nnfeat._sepInHiddenLoss);
				} else {
					unconcat(goldDenseFeature._wordRNNHiddenLoss[word_position], goldDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss,
							charRNNHiddenDummy_Loss, pGoldState->_nnfeat._sepInHiddenLoss);
				}

			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pGoldState->_nnfeat._appOutHidden, predLoss, pGoldState->_nnfeat._appOutHiddenLoss);
				_nnlayer_app_hidden.ComputeBackwardLoss(pGoldState->_nnfeat._appInHidden, pGoldState->_nnfeat._appOutHidden, pGoldState->_nnfeat._appOutHiddenLoss,
						pGoldState->_nnfeat._appInHiddenLoss);

				if (position < length) {
					unconcat(goldDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position], charLeftRNNHidden_Loss[position],
							pGoldState->_nnfeat._appInHiddenLoss);
				} else {
					unconcat(goldDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss, charRNNHiddenDummy_Loss, pGoldState->_nnfeat._appInHiddenLoss);
				}

			}

		}

		//predState
		if (pPredState->_lastAction._code == CAction::SEP || pPredState->_lastAction._code == CAction::FIN) {
			tcopy(pPredState->_nnfeat._wordPrime, predDenseFeature._wordPrime[word_position]);
			tcopy(pPredState->_nnfeat._wordPrimeMask, predDenseFeature._wordPrimeMask[word_position]);
			tcopy(pPredState->_nnfeat._wordRep, predDenseFeature._wordRep[word_position]);
			tcopy(pPredState->_nnfeat._wordHidden, predDenseFeature._wordHidden[word_position]);
			tcopy(pPredState->_nnfeat._wordRNNHidden, predDenseFeature._wordRNNHidden[word_position]);

			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				predDenseFeature._words[word_position][tmp_k] = pPredState->_curFeat._nWordFeat[tmp_k];
			}
		}

		tcopy(pPredState->_nnfeat._actionPrime, predDenseFeature._actionPrime[position]);
		tcopy(pPredState->_nnfeat._actionPrimeMask, predDenseFeature._actionPrimeMask[position]);
		tcopy(pPredState->_nnfeat._actionRep, predDenseFeature._actionRep[position]);
		tcopy(pPredState->_nnfeat._actionHidden, predDenseFeature._actionHidden[position]);
		tcopy(pPredState->_nnfeat._actionRNNHidden, predDenseFeature._actionRNNHidden[position]);
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			predDenseFeature._actions[position][tmp_k] = pPredState->_curFeat._nActionFeat[tmp_k];
		}

		//goldState
		if (pGoldState->_lastAction._code == CAction::SEP || pGoldState->_lastAction._code == CAction::FIN) {
			tcopy(pGoldState->_nnfeat._wordPrime, goldDenseFeature._wordPrime[word_position]);
			tcopy(pGoldState->_nnfeat._wordPrimeMask, goldDenseFeature._wordPrimeMask[word_position]);
			tcopy(pGoldState->_nnfeat._wordRep, goldDenseFeature._wordRep[word_position]);
			tcopy(pGoldState->_nnfeat._wordHidden, goldDenseFeature._wordHidden[word_position]);
			tcopy(pGoldState->_nnfeat._wordRNNHidden, goldDenseFeature._wordRNNHidden[word_position]);

			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				goldDenseFeature._words[word_position][tmp_k] = pGoldState->_curFeat._nWordFeat[tmp_k];
			}
		}

		tcopy(pGoldState->_nnfeat._actionPrime, goldDenseFeature._actionPrime[position]);
		tcopy(pGoldState->_nnfeat._actionPrimeMask, goldDenseFeature._actionPrimeMask[position]);
		tcopy(pGoldState->_nnfeat._actionRep, goldDenseFeature._actionRep[position]);
		tcopy(pGoldState->_nnfeat._actionHidden, goldDenseFeature._actionHidden[position]);
		tcopy(pGoldState->_nnfeat._actionRNNHidden, goldDenseFeature._actionRNNHidden[position]);
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			goldDenseFeature._actions[position][tmp_k] = pGoldState->_curFeat._nActionFeat[tmp_k];
		}

		//currently we use a uniform loss
		backPropagationStates(pPredState->_prevState, pGoldState->_prevState, predLoss, goldLoss, charLeftRNNHidden_Loss, charRightRNNHidden_Loss,
				charRNNHiddenDummy_Loss, predDenseFeature, goldDenseFeature);

	}

	bool decode(const std::vector<string>& sentence, std::vector<std::string>& words) {
		setAlphaIncreasing(false);
		if (sentence.size() >= MAX_SENTENCE_SIZE)
			return false;
		static CStateItem<xpu> lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
		static CStateItem<xpu> *lattice_index[MAX_SENTENCE_SIZE + 1];

		int length = sentence.size();
		dtype cost = 0.0;
		dtype score = 0.0;

		const static CStateItem<xpu> *pGenerator;
		const static CStateItem<xpu> *pBestGen;

		int index, tmp_i, tmp_j;
		std::vector<CAction> actions; // actions to apply for a candidate
		static NRHeap<CScoredStateAction<xpu>, CScoredStateAction_Compare<xpu> > beam(BEAM_SIZE);
		static CScoredStateAction<xpu> scored_action; // used rank actions
		static Feature feat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		vector<int> charIds(length), biCharIds(length);
		Tensor<xpu, 3, dtype> charprime = NewTensor<xpu>(Shape3(length, 1, _charDim), d_zero);
		Tensor<xpu, 3, dtype> bicharprime = NewTensor<xpu>(Shape3(length, 1, _charDim), d_zero);
		Tensor<xpu, 3, dtype> charpre = NewTensor<xpu>(Shape3(length, 1, _charDim + _biCharDim), d_zero);
		Tensor<xpu, 3, dtype> charInput = NewTensor<xpu>(Shape3(length, 1, _charRepresentDim), d_zero);
		Tensor<xpu, 3, dtype> charHidden = NewTensor<xpu>(Shape3(length, 1, _charHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charLeftRNNHidden = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 3, dtype> charRightRNNHidden = NewTensor<xpu>(Shape3(length, 1, _charRNNHiddenSize), d_zero);
		Tensor<xpu, 2, dtype> charRNNHiddenDummy = NewTensor<xpu>(Shape2(1, _charRNNHiddenSize), d_zero);

		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charIds[idx] < 0)
				charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			biCharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (biCharIds[idx] < 0)
				biCharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			_chars.GetEmb(charIds[idx], charprime[idx]);
			_bichars.GetEmb(biCharIds[idx], bicharprime[idx]);
			concat(charprime[idx], bicharprime[idx], charpre[idx]);
		}

		windowlized(charpre, charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charInput, charHidden);
		_char_left_rnn.ComputeForwardScore(charHidden, charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charHidden, charRightRNNHidden);

		while (true) {
			++index;
			lattice_index[index + 1] = lattice_index[index];
			beam.clear();
			pBestGen = 0;

			//std::cout << "check beam start" << std::endl;
			for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
				//std::cout << "new" << std::endl;
				//std::cout << pGenerator->str() << std::endl;
				pGenerator->getCandidateActions(actions);
				for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram, _actionNgram);
					_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score += pGenerator->_score;

					scored_action.nnfeat.init(_wordDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
							_actionRNNHiddenSize, _sep_hiddenInSize, _app_hiddenInSize, _sep_hiddenOutSize, _app_hiddenOutSize, true);

					//action list
					for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
						_actions.GetEmb(scored_action.feat._nActionFeat[tmp_k], scored_action.nnfeat._actionPrime[tmp_k]);
					}
					concat(scored_action.nnfeat._actionPrime, scored_action.nnfeat._actionRep);
					_nnlayer_action_hidden.ComputeForwardScore(scored_action.nnfeat._actionRep, scored_action.nnfeat._actionHidden);

					if (pGenerator->_nextPosition == 0) {
						_action_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._actionHidden, scored_action.nnfeat._actionRNNHidden);
					} else {
						_action_increased_rnn.ComputeForwardScoreIncremental(pGenerator->_nnfeat._actionRNNHidden, scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHidden);
					}

					//read word
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
						_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordRep, scored_action.nnfeat._wordHidden);
						const CStateItem<xpu> * preSepState = pGenerator->_prevSepState;
						if (preSepState == 0) {
							_word_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._wordHidden, scored_action.nnfeat._wordRNNHidden);
						} else {
							_word_increased_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._wordRNNHidden, scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHidden);
						}

						//
						if (pGenerator->_nextPosition < length) {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charLeftRNNHidden[pGenerator->_nextPosition],
									charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._sepInHidden);
						} else {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charRNNHiddenDummy, charRNNHiddenDummy,
									scored_action.nnfeat._sepInHidden);
						}
						_nnlayer_sep_hidden.ComputeForwardScore(scored_action.nnfeat._sepInHidden, scored_action.nnfeat._sepOutHidden);
						_nnlayer_sep_output.ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
					} else {
						concat(scored_action.nnfeat._actionRNNHidden, charLeftRNNHidden[pGenerator->_nextPosition], charRightRNNHidden[pGenerator->_nextPosition],
								scored_action.nnfeat._appInHidden);
						_nnlayer_app_hidden.ComputeForwardScore(scored_action.nnfeat._appInHidden, scored_action.nnfeat._appOutHidden);
						_nnlayer_app_output.ComputeForwardScore(scored_action.nnfeat._appOutHidden, score);
					}

					scored_action.score += score;

					beam.add_elem(scored_action);
				}

			}

			if (beam.elemsize() == 0) {
				std::cout << "error" << std::endl;
				for (int idx = 0; idx < sentence.size(); idx++) {
					std::cout << sentence[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				return false;
			}

			for (tmp_j = 0; tmp_j < beam.elemsize(); ++tmp_j) { // insert from
				pGenerator = beam[tmp_j].item;
				pGenerator->move(lattice_index[index + 1], beam[tmp_j].action);
				lattice_index[index + 1]->_score = beam[tmp_j].score;

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while

		}
		pBestGen->getSegResults(words);

		return true;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_splayer_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}

	void writeModel();

	void loadModel();

public:

	inline void resetEval() {
		_eval.reset();
	}

	inline void setDropValue(dtype dropOut) {
		_dropOut = dropOut;
	}

	inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
		_words.setEmbFineTune(b_wordEmb_finetune);
	}

	inline void setCharEmbFinetune(bool b_charEmb_finetune) {
		_chars.setEmbFineTune(b_charEmb_finetune);
	}

	inline void setBiCharEmbFinetune(bool b_bicharEmb_finetune) {
		_bichars.setEmbFineTune(b_bicharEmb_finetune);
	}

};

#endif /* SRC_StackLSTMBeamSearcher_H_ */
