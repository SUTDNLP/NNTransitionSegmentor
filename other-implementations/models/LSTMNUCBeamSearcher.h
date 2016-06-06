/*
 * LSTMNUCBeamSearcher.h
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_LSTMNUCBeamSearcher_H_
#define SRC_LSTMNUCBeamSearcher_H_

#include <hash_set>
#include <iostream>

#include <assert.h>
#include "Feature.h"
#include "DenseFeatureExtraction.h"
#include "DenseFeature.h"
#include "DenseFeatureChar.h"
#include "N3L.h"
#include "NeuralState.h"
#include "Action.h"
#include "SegLookupTable.h"

#define LSTM_ALG LSTM_STD

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
template<typename xpu>
class LSTMNUCBeamSearcher {
public:
	LSTMNUCBeamSearcher() {
		_dropOut = 0.5;
		_delta = 0.2;
		_oovRatio = 0.2;
		_oovFreq = 3;
		_buffer = 6;
	}
	~LSTMNUCBeamSearcher() {
	}

public:
	//SparseUniLayer1O<xpu> _splayer_output;
	UniLayer1O<xpu> _nnlayer_sep_output;
	UniLayer1O<xpu> _nnlayer_app_output;
	SegLookupTable<xpu> _words;
	LookupTable<xpu> _allwords;
	LookupTable<xpu> _chars;
	LookupTable<xpu> _keyChars;
	LookupTable<xpu> _bichars;
	LookupTable<xpu> _actions;
	LookupTable<xpu> _lengths;

	LSTM_ALG<xpu> _char_left_rnn;
	LSTM_ALG<xpu> _char_right_rnn;
	LSTM_ALG<xpu> _word_increased_rnn;
	LSTM_ALG<xpu> _action_increased_rnn;
	UniLayer<xpu> _nnlayer_sep_hidden;
	UniLayer<xpu> _nnlayer_app_hidden;
	UniLayer<xpu> _nnlayer_word_hidden;
	UniLayer<xpu> _nnlayer_char_hidden;
	UniLayer<xpu> _nnlayer_action_hidden;

	int _wordSize, _allwordSize, _lengthSize;
	int _wordDim,  _allwordDim, _lengthDim;
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

	dtype _oovRatio;

	int _oovFreq;

	int _buffer;

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

	inline void addToAllWordAlphabet(hash_map<string, int> allword_stat, int allwordCutOff = 0) {
		fe.addToAllWordAlphabet(allword_stat, allwordCutOff);
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

	inline void init(const NRMat<dtype>& wordEmb, const NRMat<dtype>& allwordEmb, const NRMat<dtype>& lengthEmb, int wordNgram, int wordHiddenSize, int wordRNNHiddenSize,
			const NRMat<dtype>& charEmb, const NRMat<dtype>& bicharEmb, int charcontext, int charHiddenSize, int charRNNHiddenSize,
			const NRMat<dtype>& actionEmb, int actionNgram, int actionHiddenSize, int actionRNNHiddenSize,
			int sep_hidden_out_size, int app_hidden_out_size, dtype delta) {
		_delta = delta;
		_linearfeatSize = 3 * fe._featAlphabet.size();
		//_splayer_output.initial(_linearfeatSize, 10);


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

		_wordSize = fe._wordAlphabet.size();
		if (_wordSize != wordEmb.nrows())
			std::cout << "word number does not match for initialization of word emb table" << std::endl;
		_allwordSize = fe._allwordAlphabet.size();
		if (_allwordSize != allwordEmb.nrows())
			std::cout << "allword number does not match for initialization of allword emb table" << std::endl;				
		_wordDim = wordEmb.ncols();
		_allwordDim = allwordEmb.ncols();

		_lengthSize = lengthEmb.nrows();
		_lengthDim = lengthEmb.ncols();

		_wordNgram = wordNgram;
		_wordRepresentDim = _wordNgram * _wordDim +  _wordNgram * _allwordDim + (2 * _wordNgram + 1) * _charDim + _wordNgram * _lengthDim;

		_wordRNNHiddenSize = wordRNNHiddenSize;
		_charRNNHiddenSize = charRNNHiddenSize;
		_actionRNNHiddenSize = actionRNNHiddenSize;
		_wordHiddenSize = wordHiddenSize;
		_charHiddenSize = charHiddenSize;
		_actionHiddenSize = actionHiddenSize;
		_sep_hiddenOutSize = sep_hidden_out_size;
		_app_hiddenOutSize = app_hidden_out_size;
		_sep_hiddenInSize = _wordRNNHiddenSize + _actionRNNHiddenSize + 2 * _charRNNHiddenSize;
		_app_hiddenInSize = _actionRNNHiddenSize + 2 * _charRNNHiddenSize;

		_nnlayer_sep_output.initial(_sep_hiddenOutSize, 10);
		_nnlayer_app_output.initial(_app_hiddenOutSize, 20);

		_words.initial(wordEmb);
		_words.setEmbFineTune(true);
		_allwords.initial(allwordEmb, false);
		_allwords.setEmbFineTune(false);
		_chars.initial(charEmb);
		_chars.setEmbFineTune(false);
		_keyChars.initial(charEmb);
		_keyChars.setEmbFineTune(false);
		_bichars.initial(bicharEmb);
		_bichars.setEmbFineTune(false);
		_actions.initial(actionEmb);
		_actions.setEmbFineTune(true);
		_lengths.initial(lengthEmb);
		_lengths.setEmbFineTune(true);

		_char_left_rnn.initial(_charRNNHiddenSize, _charHiddenSize, true, 30);
		_char_right_rnn.initial(_charRNNHiddenSize, _charHiddenSize, false, 40);
		_word_increased_rnn.initial(_wordRNNHiddenSize, _wordHiddenSize, true, 50);
		_action_increased_rnn.initial(_actionRNNHiddenSize, _actionHiddenSize, true, 60);
		_nnlayer_sep_hidden.initial(_sep_hiddenOutSize, _sep_hiddenInSize, true, 70, 0);
		_nnlayer_app_hidden.initial(_app_hiddenOutSize, _app_hiddenInSize, true, 80, 0);
		_nnlayer_word_hidden.initial(_wordHiddenSize, _wordRepresentDim, true, 90, 0);
		_nnlayer_char_hidden.initial(_charHiddenSize, _charRepresentDim, true, 100, 0);
		_nnlayer_action_hidden.initial(_actionHiddenSize, _actionRepresentDim, true, 110, 0);

	}

	inline void release() {
		//_splayer_output.release();

		_nnlayer_sep_output.release();
		_nnlayer_app_output.release();

		_words.release();
		_allwords.release();
		_chars.release();
		_bichars.release();
		_actions.release();
		_lengths.release();

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
			cost += trainOneExample(sentences[idx], goldACs[idx], sentences.size());
		}

		return cost;
	}

	// scores do not accumulate together...., big bug, refine it tomorrow or at thursday.
	dtype trainOneExample(const std::vector<std::string>& sentence, const vector<CAction>& goldAC, int num) {
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
		static DenseFeatureChar<xpu> charFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		charFeat.init(length, _charDim, _biCharDim, _charcontext, _charHiddenSize, _charRNNHiddenSize, _buffer, true);
		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charFeat._charIds[idx] < 0)
				charFeat._charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._bicharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (charFeat._bicharIds[idx] < 0)
				charFeat._bicharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			//_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			charFeat._charprime[idx] = 0.0;
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
			dropoutcol(charFeat._charpreMask[idx], _dropOut);
			charFeat._charpre[idx] = charFeat._charpre[idx] * charFeat._charpreMask[idx];
		}

		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_char_left_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden);

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
					//scored_action.clear();
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram, _actionNgram);
					//_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score = 0;
					scored_action.score += pGenerator->_score;

					scored_action.nnfeat.init(_wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
							_actionRNNHiddenSize, _sep_hiddenInSize, _app_hiddenInSize, _sep_hiddenOutSize, _app_hiddenOutSize, _buffer, true);

					//neural
					//action list
					for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
						_actions.GetEmb(scored_action.feat._nActionFeat[tmp_k], scored_action.nnfeat._actionPrime[tmp_k]);
					}

					concat(scored_action.nnfeat._actionPrime, scored_action.nnfeat._actionRep);
					dropoutcol(scored_action.nnfeat._actionRepMask, _dropOut);
					scored_action.nnfeat._actionRep = scored_action.nnfeat._actionRep * scored_action.nnfeat._actionRepMask;

					_nnlayer_action_hidden.ComputeForwardScore(scored_action.nnfeat._actionRep, scored_action.nnfeat._actionHidden);

					if (pGenerator->_nextPosition == 0) {
						_action_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
								scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
								scored_action.nnfeat._actionRNNHidden);
					} else {
						_action_increased_rnn.ComputeForwardScoreIncremental(pGenerator->_nnfeat._actionRNNHiddenBuf[4], pGenerator->_nnfeat._actionRNNHidden, scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
								scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
								scored_action.nnfeat._actionRNNHidden);
					}

					//read word
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							int unknownID = fe._wordAlphabet[fe.unknownkey];
							int curFreq = _words.getFrequency(scored_action.feat._nWordFeat[tmp_k]);
							if (curFreq >= 0 && curFreq <= _oovFreq){
								//if (1.0 * rand() / RAND_MAX < _oovRatio){
									scored_action.feat._nWordFeat[tmp_k] = unknownID;
								//}
							}
							_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k]);
							_allwords.GetEmb(scored_action.feat._nAllWordFeat[tmp_k], scored_action.nnfeat._allwordPrime[tmp_k]);
						}

						concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
						concat(scored_action.nnfeat._allwordPrime, scored_action.nnfeat._allwordRep);

						for (int tmp_k = 0; tmp_k < 2*_wordNgram+1; tmp_k++) {
							_keyChars.GetEmb(scored_action.feat._nKeyChars[tmp_k], scored_action.nnfeat._keyCharPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._keyCharPrime, scored_action.nnfeat._keyCharRep);

						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							_lengths.GetEmb(scored_action.feat._nWordLengths[tmp_k], scored_action.nnfeat._lengthPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._lengthPrime, scored_action.nnfeat._lengthRep);

						concat(scored_action.nnfeat._wordRep, scored_action.nnfeat._allwordRep, scored_action.nnfeat._keyCharRep, scored_action.nnfeat._lengthRep, scored_action.nnfeat._wordUnitRep);
						dropoutcol(scored_action.nnfeat._wordUnitRepMask, _dropOut);
						scored_action.nnfeat._wordUnitRep = scored_action.nnfeat._wordUnitRep * scored_action.nnfeat._wordUnitRepMask;

						_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordUnitRep, scored_action.nnfeat._wordHidden);

						const CStateItem<xpu> * preSepState = pGenerator->_prevSepState;
						if (preSepState == 0) {
							_word_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
									scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
									scored_action.nnfeat._wordRNNHidden);
						} else {
							_word_increased_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._wordRNNHiddenBuf[4], preSepState->_nnfeat._wordRNNHidden, scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
									scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
									scored_action.nnfeat._wordRNNHidden);
						}

						//
						if (pGenerator->_nextPosition < length) {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition],
									charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._sepInHidden);
						} else {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charRNNHiddenDummy, charFeat._charRNNHiddenDummy,
									scored_action.nnfeat._sepInHidden);
						}
						_nnlayer_sep_hidden.ComputeForwardScore(scored_action.nnfeat._sepInHidden, scored_action.nnfeat._sepOutHidden);
						_nnlayer_sep_output.ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
					} else {
						concat(scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition], charFeat._charRightRNNHidden[pGenerator->_nextPosition],
								scored_action.nnfeat._appInHidden);
						_nnlayer_app_hidden.ComputeForwardScore(scored_action.nnfeat._appInHidden, scored_action.nnfeat._appOutHidden);
						_nnlayer_app_output.ComputeForwardScore(scored_action.nnfeat._appOutHidden, score);
					}
					//std::cout << "score = " << score << std::endl;

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
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);

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
				//std::cout << "best score: " << pBestGen->_score << " , gold score: " << correctState->_score << std::endl;

				pBestGenFeat.init(pBestGen->_wordnum, index, _wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
						_actionRNNHiddenSize, _buffer);
				pGoldFeat.init(correctState->_wordnum, index, _wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
						_actionRNNHiddenSize, _buffer);
				backPropagationStates(pBestGen, correctState, 1.0/num, -1.0/num, charFeat._charLeftRNNHidden_Loss, charFeat._charRightRNNHidden_Loss, charFeat._charRNNHiddenDummy_Loss, pBestGenFeat,
						pGoldFeat);

				_char_left_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
						charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden,
						charFeat._charLeftRNNHidden_Loss, charFeat._charHidden_Loss);
				_char_right_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
						charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden,
						charFeat._charRightRNNHidden_Loss, charFeat._charHidden_Loss);
				_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
				windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
				charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;

				for(int idx = 0; idx < length; idx++){
					unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
					charFeat._charprime_Loss[idx] = 0.0;
				_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
					_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
				}
				pBestGenFeat.clear();
				pGoldFeat.clear();
				charFeat.clear();

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
			//std::cout << index << " updated" << std::endl;
			//std::cout << "best score: " << pBestGen->_score << " , gold score: " << correctState->_score << std::endl;
			pBestGenFeat.init(pBestGen->_wordnum, index, _wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize, _buffer);
			pGoldFeat.init(correctState->_wordnum, index, _wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordRepresentDim, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize, _actionRNNHiddenSize, _buffer);
			backPropagationStates(pBestGen, correctState, 1.0/num, -1.0/num, charFeat._charLeftRNNHidden_Loss, charFeat._charRightRNNHidden_Loss, charFeat._charRNNHiddenDummy_Loss, pBestGenFeat,
					pGoldFeat);
			pBestGenFeat.clear();
			pGoldFeat.clear();

			_char_left_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
					charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden,
					charFeat._charLeftRNNHidden_Loss, charFeat._charHidden_Loss);
			_char_right_rnn.ComputeBackwardLoss(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
					charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden,
					charFeat._charRightRNNHidden_Loss, charFeat._charHidden_Loss);
			_nnlayer_char_hidden.ComputeBackwardLoss(charFeat._charInput, charFeat._charHidden, charFeat._charHidden_Loss, charFeat._charInput_Loss);
			windowlized_backward(charFeat._charpre_Loss, charFeat._charInput_Loss, _charcontext);
			charFeat._charpre_Loss = charFeat._charpre_Loss * charFeat._charpreMask;

			for(int idx = 0; idx < length; idx++){
				unconcat(charFeat._charprime_Loss[idx], charFeat._bicharprime_Loss[idx], charFeat._charpre_Loss[idx]);
				charFeat._charprime_Loss[idx] = 0.0;
				_chars.EmbLoss(charFeat._charIds[idx], charFeat._charprime_Loss[idx]);
				_bichars.EmbLoss(charFeat._bicharIds[idx], charFeat._bicharprime_Loss[idx]);
			}
			pBestGenFeat.clear();
			pGoldFeat.clear();
			charFeat.clear();
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
			//predState
			_word_increased_rnn.ComputeBackwardLoss(predDenseFeature._wordHidden,
					predDenseFeature._wordRNNHiddenBuf[0], predDenseFeature._wordRNNHiddenBuf[1], predDenseFeature._wordRNNHiddenBuf[2],
					predDenseFeature._wordRNNHiddenBuf[3], predDenseFeature._wordRNNHiddenBuf[4], predDenseFeature._wordRNNHiddenBuf[5],
					predDenseFeature._wordRNNHidden, predDenseFeature._wordRNNHiddenLoss, predDenseFeature._wordHiddenLoss);
			_nnlayer_word_hidden.ComputeBackwardLoss(predDenseFeature._wordUnitRep, predDenseFeature._wordHidden, predDenseFeature._wordHiddenLoss, predDenseFeature._wordUnitRepLoss);

			for(int idx = 0; idx < predDenseFeature._wordRepLoss.size(); idx++){
				predDenseFeature._wordUnitRepLoss[idx] = predDenseFeature._wordUnitRepLoss[idx] * predDenseFeature._wordUnitRepMask[idx];
				unconcat(predDenseFeature._wordRepLoss[idx], predDenseFeature._allwordRepLoss[idx], predDenseFeature._keyCharRepLoss[idx], predDenseFeature._lengthRepLoss[idx], predDenseFeature._wordUnitRepLoss[idx]);
				unconcat(predDenseFeature._wordPrimeLoss[idx], predDenseFeature._wordRepLoss[idx]);

				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_words.EmbLoss(predDenseFeature._wordIds[idx][tmp_k], predDenseFeature._wordPrimeLoss[idx][tmp_k]);
				}

				unconcat(predDenseFeature._keyCharPrimeLoss[idx], predDenseFeature._keyCharRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < 2 * _wordNgram + 1; tmp_k++) {
					_keyChars.EmbLoss(predDenseFeature._keyCharIds[idx][tmp_k], predDenseFeature._keyCharPrimeLoss[idx][tmp_k]);
				}

				unconcat(predDenseFeature._lengthPrimeLoss[idx], predDenseFeature._lengthRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_lengths.EmbLoss(predDenseFeature._lengthIds[idx][tmp_k], predDenseFeature._lengthPrimeLoss[idx][tmp_k]);
				}

			}


			_action_increased_rnn.ComputeBackwardLoss(predDenseFeature._actionHidden,
					predDenseFeature._actionRNNHiddenBuf[0], predDenseFeature._actionRNNHiddenBuf[1], predDenseFeature._actionRNNHiddenBuf[2],
					predDenseFeature._actionRNNHiddenBuf[3], predDenseFeature._actionRNNHiddenBuf[4], predDenseFeature._actionRNNHiddenBuf[5],
					predDenseFeature._actionRNNHidden, predDenseFeature._actionRNNHiddenLoss, predDenseFeature._actionHiddenLoss);
			_nnlayer_action_hidden.ComputeBackwardLoss(predDenseFeature._actionRep, predDenseFeature._actionHidden, predDenseFeature._actionHiddenLoss, predDenseFeature._actionRepLoss);



			for(int idx = 0; idx < predDenseFeature._actionRepLoss.size(); idx++){
				predDenseFeature._actionRepLoss[idx] = predDenseFeature._actionRepLoss[idx] * predDenseFeature._actionRepMask[idx];

				unconcat(predDenseFeature._actionPrimeLoss[idx], predDenseFeature._actionRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
					_actions.EmbLoss(predDenseFeature._actionIds[idx][tmp_k], predDenseFeature._actionPrimeLoss[idx][tmp_k]);
				}
			}

			//goldState
			_word_increased_rnn.ComputeBackwardLoss(goldDenseFeature._wordHidden,
					goldDenseFeature._wordRNNHiddenBuf[0], goldDenseFeature._wordRNNHiddenBuf[1], goldDenseFeature._wordRNNHiddenBuf[2],
					goldDenseFeature._wordRNNHiddenBuf[3], goldDenseFeature._wordRNNHiddenBuf[4], goldDenseFeature._wordRNNHiddenBuf[5],
					goldDenseFeature._wordRNNHidden, goldDenseFeature._wordRNNHiddenLoss, goldDenseFeature._wordHiddenLoss);
			_nnlayer_word_hidden.ComputeBackwardLoss(goldDenseFeature._wordUnitRep, goldDenseFeature._wordHidden, goldDenseFeature._wordHiddenLoss, goldDenseFeature._wordUnitRepLoss);



			for(int idx = 0; idx < goldDenseFeature._wordRepLoss.size(); idx++){
				goldDenseFeature._wordUnitRepLoss[idx] = goldDenseFeature._wordUnitRepLoss[idx] * goldDenseFeature._wordUnitRepMask[idx];
				unconcat(goldDenseFeature._wordRepLoss[idx], goldDenseFeature._allwordRepLoss[idx], goldDenseFeature._keyCharRepLoss[idx], goldDenseFeature._lengthRepLoss[idx], goldDenseFeature._wordUnitRepLoss[idx]);

				unconcat(goldDenseFeature._wordPrimeLoss[idx], goldDenseFeature._wordRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_words.EmbLoss(goldDenseFeature._wordIds[idx][tmp_k], goldDenseFeature._wordPrimeLoss[idx][tmp_k]);
				}

				unconcat(goldDenseFeature._keyCharPrimeLoss[idx], goldDenseFeature._keyCharRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < 2 * _wordNgram + 1; tmp_k++) {
					_keyChars.EmbLoss(goldDenseFeature._keyCharIds[idx][tmp_k], goldDenseFeature._keyCharPrimeLoss[idx][tmp_k]);
				}

				unconcat(goldDenseFeature._lengthPrimeLoss[idx], goldDenseFeature._lengthRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
					_lengths.EmbLoss(goldDenseFeature._lengthIds[idx][tmp_k], goldDenseFeature._lengthPrimeLoss[idx][tmp_k]);
				}
			}

			_action_increased_rnn.ComputeBackwardLoss(goldDenseFeature._actionHidden,
					goldDenseFeature._actionRNNHiddenBuf[0], goldDenseFeature._actionRNNHiddenBuf[1], goldDenseFeature._actionRNNHiddenBuf[2],
					goldDenseFeature._actionRNNHiddenBuf[3], goldDenseFeature._actionRNNHiddenBuf[4], goldDenseFeature._actionRNNHiddenBuf[5],
					goldDenseFeature._actionRNNHidden, goldDenseFeature._actionRNNHiddenLoss, goldDenseFeature._actionHiddenLoss);
			_nnlayer_action_hidden.ComputeBackwardLoss(goldDenseFeature._actionRep, goldDenseFeature._actionHidden, goldDenseFeature._actionHiddenLoss, goldDenseFeature._actionRepLoss);

			for(int idx = 0; idx < goldDenseFeature._actionRepLoss.size(); idx++){
				goldDenseFeature._actionRepLoss[idx] = goldDenseFeature._actionRepLoss[idx] * goldDenseFeature._actionRepMask[idx];

				unconcat(goldDenseFeature._actionPrimeLoss[idx], goldDenseFeature._actionRepLoss[idx]);
				for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
					_actions.EmbLoss(goldDenseFeature._actionIds[idx][tmp_k], goldDenseFeature._actionPrimeLoss[idx][tmp_k]);
				}
			}

			return;
		}

		if (pPredState != pGoldState) {
			//sparse
			//_splayer_output.ComputeBackwardLoss(pPredState->_curFeat._nSparseFeat, predLoss);
			//_splayer_output.ComputeBackwardLoss(pGoldState->_curFeat._nSparseFeat, goldLoss);

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
							charRightRNNHidden_Loss[position], pPredState->_nnfeat._sepInHiddenLoss);
				} else {
					unconcat(predDenseFeature._wordRNNHiddenLoss[word_position], predDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss,
							charRNNHiddenDummy_Loss, pPredState->_nnfeat._sepInHiddenLoss);
				}

			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pPredState->_nnfeat._appOutHidden, predLoss, pPredState->_nnfeat._appOutHiddenLoss);
				_nnlayer_app_hidden.ComputeBackwardLoss(pPredState->_nnfeat._appInHidden, pPredState->_nnfeat._appOutHidden, pPredState->_nnfeat._appOutHiddenLoss,
						pPredState->_nnfeat._appInHiddenLoss);

				unconcat(predDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position], charRightRNNHidden_Loss[position],
							pPredState->_nnfeat._appInHiddenLoss);
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
							charRightRNNHidden_Loss[position], pGoldState->_nnfeat._sepInHiddenLoss);
				} else {
					unconcat(goldDenseFeature._wordRNNHiddenLoss[word_position], goldDenseFeature._actionRNNHiddenLoss[position], charRNNHiddenDummy_Loss,
							charRNNHiddenDummy_Loss, pGoldState->_nnfeat._sepInHiddenLoss);
				}

			} else {
				_nnlayer_app_output.ComputeBackwardLoss(pGoldState->_nnfeat._appOutHidden, goldLoss, pGoldState->_nnfeat._appOutHiddenLoss);
				_nnlayer_app_hidden.ComputeBackwardLoss(pGoldState->_nnfeat._appInHidden, pGoldState->_nnfeat._appOutHidden, pGoldState->_nnfeat._appOutHiddenLoss,
						pGoldState->_nnfeat._appInHiddenLoss);

				unconcat(goldDenseFeature._actionRNNHiddenLoss[position], charLeftRNNHidden_Loss[position], charRightRNNHidden_Loss[position],
							pGoldState->_nnfeat._appInHiddenLoss);
			}

		}

		//predState
		word_position = pPredState->_wordnum - 1;
		if (pPredState->_lastAction._code == CAction::SEP || pPredState->_lastAction._code == CAction::FIN) {
			Copy(predDenseFeature._wordPrime[word_position], pPredState->_nnfeat._wordPrime);
			Copy(predDenseFeature._wordRep[word_position], pPredState->_nnfeat._wordRep);
			Copy(predDenseFeature._allwordPrime[word_position], pPredState->_nnfeat._allwordPrime);
			Copy(predDenseFeature._allwordRep[word_position], pPredState->_nnfeat._allwordRep);			
			Copy(predDenseFeature._keyCharPrime[word_position], pPredState->_nnfeat._keyCharPrime);
			Copy(predDenseFeature._keyCharRep[word_position], pPredState->_nnfeat._keyCharRep);
			Copy(predDenseFeature._lengthPrime[word_position], pPredState->_nnfeat._lengthPrime);
			Copy(predDenseFeature._lengthRep[word_position], pPredState->_nnfeat._lengthRep);
			Copy(predDenseFeature._wordUnitRep[word_position], pPredState->_nnfeat._wordUnitRep);
			Copy(predDenseFeature._wordHidden[word_position], pPredState->_nnfeat._wordHidden);
			for(int idk = 0; idk < _buffer; idk++){
				Copy(predDenseFeature._wordRNNHiddenBuf[idk][word_position], pPredState->_nnfeat._wordRNNHiddenBuf[idk]);
			}
			Copy(predDenseFeature._wordRNNHidden[word_position], pPredState->_nnfeat._wordRNNHidden);


			Copy(predDenseFeature._wordUnitRepMask[word_position], pPredState->_nnfeat._wordUnitRepMask);

			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				predDenseFeature._wordIds[word_position][tmp_k] = pPredState->_curFeat._nWordFeat[tmp_k];
			}
			for (int tmp_k = 0; tmp_k < 2 * _wordNgram + 1; tmp_k++) {
				predDenseFeature._keyCharIds[word_position][tmp_k] = pPredState->_curFeat._nKeyChars[tmp_k];
			}
			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				predDenseFeature._lengthIds[word_position][tmp_k] = pPredState->_curFeat._nWordLengths[tmp_k];
			}
		}

		Copy(predDenseFeature._actionPrime[position], pPredState->_nnfeat._actionPrime);
		Copy(predDenseFeature._actionRep[position], pPredState->_nnfeat._actionRep);
		Copy(predDenseFeature._actionRepMask[position], pPredState->_nnfeat._actionRepMask);
		Copy(predDenseFeature._actionHidden[position], pPredState->_nnfeat._actionHidden);
		for(int idk = 0; idk < _buffer; idk++){
			Copy(predDenseFeature._actionRNNHiddenBuf[idk][word_position], pPredState->_nnfeat._actionRNNHiddenBuf[idk]);
		}
		Copy(predDenseFeature._actionRNNHidden[position], pPredState->_nnfeat._actionRNNHidden);
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			predDenseFeature._actionIds[position][tmp_k] = pPredState->_curFeat._nActionFeat[tmp_k];
		}

		//goldState
		word_position = pGoldState->_wordnum - 1;
		if (pGoldState->_lastAction._code == CAction::SEP || pGoldState->_lastAction._code == CAction::FIN) {
			Copy(goldDenseFeature._wordPrime[word_position], pGoldState->_nnfeat._wordPrime);
			Copy(goldDenseFeature._wordRep[word_position], pGoldState->_nnfeat._wordRep);
			Copy(goldDenseFeature._allwordPrime[word_position], pGoldState->_nnfeat._allwordPrime);
			Copy(goldDenseFeature._allwordRep[word_position], pGoldState->_nnfeat._allwordRep);			
			Copy(goldDenseFeature._keyCharPrime[word_position], pGoldState->_nnfeat._keyCharPrime);
			Copy(goldDenseFeature._keyCharRep[word_position], pGoldState->_nnfeat._keyCharRep);
			Copy(goldDenseFeature._lengthPrime[word_position], pGoldState->_nnfeat._lengthPrime);
			Copy(goldDenseFeature._lengthRep[word_position], pGoldState->_nnfeat._lengthRep);
			Copy(goldDenseFeature._wordUnitRep[word_position], pGoldState->_nnfeat._wordUnitRep);
			Copy(goldDenseFeature._wordHidden[word_position], pGoldState->_nnfeat._wordHidden);
			for(int idk = 0; idk < _buffer; idk++){
				Copy(goldDenseFeature._wordRNNHiddenBuf[idk][word_position], pGoldState->_nnfeat._wordRNNHiddenBuf[idk]);
			}
			Copy(goldDenseFeature._wordRNNHidden[word_position], pGoldState->_nnfeat._wordRNNHidden);


			Copy(goldDenseFeature._wordUnitRepMask[word_position], pGoldState->_nnfeat._wordUnitRepMask);

			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				goldDenseFeature._wordIds[word_position][tmp_k] = pGoldState->_curFeat._nWordFeat[tmp_k];
			}
			for (int tmp_k = 0; tmp_k < 2 * _wordNgram + 1; tmp_k++) {
				goldDenseFeature._keyCharIds[word_position][tmp_k] = pGoldState->_curFeat._nKeyChars[tmp_k];
			}
			for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
				goldDenseFeature._lengthIds[word_position][tmp_k] = pGoldState->_curFeat._nWordLengths[tmp_k];
			}
		}

		Copy(goldDenseFeature._actionPrime[position], pGoldState->_nnfeat._actionPrime);
		Copy(goldDenseFeature._actionRep[position], pGoldState->_nnfeat._actionRep);
		Copy(goldDenseFeature._actionRepMask[position], pGoldState->_nnfeat._actionRepMask);
		Copy(goldDenseFeature._actionHidden[position], pGoldState->_nnfeat._actionHidden);
		for(int idk = 0; idk < _buffer; idk++){
			Copy(goldDenseFeature._actionRNNHiddenBuf[idk][word_position], pGoldState->_nnfeat._actionRNNHiddenBuf[idk]);
		}
		Copy(goldDenseFeature._actionRNNHidden[position], pGoldState->_nnfeat._actionRNNHidden);
		for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
			goldDenseFeature._actionIds[position][tmp_k] = pGoldState->_curFeat._nActionFeat[tmp_k];
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
		static DenseFeatureChar<xpu> charFeat;

		lattice_index[0] = lattice;
		lattice_index[1] = lattice + 1;
		lattice_index[0]->clear();
		lattice_index[0]->initSentence(&sentence);

		index = 0;

		/*
		 Add Character bi rnn  here
		 */
		charFeat.init(length, _charDim, _biCharDim, _charcontext, _charHiddenSize, _charRNNHiddenSize, _buffer);
		int unknownCharID = fe._charAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._charIds[idx] = fe._charAlphabet[sentence[idx]];
			if (charFeat._charIds[idx] < 0)
				charFeat._charIds[idx] = unknownCharID;
		}

		int unknownBiCharID = fe._bicharAlphabet[fe.unknownkey];
		for (int idx = 0; idx < length; idx++) {
			charFeat._bicharIds[idx] = idx < length - 1 ? fe._bicharAlphabet[sentence[idx] + sentence[idx + 1]] : fe._bicharAlphabet[sentence[idx] + fe.nullkey];
			if (charFeat._bicharIds[idx] < 0)
				charFeat._bicharIds[idx] = unknownBiCharID;
		}

		for (int idx = 0; idx < length; idx++) {
			//_chars.GetEmb(charFeat._charIds[idx], charFeat._charprime[idx]);
			charFeat._charprime[idx] = 0.0;
			_bichars.GetEmb(charFeat._bicharIds[idx], charFeat._bicharprime[idx]);
			concat(charFeat._charprime[idx], charFeat._bicharprime[idx], charFeat._charpre[idx]);
		}

		windowlized(charFeat._charpre, charFeat._charInput, _charcontext);

		_nnlayer_char_hidden.ComputeForwardScore(charFeat._charInput, charFeat._charHidden);
		_char_left_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charLeftRNNHiddenBuf[0], charFeat._charLeftRNNHiddenBuf[1], charFeat._charLeftRNNHiddenBuf[2],
				charFeat._charLeftRNNHiddenBuf[3], charFeat._charLeftRNNHiddenBuf[4], charFeat._charLeftRNNHiddenBuf[5], charFeat._charLeftRNNHidden);
		_char_right_rnn.ComputeForwardScore(charFeat._charHidden, charFeat._charRightRNNHiddenBuf[0], charFeat._charRightRNNHiddenBuf[1], charFeat._charRightRNNHiddenBuf[2],
				charFeat._charRightRNNHiddenBuf[3], charFeat._charRightRNNHiddenBuf[4], charFeat._charRightRNNHiddenBuf[5], charFeat._charRightRNNHidden);

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
					//scored_action.clear();
					scored_action.action = actions[tmp_j];
					scored_action.item = pGenerator;
					fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat, _wordNgram, _actionNgram);
					//_splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
					scored_action.score = 0;
					scored_action.score += pGenerator->_score;

					scored_action.nnfeat.init(_wordDim, _allwordDim, _charDim, _lengthDim, _wordNgram, _wordHiddenSize, _wordRNNHiddenSize, _actionDim, _actionNgram, _actionHiddenSize,
							_actionRNNHiddenSize, _sep_hiddenInSize, _app_hiddenInSize, _sep_hiddenOutSize, _app_hiddenOutSize, _buffer);

					//neural
					//action list
					for (int tmp_k = 0; tmp_k < _actionNgram; tmp_k++) {
						_actions.GetEmb(scored_action.feat._nActionFeat[tmp_k], scored_action.nnfeat._actionPrime[tmp_k]);
					}

					concat(scored_action.nnfeat._actionPrime, scored_action.nnfeat._actionRep);

					_nnlayer_action_hidden.ComputeForwardScore(scored_action.nnfeat._actionRep, scored_action.nnfeat._actionHidden);

					if (pGenerator->_nextPosition == 0) {
						_action_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
								scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
								scored_action.nnfeat._actionRNNHidden);
					} else {
						_action_increased_rnn.ComputeForwardScoreIncremental(pGenerator->_nnfeat._actionRNNHiddenBuf[4], pGenerator->_nnfeat._actionRNNHidden, scored_action.nnfeat._actionHidden,
								scored_action.nnfeat._actionRNNHiddenBuf[0], scored_action.nnfeat._actionRNNHiddenBuf[1], scored_action.nnfeat._actionRNNHiddenBuf[2],
								scored_action.nnfeat._actionRNNHiddenBuf[3], scored_action.nnfeat._actionRNNHiddenBuf[4], scored_action.nnfeat._actionRNNHiddenBuf[5],
								scored_action.nnfeat._actionRNNHidden);
					}

					//read word
					if (scored_action.action._code == CAction::SEP || scored_action.action._code == CAction::FIN) {
						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							_words.GetEmb(scored_action.feat._nWordFeat[tmp_k], scored_action.nnfeat._wordPrime[tmp_k], fe._wordAlphabet[fe.unknownkey]);
							_allwords.GetEmb(scored_action.feat._nAllWordFeat[tmp_k], scored_action.nnfeat._allwordPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._wordPrime, scored_action.nnfeat._wordRep);
						concat(scored_action.nnfeat._allwordPrime, scored_action.nnfeat._allwordRep);

						for (int tmp_k = 0; tmp_k < 2*_wordNgram+1; tmp_k++) {
							_keyChars.GetEmb(scored_action.feat._nKeyChars[tmp_k], scored_action.nnfeat._keyCharPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._keyCharPrime, scored_action.nnfeat._keyCharRep);

						for (int tmp_k = 0; tmp_k < _wordNgram; tmp_k++) {
							_lengths.GetEmb(scored_action.feat._nWordLengths[tmp_k], scored_action.nnfeat._lengthPrime[tmp_k]);
						}
						concat(scored_action.nnfeat._lengthPrime, scored_action.nnfeat._lengthRep);

						concat(scored_action.nnfeat._wordRep, scored_action.nnfeat._allwordRep, scored_action.nnfeat._keyCharRep, scored_action.nnfeat._lengthRep, scored_action.nnfeat._wordUnitRep);

						_nnlayer_word_hidden.ComputeForwardScore(scored_action.nnfeat._wordUnitRep, scored_action.nnfeat._wordHidden);

						const CStateItem<xpu> * preSepState = pGenerator->_prevSepState;
						if (preSepState == 0) {
							_word_increased_rnn.ComputeForwardScoreIncremental(scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
									scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
									scored_action.nnfeat._wordRNNHidden);
						} else {
							_word_increased_rnn.ComputeForwardScoreIncremental(preSepState->_nnfeat._wordRNNHiddenBuf[4], preSepState->_nnfeat._wordRNNHidden, scored_action.nnfeat._wordHidden,
									scored_action.nnfeat._wordRNNHiddenBuf[0], scored_action.nnfeat._wordRNNHiddenBuf[1], scored_action.nnfeat._wordRNNHiddenBuf[2],
									scored_action.nnfeat._wordRNNHiddenBuf[3], scored_action.nnfeat._wordRNNHiddenBuf[4], scored_action.nnfeat._wordRNNHiddenBuf[5],
									scored_action.nnfeat._wordRNNHidden);
						}

						//
						if (pGenerator->_nextPosition < length) {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition],
									charFeat._charRightRNNHidden[pGenerator->_nextPosition], scored_action.nnfeat._sepInHidden);
						} else {
							concat(scored_action.nnfeat._wordRNNHidden, scored_action.nnfeat._actionRNNHidden, charFeat._charRNNHiddenDummy, charFeat._charRNNHiddenDummy,
									scored_action.nnfeat._sepInHidden);
						}
						_nnlayer_sep_hidden.ComputeForwardScore(scored_action.nnfeat._sepInHidden, scored_action.nnfeat._sepOutHidden);
						_nnlayer_sep_output.ComputeForwardScore(scored_action.nnfeat._sepOutHidden, score);
					} else {
						concat(scored_action.nnfeat._actionRNNHidden, charFeat._charLeftRNNHidden[pGenerator->_nextPosition], charFeat._charRightRNNHidden[pGenerator->_nextPosition],
								scored_action.nnfeat._appInHidden);
						_nnlayer_app_hidden.ComputeForwardScore(scored_action.nnfeat._appInHidden, scored_action.nnfeat._appOutHidden);
						_nnlayer_app_output.ComputeForwardScore(scored_action.nnfeat._appOutHidden, score);
					}

					//std::cout << "score = " << score << std::endl;

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
				lattice_index[index + 1]->_nnfeat.copy(beam[tmp_j].nnfeat);

				if (pBestGen == 0 || lattice_index[index + 1]->_score > pBestGen->_score) {
					pBestGen = lattice_index[index + 1];
				}

				++lattice_index[index + 1];
			}

			if (pBestGen->IsTerminated())
				break; // while

		}
		pBestGen->getSegResults(words);

		charFeat.clear();

		return true;
	}

	void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps, dtype clip = -1.0) {
		if(clip > 0.0) {
			dtype norm = 0.0;
			//norm += _splayer_output.squarenormAll();
			norm += _nnlayer_sep_output.squarenormAll();
			norm += _nnlayer_app_output.squarenormAll();
			norm += _words.squarenormAll();
			norm += _lengths.squarenormAll();
			norm += _keyChars.squarenormAll();
			norm += _chars.squarenormAll();
			norm += _bichars.squarenormAll();
			norm += _actions.squarenormAll();
			norm += _char_left_rnn.squarenormAll();
			norm += _char_right_rnn.squarenormAll();
			norm += _word_increased_rnn.squarenormAll();
			norm += _action_increased_rnn.squarenormAll();
			norm += _nnlayer_sep_hidden.squarenormAll();
			norm += _nnlayer_app_hidden.squarenormAll();
			norm += _nnlayer_word_hidden.squarenormAll();
			norm += _nnlayer_char_hidden.squarenormAll();
			norm += _nnlayer_action_hidden.squarenormAll();
			
			if(norm > clip * clip){
				dtype scale = clip/sqrt(norm);
				//_splayer_output.scaleGrad(scale);
				_nnlayer_sep_output.scaleGrad(scale);
				_nnlayer_app_output.scaleGrad(scale);
				_words.scaleGrad(scale);
				_lengths.scaleGrad(scale);
				_keyChars.scaleGrad(scale);
				_chars.scaleGrad(scale);
				_bichars.scaleGrad(scale);
				_actions.scaleGrad(scale);
				_char_left_rnn.scaleGrad(scale);
				_char_right_rnn.scaleGrad(scale);
				_word_increased_rnn.scaleGrad(scale);
				_action_increased_rnn.scaleGrad(scale);
				_nnlayer_sep_hidden.scaleGrad(scale);
				_nnlayer_app_hidden.scaleGrad(scale);
				_nnlayer_word_hidden.scaleGrad(scale);
				_nnlayer_char_hidden.scaleGrad(scale);
				_nnlayer_action_hidden.scaleGrad(scale);
			}
		}
		
		//_splayer_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_nnlayer_sep_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_app_output.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_lengths.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_keyChars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_chars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_bichars.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_actions.updateAdaGrad(nnRegular, adaAlpha, adaEps);

		_char_left_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_char_right_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_word_increased_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_action_increased_rnn.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_sep_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_app_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_word_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_char_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
		_nnlayer_action_hidden.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

	inline void setOOVRatio(dtype oovRatio) {
		_oovRatio = oovRatio;
	}

	inline void setOOVFreq(dtype oovFreq) {
		_oovFreq = oovFreq;
	}

	inline void setWordFreq(hash_map<string, int> word_stat){
		hash_map<int, int> wordFreq;
		hash_map<string, int>::iterator word_iter;
		for (word_iter = word_stat.begin(); word_iter != word_stat.end(); word_iter++) {
			wordFreq[fe._wordAlphabet.from_string(word_iter->first)] = word_iter->second;
		}
		_words.setFrequency(wordFreq);
	}

};

#endif /* SRC_LSTMNUCBeamSearcher_H_ */
