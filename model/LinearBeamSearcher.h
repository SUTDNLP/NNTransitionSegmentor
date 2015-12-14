/*
 * LinearBeamSearcher.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_LinearBeamSearcher_H_
#define SRC_LinearBeamSearcher_H_

#include <hash_set>
#include <iostream>

#include <assert.h>
#include "Feature.h"
#include "FeatureExtraction.h"
#include "N3L.h"
#include "State.h"
#include "Action.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//re-implementation of Yue and Clark ACL (2007)
template<typename xpu>
class LinearBeamSearcher {
public:
  LinearBeamSearcher() {
    _dropOut = 0.5;
    _delta = 0.2;
  }
  ~LinearBeamSearcher() {
  }

public:
  SparseUniLayer1O<xpu> _splayer_output;

  FeatureExtraction fe;

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

  inline void extractFeature(const CStateItem * curState, const CAction& nextAC, Feature& feat) {
    fe.extractFeature(curState, nextAC, feat);
  }

public:

  inline void init() {
    _linearfeatSize = 3*fe._featAlphabet.size();

    _splayer_output.initial(_linearfeatSize, 10);
  }

  inline void release() {
    _splayer_output.release();
  }

  dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
    fe.setFeatureFormat(false);
    setAlphaIncreasing(true);
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
    static CStateItem lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
    static CStateItem * lattice_index[MAX_SENTENCE_SIZE + 1];

    int length = sentence.size();
    dtype cost = 0.0;
    dtype score = 0.0;

    const static CStateItem *pGenerator;
    const static CStateItem *pBestGen;
    static CStateItem *correctState;

    bool bCorrect;  // used in learning for early update
    int index, tmp_i, tmp_j;
    CAction correct_action;
    bool correct_action_scored;
    std::vector<CAction> actions; // actions to apply for a candidate
    static NRHeap<CScoredStateAction, CScoredStateAction_Compare> beam(BEAM_SIZE);
    static CScoredStateAction scored_action; // used rank actions
    static CScoredStateAction scored_correct_action;

    lattice_index[0] = lattice;
    lattice_index[1] = lattice + 1;
    lattice_index[0]->clear();
    lattice_index[0]->initSentence(&sentence);

    index = 0;

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
          fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat);
          _splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
          //std::cout << "add start, action = " << actions[tmp_j] << ", cur ac score = " << scored_action.score << ", orgin score = " << pGenerator->_score << std::endl;;
          scored_action.score += pGenerator->_score;
          if(actions[tmp_j] != correct_action){
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

        ++lattice_index[index + 1];
        assert(correct_action_scored); // scored_correct_act valid
        //TRACE(index << " updated");
        //std::cout << index << " updated" << std::endl;

        cost = backPropagationStates(pBestGen, correctState, 1.0, -1.0);
        if (cost < 0) {
          std::cout << "strange ..." << std::endl;
        }
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
        assert(correct_action_scored); // scored_correct_act valid
      }

      //std::cout << "best:" << pBestGen->str() << std::endl;
      //std::cout << "gold:" << correctState->str() << std::endl;

      cost = backPropagationStates(pBestGen, correctState, 1.0, -1.0);
      if (cost < 0) {
        std::cout << "strange ..." << std::endl;
      }
      _eval.correct_label_count += length;
      _eval.overall_label_count += length + 1;
    } else {
      _eval.correct_label_count += length + 1;
      _eval.overall_label_count += length + 1;
    }

    return cost;
  }

  dtype backPropagationStates(const CStateItem *pPredState, const CStateItem *pGoldState, dtype predLoss, dtype goldLoss) {
    if (pPredState == pGoldState)
      return 0.0;

    if(pPredState->_nextPosition != pGoldState->_nextPosition){
      std::cout << "state align error" << std::endl;
    }
    dtype delta = 0.0;
    dtype predscore, goldscore;
    _splayer_output.ComputeForwardScore(pPredState->_curFeat._nSparseFeat, predscore);
    _splayer_output.ComputeForwardScore(pGoldState->_curFeat._nSparseFeat, goldscore);

    delta = predscore - goldscore;
    if(pPredState->_lastAction != pGoldState->_lastAction){
      delta += _delta;
    }

    _splayer_output.ComputeBackwardLoss(pPredState->_curFeat._nSparseFeat, predLoss);
    _splayer_output.ComputeBackwardLoss(pGoldState->_curFeat._nSparseFeat, goldLoss);

    //currently we use a uniform loss
    delta += backPropagationStates(pPredState->_prevState, pGoldState->_prevState, predLoss, goldLoss);

    dtype compare_delta = pPredState->_score - pGoldState->_score;
    if (abs(delta - compare_delta) > 0.01) {
      std::cout << "delta=" << delta << "\t, compare_delta=" << compare_delta << std::endl;
    }

    return delta;
  }

  bool decode(const std::vector<string>& sentence, std::vector<std::string>& words) {
    setAlphaIncreasing(false);
    if (sentence.size() >= MAX_SENTENCE_SIZE)
      return false;
    static CStateItem lattice[(MAX_SENTENCE_SIZE + 1) * (BEAM_SIZE + 1)];
    static CStateItem *lattice_index[MAX_SENTENCE_SIZE + 1];

    int length = sentence.size();
    dtype cost = 0.0;
    dtype score = 0.0;

    const static CStateItem *pGenerator;
    const static CStateItem *pBestGen;

    int index, tmp_i, tmp_j;
    std::vector<CAction> actions; // actions to apply for a candidate
    static NRHeap<CScoredStateAction, CScoredStateAction_Compare> beam(BEAM_SIZE);
    static CScoredStateAction scored_action; // used rank actions
    static Feature feat;

    lattice_index[0] = lattice;
    lattice_index[1] = lattice + 1;
    lattice_index[0]->clear();
    lattice_index[0]->initSentence(&sentence);

    index = 0;

    while (true) {
      ++index;
      lattice_index[index + 1] = lattice_index[index];
      beam.clear();
      pBestGen = 0;

      //std::cout << index << std::endl;
      for (pGenerator = lattice_index[index - 1]; pGenerator != lattice_index[index]; ++pGenerator) {
        pGenerator->getCandidateActions(actions);
        for (tmp_j = 0; tmp_j < actions.size(); ++tmp_j) {
          scored_action.action = actions[tmp_j];
          scored_action.item = pGenerator;
          fe.extractFeature(pGenerator, actions[tmp_j], scored_action.feat);
          _splayer_output.ComputeForwardScore(scored_action.feat._nSparseFeat, scored_action.score);
          scored_action.score += pGenerator->_score;
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

};

#endif /* SRC_LinearBeamSearcher_H_ */
