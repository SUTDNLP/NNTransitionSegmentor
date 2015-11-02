/*
 * ScoredStateAction.h
 *
 *  Created on: Oct 6, 2015
 *      Author: mszhang
 */

#ifndef BASIC_SCOREDSTATEACTION_H_
#define BASIC_SCOREDSTATEACTION_H_

#include "State.h"
#include "Feature.h"

/*===============================================================
 *
 * scored actions
 *
 *==============================================================*/

class CScoredStateAction {
public:
  CAction action;
  const CStateItem *item;
  dtype score;
  Feature feat;

public:
  CScoredStateAction() :
      item(0), action(-1), score(0) {
    feat.setFeatureFormat(false);
    feat.clear();
  }
//  void load(const CAction &action, const CStateItem *item, const dtype &score) {
//    this->action = action;
//    this->item = item;
//    this->score = score + item->_score;
//    feat.setCollection(false);
//    feat.clear();
//  }

 // CScoredStateAction& operator=(const CScoredStateAction& other){
 //   this->action = other.action;
 //   this->score = other.score;
 //   this->item = other.item;
 //   this->feat.copy(other.feat);
 // }

public:
  bool operator <(const CScoredStateAction &a1) const {
    return score < a1.score;
  }
  bool operator >(const CScoredStateAction &a1) const {
    return score > a1.score;
  }
  bool operator <=(const CScoredStateAction &a1) const {
    return score <= a1.score;
  }
  bool operator >=(const CScoredStateAction &a1) const {
    return score >= a1.score;
  }



};

class CScoredStateAction_Compare {
public:
  int operator()(const CScoredStateAction &o1, const CScoredStateAction &o2) const {

    if (o1.score < o2.score)
      return -1;
    else if (o1.score > o2.score)
      return 1;
    else
      return 0;
  }
};

#endif /* BASIC_SCOREDSTATEACTION_H_ */
