/*
 * Feature.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;

class Feature {

public:
  vector<int> _nSparseFeat;
  //vector<int> _nCharFeat;
  //vector<int> _nBiCharFeat;
  vector<int> _nWordFeat;
  vector<int> _nActionFeat;

  vector<string> _strSparseFeat;
  //vector<string> _strCharFeat;
  //vector<string> _strBiCharFeat;
  vector<string> _strWordFeat;
  vector<string> _strActionFeat;

  bool _bStringFeat;

public:
  Feature() {
    _bStringFeat = false;
    clear();
  }

  Feature(bool bCollecting) {
    _bStringFeat = bCollecting;
    clear();
  }

  /*virtual ~Feature()
   {

   }*/

  void setFeatureFormat(bool bStringFeat) {
    _bStringFeat = bStringFeat;
  }

  void copy(const Feature& other) {
    clear();
    if (other._bStringFeat) {
      for (int idx = 0; idx < other._strSparseFeat.size(); idx++) {
        _strSparseFeat.push_back(other._strSparseFeat[idx]);
      }
      //for (int idx = 0; idx < other._strCharFeat.size(); idx++) {
      //  _strCharFeat.push_back(other._strCharFeat[idx]);
      //}
      //for (int idx = 0; idx < other._strBiCharFeat.size(); idx++) {
      //  _strBiCharFeat.push_back(other._strBiCharFeat[idx]);
      //}
      for (int idx = 0; idx < other._strWordFeat.size(); idx++) {
        _strWordFeat.push_back(other._strWordFeat[idx]);
      }
      for (int idx = 0; idx < other._strActionFeat.size(); idx++) {
        _strActionFeat.push_back(other._strActionFeat[idx]);
      }
    } else {
      for (int idx = 0; idx < other._nSparseFeat.size(); idx++) {
        _nSparseFeat.push_back(other._nSparseFeat[idx]);
      }

      //for (int idx = 0; idx < other._nCharFeat.size(); idx++) {
      //  _nCharFeat.push_back(other._nCharFeat[idx]);
      //}
      //for (int idx = 0; idx < other._nBiCharFeat.size(); idx++) {
      //  _nBiCharFeat.push_back(other._nBiCharFeat[idx]);
      //}
      for (int idx = 0; idx < other._nWordFeat.size(); idx++) {
        _nWordFeat.push_back(other._nWordFeat[idx]);
      }
      for (int idx = 0; idx < other._nActionFeat.size(); idx++) {
        _nActionFeat.push_back(other._nActionFeat[idx]);
      }
    }
  }

  void clear() {
    _nSparseFeat.clear();
    //_nCharFeat.clear();
    //_nBiCharFeat.clear();
    _nWordFeat.clear();
    _nActionFeat.clear();

    _strSparseFeat.clear();
    //_strCharFeat.clear();
    //_strBiCharFeat.clear();
    _strWordFeat.clear();
    _strActionFeat.clear();
  }
};

#endif /* SRC_FEATURE_H_ */
