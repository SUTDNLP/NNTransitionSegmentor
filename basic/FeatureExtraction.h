/*
 * FeatureExtraction.h
 *
 *  Created on: Oct 7, 2015
 *      Author: mszhang
 */

#ifndef BASIC_FEATUREEXTRACTION_H_
#define BASIC_FEATUREEXTRACTION_H_
#include "N3L.h"
#include "Action.h"
#include "State.h"
#include "Utf.h"

class FeatureExtraction {
public:
  std::string nullkey;
  std::string rootdepkey;
  std::string unknownkey;
  std::string paddingtag;
  std::string seperateKey;

public:
  Alphabet _featAlphabet;
  Alphabet _wordAlphabet;
  Alphabet _charAlphabet;
  Alphabet _bicharAlphabet;
  Alphabet _actionAlphabet;

public:
  bool _bStringFeat; // string-formated features or digit-formated features

public:
  FeatureExtraction() {
    nullkey = "-null-";
    unknownkey = "-unknown-";
    paddingtag = "-padding-";
    seperateKey = "#";

    _bStringFeat = true;
  }

  FeatureExtraction(bool bCollecting) {
    nullkey = "-null-";
    unknownkey = "-unknown-";
    paddingtag = "-padding-";
    seperateKey = "#";

    _bStringFeat = bCollecting;
  }

public:

  inline void setFeatureFormat(bool bStringFeat) {
    _bStringFeat = bStringFeat;
  }


  inline void setAlphaIncreasing(bool alphaIncreasing){
    if(alphaIncreasing){
      _featAlphabet.set_fixed_flag(false);
      _wordAlphabet.set_fixed_flag(false);
      _charAlphabet.set_fixed_flag(false);
      _bicharAlphabet.set_fixed_flag(false);
      _actionAlphabet.set_fixed_flag(false);
    }
    else{
      _featAlphabet.set_fixed_flag(true);
      _wordAlphabet.set_fixed_flag(true);
      _charAlphabet.set_fixed_flag(true);
      _bicharAlphabet.set_fixed_flag(true);
      _actionAlphabet.set_fixed_flag(true);
    }
  }

  inline int getCharAlphaId(const std::string & oneChar){
    return _charAlphabet[oneChar];
  }

  inline int getBiCharAlphaId(const std::string & twoChar){
    return _bicharAlphabet[twoChar];
  }


  void extractFeature(const CStateItem * curState, const CAction& nextAC, Feature& feat) {
    feat.clear();
    feat.setFeatureFormat(_bStringFeat);
    if (nextAC._code == CAction::APP) {
      extractFeatureApp(curState, feat);
    } else if (nextAC._code == CAction::SEP) {
      extractFeatureSep(curState, feat);
    } else if (nextAC._code == CAction::FIN) {
      extractFeatureFinish(curState, feat);
    } else {

    }

    static int featId;
    if(!_bStringFeat){
      for(int idx = 0; idx < feat._strSparseFeat.size(); idx++){
        featId = _featAlphabet[feat._strSparseFeat[idx]];
        if(featId >= 0) feat._nSparseFeat.push_back(featId);
      }
      feat._strSparseFeat.clear();
    }
  }

  void addToFeatureAlphabet(hash_map<string, int> feat_stat, int featCutOff = 0) {
    _featAlphabet.set_fixed_flag(false);
    hash_map<string, int>::iterator feat_iter;
    for (feat_iter = feat_stat.begin(); feat_iter != feat_stat.end(); feat_iter++) {
      if (feat_iter->second > featCutOff) {
        _featAlphabet.from_string(feat_iter->first);
      }
    }
    _featAlphabet.set_fixed_flag(true);
  }

  void addToWordAlphabet(hash_map<string, int> word_stat, int wordCutOff = 0) {
    _wordAlphabet.set_fixed_flag(false);
    hash_map<string, int>::iterator word_iter;
    for (word_iter = word_stat.begin(); word_iter != word_stat.end(); word_iter++) {
      if (word_iter->second > wordCutOff) {
        _wordAlphabet.from_string(word_iter->first);
      }
    }
    _wordAlphabet.set_fixed_flag(true);
  }

  void addToCharAlphabet(hash_map<string, int> char_stat, int charCutOff = 0) {
    _charAlphabet.set_fixed_flag(false);
    hash_map<string, int>::iterator char_iter;
    for (char_iter = char_stat.begin(); char_iter != char_stat.end(); char_iter++) {
      if (char_iter->second > charCutOff) {
        _charAlphabet.from_string(char_iter->first);
      }
    }
    _charAlphabet.set_fixed_flag(true);
  }

  void addToBiCharAlphabet(hash_map<string, int> bichar_stat, int bicharCutOff = 0) {
    _bicharAlphabet.set_fixed_flag(false);
    hash_map<string, int>::iterator bichar_iter;
    for (bichar_iter = bichar_stat.begin(); bichar_iter != bichar_stat.end(); bichar_iter++) {
      if (bichar_iter->second > bicharCutOff) {
        _bicharAlphabet.from_string(bichar_iter->first);
      }
    }
    _bicharAlphabet.set_fixed_flag(true);
  }

  void addToActionAlphabet(hash_map<string, int> action_stat) {
    _actionAlphabet.set_fixed_flag(false);
    hash_map<string, int>::iterator action_iter;
    for (action_iter = action_stat.begin(); action_iter != action_stat.end(); action_iter++) {
      _actionAlphabet.from_string(action_iter->first);
    }
    _actionAlphabet.set_fixed_flag(true);
  }

  void initAlphabet(){
    //alphabet initialization
    _featAlphabet.clear();
    _featAlphabet.set_fixed_flag(true);

    _wordAlphabet.clear();
    _wordAlphabet.from_string(nullkey);
    _wordAlphabet.from_string(unknownkey);
    _wordAlphabet.set_fixed_flag(true);

    _charAlphabet.clear();
    _charAlphabet.from_string(nullkey);
    _charAlphabet.from_string(unknownkey);
    _charAlphabet.set_fixed_flag(true);

    _bicharAlphabet.clear();
    _bicharAlphabet.from_string(nullkey);
    _bicharAlphabet.from_string(unknownkey);
    _bicharAlphabet.set_fixed_flag(true);

    _actionAlphabet.clear();
    _actionAlphabet.set_fixed_flag(true);

    _bStringFeat = true;
  }

  void loadAlphabet(){
    _bStringFeat = false;
  }

protected:
  void extractFeatureApp(const CStateItem * curState, Feature& feat) {
    string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strlastWord;
    const std::vector<std::string> * pCharacters = curState->_pCharacters;
    string nextChar = pCharacters->at(curState->_nextPosition);
    string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
    string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd-1);
    string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

    string curWordLastCharType = curState->_lastWordEnd == -1 ? nullkey : wordtype(curWordLastChar);
    string curWordLast2CharType = curState->_lastWordEnd < 1 ? nullkey : wordtype(curWordLast2Char);
    string nextCharType = wordtype(nextChar);

    if(curState->_nextPosition != curState->_lastWordEnd+1){
      std::cout << "position error" << std::endl;
    }

    string strFeat = "";
    strFeat = "F01" + seperateKey + curWordLastChar + seperateKey + nextChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F02" + seperateKey + curWordFirstChar + seperateKey + nextChar;
    feat._strSparseFeat.push_back(strFeat);

    /*
    strFeat = "F03" + seperateKey + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F04" + seperateKey + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

     */
    strFeat = "F05" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

  }

  void extractFeatureSep(const CStateItem * curState, Feature& feat) {
    string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strlastWord;
    const std::vector<std::string> * pCharacters = curState->_pCharacters;
    string nextChar = pCharacters->at(curState->_nextPosition);
    string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
    string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd-1);
    string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

    string curWordLastCharType = curState->_lastWordEnd == -1 ? nullkey : wordtype(curWordLastChar);
    string curWordLast2CharType = curState->_lastWordEnd < 1 ? nullkey : wordtype(curWordLast2Char);
    string nextCharType = wordtype(nextChar);


    int length = curState->_lastWordEnd - curState->_lastWordStart + 1;
    if (length > 5) length = 5;
    stringstream curss;
    curss << length;
    string strCurWordLen = curss.str();

    if(curState->_nextPosition != curState->_lastWordEnd+1){
      std::cout << "position error" << std::endl;
    }

    const CStateItem * preStackState = curState->_prevStackState;
    string pre1Word = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strlastWord;
    string pre1WordLastChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordEnd);
    string pre1WordFirstChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordStart);

    length = preStackState == 0 ? 0 : preStackState->_lastWordEnd - preStackState->_lastWordStart + 1;
    if (length > 5) length = 5;
    stringstream press;
    press << length;
    string strPreWordLen = press.str();

    string strFeat = "";
    strFeat = "F11" + seperateKey + curWordLastChar + seperateKey + nextChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F12" + seperateKey + curWordFirstChar + seperateKey + nextChar;
    feat._strSparseFeat.push_back(strFeat);

    /*
    strFeat = "F13" + seperateKey + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F14" + seperateKey + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);
    */
    strFeat = "F15" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);


    strFeat = "F16" + seperateKey + curWord + seperateKey + nextChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F17" + seperateKey + curWord;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F18" + seperateKey + curWord + seperateKey + pre1Word;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F19" + seperateKey + curWord + seperateKey + pre1WordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F20" + seperateKey + curWord + seperateKey + pre1WordFirstChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F21" + seperateKey + curWordLastChar + seperateKey + pre1WordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F22" + seperateKey + curWordFirstChar + seperateKey + curWordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F23" + seperateKey + curWord + seperateKey + strPreWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F24" + seperateKey + pre1Word + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F25" + seperateKey + pre1Word + seperateKey + curWordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    if(curState->_lastWordStart != -1){
      for(int idx = curState->_lastWordStart; idx <  curState->_lastWordEnd; idx++){
        strFeat = "F26" + seperateKey + pCharacters->at(idx) + seperateKey + curWordLastChar;
        feat._strSparseFeat.push_back(strFeat);
      }
    }

    if(curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1){
      strFeat = "F27" + seperateKey + curWord;
      feat._strSparseFeat.push_back(strFeat);
    }

    strFeat = "F28" + seperateKey + curWordFirstChar + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F29" + seperateKey + curWordLastChar + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);


  }

  void extractFeatureFinish(const CStateItem * curState, Feature& feat) {
    string curWord = curState->_lastWordEnd == -1 ? nullkey : curState->_strlastWord;
    const std::vector<std::string> * pCharacters = curState->_pCharacters;
    //string nextChar = pCharacters->at(curState->_nextPosition);
    string curWordLastChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordEnd);
    string curWordLast2Char = curState->_lastWordEnd < 1 ? nullkey : pCharacters->at(curState->_lastWordEnd-1);
    string curWordFirstChar = curState->_lastWordEnd == -1 ? nullkey : pCharacters->at(curState->_lastWordStart);

    int length = curState->_lastWordEnd - curState->_lastWordStart + 1;
    if (length > 5) length = 5;
    stringstream curss;
    curss << length;
    string strCurWordLen = curss.str();

    if(curState->_nextPosition != curState->_lastWordEnd+1){
      std::cout << "position error" << std::endl;
    }

    const CStateItem * preStackState = curState->_prevStackState;
    string pre1Word = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : preStackState->_strlastWord;
    string pre1WordLastChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordEnd);
    string pre1WordFirstChar = preStackState == 0 || preStackState->_lastWordEnd == -1 ? nullkey : pCharacters->at(preStackState->_lastWordStart);

    length = preStackState == 0 ? 0 : preStackState->_lastWordEnd - preStackState->_lastWordStart + 1;
    if (length > 5) length = 5;
    stringstream press;
    press << length;
    string strPreWordLen = press.str();

    string strFeat = "";
    /*strFeat = "F11" + seperateKey + curWordLastChar + seperateKey + nullkey;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F12" + seperateKey + curWordFirstChar + seperateKey + nullkey;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F13" + seperateKey + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F14" + seperateKey + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F15" + seperateKey + curWordLast2CharType + curWordLastCharType + nextCharType;
    feat._strSparseFeat.push_back(strFeat);
    */

    strFeat = "F16" + seperateKey + curWord + seperateKey + nullkey;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F17" + seperateKey + curWord;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F18" + seperateKey + curWord + seperateKey + pre1Word;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F19" + seperateKey + curWord + seperateKey + pre1WordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F20" + seperateKey + curWord + seperateKey + pre1WordFirstChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F21" + seperateKey + curWordLastChar + seperateKey + pre1WordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F22" + seperateKey + curWordFirstChar + seperateKey + curWordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F23" + seperateKey + curWord + seperateKey + strPreWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F24" + seperateKey + pre1Word + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F25" + seperateKey + pre1Word + seperateKey + curWordLastChar;
    feat._strSparseFeat.push_back(strFeat);

    if(curState->_lastWordStart != -1){
      for(int idx = curState->_lastWordStart; idx <  curState->_lastWordEnd; idx++){
        strFeat = "F26" + seperateKey + pCharacters->at(idx) + seperateKey + curWordLastChar;
        feat._strSparseFeat.push_back(strFeat);
      }
    }

    if(curState->_lastWordEnd == curState->_lastWordStart && curState->_lastWordStart != -1){
      strFeat = "F27" + seperateKey + curWord;
      feat._strSparseFeat.push_back(strFeat);
    }

    strFeat = "F28" + seperateKey + curWordFirstChar + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);

    strFeat = "F29" + seperateKey + curWordLastChar + seperateKey + strCurWordLen;
    feat._strSparseFeat.push_back(strFeat);


  }

};

#endif /* BASIC_FEATUREEXTRACTION_H_ */
