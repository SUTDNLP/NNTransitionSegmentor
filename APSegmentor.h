/*
 * Segmentor.h
 *
 *  Created on: Mar 25, 2015
 *      Author: mszhang
 */

#ifndef SRC_PARSER_H_
#define SRC_PARSER_H_

#include "N3L.h"

#include "APBeamSearcher.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;


class Segmentor {
public:
  std::string nullkey;
  std::string rootdepkey;
  std::string unknownkey;
  std::string paddingtag;
  std::string seperateKey;

public:
  Segmentor();
  virtual ~Segmentor();

public:

#if USE_CUDA==1
  APBeamSearcher<gpu> m_classifier;
#else
  APBeamSearcher<cpu> m_classifier;
#endif

  Options m_options;

  Pipe m_pipe;

public:
  void readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb);

  void readWordClusters(const string& inFile);

  int createAlphabet(const vector<Instance>& vecInsts);

  int addTestWordAlpha(const vector<Instance>& vecInsts);

public:
  void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
      const string& wordEmbFile);
  void predict(const Instance& input, vector<string>& output);
  void test(const string& testFile, const string& outputFile, const string& modelFile);

  // static training
  void getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);


public:

  void proceedOneStepForDecode(const Instance& inputTree, CStateItem& state, int& outlab); //may be merged with train in the future

  void writeModelFile(const string& outputModelFile);
  void loadModelFile(const string& inputModelFile);

public:
  inline void getCandidateActions(const CStateItem &item, vector<CAction>& actions) {

  }



};

#endif /* SRC_PARSER_H_ */
