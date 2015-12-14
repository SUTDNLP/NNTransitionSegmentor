/*
 * Segmentor.cpp
 *
 *  Created on: Oct 23, 2015
 *      Author: mszhang
 */

#include "APSegmentor.h"

#include "Argument_helper.h"

Segmentor::Segmentor() {
  // TODO Auto-generated constructor stub
  nullkey = "-null-";
  unknownkey = "-unknown-";
  paddingtag = "-padding-";
  seperateKey = "#";
}

Segmentor::~Segmentor() {
  // TODO Auto-generated destructor stub
}

// all linear features are extracted from positive examples
int Segmentor::createAlphabet(const vector<Instance>& vecInsts) {
  cout << "Creating Alphabet..." << endl;

  int numInstance = vecInsts.size();

  hash_map<string, int> word_stat;
  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;
  hash_map<string, int> action_stat;
  hash_map<string, int> feat_stat;

  assert(numInstance > 0);

  static Metric eval;
  static CStateItem state[m_classifier.MAX_SENTENCE_SIZE];
  static Feature feat;
  static vector<string> output;
  static CAction answer;
  static int actionNum;
  m_classifier.initAlphabet();
  eval.reset();
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];
    for (int idx = 0; idx < instance.wordsize(); idx++) {
      word_stat[normalize_to_lowerwithdigit(instance.words[idx])]++;
    }
    for (int idx = 0; idx < instance.charsize(); idx++) {
      char_stat[instance.chars[idx]]++;
    }
    for (int idx = 0; idx < instance.charsize() - 1; idx++) {
      bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
    }
    bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
    bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;
    actionNum = 0;
    state[actionNum].initSentence(&instance.chars);
    state[actionNum].clear();

    while (!state[actionNum].IsTerminated()) {
      state[actionNum].getGoldAction(instance.words, answer);
      action_stat[answer.str()]++;

      m_classifier.extractFeature(state+actionNum, answer, feat);
      for (int idx = 0; idx < feat._strSparseFeat.size(); idx++) {
        feat_stat[feat._strSparseFeat[idx]]++;
      }
      state[actionNum].move(state+actionNum+1, answer);
      actionNum++;
    }

    if(actionNum-1 != instance.charsize()) {
      std::cout << "action number is not correct, please check" << std::endl;
    }
    state[actionNum].getSegResults(output);

    instance.evaluate(output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  m_classifier.addToActionAlphabet(action_stat);
  m_classifier.addToWordAlphabet(word_stat, m_options.wordEmbFineTune ? m_options.wordCutOff : 0);
  m_classifier.addToCharAlphabet(char_stat, m_options.charEmbFineTune ? m_options.charCutOff : 0);
  m_classifier.addToBiCharAlphabet(bichar_stat, m_options.bicharEmbFineTune ? m_options.bicharCutOff : 0);
  m_classifier.addToFeatureAlphabet(feat_stat, m_options.featCutOff);

  cout << numInstance << " " << endl;
  cout << "Action num: " << m_classifier.fe._actionAlphabet.size() << endl;
  cout << "Total word num: " << word_stat.size() << endl;
  cout << "Total char num: " << char_stat.size() << endl;
  cout << "Total bichar num: " << bichar_stat.size() << endl;
  cout << "Total feat num: " << feat_stat.size() << endl;

  cout << "Remain word num: " << m_classifier.fe._wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
  cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;
  cout << "Remain feat num: " << m_classifier.fe._featAlphabet.size() << endl;

  //m_classifier.setFeatureCollectionState(false);

  return 0;
}

int Segmentor::addTestWordAlpha(const vector<Instance>& vecInsts) {
  cout << "Add test Alphabet..." << endl;

  hash_map<string, int> word_stat;
  hash_map<string, int> char_stat;
  hash_map<string, int> bichar_stat;
  int numInstance;

  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];

    for (int idx = 0; idx < instance.wordsize(); idx++) {
      word_stat[normalize_to_lowerwithdigit(instance.words[idx])]++;
    }
    for (int idx = 0; idx < instance.charsize(); idx++) {
      char_stat[instance.chars[idx]]++;
    }
    for (int idx = 0; idx < instance.charsize() - 1; idx++) {
      bichar_stat[instance.chars[idx] + instance.chars[idx + 1]]++;
    }
    bichar_stat[instance.chars[instance.charsize() - 1] + m_classifier.fe.nullkey]++;
    bichar_stat[m_classifier.fe.nullkey + instance.chars[0]]++;

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }

  if (!m_options.wordEmbFineTune)
    m_classifier.addToWordAlphabet(word_stat, 0);
  if (!m_options.charEmbFineTune)
    m_classifier.addToCharAlphabet(char_stat, 0);
  if (!m_options.bicharEmbFineTune)
    m_classifier.addToBiCharAlphabet(bichar_stat, 0);

  cout << "Remain word num: " << m_classifier.fe._wordAlphabet.size() << endl;
  cout << "Remain char num: " << m_classifier.fe._charAlphabet.size() << endl;
  cout << "Remain bichar num: " << m_classifier.fe._bicharAlphabet.size() << endl;

  return 0;
}


void Segmentor::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions){
  vecActions.clear();

  static Metric eval;
  static CStateItem state[m_classifier.MAX_SENTENCE_SIZE];
  static vector<string> output;
  static CAction answer;
  eval.reset();
  static int numInstance, actionNum;
  vecActions.resize(vecInsts.size());
  for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
    const Instance &instance = vecInsts[numInstance];

    actionNum = 0;
    state[actionNum].initSentence(&instance.chars);
    state[actionNum].clear();

    while (!state[actionNum].IsTerminated()) {
      state[actionNum].getGoldAction(instance.words, answer);
      vecActions[numInstance].push_back(answer);
      state[actionNum].move(state+actionNum+1, answer);
      actionNum++;
    }

    if(actionNum-1 != instance.charsize()) {
      std::cout << "action number is not correct, please check" << std::endl;
    }
    state[actionNum].getSegResults(output);

    instance.evaluate(output, eval);

    if (!eval.bIdentical()) {
      std::cout << "error state conversion!" << std::endl;
      exit(0);
    }

    if ((numInstance + 1) % m_options.verboseIter == 0) {
      cout << numInstance + 1 << " ";
      if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
        cout << std::endl;
      cout.flush();
    }
    if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
      break;
  }
}

void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile,
    const string& wordEmbFile) {
  if (optionFile != "")
    m_options.load(optionFile);

  m_options.showOptions();
  vector<Instance> trainInsts, devInsts, testInsts;
  m_pipe.readInstances(trainFile, trainInsts, m_classifier.MAX_SENTENCE_SIZE, m_options.maxInstance);
  if (devFile != "")
    m_pipe.readInstances(devFile, devInsts, m_classifier.MAX_SENTENCE_SIZE, m_options.maxInstance);
  if (testFile != "")
    m_pipe.readInstances(testFile, testInsts, m_classifier.MAX_SENTENCE_SIZE, m_options.maxInstance);

  vector<vector<Instance> > otherInsts(m_options.testFiles.size());
  for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
    m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_classifier.MAX_SENTENCE_SIZE, m_options.maxInstance);
  }

  createAlphabet(trainInsts);

  addTestWordAlpha(devInsts);
  addTestWordAlpha(testInsts);
  for (int idx = 0; idx < otherInsts.size(); idx++) {
    addTestWordAlpha(otherInsts[idx]);
  }


  m_classifier.init();
  m_classifier.setDropValue(m_options.dropProb);

  vector<vector<CAction> > trainInstGoldactions;
  getGoldActions(trainInsts, trainInstGoldactions);
  double bestFmeasure = 0;

  int inputSize = trainInsts.size();

  std::vector<int> indexes;
  for (int i = 0; i < inputSize; ++i)
    indexes.push_back(i);

  static Metric eval, metric_dev, metric_test;

  int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
  int oneIterMaxRound = (inputSize + m_options.batchSize -1) / m_options.batchSize;
  std::cout << "maxIter = " << maxIter << std::endl;
  int devNum = devInsts.size(), testNum = testInsts.size();

  static vector<vector<string> > decodeInstResults;
  static vector<string> curDecodeInst;
  static bool bCurIterBetter;
  static vector<vector<string> > subInstances;
  static vector<vector<CAction> > subInstGoldActions;

  //m_classifier.setAlphaIncreasing(true);
  for (int iter = 0; iter < maxIter; ++iter) {
    std::cout << "##### Iteration " << iter << std::endl;
    srand(iter);
    //random_shuffle(indexes.begin(), indexes.end());
    std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
    eval.reset();
    for (int updateIter = 0; updateIter < oneIterMaxRound; updateIter++) {
      int start_pos = updateIter * m_options.batchSize;
      int end_pos = (updateIter + 1) * m_options.batchSize;
      if (end_pos > inputSize)
      end_pos = inputSize;
      subInstances.clear();
      subInstGoldActions.clear();
      for (int idy = start_pos; idy < end_pos; idy++) {
        subInstances.push_back(trainInsts[indexes[idy]].chars);
        subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
      }

      double cost = m_classifier.train(subInstances, subInstGoldActions);

      eval.overall_label_count += m_classifier._eval.overall_label_count;
      eval.correct_label_count += m_classifier._eval.correct_label_count;

      //if ((updateIter + 1) % (m_options.verboseIter*10) == 0) {
        //std::cout << "current: " << updateIter + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
      //}
      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
    }

    std::cout << "current: " << iter + 1  << ", Correct(%) = " << eval.getAccuracy() << std::endl;

    if (devNum > 0) {
      bCurIterBetter = false;
      if (!m_options.outBest.empty())
        decodeInstResults.clear();
      metric_dev.reset();
      for (int idx = 0; idx < devInsts.size(); idx++) {
        predict(devInsts[idx], curDecodeInst);
        devInsts[idx].evaluate(curDecodeInst, metric_dev);
        if (!m_options.outBest.empty()) {
          decodeInstResults.push_back(curDecodeInst);
        }
      }
      std::cout << "dev:" << std::endl;
      metric_dev.print();

      if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestFmeasure) {
        m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
        bCurIterBetter = true;
      }

      if (testNum > 0) {
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idx = 0; idx < testInsts.size(); idx++) {
          predict(testInsts[idx], curDecodeInst);
          testInsts[idx].evaluate(curDecodeInst, metric_test);
          if (bCurIterBetter && !m_options.outBest.empty()) {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
        }
      }

      for (int idx = 0; idx < otherInsts.size(); idx++) {
        std::cout << "processing " << m_options.testFiles[idx] << std::endl;
        if (!m_options.outBest.empty())
          decodeInstResults.clear();
        metric_test.reset();
        for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
          predict(otherInsts[idx][idy], curDecodeInst);
          otherInsts[idx][idy].evaluate(curDecodeInst, metric_test);
          if (bCurIterBetter && !m_options.outBest.empty()) {
            decodeInstResults.push_back(curDecodeInst);
          }
        }
        std::cout << "test:" << std::endl;
        metric_test.print();

        if (!m_options.outBest.empty() && bCurIterBetter) {
          m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
        }
      }


      if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestFmeasure) {
        std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
        bestFmeasure = metric_dev.getAccuracy();
        writeModelFile(modelFile);
      }
    }
  }
}

void Segmentor::predict(const Instance& input, vector<string>& output) {
  m_classifier.decode(input.chars, output);
}

void Segmentor::test(const string& testFile, const string& outputFile, const string& modelFile) {
  loadModelFile(modelFile);
  vector<Instance> testInsts;
  m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

  vector<vector<string> > testInstResults(testInsts.size());
  Metric metric_test;
  metric_test.reset();
  for (int idx = 0; idx < testInsts.size(); idx++) {
    vector<string> result_labels;
    predict(testInsts[idx], testInstResults[idx]);
    testInsts[idx].evaluate(testInstResults[idx], metric_test);
  }
  std::cout << "test:" << std::endl;
  metric_test.print();

  std::ofstream os(outputFile.c_str());

  for (int idx = 0; idx < testInsts.size(); idx++) {
    for(int idy = 0; idy < testInstResults[idx].size(); idy++){
      os << testInstResults[idx][idy] << " ";
    }
    os << std::endl;
  }
  os.close();
}

/*
void Segmentor::readWordEmbeddings(const string& inFile, NRMat<dtype>& wordEmb) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordDim = vecInfo.size() - 1;

  std::cout << "word embedding dim is " << wordDim << std::endl;
  m_options.wordEmbSize = wordDim;

  wordEmb.resize(m_wordAlphabet.size(), wordDim);
  wordEmb = 0.0;
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  dtype sum[wordDim];
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordDim; idx++) {
      dtype curValue = atof(vecInfo[idx + 1].c_str());
      sum[idx] = curValue;
      wordEmb[wordId][idx] = curValue;
    }

  } else {
    for (int idx = 0; idx < wordDim; idx++) {
      sum[idx] = 0.0;
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordDim + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);

      for (int idx = 0; idx < wordDim; idx++) {
        dtype curValue = atof(vecInfo[idx + 1].c_str());
        sum[idx] = curValue;
        wordEmb[wordId][idx] += curValue;
      }
    }

  }

  if (!bHasUnknown) {
    for (int idx = 0; idx < wordDim; idx++) {
      wordEmb[unknownId][idx] = sum[idx] / count;
    }
    count++;
    std::cout << unknownkey << " not found, using averaged value to initialize." << std::endl;
  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordDim; idx++) {
        wordEmb[id][idx] = wordEmb[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", embedding oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}

void Segmentor::readWordClusters(const string& inFile) {
  static ifstream inf;
  if (inf.is_open()) {
    inf.close();
    inf.clear();
  }
  inf.open(inFile.c_str());

  static string strLine, curWord;
  static int wordId;

  //find the first line, decide the wordDim;
  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (!strLine.empty())
      break;
  }

  int unknownId = m_wordAlphabet.from_string(unknownkey);

  static vector<string> vecInfo;
  split_bychar(strLine, vecInfo, ' ');
  int wordClusterNum = vecInfo.size() - 1;

  std::cout << "word cluster number is " << wordClusterNum << std::endl;

  m_wordClusters.resize(m_wordAlphabet.size(), wordClusterNum);
  m_wordClusters = "0";
  curWord = normalize_to_lowerwithdigit(vecInfo[0]);
  wordId = m_wordAlphabet.from_string(curWord);
  hash_set<int> indexers;
  int count = 0;
  bool bHasUnknown = false;
  if (wordId >= 0) {
    count++;
    if (unknownId == wordId)
      bHasUnknown = true;
    indexers.insert(wordId);
    for (int idx = 0; idx < wordClusterNum; idx++) {
      m_wordClusters[wordId][idx] = vecInfo[idx + 1];
    }
  }

  while (1) {
    if (!my_getline(inf, strLine)) {
      break;
    }
    if (strLine.empty())
      continue;
    split_bychar(strLine, vecInfo, ' ');
    if (vecInfo.size() != wordClusterNum + 1) {
      std::cout << "error embedding file" << std::endl;
    }
    curWord = normalize_to_lowerwithdigit(vecInfo[0]);
    wordId = m_wordAlphabet.from_string(curWord);
    if (wordId >= 0) {
      count++;
      if (unknownId == wordId)
        bHasUnknown = true;
      indexers.insert(wordId);
      for (int idx = 0; idx < wordClusterNum; idx++) {
        m_wordClusters[wordId][idx] = vecInfo[idx + 1];
      }
    }

  }

  int oovWords = 0;
  int totalWords = 0;
  for (int id = 0; id < m_wordAlphabet.size(); id++) {
    if (indexers.find(id) == indexers.end()) {
      oovWords++;
      for (int idx = 0; idx < wordClusterNum; idx++) {
        m_wordClusters[id][idx] = m_wordClusters[unknownId][idx];
      }
    }
    totalWords++;
  }

  std::cout << "OOV num is " << oovWords << ", total num is " << m_wordAlphabet.size() << ", cluster oov ratio is " << oovWords * 1.0 / m_wordAlphabet.size()
      << std::endl;

}
*/

void Segmentor::loadModelFile(const string& inputModelFile) {

}

void Segmentor::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
  std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
  std::string wordEmbFile = "", optionFile = "";
  std::string outputFile = "";
  bool bTrain = false;
  dsr::Argument_helper ah;

  ah.new_flag("l", "learn", "train or test", bTrain);
  ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
  ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
  ah.new_named_string("test", "testCorpus", "named_string",
      "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
  ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
  ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
  ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
  ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

  ah.process(argc, argv);

  Segmentor segmentor;
  if (bTrain) {
    segmentor.train(trainFile, devFile, testFile, modelFile, optionFile, wordEmbFile);
  } else {
    segmentor.test(testFile, outputFile, modelFile);
  }

  //test(argv);
  //ah.write_values(std::cout);

}
