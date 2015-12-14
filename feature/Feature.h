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
	vector<int> _nWordFeat;
	vector<int> _nActionFeat;

	vector<string> _strSparseFeat;
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

	~Feature() {
		clear();
	}

	void setFeatureFormat(bool bStringFeat) {
		_bStringFeat = bStringFeat;
	}

	void copy(const Feature& other) {
		clear();
		if (other._bStringFeat) {
			for (int idx = 0; idx < other._strSparseFeat.size(); idx++) {
				_strSparseFeat.push_back(other._strSparseFeat[idx]);
			}
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

			for (int idx = 0; idx < other._nWordFeat.size(); idx++) {
				_nWordFeat.push_back(other._nWordFeat[idx]);
			}
			for (int idx = 0; idx < other._nActionFeat.size(); idx++) {
				_nActionFeat.push_back(other._nActionFeat[idx]);
			}
		}
	}

	inline Feature& operator=(const Feature &rhs) {
		// Check for self-assignment!
		if (this == &rhs)      // Same object?
			return *this;        // Yes, so skip assignment, and just return *this.
		copy(rhs);
		return *this;
	}

	void clear() {
		_nSparseFeat.clear();
		_nWordFeat.clear();
		_nActionFeat.clear();

		_strSparseFeat.clear();
		_strWordFeat.clear();
		_strActionFeat.clear();
	}
};

#endif /* SRC_FEATURE_H_ */
