//
//  classifier.hpp
//  lab2
//
//  Created by Анастасия Попова on 04.11.16.
//  Copyright © 2016 Анастасия Попова. All rights reserved.
//

#ifndef classifier_hpp
#define classifier_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

using namespace std;

struct d{
    int heigh, weight, label;
};

class Classifier{
private:
    double p[2];
    double pWeight[2][250], pHeight[2][250];
    int counts[2];
    
    vector<d> readTrainFile(string path);
    vector<pair<int, int>> readTestFile(string path);
    double probability(int h, int w, int label);
    int findNext(int i, double *arr);
public:
    Classifier();
    ~Classifier();
    void train(string path);
    void train(vector<d>input);
    vector<int> classify(string path);
    vector<int> classify(vector<pair<int, int>>input);
};

#endif /* classifier_hpp */
