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
#include <algorithm>

using namespace std;

struct d{
    double height, weight;
    int label;
};

class Classifier{
private:
    int mathModel;
    double p[2];
    double pWeight[2][250], pHeight[2][250];
    int counts[2];
    double ExW[2], DxW[2], ExH[2], DxH[2]; //матожидание&дисперсия
    double H; //коэфф сглаживания
    vector<d>train_set;
    
    void readTrainFile(ifstream &F);
    vector<pair<double, double>> readTestFile(ifstream &F);
    double probability(double h, double w, int label);
    int findNext(int i, double *arr);
public:
    Classifier();
    ~Classifier();
    friend Classifier barChart(Classifier A);
    friend Classifier normalDistribution (Classifier A);
    friend Classifier parzanRozenblatt (Classifier A);
    void train(ifstream &F, Classifier (*f)(Classifier A));
    void train(vector<d>input, Classifier (*f)(Classifier A));
    vector<int> classify(ifstream &F);
    vector<int> classify(vector<pair<double, double>>input);    
};
Classifier barChart(Classifier A);
Classifier normalDistribution (Classifier A);
Classifier parzanRozenblatt (Classifier A);

#endif /* classifier_hpp */
