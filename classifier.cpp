//
//  classifier.cpp
//  lab2
//
//  Created by Анастасия Попова on 16.11.16.
//  Copyright © 2016 Анастасия Попова. All rights reserved.
//

#include "classifier.hpp"

Classifier::Classifier(){
    p[0]=0.0;
    p[1]=0.0;
    counts[0]=0;
    counts[1]=0;
}
Classifier::~Classifier(){}

vector<d> Classifier:: readTrainFile(string path){
    vector<d>data;
    fstream F;
    F.open(path);
    while (!F.eof()){
        d tmp;
        F>>tmp.heigh>>tmp.weight>>tmp.label;
        data.push_back(tmp);
    }
    return data;
}

vector<pair<int, int>> Classifier::readTestFile(string path){
    vector<pair<int, int>> r;
    fstream F;
    F.open(path);
    while (!F.eof()){
        int x, y;
        F>>x>>y;
        r.push_back(make_pair(x, y));
    }
    return r;
}

int Classifier::findNext(int i, double *arr){
    for (; i<250; i++){
        if (arr[i] > 10e-8) return i;
    }
    return 0;
}

double Classifier::probability(int h, int w, int label){
    return pWeight[label][w]*pHeight[label][h]*p[label];
}

void Classifier::train(string path){
    train(readTrainFile(path));
}
void Classifier::train(vector<d>input){
    int weight[2][250], heihgt[2][250];
    for (int i=0; i<2; i++){
        for (int j=0; j<250; j++){
            weight[i][j]=0;
            heihgt[i][j]=0;
        }
    }
    for (int i=0; i<input.size(); i++){
        counts[input[i].label]++;
        heihgt[input[i].label][input[i].heigh]++;
        weight[input[i].label][input[i].weight]++;
    }
    
    p[0] = double(counts[0])/double(input.size());
    p[1] = double(counts[1])/double(input.size());
    

    for (int i=0; i<2; i++)
        for (int j=0; j<250; j++){
            pWeight[i][j] = double(weight[i][j])/double(counts[i]);
            pHeight[i][j] = double(heihgt[i][j])/double(counts[i]);
        }
    pWeight[0][0]=10e-7;
    pWeight[1][0]=10e-7;
    pWeight[0][249]=10e-7;
    pWeight[1][249]=10e-7;
    pHeight[0][0]=10e-7;
    pHeight[1][0]=10e-7;
    pHeight[0][249]=10e-7;
    pHeight[1][249]=10e-7;
    int c=0;
    for (int i=0; i<2; i++)
        for (int j=0; j<250; j++){
            if (j==71){
                j=j;
            }
            if (pWeight[i][j] < 10e-8){
                c = findNext(j, pWeight[i]);
                for (int k=j; k<c; k++)
                    pWeight[i][k] = pWeight[i][j-1] + (k-j+1)*(pWeight[i][c] - pWeight[i][j-1])/(c-j+1);
            }
            if (pHeight[i][j] < 10e-8){
                c = findNext(j, pHeight[i]);
                for (int k=j; k<c; k++)
                    pHeight[i][k] = pHeight[i][j-1] + (k-j+1)*(pHeight[i][c] - pHeight[i][j-1])/(c-j+1);
            }
        }
}

vector<int> Classifier::classify(string path){
    return classify(readTestFile(path));
}

vector<int> Classifier::classify(vector<pair<int, int>>input){
    vector<int>rez;
    for (int i=0; i<input.size(); i++){
        if (probability(input[i].first, input[i].second, 0) > probability(input[i].first, input[i].second, 1)) rez.push_back(0);
        else rez.push_back(1);
    }
    return rez;
}
