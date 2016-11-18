//
//  classifier.cpp
//  lab2
//
//  Created by Анастасия Попова on 16.11.16.
//  Copyright © 2016 Анастасия Попова. All rights reserved.
//

#include "classifier.hpp"

Classifier::Classifier(){
    p0=0.0;
    p1=0.0;
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

double Classifier::probability(int h, int w, int label){
    double p;
    if (label==0) p=p0;
    else p=p1;
    return (pWeight[label][w-2] + pWeight[label][w-1] + pWeight[label][w] + pWeight[label][w+1] + pWeight[label][w+2]) * (pHeight[label][h-2] + pHeight[label][h-1] + pHeight[label][h] + pHeight[label][h+1] + pHeight[label][h+2]) * p;
    
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
    
    p0 = double(counts[0])/double(input.size());
    p1 = double(counts[1])/double(input.size());
    

    for (int i=0; i<2; i++)
        for (int j=0; j<250; j++){
            pWeight[i][j] = double(weight[i][j])/double(counts[i]);
            pHeight[i][j] = double(heihgt[i][j])/double(counts[i]);
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
