//
//  main.cpp
//  lab2
//
//  Created by Анастасия Попова on 04.11.16.
//  Copyright © 2016 Анастасия Попова. All rights reserved.
//

#include <iostream>
#include "classifier.hpp"

using namespace std;

int main() {
    Classifier my;
    
    vector<pair<double, double>> myTest;
    myTest.push_back(make_pair(190, 90));
    myTest.push_back(make_pair(185, 86));
    myTest.push_back(make_pair(165, 46));
    myTest.push_back(make_pair(164, 49));
    myTest.push_back(make_pair(179, 76));
    myTest.push_back(make_pair(155, 42));
    
    vector<d>test1;
    
    ifstream F;
    F.open("/Users/anastasiapopova/Desktop/YANDEX/lab2_Bayes/lab2/train_test_set");
    
    my.train(F, barChart);
    vector<int>ans = my.classify(myTest);
    for (int i=0; i<ans.size(); i++){
        cout<<ans[i]<<" ";
    }
    cout<<endl;
    
    my.train(F, normalDistribution);
    ans = my.classify(myTest);
    for (int i=0; i<ans.size(); i++){
        cout<<ans[i]<<" ";
    }
    cout<<endl;
    
    my.train(F, parzanRozenblatt);
    ans = my.classify(myTest);
    
    for (int i=0; i<ans.size(); i++){
        cout<<ans[i]<<" ";
    }
    cout<<endl;
    
    return 0;
}
