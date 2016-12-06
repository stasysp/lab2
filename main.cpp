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
    
    ifstream F, F1;
    ofstream A;
    F.open("/Users/anastasiapopova/Desktop/YANDEX/lab2_Bayes/lab2/train_set.txt");
    F1.open("/Users/anastasiapopova/Desktop/YANDEX/lab2_Bayes/lab2/test_set.txt");
    
    A.open("/Users/anastasiapopova/Desktop/YANDEX/lab2_Bayes/lab2/new");
    my.train(F, parzanRozenblatt2);
    cout<<"train"<<endl;
    vector<int>ans = my.classify(F1);
    cout<<"classify"<<endl;
    for (int i=0; i<ans.size(); i++){
            A<<ans[i]<<endl;
    }
    A.close();
  
    return 0;
}
