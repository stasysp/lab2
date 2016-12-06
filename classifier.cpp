#include "classifier.hpp"
#include <cassert>
#include <map>


Classifier::Classifier(){
    mathModel = 0;
    p[0]=0.0;
    p[1]=0.0;
    counts[0]=0;
    counts[1]=0;
    ExW[0]=0; ExW[1]=0;
    DxW[0]=0; DxW[1]=0;
    ExH[0]=0; ExH[1]=0;
    DxH[0]=0; DxH[1]=0;
    H = 1e-7;
    k = 0;
}
Classifier::~Classifier(){}

Classifier::Classifier(const Classifier &A){
    mathModel = A.mathModel;
    H=A.H;
    for (int i=0; i<2; i++){
        p[i]=A.p[i];
        counts[i]=A.counts[i];
        ExW[i]=A.ExW[i];
        DxW[i]=A.DxW[i];
        ExH[i]=A.ExH[i];
        DxH[i]=A.DxH[i];
        for (int j=0; j<250; j++){
            pWeight[i][j]=A.pWeight[i][j];
            pHeight[i][j]=A.pHeight[i][j];
        }
    }
    for (int i=0; i<A.train_set.size(); i++)
        train_set.push_back(A.train_set[i]);
}

void Classifier:: readTrainFile(ifstream &F){
    while (!F.eof()){
        d tmp;
        F>>tmp.height>>tmp.weight>>tmp.label;
        train_set.push_back(tmp);
    }
}

vector<pair<double, double>> Classifier::readTestFile(ifstream &F){
    vector<pair<double, double>> r;
    while (!F.eof()){
        double x, y;
        F>>x>>y;
        r.push_back(make_pair(x, y));
    }
    F.close();
    return r;
}

int Classifier::findNext(int i, double *arr){
    for (; i<250; i++){
        if (arr[i] > 1e-9) return i;
    }
    return 0;
}
double kernel(double _h, int index, int label, vector<d>data){
    double f=0.0;
    const double exp = 2.718;
    for (int i=0; i<data.size(); i++){
        if (i!=index && data[i].label==label){
            f+=pow(exp, -sqrt(pow(data[index].weight-data[i].weight, 2)+pow(data[index].height-data[i].height, 2))/_h);
        }
    }
    return f;
}
double kernel1(double _h, int index, int label, vector<d>data){
    //     sqrt((x-xi)^2+(y-yi)^2)
    //K(x)=----------------------- if sqrt((x-xi)^2+(y-yi)^2) < h
    //             pi*h^2
    //
    const double pi = 3.14159265;
    double f=0.0;
    for (int i=0; i<data.size(); i++){
        if (i!=index && data[i].label==label){
            double tmp = sqrt(pow(data[index].weight-data[i].weight, 2)+pow(data[index].height-data[i].height, 2));
            if (tmp<_h) f+= tmp/(pi*pow(_h, 2));
        }
    }
    return f;
}
double Classifier::probability(double h, double w, int label){
    // P(x|yi)=P(x)*P(yi|x)/P(yi)
    // события yi независимые 
    //      pW[label]*pH[label]*p[label]
    //p(x)=------------------------------
    //      ((pW[0]+pW[1])*(pH[0]+pH[1]))
    //знаменатели одинаковые, достаточно сравнить только числители pW[label]*pH[label]*p[label]
    if (mathModel == 1){
        return pWeight[label][int(w)]*pHeight[label][int(h)]*p[label];
    }
    else if (mathModel == 2){
        const double pi = 3.14159265;
        const double exp = 2.718;
        double pW = pow(exp, (-pow(w-ExW[label], 2)/(2*DxW[label])))/(sqrt(2*pi*DxW[label]));
        double pH = pow(exp, (-pow(h-ExH[label], 2)/(2*DxH[label])))/(sqrt(2*pi*DxH[label]));
        return pW*pH*p[label];
    }
    else if (mathModel == 3){
        double f=0.0;
        const double exp = 2.718;
        for (int i=0; i<train_set.size(); i++){
            if (train_set[i].label==label){
                f+=pow(exp,-sqrt(pow(h-train_set[i].height, 2)+pow(w-train_set[i].weight, 2))/H);
            }
        }
        return p[label]*f/(counts[label]*H);
    } else{ //mathModel == 5
        const double pi = 3.14159265;
        double f=0.0;
        for (int i=0; i<train_set.size(); i++){
            if (train_set[i].label==label){
            double tmp = sqrt(pow(h-train_set[i].height, 2)+pow(w-train_set[i].weight, 2));
                if (tmp<H) f+= tmp/(pi*pow(H, 2));
            }
        }
        return p[label]*f/(counts[label]*(pi*pow(H, 2)));
    }
}
int findAns(int _k, vector<pair<double, int>> data){
    int count = 0;
    for (int i=0; i<_k; i++)
        if (data[i].second == 1) count++;
    if (count > _k-count) return 1;
    return 0;
}

Classifier barChart(Classifier A){
    A.mathModel = 1;
    int weight[2][250], height[2][250];
    const double eps = 1e-8;
    for (int i=0; i<2; i++){
        for (int j=0; j<250; j++){
            weight[i][j]=0;
            height[i][j]=0;
        }
    }
    for (int i=0; i<A.train_set.size(); i++){
        A.counts[A.train_set[i].label]++;
   //     assert(int(A.train_set[i].height) >= 0 && int(A.train_set[i].height) < 250);
   //     assert(int(A.train_set[i].weight) >= 0 && int(A.train_set[i].weight) < 250);
        if (A.train_set[i].height>249) A.train_set[i].height = 249;
        if (A.train_set[i].weight>249) A.train_set[i].weight = 249;
        height[A.train_set[i].label][int(A.train_set[i].height)]++;
        weight[A.train_set[i].label][int(A.train_set[i].weight)]++;
    }
    
    A.p[0] = double(A.counts[0])/double(A.train_set.size());
    A.p[1] = double(A.counts[1])/double(A.train_set.size());
    
    
    for (int i=0; i<2; i++)
        for (int j=0; j<250; j++){
            A.pWeight[i][j] = double(weight[i][j])/double(A.counts[i]);
            A.pHeight[i][j] = double(height[i][j])/double(A.counts[i]);
        }
    A.pWeight[0][0]=eps;
    A.pWeight[1][0]=eps;
    A.pWeight[0][249]=eps;
    A.pWeight[1][249]=eps;
    A.pHeight[0][0]=eps;
    A.pHeight[1][0]=eps;
    A.pHeight[0][249]=eps;
    A.pHeight[1][249]=eps;
    int c=0;
    // все нули посередине усредняются с соседними столбиками
    for (int i=0; i<2; i++)
        for (int j=0; j<250; j++){
            if (A.pWeight[i][j] < eps*0.1){
                c = A.findNext(j, A.pWeight[i]);
                for (int k=j; k<c; k++)
                    A.pWeight[i][k] = A.pWeight[i][j-1] + (k-j+1)*(A.pWeight[i][c] - A.pWeight[i][j-1])/(c-j+1);
            }
            if (A.pHeight[i][j] < eps*0.1){
                c = A.findNext(j, A.pHeight[i]);
                for (int k=j; k<c; k++)
                    A.pHeight[i][k] = A.pHeight[i][j-1] + (k-j+1)*(A.pHeight[i][c] - A.pHeight[i][j-1])/(c-j+1);
            }
        }
    return A;
}
Classifier normalDistribution (Classifier A){
    //      exp{-(x-Ex)^2/2Dx}
    // p(x)=-------------------
    //       (sqrt(2*pi*Dx)
    // Ex - матожидание
    // Dx - дисперсия
    A.mathModel = 2;
    double ExW2[2] = {0, 0};
    double ExH2[2] = {0, 0};
    if (A.counts[0] == 0 && A.counts[1]==0)
        for (int i=0; i<A.train_set.size(); i++)
            A.counts[A.train_set[i].label]++;
    A.p[0] = double(A.counts[0])/double(A.train_set.size());
    A.p[1] = double(A.counts[1])/double(A.train_set.size());
    for (int i=0; i<A.train_set.size(); i++){
        A.ExW[A.train_set[i].label] += A.train_set[i].weight / A.counts[A.train_set[i].label]; // 1/n * weight;
        A.ExH[A.train_set[i].label] += A.train_set[i].height / A.counts[A.train_set[i].label];
        ExW2[A.train_set[i].label] += A.train_set[i].weight * A.train_set[i].weight / A.counts[A.train_set[i].label];
        ExH2[A.train_set[i].label] += A.train_set[i].height * A.train_set[i].height / A.counts[A.train_set[i].label];
    }
    for (int i=0; i<2; i++){
        A.DxW[i] = ExW2[i] - A.ExW[i]*A.ExW[i];
        A.DxH[i] = ExH2[i] - A.ExH[i]*A.ExH[i];
    }
    return A;
}

Classifier parzanRozenblatt (Classifier A){
    //      1           x-xi
    //P(x)=--- * sum K(-----)
    //     n*h           h
    //
    //       exp{-sqrt((x-xi)^2+(y-yi)^2)}
    //K(x)=----------------------------------
    //                    h
    //
    A.mathModel = 3;
    if (A.counts[0] == 0 && A.counts[1]==0)
        for (int i=0; i<A.train_set.size(); i++)
            A.counts[A.train_set[i].label]++;
    A.p[0] = double(A.counts[0])/double(A.train_set.size());
    A.p[1] = double(A.counts[1])/double(A.train_set.size());
    srand(time(NULL));
   // int c = (int)(A.train_set.size()/10);
    int c = 30;
    int count = 0;
    double countTMP = 0;
    double tmpH = A.H;
    double const eps = 0.1;
    for (int i=0; i<(10.00-A.H)/eps; i++){
        countTMP = 0;
        for (int j = 0; j < c; j++){
            int a = rand()%(A.train_set.size());
            //сократим на 1/tmpH
            if (kernel(tmpH, a, 0, A.train_set)/A.counts[0] >= kernel(tmpH, a, 1, A.train_set)/A.counts[1] && A.train_set[a].label == 0) countTMP++;
            if (kernel(tmpH, a, 1, A.train_set)/A.counts[1] > kernel(tmpH, a, 0, A.train_set)/A.counts[0] && A.train_set[a].label == 1) countTMP++;
        }
        if (countTMP > count){
            count = countTMP;
            A.H = tmpH;
        }
        tmpH+=eps;
    }
    return A;
}

Classifier KNN(Classifier A){
    A.mathModel = 4;
    random_shuffle(A.train_set.begin(), A.train_set.end());
    int c = int(A.train_set.size()/10);
    vector<pair<double, int>> *mydata;
    mydata = new vector<pair<double, int>> [c];
    for (int i=0; i<c; i++){
        for (int j=0; j<A.train_set.size(); j++){
            if (i!=j) {
                mydata[i].push_back(make_pair(sqrt(pow(A.train_set[i].height-A.train_set[j].height, 2)+pow(A.train_set[i].weight-A.train_set[j].weight, 2)), A.train_set[j].label));
            }
        }
        sort(mydata[i].begin(), mydata[i].end());
    }
    int **ans;
    *ans = new int [c];
    for (int i=0; i<c; i++)
        ans[i]= new int [15];
    for (int i=0; i<c; i++){
        for (int j=0; j<15; j++){
            ans[i][j] = findAns(j, mydata[i]);
        }
    }
    int mycounts[15];
    int tmpK = 0;
    int max = 0;
    for (int i=1; i<15; i++){
        mycounts[i]=0;
        for (int j=0; j<c; j++){
            if (ans[j][i] == A.train_set[j].label) mycounts[i]++;
        }
        if (mycounts[i]>max) {
            max = mycounts[i];
            tmpK = i;
        }
    }
    A.k = tmpK;
    return A;
}

Classifier parzanRozenblatt2 (Classifier A){
    //        1              X-Xi
    //P(x)=--------- * sum K(-----)
    //     n*pi*h^2            h
    //
    //     sqrt((x-xi)^2+(y-yi)^2)
    //K(x)=----------------------- if sqrt((x-xi)^2+(y-yi)^2) < h
    //             pi*h^2
    //
    const double pi = 3.14159265;
    A.mathModel = 5;
    if (A.counts[0] == 0 && A.counts[1]==0)
        for (int i=0; i<A.train_set.size(); i++)
            A.counts[A.train_set[i].label]++;
    A.p[0] = double(A.counts[0])/double(A.train_set.size());
    A.p[1] = double(A.counts[1])/double(A.train_set.size());
    // int c = (int)(A.train_set.size()/10);
    int c = 30;
    int count = 0;
    double countTMP = 0;
    double tmpH = A.H;
    double const eps = 0.1;
    for (int i=0; i<(10.00-A.H)/eps; i++){
        countTMP = 0;
        for (int j = 0; j < c; j++){
            //сократим на 1/pi*h^2
            if (kernel1(tmpH, j, 0, A.train_set)/A.counts[0] >= kernel(tmpH, j, 1, A.train_set)/A.counts[1] && A.train_set[j].label == 0) countTMP++;
            if (kernel1(tmpH, j, 1, A.train_set)/A.counts[1] > kernel(tmpH, j, 0, A.train_set)/A.counts[0] && A.train_set[j].label == 1) countTMP++;
        }
        if (countTMP > count){
            count = countTMP;
            A.H = tmpH;
        }
        tmpH+=eps;
    }
    
    return A;
}

void Classifier::train(ifstream &F, Classifier (*f)(Classifier A)){
    readTrainFile(F);
    train(this->train_set, f);
}
void Classifier::train(vector<d> input, Classifier (*f)(Classifier A)){
    *this = f(*this);

}

vector<int> Classifier::classify(ifstream &F){
    return classify(readTestFile(F));
}

vector<int> Classifier::classify(vector<pair<double, double>>input){
    vector<int>rez;
    if (mathModel == 4){
        vector<pair<double, int>> mydata;
        for (int i=0; i<input.size(); i++){
            for (int j=0; j<train_set.size(); j++){
                    mydata.push_back(make_pair(sqrt(pow(train_set[j].height-input[i].first, 2)+pow(train_set[j].weight-input[i].second, 2)), train_set[j].label));
                }
            sort(mydata.begin(), mydata.end());
            rez.push_back(findAns(13, mydata));
            mydata.clear();
        }
    } else {
    for (int i=0; i<input.size(); i++){
        double a = probability(input[i].first, input[i].second, 0);
        double b = probability(input[i].first, input[i].second, 1);
        if (a > b) rez.push_back(0);
        else rez.push_back(1);
        }
    }
    return rez;
}
