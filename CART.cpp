#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <random>

using namespace std;

#define BREAST_CANCER_ATTRIBUTE_NUMBER 9

struct OneBreastCancerSample{
    double input[BREAST_CANCER_ATTRIBUTE_NUMBER];
    string output;
    int id;
};

struct DecisionTreeNode{
    int whichAttribute, parentIndex, leftChildIndex, rightChildIndex;
    double threshold;
    string output;
    vector<OneBreastCancerSample> remainingSamples;
};

struct Sum{
    string output;
    int number;
};

struct SplitInfomation{
    int whichAttribute, index;
    double threshold;
    string category;
};

unsigned int compareIndex;

bool Compare_By_Index(OneBreastCancerSample s1, OneBreastCancerSample s2){
    return s1.input[compareIndex] < s2.input[compareIndex];
}
bool By_name(OneBreastCancerSample s1, OneBreastCancerSample s2){
    if(s1.output.compare(s2.output) > 0){
        return 1;
    }
    else{
        return 0;
    }
}

class RandomForest{
    public:
        vector<OneBreastCancerSample> wholeSamples;
        int wholeSamplesSize;
        RandomForest(string fileName, unsigned int BSS, unsigned int TN, unsigned int HMCV){

            BAGGING_SAMPLE_SIZE = BSS;
            TREE_NUMBER = TN;
            HOW_MANY_CROSS_VALIDATION = HMCV;

            // open file
            fstream file;
            file.open(fileName, ios::in);
            if(!file){
                cout<<"File didn't open!"<<endl;
                exit(1);
            }

            // read data
            OneBreastCancerSample sampleBuff;
            sampleBuff.id = 0;
            int uselessAttribute;
            char comma;

            while(file>>sampleBuff.input[0]){
                file>>comma>>uselessAttribute>>comma;
                for(int i = 1; i < BREAST_CANCER_ATTRIBUTE_NUMBER; i++){
                    file>>sampleBuff.input[i]>>comma;
                }
                file>>sampleBuff.output;
                wholeSamples.push_back(sampleBuff);
                sampleBuff.id++;
            }
            wholeSamplesSize = wholeSamples.size();

            // shuffle the samples to randomize the way of getting validation set
            shuffle(wholeSamples.begin(), wholeSamples.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
        }
        void cross_validation(){
            int validationSetNumber = wholeSamplesSize / HOW_MANY_CROSS_VALIDATION, trainingSetNumber = wholeSamplesSize - wholeSamplesSize / HOW_MANY_CROSS_VALIDATION;
            double accuracyForOneValidation = 0;
            for(int i = 0; i < HOW_MANY_CROSS_VALIDATION; i++){
                cout<<"---------------Cross "<<i<<"---------------"<<endl;
                // split samples into training set and validation set
                vector<OneBreastCancerSample> trainingSet, validationSet;
                for(int j = 0; j < wholeSamplesSize; j++){
                    if(validationSetNumber * i <= j && j < validationSetNumber * (i + 1)){
                        validationSet.push_back(wholeSamples[j]);
                    }
                    else{
                        trainingSet.push_back(wholeSamples[j]);
                    }
                }
            
                // form forest
                roots.clear();
                for(int j = 0; j < TREE_NUMBER; j++){
                    DecisionTreeNode root;
                    root.output = "";
                    root.parentIndex = -1;
                    root.leftChildIndex = 0;
                    for(int k = 0; k < BAGGING_SAMPLE_SIZE; k++){
                        root.remainingSamples.push_back(trainingSet[rand() % trainingSetNumber]);
                    };
                    Build_A_Decision_Tree(root);
                }

                // validating
                int hit = 0;
                for(int j = 0; j < validationSet.size(); j++){
                    vector<Sum> result;
                    for(int k = 0; k < roots.size(); k++){

                        // get the result from a tree
                        DecisionTreeNode currentNode = roots[k][0];
                        while(currentNode.leftChildIndex != 0){
                            if(validationSet[j].input[currentNode.whichAttribute] < currentNode.threshold){
                                currentNode = roots[k][currentNode.leftChildIndex];
                            }
                            else{
                                currentNode = roots[k][currentNode.rightChildIndex];
                            }
                        }

                        // note down the result
                        for(int m = 0; ;m++){
                            if(m == result.size()){
                                Sum temp;
                                temp.number = 1;
                                temp.output = currentNode.output;
                                result.push_back(temp);
                                break;
                            }
                            else if(result[m].output == currentNode.output){
                                result[m].number++;
                                break;
                            }
                        }
                    }

                    // pick the most possible output and check if it was a hit
                    int theMostOutputIndex = 0, theMostOutputNumber = 0;
                    for(int k = 0; k < result.size(); k++){
                        if(result[k].number > theMostOutputNumber){
                            theMostOutputIndex = k;
                            theMostOutputNumber = result[k].number;
                        }
                    }
                    if(result[theMostOutputIndex].output.compare(validationSet[j].output) == 0){
                        hit++;
                    }
                }
                cout<<"Total samples: "<<wholeSamplesSize<<endl;
                cout<<"Bagging samples: "<<BAGGING_SAMPLE_SIZE<<endl;
                cout<<"Validation samples: "<<validationSetNumber<<endl;
                cout<<"Hit: "<<hit<<endl;
                accuracyForOneValidation += (double) hit / (double) validationSetNumber;
            }
            cout<<endl<<"------------------------------------------------"<<endl;
            cout<<"Accuracy: "<<accuracyForOneValidation / HOW_MANY_CROSS_VALIDATION<<endl;
            cout<<"------------------------------------------------"<<endl;
        }

    private:
        unsigned int BAGGING_SAMPLE_SIZE, TREE_NUMBER, HOW_MANY_CROSS_VALIDATION;
        vector<vector<DecisionTreeNode>> roots;
        
        SplitInfomation Where_To_Split(vector<OneBreastCancerSample> samples){
            SplitInfomation splitInfo;
            int samplesSize = samples.size();

            vector<Sum> sum;
            for(int i = 0; i < samplesSize; i++){
                for(int j = 0; ; j++){
                    if(j == sum.size()){
                        Sum s;
                        s.output = samples[i].output;
                        s.number = 1;
                        sum.push_back(s);
                        break;
                    }
                    else if(sum[j].output.compare(samples[i].output) == 0){
                        sum[j].number++;
                        break;
                    }
                }
            }

            if(sum.size() == 1){ // no need to split
                splitInfo.whichAttribute = -1;
                return splitInfo;
            }

            // find the best attribute
            double biggerGini = -1013;
            for(int i = 0; i < BREAST_CANCER_ATTRIBUTE_NUMBER; i++){
                sort(samples.begin(), samples.end(), By_name);
                compareIndex = (unsigned int) i;
                sort(samples.begin(), samples.end(), Compare_By_Index);

                // find the best threshold for an attribute
                for(int j = 1; j < samplesSize; j++){ // j is the firt sample on the right and indicates how many samples on the left.
                    double leftGini = 0, rightGini = 0; // not the gini impurity, but for find the best split point.

                    vector<Sum> sumAtLeft(sum.begin(), sum.end());
                    for(int k = 0; k < sumAtLeft.size(); k++){
                        sumAtLeft[k].number = 0;
                    }
                    for(int k = 0; k < j; k++){
                        for(int l = 0; l < sumAtLeft.size(); l++){
                            if(samples[k].output.compare(sumAtLeft[l].output) == 0){
                                sumAtLeft[l].number++;
                            }
                        }
                    }

                    for(int k = 0; k < sum.size(); k++){
                        leftGini += (double) sumAtLeft[k].number / (double) j * (double) sumAtLeft[k].number / (double) j;
                        rightGini += (double) (sum[k].number - sumAtLeft[k].number) / (double) (samplesSize - j) * (double) (sum[k].number - sumAtLeft[k].number) / (double) (samplesSize - j);
                    }
                    if(biggerGini < leftGini * j / samplesSize + rightGini * (samplesSize - j) / samplesSize){
                        biggerGini = leftGini * j / samplesSize + rightGini * (samplesSize - j) / samplesSize;
                        splitInfo.whichAttribute = i;
                        splitInfo.index = j;
                        splitInfo.threshold = (samples[j-1].input[i] + samples[j].input[i]) / 2;
                    }
                }
            }

            return splitInfo;
        }
        void Build_A_Decision_Tree(DecisionTreeNode root){

            vector<DecisionTreeNode> unsplittedNodes, decisionTreeNodes;
            unsplittedNodes.push_back(root);
            while(unsplittedNodes.size() != 0){
                SplitInfomation splitInfomation = Where_To_Split(unsplittedNodes.back().remainingSamples);

                if(splitInfomation.whichAttribute == -1){ // leaf node
                    unsplittedNodes.back().output = unsplittedNodes.back().remainingSamples[0].output;
                    unsplittedNodes.back().remainingSamples.clear();
                    if(decisionTreeNodes[unsplittedNodes.back().parentIndex].leftChildIndex == 0){
                        decisionTreeNodes[unsplittedNodes.back().parentIndex].leftChildIndex = decisionTreeNodes.size();
                    }
                    else{
                        decisionTreeNodes[unsplittedNodes.back().parentIndex].rightChildIndex = decisionTreeNodes.size();
                    }
                    decisionTreeNodes.push_back(unsplittedNodes.back());
                    unsplittedNodes.pop_back();
                }
                else{ // non-leaf node
                    DecisionTreeNode leftChild, rightChild;
                    leftChild.parentIndex = decisionTreeNodes.size();
                    leftChild.leftChildIndex = 0;
                    rightChild.parentIndex = decisionTreeNodes.size();
                    rightChild.leftChildIndex = 0;

                    // sort according to spliting information
                    sort(unsplittedNodes.back().remainingSamples.begin(), unsplittedNodes.back().remainingSamples.end(), By_name);
                    compareIndex = (unsigned int) splitInfomation.whichAttribute;
                    sort(unsplittedNodes.back().remainingSamples.begin(), unsplittedNodes.back().remainingSamples.end(), Compare_By_Index);
                    
                    leftChild.remainingSamples.assign(unsplittedNodes.back().remainingSamples.begin(), unsplittedNodes.back().remainingSamples.begin() + splitInfomation.index);
                    rightChild.remainingSamples.assign(unsplittedNodes.back().remainingSamples.begin() + splitInfomation.index, unsplittedNodes.back().remainingSamples.end());
                    
                    unsplittedNodes.back().whichAttribute = splitInfomation.whichAttribute;
                    unsplittedNodes.back().threshold = splitInfomation.threshold;
                    unsplittedNodes.back().remainingSamples.clear();
                    
                    if(unsplittedNodes.back().parentIndex != -1){ // not the root
                        // Because left children will always be visited before the right one.
                        if(decisionTreeNodes[unsplittedNodes.back().parentIndex].leftChildIndex == 0){
                            decisionTreeNodes[unsplittedNodes.back().parentIndex].leftChildIndex = decisionTreeNodes.size();
                        }
                        else{
                            decisionTreeNodes[unsplittedNodes.back().parentIndex].rightChildIndex = decisionTreeNodes.size();
                        }
                    }

                    decisionTreeNodes.push_back(unsplittedNodes.back());
                    unsplittedNodes.pop_back();
                    unsplittedNodes.push_back(rightChild);
                    unsplittedNodes.push_back(leftChild);
                }
            }
            /*for(int i = 0; i < decisionTreeNodes.size(); i++){
                cout<<endl<<"---Node "<<i<<"---"<<endl;
                if(decisionTreeNodes[i].leftChildIndex != 0){
                    cout<<"attribute: "<<decisionTreeNodes[i].whichAttribute<<endl;
                    cout<<"threshold: "<<decisionTreeNodes[i].threshold<<endl;
                    cout<<"left child node: "<<decisionTreeNodes[i].leftChildIndex<<endl;
                    cout<<"right child node: "<<decisionTreeNodes[i].rightChildIndex<<endl;
                }
                else{
                    cout<<"Output is ";
                    cout<<decisionTreeNodes[i].output<<endl;
                } 
            }*/
            roots.push_back(decisionTreeNodes);
        }
};

int main(int argc, char const *argv[]){

    srand(time(NULL));
    
    cout<<"Type in: (1) (2) (3)"<<endl;
    cout<<"--(1)how many samples is a random tree should train on"<<endl;
    cout<<"--(2)how many tree should be constructed"<<endl;
    cout<<"--(3)how many times should cross validation perform"<<endl;
    cout<<"or Type in \"-1\" to exit---"<<endl;
    
    int p[3];
    while(1){
        cin>>p[0];
        if(p[0] == -1){
            break;
        }
        cin>>p[1]>>p[2];
        RandomForest forest("breast_cancer.txt", p[0], p[1], p[2]);
        forest.cross_validation();
    }
    
    
    return 0;
}
