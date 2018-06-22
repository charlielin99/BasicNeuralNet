#include <iostream>
#include <vector>
using namespace std;

class Net {
public:

    Net(const vector<unsigned> &topology);
    void feedForward (const vector<double> &inputVals);
    void backProp (const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;

private:
};


int main()
{
    //CONSTURCTOR # of layers
    vector<unsigned> topology;
    Net myNet(topology);

    vector<double> inputVals;
    vector<double> targetVals;
    vector<double> resultVals;

    ////TRAINING//// 
    myNet.feedForward(inputVals);
    myNet.backProp(targetVals);
    ////////

    myNet.getResults(resultVals);
}