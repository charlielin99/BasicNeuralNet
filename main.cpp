#include <iostream>
#include <vector>
using namespace std;

class Neuron {};
typedef vector<Neuron> Layer;



class Net {
public:

    Net(const vector<unsigned> &topology);
    void feedForward (const vector<double> &inputVals) {};
    void backProp (const vector<double> &targetVals) {};
    void getResults(vector<double> &resultVals) const {};

private:
    vector<Layer> m_layers; // [layerNumber][neuronNumber]
};





Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for (unsigned layerNum =0; layerNum < numLayers; ++ layerNum){
        //this loop creates layers
        m_layers.push_back(Layer());
        for (unsigned neuronNum =0; neuronNum <= topology[layerNum]; ++neuronNum){
            //this loop creates neurons for each layer
            m_layers.back().push_back(Neuron());
            cout << "Made a Neuron!" << endl;
        }
    }
}





int main()
{
    //CONSTURCTOR # of layers
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
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