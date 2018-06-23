#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace std;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;



////////////   CLASS NEURON   ////////////
class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta; //overall net training rate
    static double alpha; //multiplier of last weight change
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double sumDOW (const Layer &nextLayer) const;
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    double m_gradient;
};


double Neuron::eta = 0.15; //overall net learning rate
double Neuron::alpha = 0.5; //the momentum, multiplier of last deltaWeight


void Neuron::updateInputWeights(Layer &prevLayer){
    //the weights to be updated are in connection container in the neurons in the preceding layer
    for(unsigned n=0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                //individual input, magnified by the gradient and train rate:
                eta
                * neuron.getOutputVal()
                * m_gradient
                //Also add momentum = a fraction of the previous delta weight
                + alpha
                  * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}


double Neuron::sumDOW (const Layer &nextLayer) const{
    double sum = 0.0;

    //sum our contributions of the errors at the nodes we feed

    for (unsigned n=0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}


void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}


void Neuron::calcOutputGradients(double targetVals) {
    double delta = targetVals - m_outputVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}


double Neuron::activationFunction(double x) {
    // using the tanh
    return tanh(x);
}


double Neuron::activationFunctionDerivative(double x) {
    return 1.0 - x*x;
}


void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;

    // sum the previous layer's outputs (which are inputs_
    // include bias node from previous layer

    for (unsigned n=0; n<prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::activationFunction(sum);
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}




////////////   CLASS NET   ////////////
class Net {
public:
    Net(const vector<unsigned> &topology);
    void feedForward (const vector<double> &inputVals);
    void backProp (const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // [layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::getResults(vector<double> &resultVals) const{
    resultVals.clear();

    for (unsigned n=0; n < m_layers.back().size() - 1; ++n){
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::backProp (const vector<double> &targetVals) {
    // calculate the RMS error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n =0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; //average error squared
    m_error = sqrt(m_error); //RMS

    // THIS IS A RUNNING AVERAGE OF SERVERAL RUNS:
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
                           / (m_recentAverageSmoothingFactor + 1.0);

    // calculate output layer gradients
    for (unsigned n=0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n =0; n<hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    // for all layers from outputs to first hidden layer update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n=0; n < layer.size() -1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }
};


void Net::feedForward (const vector<double> &inputVals) {
    // Check the num of inputVals euqal to neuronnum expect bias
    assert(inputVals.size() == m_layers[0].size() - 1);
    // Assign the input values into the input neurons
    for (unsigned i=0; i<inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward propagation
    for(unsigned layerNum =1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n=0; n < m_layers[layerNum].size() - 1; ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }

}


Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for (unsigned layerNum =0; layerNum < numLayers; ++ layerNum){
        //this loop creates layers
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum =0; neuronNum <= topology[layerNum]; ++neuronNum){
            //this loop creates neurons for each layer
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }
        //force bias node output to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}










int main()
{
    ////////////   CONSTRUCTOR with # OF LAYERS   ////////////
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);


    vector<double> inputVals;
    vector<double> targetVals;
    vector<double> resultVals;

    ////////////   TRAINING   ////////////
    myNet.feedForward(inputVals);
    myNet.backProp(targetVals);

    myNet.getResults(resultVals);
}
