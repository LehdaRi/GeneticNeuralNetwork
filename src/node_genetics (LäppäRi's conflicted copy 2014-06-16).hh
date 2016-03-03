#ifndef NODE_GENETICS_HH
#define NODE_GENETICS_HH


#include "extended_neural_network.hh"

#include <map>
#include <vector>
#include <memory>


class NodeGeneticNeuralNetwork;

typedef NodeGeneticNeuralNetwork NGNN;


class NodeGeneticNeuralNetwork : public ExtendedNeuralNetwork {
public:
    enum Exception {
        UNABLE_TO_CROSSOVER
    };

    struct HistoryEntry {
        /*  0: backpropagation
            1: mutation
            (2: crossover) */
        unsigned int type;
        union {
            struct {
                /*  0: add a node
                    1: remove a node
                    2: add a connection
                    3: remove a connection
                    4: manipulate a connection */
                unsigned int type;
                union {
                    struct {
                        unsigned int nodeId;
                    } type1;

                    struct {
                        unsigned int fromNodeId, toNodeId;
                        double weight;
                    } type2;

                    struct {
                        unsigned int nodeId, connId;
                    } type3;

                    struct {
                        unsigned int nodeId, connId;
                        double weight;
                    } type4;
                } data;
            } mutation;
        } data;
    };

    union HistoryEntryDataWrapper {
        HistoryEntry entry;
        unsigned char data[sizeof(HistoryEntry)];
    };

    NodeGeneticNeuralNetwork(double learningRate_, double momentum_);

    /*  random network constructor */
    NodeGeneticNeuralNetwork(double learningRate_,
                             double momentum_,
                             const size_t numInputs,
                             const size_t numOutputs,
                             const size_t minNumHiddens,
                             const size_t maxNumHiddens,
                             const size_t numMaxConnections,
                             const double minNodeValue,
                             const double maxNodeValue,
                             const double minWeight,
                             const double maxWeight);

    /*  mutation */
    void mutate(void);

    /*  crossover */
    friend NGNN crossover(NGNN& nn1, NGNN& nn2);

    /*  some debug stuff */
    void printGenome(void);
    void printHistory(void);

private:
    std::vector<HistoryEntryDataWrapper> history;
};


class NGNNEvolver {
public:
    NGNNEvolver(double learningRate,
                double momentum,
                const size_t numInputs,
                const size_t numOutputs,
                const size_t minNumHiddens,
                const size_t maxNumHiddens,
                const size_t numMaxConnections,
                const double minNodeValue,
                const double maxNodeValue,
                const double minWeight,
                const double maxWeight,
                DataLoader& dataLoader_);

    NGNNEvolver(const NGNN& network_,
                DataLoader& dataLoader_);

    NGNN mutate(size_t populationSize,
                size_t individualsToMutate,
                size_t numTrainingRounds,
                size_t numEvaluationRounds,
                double learningSpeedConstraint,
                size_t numEpochsConstraint);

private:
    NGNN network;
    DataLoader& dataLoader_;
};


#endif // NODE_GENETICS_HH
