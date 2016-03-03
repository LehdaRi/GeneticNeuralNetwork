#include "node_genetics.hh"
#include "sorter_container.hh"
#include "util.hh"


#include <iostream>
#include <algorithm>
#include <SFML/Graphics.hpp> //TEMP


/*  NodeGeneticNeuralNetwork - public member functions */


NodeGeneticNeuralNetwork::NodeGeneticNeuralNetwork(double learningRate_, double momentum_) :
    ExtendedNeuralNetwork(learningRate_, momentum_)
    {}


NodeGeneticNeuralNetwork::NodeGeneticNeuralNetwork(double learningRate_,
                                                   double momentum_,
                                                   const size_t numInputs,
                                                   const size_t numOutputs,
                                                   const size_t minNumHiddens,
                                                   const size_t maxNumHiddens,
                                                   const size_t numMaxConnections,
                                                   const double minNodeValue,
                                                   const double maxNodeValue,
                                                   const double minWeight,
                                                   const double maxWeight) :
    ExtendedNeuralNetwork(learningRate_, momentum_)
    {
        ut::Rand rand;
        const size_t numHiddens = rand.iRand(minNumHiddens, maxNumHiddens);
        unsigned int newNodeId[numInputs + numHiddens + numOutputs];

        /*  input nodes */
        for (size_t i=0; i<numInputs; ++i)
            newNodeId[i] = addInputNode(0.0, 0.0);

        /*  hidden nodes */
        for (size_t i=0; i<numHiddens; ++i)
            newNodeId[numInputs + i] = addNode(rand.dRand(minNodeValue, maxNodeValue));

        /*  output nodes */
        for (size_t i=0; i<numOutputs; ++i)
            newNodeId[numInputs + numHiddens + i] = addOutputNode(0.0, 0.0);

        /*  connections for hidden nodes */
        for (size_t i=0; i<numHiddens; ++i) {
            const size_t numConnections = rand.iRand(0, std::min(numMaxConnections, numInputs + numHiddens - 1));

            for (size_t j=0; j<numConnections; ++j)
                while (!setConnection(newNodeId[numInputs + i], // TODO could be optimized
                                      newNodeId[rand.iRand(0, numInputs + numHiddens - 1)],
                                      rand.dRand(minWeight, maxWeight)));
        }

        /*  connections for output nodes */
        for (size_t i=0; i<numOutputs; ++i) {
            const size_t numConnections = rand.iRand(0, std::min(numMaxConnections, numInputs + numHiddens + numOutputs - 1));
            for (size_t j=0; j<numConnections; ++j)
                while (!setConnection(newNodeId[numInputs + numHiddens + i], // TODO could be optimized
                                      newNodeId[rand.iRand(0, numInputs + numHiddens + numOutputs - 1)],
                                      rand.dRand(minWeight, maxWeight)));
        }
    }

NodeGeneticNeuralNetwork::NodeGeneticNeuralNetwork(double learningRate_,
                                                   double momentum_,
                                                   const size_t numInputs,
                                                   const size_t numOutputs,
                                                   const std::vector<size_t>& layers,
                                                   const double minWeight,
                                                   const double maxWeight) :
    ExtendedNeuralNetwork(learningRate_, momentum_)
{
    ut::Rand rand;
    unsigned inputIds[numInputs+1], outputIds[numOutputs];

    /*  input nodes */
    for (size_t i=0; i<numInputs; ++i)
        inputIds[i] = addInputNode(0.0, 0.0);
    inputIds[numInputs] = addNode(1.0);

    /*  output nodes */
    for (size_t i=0; i<numOutputs; ++i)
        outputIds[i] = addOutputNode(0.0, 0.0);

    std::vector<std::vector<unsigned>> hiddenIds;

    /*  hidden layers */
    for (unsigned l=0; l<layers.size(); ++l) {
        std::vector<unsigned> layerIds;

        for (size_t i=0; i<layers[l]; ++i) {
            unsigned newId = addNode(0.0);
            layerIds.push_back(newId);

            if (l == 0) {
                for (size_t j=0; j<=numInputs; ++j)
                    setConnection(newId, inputIds[j], rand.dRand(minWeight, maxWeight));
            }
            else {
                for (auto& id : hiddenIds[l-1])
                    setConnection(newId, id, rand.dRand(minWeight, maxWeight));
            }
        }

        unsigned biasId = addNode(1.0);
        layerIds.push_back(biasId);

        hiddenIds.push_back(layerIds);
    }

    for (size_t i=0; i<numOutputs; ++i)
        for (auto& id : hiddenIds[hiddenIds.size()-1])
            setConnection(outputIds[i], id, rand.dRand(minWeight, maxWeight));
}


void NodeGeneticNeuralNetwork::mutate(void) {
    ut::Rand rand;

    /*  random mutation */
    NEWMUTATION:
    switch(rand.iRand(0,4)) {
    case 0: // add a node
    {
        //TEMP std::cout << "MUTATION: add a node" << std::endl;
        unsigned int newNodeId = addNode(rand.dRand(-2.0, 2.0));
        moveNode(newNodeId, *std::min_element(outputIds.begin(), outputIds.end()));

        HistoryEntryDataWrapper historyEntry;
        historyEntry.entry.type = 1;
        historyEntry.entry.data.mutation.type = 0;
        history.push_back(historyEntry);
    }
    break;
    case 1: // remove a node
    {
        if (nodes.size() <= inputIds.size() + outputIds.size())
            break;

        unsigned int nodeId = rand.iRand(0, nodes.size()-1);

        while (std::find(inputIds.begin(), inputIds.end(), nodeId) != inputIds.end() ||
               std::find(outputIds.begin(), outputIds.end(), nodeId) != outputIds.end())
            nodeId = rand.iRand(0, nodes.size()-1);

        removeNode(nodeId);

        HistoryEntryDataWrapper historyEntry;
        historyEntry.entry.type = 1;
        historyEntry.entry.data.mutation.type = 1;
        historyEntry.entry.data.mutation.data.type1.nodeId = nodeId;
        history.push_back(historyEntry);
    }
    break;
    case 2: // add a connection
    {
        //TEMP std::cout << "MUTATION: add a connection" << std::endl;
        if (nodes.size() < 2)
            break;

        if (hasMaximumConnections())
            goto NEWMUTATION;

        unsigned int
            fromNodeId(rand.iRand(0, nodes.size()-1)),
            toNodeId(rand.iRand(0, nodes.size()-1));
        double weight(rand.dRand(-0.5, 0.5));

        while (!setConnection(fromNodeId, toNodeId, weight)) {
            fromNodeId = rand.iRand(0, nodes.size()-1);
            toNodeId = rand.iRand(0, nodes.size()-1);
        }

        HistoryEntryDataWrapper historyEntry;
        historyEntry.entry.type = 1;
        historyEntry.entry.data.mutation.type = 2;
        historyEntry.entry.data.mutation.data.type2.fromNodeId = fromNodeId;
        historyEntry.entry.data.mutation.data.type2.toNodeId = toNodeId;
        historyEntry.entry.data.mutation.data.type2.weight = weight;
        history.push_back(historyEntry);
    }
    break;
    case 3: // remove a connection
    {
        //TEMP std::cout << "MUTATION: remove a connection" << std::endl;
        if (nodes.size() == 0 || getConnectionsNumber() <= 1)
            break;

        unsigned int nodeId = rand.iRand(0, nodes.size()-1);

        while (nodes[nodeId].connections.size() == 0)
            nodeId = rand.iRand(0, nodes.size()-1);

        auto iter = nodes[nodeId].connections.begin();
        unsigned int connId = rand.iRand(0, nodes[nodeId].connections.size()-1);
        for (unsigned int i=0; i<connId; ++i)
            ++iter;

        nodes[nodeId].connections.erase(iter);

        HistoryEntryDataWrapper historyEntry;
        historyEntry.entry.type = 1;
        historyEntry.entry.data.mutation.type = 3;
        historyEntry.entry.data.mutation.data.type3.nodeId = nodeId;
        historyEntry.entry.data.mutation.data.type3.connId = connId;
        history.push_back(historyEntry);
    }
    break;
    case 4: // manipulate a connection
    {
        //TEMP std::cout << "MUTATION: modify a weight" << std::endl;
        if (getConnectionsNumber() == 0)
            break;

        unsigned int nodeId = rand.iRand(0, nodes.size()-1);
        while (nodes[nodeId].connections.size() == 0)
            nodeId = rand.iRand(0, nodes.size()-1);

        auto iter = nodes[nodeId].connections.begin();
        unsigned int connId = rand.iRand(0, nodes[nodeId].connections.size()-1);
        for (unsigned int i=0; i<connId; ++i)
            ++iter;

        iter->second.weight = rand.dRand(-0.5, 0.5);
        iter->second.delta = 0.0;

        HistoryEntryDataWrapper historyEntry;
        historyEntry.entry.type = 1;
        historyEntry.entry.data.mutation.type = 4;
        historyEntry.entry.data.mutation.data.type4.nodeId = nodeId;
        historyEntry.entry.data.mutation.data.type4.connId = connId;
        historyEntry.entry.data.mutation.data.type4.weight = iter->second.weight;
        history.push_back(historyEntry);
    }
    break;
    }

    // TEMP: check if produced nn is invalid
    for (auto& node : nodes) {
        for (auto& conn : node.connections) {
            if (nodes[conn.first].layer >= node.layer) {
                std::cout << "INVALID NETWORK PRODUCED!" << std::endl << *this;
                int asd;
                std::cin >> asd;
            }
        }
    }
}


void NodeGeneticNeuralNetwork::printGenome(void) {
    for (unsigned int nodeId=0; nodeId<nodes.size(); ++nodeId) {
        std::cout << "(" << nodeId;

        if (std::find(inputIds.begin(), inputIds.end(), nodeId) != inputIds.end())
            std::cout << "i";
        if (std::find(outputIds.begin(), outputIds.end(), nodeId) != outputIds.end())
            std::cout << "o";

        std::cout << ") ";

        for (auto& conn : nodes[nodeId].connections)
            std::cout << conn.first << " ";
    }

    std::cout << std::endl;
}


void NodeGeneticNeuralNetwork::printHistory(void) {
    for (auto& entry : history) {
        switch (entry.entry.type) {
        case 1: // mutation
            std::cout << "Entry type: Mutation" << std::endl;
            switch (entry.entry.data.mutation.type) {
            case 0: // add a node
                std::cout << "  Added a node" << std::endl;
            break;
            case 1: // remove a node
                std::cout
                    << "  Removed a node (nodeId: " << entry.entry.data.mutation.data.type1.nodeId << ")" << std::endl;
            break;
            case 2: // add a connection
                std::cout
                    << "  Added a connection (fromNodeId: " << entry.entry.data.mutation.data.type2.fromNodeId
                    << " toNodeId: " << entry.entry.data.mutation.data.type2.toNodeId
                    << " weight: " << entry.entry.data.mutation.data.type2.weight << ")" << std::endl;
            break;
            case 3: // remove a connection
                std::cout
                    << "  Removed a connection (nodeId: " << entry.entry.data.mutation.data.type3.nodeId
                    << " connId: " << entry.entry.data.mutation.data.type3.connId << ")" << std::endl;
            break;
            case 4: // manipulate a connection
                std::cout
                    << "  Manipulated a connection (nodeId: " << entry.entry.data.mutation.data.type4.nodeId
                    << " connId: " << entry.entry.data.mutation.data.type4.connId
                    << " weight: " << entry.entry.data.mutation.data.type4.weight << ")" << std::endl;
            break;
            }
        break;
        }
    }
};


/*  NodeGeneticNeuralNetwork - functions using class */


NGNN crossover(NGNN& nn1, NGNN& nn2) {
    ut::Rand rand;
    /*  same amount of inputs and outputs are required */
    // TEMP std::cout << "nn1in:" << nn1.inputIds.size() << " nn1out:" << nn1.outputIds.size() << " nn2in:" << nn2.inputIds.size() << " nn2out:" << nn2.outputIds.size() << std::endl;

    if (nn1.inputIds.size() == 0 ||
        nn2.inputIds.size() == 0 ||
        nn1.inputIds.size() != nn2.inputIds.size() ||
        nn1.outputIds.size() != nn2.outputIds.size())
        throw NGNN::UNABLE_TO_CROSSOVER;

    /*  find minimum output id to find out number of hidden neurons */
    unsigned int
        minOutputId1(nn1.outputIds[0]),
        minOutputId2(nn2.outputIds[0]);

    for (auto outputId : nn1.outputIds)
        if (outputId < minOutputId1)
            minOutputId1 = outputId;

    for (auto outputId : nn2.outputIds)
        if (outputId < minOutputId2)
            minOutputId2 = outputId;

    unsigned int crossoverId = rand.iRand(nn1.inputIds.size(), std::min(minOutputId1, minOutputId2));

    NGNN newNN((nn1.learningRate + nn2.learningRate) / 2.0, (nn1.momentum + nn2.momentum) / 2.0);

    /*  add nodes */
    for (size_t i=0; i<nn2.nodes.size(); ++i) {
        if (i < nn1.inputIds.size())
            newNN.addInputNode();
        else if (i < minOutputId2)
            newNN.addNode();
        else
            newNN.addOutputNode();
    }

    /*  set connections */
    for (size_t i=0; i<newNN.nodes.size(); ++i) {
        if (i < crossoverId) {
            for (auto& conn : nn1.nodes[i].connections) {
                if (conn.first >= newNN.nodes.size()) // if trying to connect to node which wasn't copied
                    break;

                newNN.setConnection(i, conn.first, conn.second.weight);
            }
        }
        else {
            for (auto& conn : nn2.nodes[i].connections) {
                if (conn.first >= newNN.nodes.size()) // if trying to connect to node which wasn't copied
                    break;

                newNN.setConnection(i, conn.first, conn.second.weight);
            }
        }
    }

    if (newNN.outputIds.size() > nn2.outputIds.size()) {
        std::cout << "INVALID NETWORK PRODUCED!" << std::endl << nn1 << nn2 << newNN;
        int asd;
        std::cin >> asd;
    }


    return newNN;
}

/*  NGNNEvolver - public member functions */


NGNNEvolver::NGNNEvolver(NGNN* network_, DataLoader<double>* dataLoader_) :
    network(network_), dataLoader(dataLoader_)
    {}

void NGNNEvolver::evolve(size_t numTrainingRounds,
                         size_t numEvaluationRounds,
                         double MSEConstraint,
                         double mutationThreshold,
                         double learningSpeedEpsilon) {
    /*  get initial MSE and learning speed */

    double MSE(0.0), newMSE(0.0), learningSpeed(0.0), newLearningSpeed(0.0);
    for (unsigned int i=0; i<numEvaluationRounds; ++i) {
        dataLoader->loadNewEntry();
        network->setInputValues(dataLoader->getInput());
        network->setDesiredOutputValues(dataLoader->getOutput());
        network->feedForward();
        MSE += network->getMeanSquareError();
    }
    MSE /= numEvaluationRounds;
    for (unsigned int i=0; i<numTrainingRounds; ++i) {
        dataLoader->loadNewEntry();
        network->setInputValues(dataLoader->getInput());
        network->setDesiredOutputValues(dataLoader->getOutput());
        network->feedForward();
        network->backpropagate();
    }
    newMSE = 0.0;
    for (unsigned int i=0; i<numEvaluationRounds; ++i) {
        dataLoader->loadNewEntry();
        network->setInputValues(dataLoader->getInput());
        network->setDesiredOutputValues(dataLoader->getOutput());
        network->feedForward();
        newMSE += network->getMeanSquareError();
    }
    newMSE /= numEvaluationRounds;
    learningSpeed = (MSE-newMSE)/numTrainingRounds;

    while (MSE > MSEConstraint) {
        bool refreshLearningSpeed = false;
        for (unsigned int i=0; i<numTrainingRounds; ++i) {
            dataLoader->loadNewEntry();
            network->setInputValues(dataLoader->getInput());
            network->setDesiredOutputValues(dataLoader->getOutput());
            network->feedForward();
            network->backpropagate();
        }
        newMSE = 0.0;
        for (unsigned int i=0; i<numEvaluationRounds; ++i) {
            dataLoader->loadNewEntry();
            network->setInputValues(dataLoader->getInput());
            network->setDesiredOutputValues(dataLoader->getOutput());
            network->feedForward();
            newMSE += network->getMeanSquareError();
        }
        newMSE /= numEvaluationRounds;
        newLearningSpeed = (MSE-newMSE)/numTrainingRounds;
        if (refreshLearningSpeed)
            learningSpeed = newLearningSpeed;
        MSE = newMSE;

        std::cout
            << "MSE: " << MSE << std::endl
            << "newMSE: " << newMSE << std::endl
            << "learningSpeed: " << learningSpeed << std::endl
            << "newLearningSpeed: " << newLearningSpeed << std::endl;

        /*  if the learning speed reaches the mutation threshold, start mutation */
        if (newLearningSpeed < learningSpeed*mutationThreshold) {
            mutate(100, 10, numTrainingRounds, numEvaluationRounds, newLearningSpeed*2, 100);
            refreshLearningSpeed = true;
        }

        if (learningSpeed < learningSpeedEpsilon || newLearningSpeed < learningSpeedEpsilon) {
            mutate(100, 10, numTrainingRounds, numEvaluationRounds, learningSpeedEpsilon, 100);
            refreshLearningSpeed = true;
        }
    }
}


void NGNNEvolver::mutate(size_t populationSize,
                         size_t individualsToMutate,
                         size_t numTrainingRounds,
                         size_t numEvaluationRounds,
                         double learningSpeedConstraint,
                         size_t numEpochsConstraint) {
    //TEMP begin
    sf::RenderWindow window;
    window.create(sf::VideoMode(800, 600), "Genetic Neural Networks");
    //TEMP end

    ut::Rand rand;

    /*  first find out the current MSE */
    double initMSE;
    for (unsigned int i=0; i<numEvaluationRounds; ++i) {
        dataLoader->loadNewEntry();
        network->setInputValues(dataLoader->getInput());
        network->setDesiredOutputValues(dataLoader->getOutput());
        network->feedForward();
        initMSE += network->getMeanSquareError();
    }
    initMSE /= numEvaluationRounds;

    /*  Construct the population from clones of the original network */
    std::vector<NGNN*> population;
    SorterContainer<NGNN, double, 2> sorter;
    for (unsigned int i=0; i<populationSize; ++i) {
        population.push_back(new NGNN(*network));
        sorter.addEntry(population[i],
                        { initMSE,/*
                          0.0,
                          (double)population[i]->getConnectionsNumber() / population[i]->getNodesNumber(),
                          */getConnectionHandicap(population[i]) });
        if (i>0) {
            unsigned int nMutations = rand.iRand(1, 10);
            for (unsigned int j=0; j<nMutations; ++j)
                population[i]->mutate();
        }
    }

    std::vector<std::vector<double>> inputs, outputs;
    std::vector<NGNN*> networksToReplace;
    double maxLearningSpeed = 0.0;

    for (unsigned int epoch=0; epoch<numEpochsConstraint && maxLearningSpeed<learningSpeedConstraint; ++epoch) {
        /*  TEMP: debug printing */
        std::cout << "Epoch " << epoch << ":" << std::endl;

        /*  load the data */
        inputs.clear();
        outputs.clear();

        for (unsigned int i=0; i<numTrainingRounds+numEvaluationRounds; ++i) {
            dataLoader->loadNewEntry();
            inputs.push_back(dataLoader->getInput());
            outputs.push_back(dataLoader->getOutput());
        }

        /*  train the networks and retrieve sort data */
        for (auto& individual : population) {
            for (unsigned int i=0; i<numTrainingRounds; ++i) {
                individual->setInputValues(inputs[i]);
                individual->setDesiredOutputValues(outputs[i]);
                individual->feedForward();
                individual->backpropagate();
            }
            double newMSE = 0.0;
            for (unsigned int i=0; i<numEvaluationRounds; ++i) {
                individual->setInputValues(inputs[numTrainingRounds+i]);
                individual->setDesiredOutputValues(outputs[numTrainingRounds+i]);
                individual->feedForward();
                newMSE += individual->getMeanSquareError();
            }
            newMSE /= numEvaluationRounds;

            double
                newLearningSpeed((sorter[individual][0]-newMSE)/numTrainingRounds),
                newConnsToNodesRatio((double)individual->getConnectionsNumber() / individual->getNodesNumber()),
                newConnsHandicap(getConnectionHandicap(individual));

            if (maxLearningSpeed < newLearningSpeed)
                maxLearningSpeed = newLearningSpeed;

            sorter.modifyEntry(individual,
                               { newMSE,/*
                                 newLearningSpeed,
                                 newConnsToNodesRatio,*/
                                 newConnsHandicap });
        }

        /*  sort the data */
        sorter.sort(0, SorterContainer<NGNN, double, 2>::DESCENDING);
        //sorter.sort(1, SorterContainer<NGNN, double, 4>::ASCENDING);
        //sorter.sort(2, SorterContainer<NGNN, double, 4>::ASCENDING);
        sorter.sort(1, SorterContainer<NGNN, double, 2>::DESCENDING);

        /*  some debug printing */
        std::cout << "Best MSE: " << sorter.getSortedComparisonValue(0, populationSize-1) << std::endl;
        sorter.getSortedEntry(0, populationSize-1)->printGenome();
        //std::cout << "Best learning speed: " << sorter.getSortedComparisonValue(1, populationSize-1) << std::endl;
        //sorter.getSortedEntry(1, populationSize-1)->printGenome();
        //std::cout << "Best conns / nodes ratio: " << sorter.getSortedComparisonValue(2, populationSize-1) << std::endl;
        //sorter.getSortedEntry(2, populationSize-1)->printGenome();
        std::cout << "Smallest handicap: " << sorter.getSortedComparisonValue(1, populationSize-1) << std::endl;
        sorter.getSortedEntry(1, populationSize-1)->printGenome();

        std::cout << "Worst MSE: " << sorter.getSortedComparisonValue(0, 0) << std::endl;
        sorter.getSortedEntry(0, 0)->printGenome();
        //std::cout << "Worst learning speed: " << sorter.getSortedComparisonValue(1, 0) << std::endl;
        //sorter.getSortedEntry(1, 0)->printGenome();
        //std::cout << "Worst conns / nodes ratio: " << sorter.getSortedComparisonValue(2, 0) << std::endl;
        //sorter.getSortedEntry(2, 0)->printGenome();
        std::cout << "Biggest handicap: " << sorter.getSortedComparisonValue(1, 0) << std::endl;
        sorter.getSortedEntry(1, 0)->printGenome();

        //TEMP begin
        window.clear(sf::Color(0, 0, 0));
        sorter.getSortedEntry(0, populationSize-1)->draw(window);
        window.display();
        //TEMP end

        /*  replace the worst performing individuals */
        std::vector<NGNN*> networksToReplace = sorter.getBestEntries(individualsToMutate);

        for (auto& n : networksToReplace) {
            NGNN* replacement;
            do
                replacement = population[rand.iRand(0, population.size()-1)];
            while (std::find(networksToReplace.begin(), networksToReplace.end(), replacement) != networksToReplace.end());
            *n = *replacement;

            unsigned int nMutations = 1;//rand.iRand(1, 10);
            for (unsigned int i=0; i<nMutations; ++i)
                n->mutate();
        }

        *network = *(sorter.getSortedEntry(1, populationSize-1));
    }

    for (auto& individual : population)
        delete individual;

    /*std::cout
        << "learningSpeedSort.size(): " << learningSpeedSort.size() << std::endl
        << "populationSize: " << populationSize << std::endl;*/

    window.close(); // TEMP
}


/*  NGNNEvolver - private member functions */

double NGNNEvolver::getConnectionHandicap(NGNN* n) {
    double
        nodes(n->getNodesNumber()),
        conns(n->getConnectionsNumber()),
        a = nodes*nodes+conns;

    return (102.4/(a+1) + 0.1*a);
}
