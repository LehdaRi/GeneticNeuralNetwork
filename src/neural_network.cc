#include "neural_network.hh"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdint>

//TEMP
size_t findLevel = 0;


/*  NeuralNetwork - public member functions */


NeuralNetwork::NeuralNetwork(double learningRate_, double momentum_) :
    learningRate(learningRate_),
    momentum(momentum_)
{
}


NeuralNetwork::NeuralNetwork(const std::string& fileName)
{
    loadFromFile(fileName);
}


void NeuralNetwork::setLearningRate(double learningRate_) {
    learningRate = learningRate_;
}


void NeuralNetwork::setMomentum(double momentum_) {
    momentum = momentum_;
}


unsigned int NeuralNetwork::addNode(double value) {
    Node newNode;
    newNode.value = value;
    newNode.inputSum = 0.0;
    newNode.layer = -1;
    nodes.push_back(newNode);
    layers[-1].push_back(nodes.size()-1);

    return nodes.size()-1;
}


unsigned int NeuralNetwork::addNode(std::map<unsigned int, double>& connections, double value) {
    unsigned int newNodeId = addNode(value);

    for (auto& conn : connections)
        setConnection(newNodeId, conn.first, conn.second);

    return newNodeId;
}


unsigned int NeuralNetwork::addInputNode(double value, double inputValue) {
    unsigned int newNodeId = addNode(value);
    setNodeLayer(newNodeId, 0);
    inputIds.push_back(newNodeId);
    inputValues[newNodeId] = inputValue;
    return newNodeId;
}


unsigned int NeuralNetwork::addOutputNode(double value, double desiredValue) {
    unsigned int newNodeId = addNode(value);
    outputIds.push_back(newNodeId);
    outputDesiredValues[newNodeId] = desiredValue;
    return newNodeId;
}


unsigned int NeuralNetwork::addOutputNode(std::map<unsigned int, double>& connections, double value, double desiredValue) {
    unsigned int newNodeId = addNode(connections, value);
    outputIds.push_back(newNodeId);
    outputDesiredValues[newNodeId] = desiredValue;
    return newNodeId;
}


void NeuralNetwork::removeNode(unsigned int nodeId) {
    if (nodeId >= nodes.size())
        throw INVALID_NODE_ID;

    /*  first erase node from data structures */
    nodes.erase(nodes.begin()+nodeId);

    auto it = std::find(inputIds.begin(), inputIds.end(), nodeId);
    if (it != inputIds.end())
        inputIds.erase(it);

    it = std::find(outputIds.begin(), outputIds.end(), nodeId);
    if (it != outputIds.end())
        outputIds.erase(it);

    for (auto layerIt = layers.begin(); layerIt != layers.end(); ++layerIt) {
        it = std::find(layerIt->second.begin(), layerIt->second.end(), nodeId);
        if (it != layerIt->second.end()) {
            layerIt->second.erase(it);

            break;
        }
    }

    auto it2 = inputValues.find(nodeId);
    if (it2 != inputValues.end())
        inputValues.erase(it2);

    it2 = outputDesiredValues.find(nodeId);
    if (it2 != outputDesiredValues.end())
        outputDesiredValues.erase(it2);

    for (auto& node : nodes) {
        for (auto it = node.connections.begin(); it != node.connections.end();)
            if (it->first == nodeId)
                it = node.connections.erase(it);
            else
                ++it;
    }

    /*  then shift all of the ids greater than nodeId */
    for (auto& inputId : inputIds)
        if (inputId > nodeId)
            --inputId;

    for (auto& outputId : outputIds)
        if (outputId > nodeId)
            --outputId;

    for (auto& layer : layers) {
        for (auto& layerNodeId : layer.second) {
            if (layerNodeId > nodeId)
                --layerNodeId;
        }
    }

    for (auto it = inputValues.begin(); it != inputValues.end(); ++it)
        if (it->first > nodeId) {
            it = ++inputValues.emplace(it->first-1, it->second).first;
            it = --inputValues.erase(it);
        }


    for (auto it = outputDesiredValues.begin(); it != outputDesiredValues.end(); ++it)
        if (it->first > nodeId) {
            it = ++outputDesiredValues.emplace(it->first-1, it->second).first;
            it = --outputDesiredValues.erase(it);
        }

    for (auto& node : nodes)
        for (auto it = node.connections.begin(); it != node.connections.end(); ++it)
            if (it->first > nodeId) {
                it = ++node.connections.emplace(it->first-1, it->second).first;
                it = --node.connections.erase(it);
            }

    truncateLayers();
}


void NeuralNetwork::moveNode(unsigned int fromNodeId, unsigned int toNodeId) {
    if (fromNodeId >= nodes.size() ||
        toNodeId >= nodes.size())
        throw INVALID_NODE_ID;

    if (fromNodeId == toNodeId)
        return;

    Node tempNode = nodes[fromNodeId];

    /*  move the node */
    nodes.erase(nodes.begin()+fromNodeId);
    nodes.insert(nodes.begin()+toNodeId, tempNode);

    /*  next correct all the ids */
    for (auto& inputId : inputIds) {
        if (inputId == fromNodeId)
            inputId = toNodeId;
        else {
            if (toNodeId > fromNodeId) {
                if (inputId > fromNodeId && inputId <= toNodeId)
                    --inputId;
            }
            else {
                if (inputId < fromNodeId && inputId >= toNodeId)
                    ++inputId;
            }
        }
    }

    for (auto& outputId : outputIds) {
        if (outputId == fromNodeId)
            outputId = toNodeId;
        else {
            if (toNodeId > fromNodeId) {
                if (outputId > fromNodeId && outputId <= toNodeId)
                    --outputId;
            }
            else {
                if (outputId < fromNodeId && outputId >= toNodeId)
                    ++outputId;
            }
        }
    }

    for (auto& layer : layers) {
        for (auto& layerNodeId : layer.second)
            if (layerNodeId == fromNodeId)
                layerNodeId = toNodeId;
            else {
                if (toNodeId > fromNodeId) {
                    if (layerNodeId > fromNodeId && layerNodeId <= toNodeId)
                        --layerNodeId;
                }
                else {
                    if (layerNodeId < fromNodeId && layerNodeId >= toNodeId)
                        ++layerNodeId;
                }
            }
    }

    std::map<unsigned int, double> tempInputValues;
    for (auto it = inputValues.begin(); it != inputValues.end();)
        if (it->first == fromNodeId) {
            tempInputValues[toNodeId] = it->second;
            it = inputValues.erase(it);
        }
        else {
            if (toNodeId > fromNodeId) {
                if (it->first > fromNodeId && it->first <= toNodeId) {
                    tempInputValues[it->first-1] = it->second;
                    it = inputValues.erase(it);
                }
                else
                    ++it;
            }
            else {
                if (it->first < fromNodeId && it->first >= toNodeId) {
                    tempInputValues[it->first+1] = it->second;
                    it = inputValues.erase(it);
                }
                else
                    ++it;
            }
        }
    inputValues.insert(tempInputValues.begin(), tempInputValues.end());

    std::map<unsigned int, double> tempOutputDesiredValues;
    for (auto it = outputDesiredValues.begin(); it != outputDesiredValues.end();)
        if (it->first == fromNodeId) {
            tempOutputDesiredValues[toNodeId] = it->second;
            it = outputDesiredValues.erase(it);
        }
        else {
            if (toNodeId > fromNodeId) {
                if (it->first > fromNodeId && it->first <= toNodeId) {
                    tempOutputDesiredValues[it->first-1] = it->second;
                    it = outputDesiredValues.erase(it);
                }
                else
                    ++it;
            }
            else {
                if (it->first < fromNodeId && it->first >= toNodeId) {
                    tempOutputDesiredValues[it->first+1] = it->second;
                    it = outputDesiredValues.erase(it);
                }
                else
                    ++it;
            }
        }
    outputDesiredValues.insert(tempOutputDesiredValues.begin(), tempOutputDesiredValues.end());

    std::map<unsigned int, Connection> tempConnections;
    for (auto& node : nodes) {
        for (auto it = node.connections.begin(); it != node.connections.end();)
            if (it->first == fromNodeId) {
                tempConnections[toNodeId] = it->second;
                it = node.connections.erase(it);
            }
            else {
                if (toNodeId > fromNodeId) {
                    if (it->first > fromNodeId && it->first <= toNodeId) {
                        tempConnections[it->first-1] = it->second;
                        it = node.connections.erase(it);
                    }
                    else
                        ++it;
                }
                else {
                    if (it->first < fromNodeId && it->first >= toNodeId) {
                        tempConnections[it->first+1] = it->second;
                        it = node.connections.erase(it);
                    }
                    else
                        ++it;
                }
            }
        node.connections.insert(tempConnections.begin(), tempConnections.end());
        tempConnections.clear();
    }
}


size_t NeuralNetwork::getInputNodesNumber(void) {
    return inputIds.size();
}


size_t NeuralNetwork::getOutputNodesNumber(void) {
    return outputIds.size();
}


unsigned int NeuralNetwork::getInputNodeId(unsigned int inputNodeId) {
    return inputIds[inputNodeId];
}


unsigned int NeuralNetwork::getOutputNodeId(unsigned int outputNodeId) {
    return outputIds[outputNodeId];
}


size_t NeuralNetwork::getNodesNumber(void) {
    return nodes.size();
}


size_t NeuralNetwork::getConnectionsNumber(void) {
    size_t numConns = 0;
    for (auto& node : nodes)
        numConns += node.connections.size();

    return numConns;
}


std::vector<int> NeuralNetwork::getLayerIds(void) {
    std::vector<int> layerIds;

    for (auto& layer : layers)
        layerIds.push_back(layer.first);

    return layerIds;
}


std::vector<unsigned int> NeuralNetwork::getLayerNodeIds(int layer) {
    return layers[layer];
}


std::vector<double> NeuralNetwork::getLayerNodeValues(int layer) {
    std::vector<double> layerNodeValues;

    for (auto& layerNodeId : layers[layer])
        layerNodeValues.push_back(nodes[layerNodeId].value);

    return layerNodeValues;
}


bool NeuralNetwork::hasMaximumConnections(void) {
    unsigned int nNodesBelow = 0;

    for (auto& layer : layers) {
        for (auto& nodeId : layer.second)
            if (nodes.at(nodeId).connections.size() < nNodesBelow)
                return false;
        nNodesBelow += layer.second.size();
    }

    return true;
}


void NeuralNetwork::setNodeValue(unsigned int nodeId, double value) {
    nodes[nodeId].value = value;
}


double NeuralNetwork::getNodeValue(unsigned int nodeId) {
    return nodes[nodeId].value;
}


void NeuralNetwork::setInputValue(unsigned int nodeId, double inputValue) {
    if (inputValues.find(nodeId) != inputValues.end())
        inputValues[nodeId] = inputValue;
}


void NeuralNetwork::setDesiredOutputValue(unsigned int nodeId, double desiredValue) {
    if (outputDesiredValues.find(nodeId) != outputDesiredValues.end())
        outputDesiredValues[nodeId] = desiredValue;
}


void NeuralNetwork::setInputValues(const std::vector<double>& inputs) {
    if (inputs.size() != inputIds.size())
        throw INVALID_VECTOR_SIZE;

    for (unsigned int i=0; i<inputs.size(); ++i)
        inputValues[inputIds[i]] = inputs[i];
}


void NeuralNetwork::setDesiredOutputValues(const std::vector<double>& outputs) {
    if (outputs.size() != outputIds.size())
        throw INVALID_VECTOR_SIZE;

    for (unsigned int i=0; i<outputs.size(); ++i)
        outputDesiredValues[outputIds[i]] = outputs[i];
}


void NeuralNetwork::setInputValues(const std::vector<float>& inputs) {
    if (inputs.size() != inputIds.size())
        throw INVALID_VECTOR_SIZE;

    for (unsigned int i=0; i<inputs.size(); ++i)
        inputValues[inputIds[i]] = inputs[i];
}


void NeuralNetwork::setDesiredOutputValues(const std::vector<float>& outputs) {
    if (outputs.size() != outputIds.size())
        throw INVALID_VECTOR_SIZE;

    for (unsigned int i=0; i<outputs.size(); ++i)
        outputDesiredValues[outputIds[i]] = outputs[i];
}


bool NeuralNetwork::setConnection(unsigned int nodeId, unsigned int connNodeId, double weight) {
    if (nodeId >= nodes.size() || connNodeId >= nodes.size())
        throw INVALID_NODE_ID;

    Node& node(nodes[nodeId]), connNode(nodes[connNodeId]);

    // std::cout << "node " << connNodeId << " (layer " << connNode.layer << ") to node " << nodeId << " (layer " << node.layer << ")" << std::endl;
    // std::cout << *this;

    findLevel = 0;

    /*  connections to node itself or connections between unconnected nodes (layer -1) not allowed.
        also to avoid creating loops, find out if node is feed forwarded to connective node */
    if (nodeId == connNodeId ||
        (node.layer == -1 && connNode.layer == -1) ||
        findConnection(connNodeId, nodeId, node.layer))
        return false;

    /*  add/update connection */
    node.connections[connNodeId].weight = weight;
    node.connections[connNodeId].delta = 0.0;

    /*  if node is unconnected, update its layer, no further layer updates required. */
    if (node.layer == -1) {
        setNodeLayer(nodeId, connNode.layer+1);
        return true;
    }

    /*  if connective node is unconnected and node is above layer 0, update its layer
        to layer beneath the node and return. otherwise update it to same layer
        and let the layer handling do its job. */
    if (connNode.layer == -1) {
        if (node.layer > 0) {
            setNodeLayer(connNodeId, node.layer-1);
            return true;
        }
        else
            setNodeLayer(connNodeId, node.layer);
    }

    /*  layer handling:
        check if the layer of the already connected node must be updated. (1)
        if so, all the layers following it must be updated too (2) */
    bool updated = false;
    int lastLayer = -1;

    /*  (1) */
    for (auto& conn : node.connections)
        if (nodes[conn.first].layer >= node.layer) {
            lastLayer = node.layer;
            setNodeLayer(nodeId, nodes[conn.first].layer+1);
            updated = true;
        }

    if (!updated)
        return true;

    /*  (2) */
    int layerId = lastLayer+1; // first layer to check
    for (auto layerIt = layers.find(layerId);
         layerIt != layers.end();
         layerIt = layers.find(layerId)) {
        /*std::cout
            << "layerId " << layerId << std::endl
            << "layers.size() " << layers.size() << std::endl;*/
        recheckLayer: // required if layer is modified (setNodeLayer)
        for (auto layerNodeId : layerIt->second) {
            /*std::cout
                << "  nodes[" << layerNodeId << "].layer " << nodes[layerNodeId].layer << std::endl;*/
            for (auto& conn : nodes[layerNodeId].connections) {
                /*std::cout
                    << "    nodes[" << conn.first << "].layer " << nodes[conn.first].layer << std::endl;*/
                if (nodes[conn.first].layer >= nodes[layerNodeId].layer) {
                    setNodeLayer(layerNodeId, nodes[conn.first].layer+1);
                    layerIt = layers.find(layerId);
                    goto recheckLayer;
                }
            }
        }
        ++layerId;
    }

    /*  finally truncate layers in case of empty layer creation due to layer handling */
    truncateLayers();

    return true;
}


bool NeuralNetwork::findConnection(unsigned int nodeId, unsigned int nodeIdToFind, int minLayer) {
    if (nodeId >= nodes.size() || nodeIdToFind >= nodes.size())
        throw INVALID_NODE_ID;

    if (minLayer < 0)
        minLayer = 0;

    //for (int i=0; i<findLevel; ++i)
        //std::cout << " ";
    //std::cout << "nodeId: " << nodeId << std::endl;

    if (nodeId == nodeIdToFind || nodes[nodeId].layer < minLayer)
        return false;

    for (auto& conn : nodes[nodeId].connections) {
        if (conn.first == nodeIdToFind && nodes[conn.first].layer >= minLayer)
            return true;

        ++findLevel;

        if (findConnection(conn.first, nodeIdToFind, minLayer))
            return true;

        --findLevel;
    }
    return false;
}


void NeuralNetwork::feedForward(void) {
    /*  first calculate input nodes */
    for (auto& input : inputValues)
        nodes[input.first].value = input.second;//1.0 / (1.0 + exp(-input.second)); // ACTIVATIONFUNC

    /*  layers -1 and 0 don't need to be iterated through */
    for (auto layerIt = layers.find(1); layerIt != layers.end(); ++layerIt) {
        if (layerIt->second.size() == 0)
            break;

        for (auto layerNodeId : layerIt->second) {
            /*  TODO: investigate if using this provides a performance gain */
            Node& layerNode = nodes[layerNodeId];

            if (layerNode.connections.size() > 0) {
                /*  calculate weighted input sum */
                layerNode.inputSum = 0.0;
                for (auto& conn : layerNode.connections)
                    layerNode.inputSum += nodes[conn.first].value * conn.second.weight;

                /*  calculate new node value using sigmoid activation function */
                layerNode.value = 1.0 / (1.0 + exp(-layerNode.inputSum)); // ACTIVATIONFUNC
            }
        }
    }

    return;
}


void NeuralNetwork::backpropagate(void) {
    /*  initially set the error derivatives to 0 */
    for (auto& node : nodes)
        node.errorDerivative = 0.0;

    /*  using reverse iterator */
    for (auto layerIt = layers.rbegin(); layerIt->first > 0; ++layerIt) {
        for (auto layerNodeId : layerIt->second) {
            /*  TODO: investigate if using this provides a performance gain */
            Node& layerNode = nodes[layerNodeId];

            /*  initial error derivative for outputs
                for hidden layers it has been calculated by the upper one */
            if (std::find(outputIds.begin(), outputIds.end(), layerNodeId) != outputIds.end())
                layerNode.errorDerivative = (outputDesiredValues[layerNodeId] - layerNode.value);

            /*  derivative of the activation function with respect to weighted input sum */
            layerNode.errorDerivative *= layerNode.value * (1 - layerNode.value);

            /*  backwards propagate the error derivative */
            for (auto& conn : layerNode.connections) {
                nodes[conn.first].errorDerivative += layerNode.errorDerivative * conn.second.weight;

                /*  weight update */
                conn.second.delta = learningRate * nodes[conn.first].value * layerNode.errorDerivative +
                                    momentum * conn.second.delta;
                conn.second.weight += conn.second.delta;
            }
        }
    }
}


double NeuralNetwork::getMeanSquareError(void) {
    double MSE = 0.0;
    for (auto& output : outputDesiredValues)
        MSE += pow(nodes[output.first].value - output.second, 2);

    return MSE / outputDesiredValues.size();
}


void NeuralNetwork::saveToFile(const std::string& fileName) {
    // connection list
    std::vector<ConnectionListEntry> connList;
    std::map<unsigned int, uint64_t> connListStarts;

    for (unsigned int i=0; i<nodes.size(); ++i) {
        connListStarts[i] = headerSize + connList.size()*connectionListEntrySize;
        for (auto& conn : nodes[i].connections) {
            ConnectionListEntry connListEntry;

            connListEntry.connNodeId = conn.first;
            connListEntry.weight = conn.second.weight;

            connList.push_back(connListEntry);
        }
    }

    // node list
    std::vector<NodeListEntry> nodeList;

    for (auto& layer : layers)
        for (auto& layerNodeId : layer.second) {
            NodeListEntry nodeListEntry;

            nodeListEntry.nodeId = layerNodeId;
            nodeListEntry.connListStart = connListStarts[layerNodeId];
            nodeListEntry.numConnEntries = nodes[layerNodeId].connections.size();
            if (std::find(inputIds.begin(), inputIds.end(), layerNodeId) != inputIds.end())
                nodeListEntry.type = 1;
            else if (std::find(outputIds.begin(), outputIds.end(), layerNodeId) != outputIds.end())
                nodeListEntry.type = 2;
            else
                nodeListEntry.type = 0;
            nodeListEntry.value = nodes[layerNodeId].value;

            nodeList.push_back(nodeListEntry);
        }

    // header
    Header header;
    header.learningRate = learningRate;
    header.momentum = momentum;
    header.weightPrecision = 1;
    header.nodeListStart = headerSize + connList.size()*connectionListEntrySize;
    header.nodeEntrySize = nodeListEntrySize;
    header.numNodeEntries = nodeList.size();

    // write to file
    std::ofstream file(fileName, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!file.is_open())
        throw CANNOT_OPEN_FILE;

    file.write((char*)&header.learningRate, sizeof(double));
    file.write((char*)&header.momentum, sizeof(double));
    file.write((char*)&header.weightPrecision, sizeof(uint16_t));
    file.write((char*)&header.nodeListStart, sizeof(uint64_t));
    file.write((char*)&header.nodeEntrySize, sizeof(uint16_t));
    file.write((char*)&header.numNodeEntries, sizeof(uint32_t));

    for (auto& connListEntry : connList) {
        file.write((char*)&connListEntry.connNodeId, sizeof(uint32_t));
        file.write((char*)&connListEntry.weight, sizeof(double));
    }

    for (auto& nodeListEntry : nodeList) {
        file.write((char*)&nodeListEntry.nodeId, sizeof(uint32_t));
        file.write((char*)&nodeListEntry.connListStart, sizeof(uint64_t));
        file.write((char*)&nodeListEntry.numConnEntries, sizeof(uint32_t));
        file.write((char*)&nodeListEntry.type, sizeof(uint16_t));
        file.write((char*)&nodeListEntry.value, sizeof(double));
    }

    file.close();
}


void NeuralNetwork::loadFromFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::in | std::ios::binary);

    if (!file.is_open())
        throw CANNOT_OPEN_FILE;

    // header
    char headerData[headerSize];
    file.read(headerData, headerSize);

    Header header;
    memcpy(&header.learningRate, &headerData[0], sizeof(double));
    memcpy(&header.momentum, &headerData[sizeof(double)], sizeof(double));
    memcpy(&header.weightPrecision, &headerData[2*sizeof(double)], sizeof(uint16_t));
    memcpy(&header.nodeListStart, &headerData[2*sizeof(double) + 2], sizeof(uint64_t));
    memcpy(&header.nodeEntrySize, &headerData[2*sizeof(double) + 10], sizeof(uint16_t));
    memcpy(&header.numNodeEntries, &headerData[2*sizeof(double) + 12], sizeof(uint32_t));

    // nodes
    char* nodeListData = new char[header.nodeEntrySize*header.numNodeEntries];
    file.seekg(header.nodeListStart, std::ios::beg);
    file.read(nodeListData, header.nodeEntrySize*header.numNodeEntries);

    std::vector<NodeListEntry> nodeList;
    std::map<uint32_t, unsigned int> ids;
    for (unsigned i=0; i<header.numNodeEntries; ++i) {
        NodeListEntry nodeListEntry;
        memcpy(&nodeListEntry.nodeId, &nodeListData[i*header.nodeEntrySize], sizeof(uint32_t));
        memcpy(&nodeListEntry.connListStart, &nodeListData[i*header.nodeEntrySize + 4], sizeof(uint64_t));
        memcpy(&nodeListEntry.numConnEntries, &nodeListData[i*header.nodeEntrySize + 12], sizeof(uint32_t));
        memcpy(&nodeListEntry.type, &nodeListData[i*header.nodeEntrySize + 16], sizeof(uint16_t));
        memcpy(&nodeListEntry.value, &nodeListData[i*header.nodeEntrySize + 18], sizeof(double));
        nodeList.push_back(nodeListEntry);

        switch (nodeListEntry.type) {
        case 0:
            ids[nodeListEntry.nodeId] = addNode(nodeListEntry.value);
        break;
        case 1:
            ids[nodeListEntry.nodeId] = addInputNode(nodeListEntry.value, nodeListEntry.value);
        break;
        case 2:
            ids[nodeListEntry.nodeId] = addOutputNode(nodeListEntry.value, nodeListEntry.value);
        break;
        }
    }

    // connections
    for (auto& nodeListEntry : nodeList) {
        char* connListData = new char[connectionListEntrySize*nodeListEntry.numConnEntries];
        file.seekg(nodeListEntry.connListStart, std::ios::beg);
        file.read(connListData, connectionListEntrySize*nodeListEntry.numConnEntries);

        for (unsigned i=0; i<nodeListEntry.numConnEntries; ++i) {
            ConnectionListEntry connListEntry;
            memcpy(&connListEntry.connNodeId, &connListData[i*connectionListEntrySize + 0], sizeof(uint32_t));
            memcpy(&connListEntry.weight, &connListData[i*connectionListEntrySize + 4], sizeof(double));

            setConnection(ids[nodeListEntry.nodeId], ids[connListEntry.connNodeId], connListEntry.weight);
        }

        delete connListData;
    }

    file.close();

    delete nodeListData;
}


/*  NeuralNetwork - protected member functions */


void NeuralNetwork::setNodeLayer(unsigned int nodeId, int layer) {
    /*  first erase old layer data */
    auto& layerVec = layers[nodes[nodeId].layer];
    auto old = std::find(layerVec.begin(), layerVec.end(), nodeId);
    if (old != layerVec.end())
        layerVec.erase(old);

    /*  then add the new one and update nodes layer member */
    layers[layer].push_back(nodeId);
    nodes[nodeId].layer = layer;
}


void NeuralNetwork::truncateLayers(void) {
    std::map<int, std::vector<unsigned int>> tempLayers;
    int tempTopLayer = 0;

    for (auto& layer : layers) {
        if (layer.first == -1) {
            tempLayers[-1] = layers[-1];
            continue;
        }

        for (auto& layerNodeId : layer.second)
            for (auto& conn : nodes[layerNodeId].connections)
                if (nodes[conn.first].layer == tempTopLayer) {
                    ++tempTopLayer;
                    goto stop;
                }
        stop:
        tempLayers[tempTopLayer].insert(tempLayers[tempTopLayer].end(),
                                        layer.second.begin(),
                                        layer.second.end());
        /* update internal layer members too */
        for (auto tempLayerNodeId : tempLayers[tempTopLayer])
            nodes[tempLayerNodeId].layer = tempTopLayer;
    }

    std::swap(layers, tempLayers);
}



/*  NeuralNetwork - functions using class */


std::ostream& operator<<(std::ostream& out, NeuralNetwork& nn) {
    for (auto& layer : nn.layers) {
        out << "Layer #" << layer.first << ":" << std::endl;
        for (auto& layerNodeId : layer.second) {
            out << "   Node #" << layerNodeId << ":";

            if (std::find(nn.inputIds.begin(), nn.inputIds.end(), layerNodeId) != nn.inputIds.end())
                out << " (input)" << std::endl
                    << "      Input value: " << nn.inputValues[layerNodeId];
            if (std::find(nn.outputIds.begin(), nn.outputIds.end(), layerNodeId) != nn.outputIds.end())
                out << " (output)" << std::endl
                    << "      Desired output value: " << nn.outputDesiredValues[layerNodeId];

            out << std::endl
                << "      Value: " << nn.nodes[layerNodeId].value << std::endl
                << "      Weighted input sum: " << nn.nodes[layerNodeId].inputSum << std::endl;

            if (nn.nodes[layerNodeId].connections.size() > 0)
                out << "      Connections: " << std::endl;

            for (auto& conn : nn.nodes[layerNodeId].connections)
                out << "         From node #" << conn.first << " (layer " << nn.nodes[conn.first].layer << "):  Weight: " << conn.second.weight << std::endl;
        }
    }

    return out;
}
