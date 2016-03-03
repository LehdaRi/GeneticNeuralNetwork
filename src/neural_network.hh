/**
    neural_network.hh

    Neural network impementation, version 3.

    TODO add description here

    @author     Miika Lehtimäki
    @version    3.0
    @date       2014-03-19
**/


#ifndef NEURAL_NETWORK_HH
#define NEURAL_NETWORK_HH


#include <map>
#include <vector>


class NeuralNetwork {
public:
    /*  exception */
    enum Exception {
        INVALID_NODE_ID,
        INVALID_VECTOR_SIZE,
        CANNOT_OPEN_FILE
    };

    /*  network connection struct */
    struct Connection {
        double weight;
        double delta;
    };

    /*  network node struct */
    struct Node {
        double value, inputSum, errorDerivative;
        /*  layer, negative if the node isn't connected to the network */
        int layer;

        /*  input connections aka. weight / weight delta values bound to input node Id:s
            weight  weight delta */
        std::map<unsigned int, Connection> connections;
    };

    /*  file handling structs */
    struct Header {
        double   learningRate;
        double   momentum;
        uint16_t weightPrecision;   // 0: float, 1: double
        uint64_t nodeListStart;     // position in file
        uint16_t nodeEntrySize;     // node entry size in bytes
        uint32_t numNodeEntries;    // number of node entries
    };
    unsigned int headerSize = 2*sizeof(double) + 16;

    struct NodeListEntry {
        uint32_t nodeId;            // id of the node
        uint64_t connListStart;     // position in file
        uint32_t numConnEntries;    // number of weight entries
        uint16_t type;              // 0: none(hidden), 1: input, 2: output
        double   value;
    };
    unsigned int nodeListEntrySize = 26;

    struct ConnectionListEntry {
        uint32_t connNodeId;        // id of the node the weight is connected to
        double   weight;
    };
    unsigned int connectionListEntrySize = 4 + sizeof(double);

    /*  constructors/destructors */
    NeuralNetwork(double learningRate_, double momentum_);
    NeuralNetwork(const std::string& fileName);
    virtual ~NeuralNetwork(void) {}

    /*  set learning rate/momentum */
    void setLearningRate(double learningRate_);
    void setMomentum(double momentum_);

    /*  add node to network
        input nodes are created on layer 0
            return value: id of the added node */
    unsigned int addNode(double value = 0.0);
    unsigned int addNode(std::map<unsigned int, double>& connections, double value = 0.0);
    unsigned int addInputNode(double value = 0.0, double inputValue = 0.0);
    unsigned int addOutputNode(double value = 0.0, double desiredValue = 0.0);
    unsigned int addOutputNode(std::map<unsigned int, double>& connections, double value = 0.0, double desiredValue = 0.0);

    /*  remove node from network
        note: changes node ids */
    void removeNode(unsigned int nodeId);

    /*  move node from ID to another
        note: changes node ids */
    void moveNode(unsigned int fromNodeId, unsigned int toNodeId);

    /*  getters for input/output node
        note: inputNodeId/outputNodeId don't refer to indices of nodes vector
        but to indices of vectors inputIds and outputIds
        max value for these are returned by member functions
        getInputNodesNumber and getOutputNodesNumber */
    size_t getInputNodesNumber(void);
    size_t getOutputNodesNumber(void);
    unsigned int getInputNodeId(unsigned int inputNodeId);
    unsigned int getOutputNodeId(unsigned int outputNodeId);

    size_t getNodesNumber(void);
    size_t getConnectionsNumber(void);

    /*  getters for entire layers */
    std::vector<int> getLayerIds(void);
    std::vector<unsigned int> getLayerNodeIds(int layer);
    std::vector<double> getLayerNodeValues(int layer);

    bool hasMaximumConnections(void);

    /*  set/get node value / get node layer */
    void setNodeValue(unsigned int nodeId, double value);
    double getNodeValue(unsigned int nodeId);
    unsigned int getNodeLayer(unsigned int nodeId);

    /*  set input value / desired output value */
    void setInputValue(unsigned int nodeId, double inputValue);
    void setDesiredOutputValue(unsigned int nodeId, double desiredValue);
    /*  set values with a vector of doubles
        NOTE: its size must match the amount of input/output neurons
        in the network */
    void setInputValues(const std::vector<double>& inputs);
    void setInputValues(const std::vector<float>& inputs);
    void setDesiredOutputValues(const std::vector<double>& outputs);
    void setDesiredOutputValues(const std::vector<float>& outputs);

    /*  Set node connection, can be used for adding new connections or
        changing the weight of an existing one.
        Since only layered, feed-forward networks can be constructed,
        the connection can be made only if nodes layer is equal or greater than
        the layer of the connective node.
            nodeId: ID of the node
            connNodeId: ID of connective node
            weight: weight of the connection */
    bool setConnection(unsigned int nodeId, unsigned int connNodeId, double weight);

    /*  recursively finds node id from connections and sub-connections.
        you can specify minimum layer from wich to search from.
        returns true if connection is found, false otherwise.
            nodeId: node to start search from
            nodeIdToFind: node id to find (yeah, really)
            minLayer: minimum layer to search from */
    bool findConnection(unsigned int nodeId, unsigned int nodeIdToFind, int minLayer = -1);

    /*  feed-forward & backpropagation */
    void feedForward(void);
    void backpropagate(void);

    /*  mean square error */
    double getMeanSquareError(void);

    /*  save/load network into file */
    void saveToFile(const std::string& fileName);
    void loadFromFile(const std::string& fileName);

    /*  operator overloads */
    friend std::ostream& operator<<(std::ostream& out, NeuralNetwork& nn);

    /*  rule of 3 */
    //NeuralNetwork(const NeuralNetwork&) = delete;
    //NeuralNetwork& operator=(const NeuralNetwork&) = delete;

protected:
    double learningRate, momentum;

    /*  nodes */
    std::vector<Node> nodes;

    /*  input/output node ids */
    std::vector<unsigned int> inputIds, outputIds;
    /*  node ids arranged by layers */
    std::map<int, std::vector<unsigned int>> layers;
    /*  input values / desired output values - node ids mapped to values */
    std::map<unsigned int, double> inputValues;
    std::map<unsigned int, double> outputDesiredValues;

    /*  member function to set node layer (never modify layers map or Nodes layer member directly!) */
    void setNodeLayer(unsigned int nodeId, int layer);

    /*  should be called whenever connections are added/removed */
    void truncateLayers(void);
};


#endif // NEURAL_NETWORK_HH
