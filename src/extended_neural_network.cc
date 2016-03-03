#include "extended_neural_network.hh"


ExtendedNeuralNetwork::ExtendedNeuralNetwork(double learningRate_, double momentum_) :
    NeuralNetwork(learningRate_, momentum_)
{}

double ExtendedNeuralNetwork::getConnectivityPercentage(void) {
    unsigned int numConns = 0;
    for (auto& outputNodeId : outputIds) {
        for (auto& inputNodeId : inputIds) {
            if (findConnection(outputNodeId, inputNodeId)) {
                ++numConns;
                break;
            }
        }
    }
    return (double)numConns/outputIds.size();
}

void ExtendedNeuralNetwork::draw(sf::RenderTarget& target) {
    sf::CircleShape circle;
    circle.setRadius(10.0f);
    sf::VertexArray lines(sf::Lines, getConnectionsNumber()*2);

    std::map<unsigned int, std::pair<float, float>> coords;

    int xPos(0);
    size_t nLines = 0;

    for (auto& layer : layers) {
        unsigned yPos = 0;
        for (auto& nodeId : layer.second) {
            coords[nodeId] = std::make_pair(25.0f + 50.0f*xPos, 25.0f + 30.0f*yPos++);

            for (auto& conn : nodes[nodeId].connections) {
                lines[nLines*2].position = sf::Vector2f(coords[conn.first].first+10.0f, coords[conn.first].second+10.0f);
                lines[nLines*2+1].position = sf::Vector2f(coords[nodeId].first+10.0f, coords[nodeId].second+10.0f);
                lines[nLines*2].color = sf::Color(128+conn.second.weight*128, 128+conn.second.weight/10*128, 128+conn.second.weight/100*128);
                lines[nLines*2+1].color = lines[nLines*2].color;
                ++nLines;
            }
        }
        ++xPos;
    }

    target.draw(lines);

    for (auto& layer : layers) {
        for (auto& nodeId : layer.second) {
            circle.setPosition(coords[nodeId].first, coords[nodeId].second);
            circle.setFillColor(sf::Color(255, 0, 255*nodes[nodeId].value));

            if (std::find(inputIds.begin(), inputIds.end(), nodeId) != inputIds.end()) {
                circle.setOutlineThickness(3.0f);
                circle.setOutlineColor(sf::Color(255, 128, 0));
            }
            else if (std::find(outputIds.begin(), outputIds.end(), nodeId) != outputIds.end()) {
                circle.setOutlineThickness(3.0f);
                circle.setOutlineColor(sf::Color(0, 128, 255));
            }
            else
                circle.setOutlineThickness(0.0f);

            target.draw(circle);
        }
    }
}

void ExtendedNeuralNetwork::drawImage(sf::RenderTarget& target,
                                      const float xPos,
                                      const float yPos,
                                      const float xSize,
                                      const float ySize,
                                      const unsigned int xDivs,
                                      const unsigned int yDivs) {
    float xs(xSize/xDivs), ys(ySize/yDivs);

    sf::VertexArray arr(sf::TrianglesStrip, 2*xDivs+2);

    for (unsigned int y=0; y<yDivs; ++y) {
        for (unsigned int x=0; x<xDivs+1; ++x) {
            setInputValue(inputIds[0], (1.0f/xDivs)*x);
            setInputValue(inputIds[1], (1.0f/yDivs)*y);
            feedForward();
            arr[2*x].position = sf::Vector2f(xPos + xs*x, yPos + ys*y);
            arr[2*x].color = sf::Color(getNodeValue(outputIds[0])*255,
                                       getNodeValue(outputIds[1])*255,
                                       getNodeValue(outputIds[2])*255);

            setInputValue(inputIds[0], (1.0f/xDivs)*x);
            setInputValue(inputIds[1], (1.0f/yDivs)*(y+1));
            feedForward();
            arr[2*x+1].position = sf::Vector2f(xPos + xs*x, yPos + ys*(y+1));
            arr[2*x+1].color = sf::Color(getNodeValue(outputIds[0])*255,
                                         getNodeValue(outputIds[1])*255,
                                         getNodeValue(outputIds[2])*255);
        }
        target.draw(arr);
    }
}
