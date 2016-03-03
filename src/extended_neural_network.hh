#ifndef EXTENDED_NEURAL_NETWORK_HH
#define EXTENDED_NEURAL_NETWORK_HH


#include "neural_network.hh"
#include "data_loader.hh"

#include <SFML/Graphics.hpp>


class ExtendedNeuralNetwork : public NeuralNetwork {
public:
    /*  constructors/destructors */
    ExtendedNeuralNetwork(double learningRate_, double momentum_);
    virtual ~ExtendedNeuralNetwork(void) {}

    /*  returns a percentage of output nodes connected to inputs (0-1) */
    double getConnectivityPercentage(void);

    void draw(sf::RenderTarget& target);
    void drawImage(sf::RenderTarget& target,
                   const float xPos,
                   const float yPos,
                   const float xSize,
                   const float ySize,
                   const unsigned int xDivs,
                   const unsigned int yDivs);
};


#endif // EXTENDED_NEURAL_NETWORK_HH
