using System;
using System.Diagnostics.Metrics;
using System.Diagnostics;

class MLPApproximator
{
    // Hyperparameters
    private const int inputNodes = 1;
    private const int hiddenNodes = 6; // Adjust between 4 to 8 as needed
    private const int outputNodes = 1;
    private const double learningRate = 0.01;
    private const int epochs = 10000;

    // Weights and biases
    private double[,] weightsInputHidden;
    private double[,] weightsHiddenOutput;
    private double[] biasHidden;
    private double[] biasOutput;

    // Activation functions
    private static Random random = new Random();

    public MLPApproximator()
    {
        // Initialize weights and biases with random values
        weightsInputHidden = new double[inputNodes, hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes, outputNodes];
        biasHidden = new double[hiddenNodes];
        biasOutput = new double[outputNodes];

        InitializeWeights();
    }

    private void InitializeWeights()
    {
        // Initialize weights and biases with random small values
        for (int i = 0; i < inputNodes; i++)
        {
            for (int j = 0; j < hiddenNodes; j++)
            {
                weightsInputHidden[i, j] = random.NextDouble() * 2 - 1; // Values between -1 and 1
            }
        }
        for (int i = 0; i < hiddenNodes; i++)
        {
            for (int j = 0; j < outputNodes; j++)
            {
                weightsHiddenOutput[i, j] = random.NextDouble() * 2 - 1; // Values between -1 and 1
            }
        }
        for (int i = 0; i < hiddenNodes; i++)
        {
            biasHidden[i] = random.NextDouble() * 2 - 1;
        }
        for (int i = 0; i < outputNodes; i++)
        {
            biasOutput[i] = random.NextDouble() * 2 - 1;
        }
    }

    // Sigmoid activation function
    private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    // Derivative of sigmoid for backpropagation
    private double SigmoidDerivative(double x) => x * (1.0 - x);

    // Hyperbolic Tangent activation function
    private double Tanh(double x) => Math.Tanh(x);

    // Derivative of Tanh
    private double TanhDerivative(double x) => 1 - Math.Pow(x, 2);

    // Linear activation function (used for output neuron)
    private double Linear(double x) => x;

    // Feedforward step
    public double FeedForward(double x)
    {
        // Input to hidden layer
        double[] hiddenInputs = new double[hiddenNodes];
        double[] hiddenOutputs = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++)
        {
            hiddenInputs[i] = x * weightsInputHidden[0, i] + biasHidden[i];
            hiddenOutputs[i] = Tanh(hiddenInputs[i]); // Using tanh activation function
        }

        // Hidden to output layer
        double outputInput = 0.0;
        for (int i = 0; i < hiddenNodes; i++)
        {
            outputInput += hiddenOutputs[i] * weightsHiddenOutput[i, 0];
        }
        outputInput += biasOutput[0];

        return Linear(outputInput); // Linear output
    }

    // Backpropagation and weight update
    public void Train(double[] inputs, double[] expectedOutputs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                // Feedforward pass
                double[] hiddenInputs = new double[hiddenNodes];
                double[] hiddenOutputs = new double[hiddenNodes];
                for (int j = 0; j < hiddenNodes; j++)
                {
                    hiddenInputs[j] = inputs[i] * weightsInputHidden[0, j] + biasHidden[j];
                    hiddenOutputs[j] = Tanh(hiddenInputs[j]);
                }

                double outputInput = 0.0;
                for (int j = 0; j < hiddenNodes; j++)
                {
                    outputInput += hiddenOutputs[j] * weightsHiddenOutput[j, 0];
                }
                outputInput += biasOutput[0];

                double output = Linear(outputInput);

                // Calculate output error
                double error = expectedOutputs[i] - output;

                // Backpropagate the error
                // Output to hidden layer weight update
                for (int j = 0; j < hiddenNodes; j++)
                {
                    double deltaOutput = error * 1; // Derivative of Linear is 1
                    weightsHiddenOutput[j, 0] += learningRate * deltaOutput * hiddenOutputs[j];
                }
                biasOutput[0] += learningRate * error;

                // Hidden to input layer weight update
                for (int j = 0; j < hiddenNodes; j++)
                {
                    double deltaHidden = error * weightsHiddenOutput[j, 0] * TanhDerivative(hiddenOutputs[j]);
                    weightsInputHidden[0, j] += learningRate * deltaHidden * inputs[i];
                    biasHidden[j] += learningRate * deltaHidden;
                }
            }

            // Optional: Print the error at each epoch
            if (epoch % 1000 == 0)
            {
                double totalError = 0.0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    double prediction = FeedForward(inputs[i]);
                    totalError += Math.Pow(expectedOutputs[i] - prediction, 2);
                }
                Console.WriteLine($"Epoch {epoch}, Error: {totalError}");
            }
        }
    }

    static void Main(string[] args)
    {
        // Training data
        double[] inputs = new double[20];
        double[] expectedOutputs = new double[20];

        for (int i = 0; i < 20; i++)
        {
            inputs[i] = 0.1 + (1.0 / 22) * i;
            expectedOutputs[i] = (1 + 0.6 * Math.Sin(2 * Math.PI * inputs[i] / 0.7) + 0.3 * Math.Sin(2 * Math.PI * inputs[i])) / 2;
        }

        // Create and train the MLP
        MLPApproximator mlp = new MLPApproximator();
        mlp.Train(inputs, expectedOutputs);

        // Test the network
        Console.WriteLine("Predictions after training:");
        foreach (var input in inputs)
        {
            Console.WriteLine($"Input: {input}, Predicted Output: {mlp.FeedForward(input)}");
        }
    }
}


                                    //E   X    P    L    A   N   A   T   I   O    N 
//Network Structure:

//One input node, a hidden layer with 6 neurons (can adjust between 4 to 8), and one output node.
//The hidden layer uses the hyperbolic tangent activation function (Tanh), and the output neuron uses a linear activation function (Linear).
//Training:

//The training process uses backpropagation. We compute the error at the output, propagate it backward, and adjust the weights using gradient descent.
//We train the network for a fixed number of epochs (epochs = 10000) with a learning rate of 0.01.
//Input Data:

//The inputs are 20 values between 0.1 and 1 evenly spaced.
//The expected outputs are computed based on the formula y = (1 + 0.6 * sin(2 * pi * x / 0.7)) +0.3 * sin(2 * pi * x)) / 2.
//Prediction:

//After training, the network predicts the output for the same inputs used during training.