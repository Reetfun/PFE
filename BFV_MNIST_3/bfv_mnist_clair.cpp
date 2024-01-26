/*
    Spécifités :
        - BFV
        - MNIST
    TODO: 
        - Il faut mettre les poids et biais sur tout un batch
*/

#include "openfhe.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace lbcrypto;

string initDirectory(string path, int nb_test) {

    auto currentTimePoint = chrono::system_clock::now();
    time_t currentTime = chrono::system_clock::to_time_t(currentTimePoint);
    tm* localTime = localtime(&currentTime);
    stringstream timestamp;
    timestamp << put_time(localTime, "%Y%m%d_%H%M%S");

    string folderName = "Results_" + timestamp.str() + "_deg" + "_nb" + to_string(nb_test);
    string full_path = path + "/" + folderName;
    filesystem::create_directory(full_path);

    return full_path;
}

vector<double> splitBias(const string &line, char delimiter)
{
    vector<double> values;
    istringstream iss(line);
    string item;
    while (getline(iss, item, delimiter))
    {
        values.push_back(stod(item));
    }
    return values;
}

vector<vector<double>> splitWeights(const string &line, char delimiter, int row, int col)
{
    vector<vector<double>> values;
    istringstream iss(line);
    string item;
    int i = 0;

    while (i < row)
    {
        getline(iss, item, delimiter);
        vector<double> row_vector;
        for (int j = 0; j < col; j++)
        {
            row_vector.push_back(stod(item));
            getline(iss, item, delimiter);
        }
        values.push_back(row_vector);
        i++;
    }
    return values;
}

vector<vector<vector<double>>> splitWeightsMap(const string &line, char delimiter, int row, int col, int map_count)
{
    vector<vector<vector<double>>> values;
    istringstream iss(line);
    string item;
    for (int k = 0; k < map_count; k++) {
        vector<vector<double>> map;
        for (int j = 0; j < row; j++)
        {
            vector<double> row_vector;
            for (int l = 0; l < col; l++)
            {
                getline(iss, item, delimiter);
                row_vector.push_back(stod(item));
            }
            map.push_back(row_vector);
        }
        values.push_back(map);
    }
    return values;
}

vector<double> splitWeightsFlatten(const string &s, char delim) {
    vector<double> elems;
    istringstream iss(s);
    string item;
    while (getline(iss, item, delim)) {
        elems.push_back(stod(item));
    }
    return elems;

}

void getWeights(vector<vector<vector<double>>> &weights_0, vector<double> &bias_0, vector<double> &weights_1, vector<double> &bias_1, vector<double> &weights_2, vector<double> &bias_2) {
    ifstream file("../BFV_MNIST_3/weights_biases.txt");
    if (!file.is_open()) {
        cerr << "Error in getWeights !" << endl;
    }
    string line;

    getline(file, line);
    weights_0 = splitWeightsMap(line, ' ', 5, 5, 5);

    getline(file, line);
    bias_0 = splitBias(line, ' ');

    getline(file, line);
    weights_1 = splitWeightsFlatten(line, ' ');

    getline(file, line);
    bias_1 = splitBias(line, ' ');

    getline(file, line);
    weights_2 = splitWeightsFlatten(line, ' ');

    getline(file, line);
    bias_2 = splitBias(line, ' ');

    file.close();    
}

vector<int> getLabels(int batch_size) {
    vector<int> labelsResult;
    ifstream inputFile("../BFV_MNIST_3/MNIST-28x28-test.txt");

    while (!inputFile.eof() && int(labelsResult.size()) < batch_size)
    {
        string line;
        getline(inputFile, line);

        if (line.empty())
            continue;

        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        labelsResult.push_back(stoi(tokens[0]));
    }

    return labelsResult;
}

vector<vector<double>> getInputs(int batch_size, double normalizationFactor, double scale) {
    vector<vector<double>> inputData;
    vector<vector<pair<int64_t, double>>> input;
    ifstream file("../BFV_MNIST_3/MNIST-28x28-test.txt");

    while (!file.eof() && int(inputData.size()) < batch_size)
    {
        string line;
        getline(file, line);

        if (line.empty())
            continue;

        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};

        vector<pair<int, double>> features;

        for (size_t k = 2; k < tokens.size(); ++k)
        {
            string sub = tokens[k];
            size_t pos = sub.find(':');
            int coordinate = stoi(sub.substr(0, pos));
            double value = stod(sub.substr(pos + 1));
            features.push_back(make_pair(coordinate, value * normalizationFactor));
        }

        vector<double> row;
        for (int i = 0; i < 28*28; i++) {
            if (i == features[0].first) {
                row.push_back(round(features[0].second * scale));
                features.erase(features.begin());
            } else {
                row.push_back(0);
            }
        }
        inputData.push_back(row);
    }
    return inputData;
}

vector<vector<double>> transposeData(vector<vector<double>> data) {
   vector<vector<double>> transposed_data;
   for (int i = 0; i < int(data[0].size()); i++) {
      vector<double> column;
      for (int j = 0; j < int(data.size()); j++) {
         column.push_back(data[j][i]);
      }
      transposed_data.push_back(column);
   }
   return transposed_data;
}

vector<double> multVectorAndInt(vector<double> data, double weight) {
    vector<double> result;
    for (int i = 0; i < int(data.size()); i++) {
        result.push_back(data[i] * round(weight));
    }
    return result;
} 

vector<double> addTwoVectors(vector<double> data1, vector<double> data2) {
    vector<double> result;
    for (int i = 0; i < int(data1.size()); i++) {
        result.push_back(data1[i] + data2[i]);
    }
    return result;
}

vector<double> addVectorAndBias(vector<double> data, double bias) {
    vector<double> result;
    for (int i = 0; i < int(data.size()); i++) {
        result.push_back(data[i] + bias);
    }
    return result;
}

vector<vector<double>> ptConvLayer(vector<vector<double>> inputs, vector<vector<vector<double>>> weights, vector<double> bias, int kernel_dim, int stride, int dim, double scale, int padding, int batch_size) {
    // Padding en bas et à droite
    vector<vector<double>> conv_results;
    int output_dim = (dim - kernel_dim + padding) / stride + 1; // = 13

    for (int k = 0; k < int(weights.size()); k++) { // Pour chaque filtre
        for (int j = 0; j < output_dim; j++) { // Pour chaque ligne de l'image de sortie
            for (int i = 0; i < output_dim; i++) { // Pour chaque colonne de l'image de sortie
                vector<double> kernel_result(batch_size, 0.0);
                for (int x = 0; x < kernel_dim; x++) { // Pour chaque ligne du filtre
                    for (int y = 0; y < kernel_dim; y++) { // Pour chaque colonne du filtre
                        double weight = weights[k][x][y];
                        int row = stride * j + x; // Ligne de l'image d'entrée
                        int col = stride * i + y; // Colonne de l'image d'entrée
                        if (row < dim && col < dim) { // Padding en bas et à droite !
                            // vector<double> mult_result = multVectorAndInt(inputs[row*(dim+1) + col], weight * scale);
                            vector<double> mult_result = multVectorAndInt(inputs[row*dim + col], weight * scale);
                            kernel_result = addTwoVectors(kernel_result, mult_result);
                        }
                    }
                }
                kernel_result = addVectorAndBias(kernel_result, bias[k]);
                conv_results.push_back(kernel_result);
            }
        }
    }
    return conv_results;
}

vector<vector<double>> ptSquareActivation(vector<vector<double>> inputs) {
    vector<vector<double>> output;
    for (int i = 0; i < int(inputs.size()); i++) {
        vector<double> row;
        for (int j = 0; j < int(inputs[0].size()); j++) {
            row.push_back(inputs[i][j] * inputs[i][j]);
        }
        output.push_back(row);
    }
    return output;
}

void saveResults(string path, vector<vector<double>> decrypted_results, vector<int> labels, int batch_size, double scale, int nb_true, int nb_false, int nb_total, vector<double> time_results) {
    ofstream file;
    file.open(path + "/" + "final_result.csv");
    if (file.is_open()) {
        for (int i = 0; i < batch_size; i++) {
            file << i << "," << decrypted_results[0][i] << "," << decrypted_results[1][i] << "," << decrypted_results[2][i] << "," << decrypted_results[3][i] << "," << decrypted_results[4][i] << "," << decrypted_results[5][i] << "," << decrypted_results[6][i]  << "," << decrypted_results[7][i] << "," << decrypted_results[8][i] << "," << decrypted_results[9][i] << "," << labels[i] << endl;
        }
        file.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    ofstream file2;
    file2.open(path + "/" + "time.csv");
    if (file2.is_open()) {
        file2 << "step,time" << endl;
        file2 << "convolution," << time_results[0] << endl;
        file2 << "square_activation_1," << time_results[1] << endl;
        file2 << "fully_connected_layer_2," << time_results[2] << endl;
        file2 << "square_activation_2," << time_results[3] << endl;
        file2 << "fully_connected_layer_3," << time_results[4] << endl;
        file2.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    ofstream file3;
    file3.open(path + "/" + "recap.csv");
    if (file3.is_open()) {
        file3 << "nb_true,nb_false,nb_total,precision,scale" << endl;
        file3 << nb_true << "," << nb_false << "," << nb_total << "," << double(nb_true)/double(nb_total) << "," << scale << endl;
        file3.close();
    } else {
        cerr << "Error opening the file." << endl;
    }
}

vector<double> scaleBias(vector<double> bias, double scale) {
    vector<double> scaled_bias;
    for (int i = 0; i < int(bias.size()); i++) {
        scaled_bias.push_back(round(bias[i] * scale));
    }
    return scaled_bias;
}

void saveResultsClear(string path, vector<vector<double>> results, vector<int> labels, int batch_size) {
    ofstream file;
    file.open(path + "/" + "final_result.csv");
    if (file.is_open()) {
        for (int i = 0; i < batch_size; i++) {
            file << i << "," << results[0][i] << "," << results[1][i] << "," << results[2][i] << "," << results[3][i] << "," << results[4][i] << "," << results[5][i] << "," << results[6][i] << "," << results[7][i] << "," << results[8][i] << "," << results[9][i] << "," << labels[i] << endl;
        }
        file.close();
    } else {
        cerr << "Error opening the file." << endl;
    }
}

void add_many(vector<double> &dest, vector<vector<double>> src) {
    for (int i = 0; i < int(src.size()); i++) {
        for (int j = 0; j < int(src[0].size()); j++) {
            dest[j] += src[i][j];
        }
    }
}

vector<vector<double>> ptFCLayer(vector<vector<double>> inputs, vector<double> weights, vector<double> bias, double scale, int batch_size) {
    vector<vector<double>> fc_results(weights.size() / inputs.size(), vector<double>(batch_size, 0));

    for (int i = 0; i < int(weights.size()); i += int(inputs.size())) { 
        vector<vector<double>> dots;
        for (int x = 0; x < int(inputs.size()); x++) {
            vector<double> row = multVectorAndInt(inputs[x], weights[i + x]*scale);
            dots.push_back(row);
        }
        int index = (i / int(inputs.size()));
        add_many(fc_results[index], dots);
        fc_results[index] = addVectorAndBias(fc_results[index], bias[i/int(inputs.size())]);
    }
    return fc_results;
}

int main(int argc, char *argv[]) {
    TimeVar t;
    double processing_time(0.0);
    double input_scale = 2.0; // 16
    double conv_scale = 2.0; // 32
    double fc_scale = 32.0; // 32
    double output_scale = 16.0; // 32
    vector<double> time_results;

    // // ==================================================================================
    // // Etape 1
    // // Initialisation des paramètres
    // // ==================================================================================

    int batch_size = 8192; 
    double normalization_factor = 1.0/256.0;

    // // ==================================================================================      
    // // Etape 1.1
    // // Initialisation des répertoires de sauvegarde des résultats
    // // ==================================================================================
    string dir_path = initDirectory("../BFV_MNIST_3/results_pt", batch_size);

    // ==================================================================================
    // Etape 2
    // Récupération des données
    // ==================================================================================
    cout << "\n{...} Getting data" << endl;
    vector<int> labels = getLabels(batch_size);
    vector<vector<double>> input = transposeData(getInputs(batch_size, normalization_factor, input_scale));


    cout << "\n\ninput.size() : " << input.size() << endl;
    cout << "input[0].size() : " << input[0].size() << endl;

    cout << "\n=================\n" << endl;

    for (int i = 0; i < 28; i++) {
        if (i+1 < 10) cout << i+1 << "  ";
        else cout << i+1 << " ";
    }
    cout << endl;


    for (int j = 0; j < 28; j++) {
        for (int i = 0; i < 28; i++) {
            cout << input[j*28 + i][1] << "  ";
        }
        cout << endl;
    }

    // ==================================================================================
    // Etape 4
    // Récupération des poids et des biais
    // ==================================================================================
    cout << "\n{...} Getting weights and biases" << endl;
    vector<vector<vector<double>>> weights_0;
    vector<double> bias_0, bias_1, bias_2, weights_1_flatten, weights_2_flatten;
    getWeights(weights_0, bias_0, weights_1_flatten, bias_1, weights_2_flatten, bias_2);
    vector<double> bias_scale_0 = scaleBias(bias_0, conv_scale*input_scale);
    vector<double> bias_scale_1 = scaleBias(bias_1, fc_scale*conv_scale*input_scale);
    vector<double> bias_scale_2 = scaleBias(bias_2, output_scale*fc_scale*conv_scale*input_scale);

    cout << "\n=================\n" << endl;


    for (int j = 0; j < int(weights_0[0].size()); j++) {
        for (int k = 0; k < int(weights_0[0][0].size()); k++) {
            cout << weights_0[0][j][k] << " ";
            // cout << weights_0[1][j][k] << " ";
        }
        cout << endl;
    }
    cout << endl;

    // ==================================================================================
    // Etape 5
    // ConvLayer 1
    // ==================================================================================
    cout << "\n{...} ConvLayer 1" << endl;
    int kernel_dim = 5; // Dimension du filtre
    int stride = 2; // Pas de déplacement du filtre (2, 2) : 2 en bas et 2 à droite
    int padding = 1; // Padding à appliquer aux bords de l'image (nbre de rangée de 0 à ajouter) en haut et à gauche  
    int dim = 28; // Dimension de l'image

    TIC(t);
    vector<vector<double>> conv_results = ptConvLayer(input, weights_0, bias_scale_0, kernel_dim, stride, dim, conv_scale, padding, batch_size);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> Convolution done [" << processing_time << " ms]" << endl;

    cout << "\n=================\n" << endl;

    cout << conv_results.size() << "\n" << endl;

    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 13; j++) {
            cout << conv_results[i*13 + j][1] << " ";
        }
        cout << endl;
    }

    // ==================================================================================
    // Etape 6
    // ActivationLayer Square 1
    // ==================================================================================
    cout << "\n{...} ActivationLayer Square 1" << endl;
    TIC(t);
    vector<vector<double>> activation_results_1 = ptSquareActivation(conv_results);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> Square done [" << processing_time << " ms]" << endl;    

    // ==================================================================================
    // Etape 7
    // ConvLayer 2
    // ==================================================================================
    cout << "\n{...} FCLayer 2" << endl;
    TIC(t);
    vector<vector<double>> fc_results_2 = ptFCLayer(activation_results_1, weights_1_flatten, bias_scale_1, fc_scale, batch_size);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> FCLayer done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 8
    // ActivationLayer Square 2
    // ==================================================================================
    cout << "\n{...} ActivationLayer Square 2" << endl;
    TIC(t);
    vector<vector<double>> activation_results_2 = ptSquareActivation(fc_results_2);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> Square done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 9
    // ConvLayer 3
    // ==================================================================================
    cout << "\n{...} FCLayer 3" << endl;
    TIC(t);
    vector<vector<double>> fc_results_3 = ptFCLayer(activation_results_2, weights_2_flatten, bias_scale_2, output_scale, batch_size);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> FCLayer done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 10
    // Calcul Precision
    // ==================================================================================
    cout << "\n{...} Calculating precision" << endl;
    int nb_total = 0;
    int nb_true = 0;
    int nb_false = 0;

    for (int i = 0; i < batch_size; i++) {
        int64_t max = fc_results_3[0][i] / (output_scale * fc_scale * conv_scale * input_scale);
        int index_max = 0;
        for (int j = 1; j < 10; j++) {
            if (fc_results_3[j][i] / (output_scale * fc_scale * conv_scale * input_scale) > max) {
                max = fc_results_3[j][i] / (output_scale * fc_scale * conv_scale * input_scale);
                index_max = j;
            }
        }
        if (index_max == labels[i]) {
            nb_true++;
        } else {
            nb_false++;
        }
        nb_total++;
    }

    cout << "nb_true : " << nb_true << endl;
    cout << "nb_false : " << nb_false << endl;
    cout << "nb_total : " << nb_total << endl;
    cout << "Precision : " << double(nb_true)/double(nb_total) << endl;

    // ==================================================================================
    // Etape 10
    // Save results
    // ==================================================================================

    saveResults(dir_path, fc_results_3, labels, batch_size, output_scale * fc_scale * conv_scale * input_scale, nb_true, nb_false, nb_total, time_results);

    cout << "Done ?" << endl;


    return 0;

}