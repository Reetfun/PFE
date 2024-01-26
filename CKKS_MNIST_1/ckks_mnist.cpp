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
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"


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

CryptoContext<DCRTPoly> initSystem(int batch_size, int mult_depth) {
    CCParams<CryptoContextCKKSRNS> parametersCKKS;
    parametersCKKS.SetBatchSize(batch_size);
    parametersCKKS.SetMultiplicativeDepth(mult_depth);
    parametersCKKS.SetScalingModSize(50);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parametersCKKS);  
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

vector<Ciphertext<DCRTPoly>> encryptBatch(vector<vector<double>> data, CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> pk) {
    vector<Ciphertext<DCRTPoly>> encrypted_data;
    for (int i = 0; i < int(data.size()); i++) {
        Plaintext pt = cc->MakeCKKSPackedPlaintext(data[i]);
        Ciphertext<DCRTPoly> ct = cc->Encrypt(pk, pt);
        encrypted_data.push_back(ct);
    }
    return encrypted_data;
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
    while (getline(iss, item, delimiter))
    {
        vector<double> row_vector;
        for (int j = 0; j < col; j++)
        {
            row_vector.push_back(stod(item));
            getline(iss, item, delimiter);
        }
        values.push_back(row_vector);
        i++;
        if (i == row)
            break;
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
    ifstream file("../CKKS_MNIST_1/weights_biases.txt");
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
    ifstream inputFile("../BFV_MNIST_2/MNIST-28x28-test.txt");

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
    ifstream file("../CKKS_MNIST_1/MNIST-28x28-test.txt");

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
                row.push_back(0.0);
            }
        }
        inputData.push_back(row);
    }
    return inputData;
}

vector<Plaintext> decryptVector(vector<Ciphertext<DCRTPoly>> ciphers, CryptoContext<DCRTPoly> cc, PrivateKey<DCRTPoly> sk, int batch_size) {
    vector<Plaintext> pt_vector;
    for (int i = 0; i < int(ciphers.size()); i++) {
        Plaintext pt;
        cc->Decrypt(sk, ciphers[i], &pt);
        pt_vector.push_back(pt);
    }
    return pt_vector;
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

vector<double> fillVector(double weight, int scale, int batch_size) {
    vector<double> weight_vector;
    for (int i = 0; i < batch_size; i++) {
        weight_vector.push_back(weight * scale);
    }
    return weight_vector;
}

void encodeParams(vector<double> data, vector<Plaintext> &dest, CryptoContext<DCRTPoly> cc, int scale, int batch_size) {
    std::cout << "a" << endl;
    for (int i = 0; i < int(data.size()); i++) {
        try {
            dest.push_back(cc->MakeCKKSPackedPlaintext(fillVector(data[i], scale, batch_size)));
        } catch (const exception &e) {
            cerr << "Error in encodeParamsOneByOne ! " << i << e.what() << endl;
        }
    }
    std::cout << "e" << endl;
}

vector<Ciphertext<DCRTPoly>> ConvLayer(vector<Ciphertext<DCRTPoly>> ct_inputs, vector<vector<vector<double>>> weights, vector<Plaintext> pt_bias_0, int kernel_dim, int stride, int dim, CryptoContext<DCRTPoly> cc, double conv_scale, PublicKey<DCRTPoly> pk, int batch_size, int padding) {
    // Padding en bas et à droite
    vector<Ciphertext<DCRTPoly>> conv_results;
    int output_dim = (dim - kernel_dim + padding) / stride + 1; // = 13

    for (int k = 0; k < int(weights.size()); k++) { // Pour chaque filtre

        cout << k << "eme filtre" << endl; 

        for (int j = 0; j < output_dim; j++) { // Pour chaque ligne de l'image de sortie

            for (int i = 0; i < output_dim; i++) { // Pour chaque colonne de l'image de sortie

                cout << "ligne " << j << " colonne " << i << endl;

                Ciphertext<DCRTPoly> kernel_result = cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(fillVector(0, conv_scale, batch_size)));
                    
                    for (int x = 0; x < kernel_dim; x++) { // Pour chaque ligne du filtre

                        for (int y = 0; y < kernel_dim; y++) { // Pour chaque colonne du filtre

                            double weight = weights[k][x][y];
                            Plaintext pt_weight = cc->MakeCKKSPackedPlaintext(fillVector(weight, conv_scale, batch_size));

                            int row = stride * j + x; // Ligne de l'image d'entrée
                            int col = stride * i + y; // Colonne de l'image d'entrée

                            if (row < dim && col < dim) { // Padding en bas et à droite !
                                kernel_result = cc->EvalAdd(kernel_result, cc->EvalMult(ct_inputs[row*dim + col], pt_weight));
                            }
                        }
                    }
                conv_results.push_back(cc->EvalAdd(kernel_result, pt_bias_0[k]));
            }
        }
    }
    return conv_results;
}

vector<Ciphertext<DCRTPoly>> FCLayer(vector<Ciphertext<DCRTPoly>> ct_inputs, vector<double> weights, vector<Plaintext> pt_bias_1, CryptoContext<DCRTPoly> cc, double fc_scale, PublicKey<DCRTPoly> pk, int batch_size) {
    vector<Ciphertext<DCRTPoly>> fc_results;
    // vector<Plaintext> pt_weights;
    // encodeParams(weights, pt_weights, cc, fc_scale, batch_size);

    for (int i = 0; i < int(weights.size()); i += int(ct_inputs.size())) { 
        Ciphertext<DCRTPoly> row = cc->EvalMult(ct_inputs[0], cc->MakeCKKSPackedPlaintext(fillVector(weights[i], fc_scale, batch_size)));
        cout << "i = " << i << endl;
        // Ciphertext<DCRTPoly> row = cc->EvalMult(ct_inputs[0], pt_weights[i]);
        for (int x = 1; x < int(ct_inputs.size()); x++) { 
            row = cc->EvalAdd(row, cc->EvalMult(ct_inputs[x], cc->MakeCKKSPackedPlaintext(fillVector(weights[i + x], fc_scale, batch_size))));
            // row = cc->EvalAdd(row, cc->EvalMult(ct_inputs[x], pt_weights[i + x]));
        }
        fc_results.push_back(cc->EvalAdd(row, pt_bias_1[i / int(ct_inputs.size())]));
    }
    return fc_results;
}


vector<Ciphertext<DCRTPoly>> SquareActLayer(vector<Ciphertext<DCRTPoly>> input, CryptoContext<DCRTPoly> cc) {
    vector<Ciphertext<DCRTPoly>> output;
    for (int i = 0; i < int(input.size()); i++) {
        output.push_back(cc->EvalMult(input[i], input[i]));
    }
    return output;
}

void saveResultsClear(string path, vector<Plaintext> results, vector<int> labels, int batch_size, int nb_true, int nb_false, int nb_total, double precision, vector<double> time_results) {
    ofstream file;
    file.open(path + "/" + "final_result.csv");
    if (file.is_open()) {
        for (int i = 0; i < batch_size; i++) {
            file << i << "," << results[0]->GetRealPackedValue()[i] << "," << results[1]->GetRealPackedValue()[i] << "," << results[2]->GetRealPackedValue()[i] << "," << results[3]->GetRealPackedValue()[i] << "," << results[4]->GetRealPackedValue()[i] << "," << results[5]->GetRealPackedValue()[i] << "," << results[6]->GetRealPackedValue()[i] << "," << results[7]->GetRealPackedValue()[i] << "," << results[8]->GetRealPackedValue()[i] << "," << results[9]->GetRealPackedValue()[i] << "," << labels[i] << endl;
        }
        file.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    ofstream file2;
    file2.open(path + "/" + "recap.csv");
    if (file2.is_open()) {
        file2 << "nb_true,nb_false,nb_total,precision" << endl;
        file2 << nb_true << "," << nb_false  << "," << nb_total << "," << precision << endl;
        file2.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    ofstream file3;
    file3.open(path + "/" + "time.csv");
    if (file3.is_open()) {
        file3 << "step,time" << endl;
        file3 << "init," << time_results[0] << endl;
        file3 << "encrypt," << time_results[1] << endl;
        file3 << "conv1," << time_results[2] << endl;
        file3 << "square1," << time_results[3] << endl;
        file3 << "fc2," << time_results[4] << endl;
        file3 << "square2," << time_results[5] << endl;
        file3 << "fc3," << time_results[6] << endl;
        file3.close();
    } else {
        cerr << "Error opening the file." << endl;
    }
}

int main(int argc, char *argv[]) {
    TimeVar t;
    double processing_time(0.0);
    double input_scale = 1.0; // 16
    double conv_scale = 1.0; // 32
    double fc_scale = 1.0; // 32
    double output_scale = 1.0; // 32
    vector<double> time_results;

    // // ==================================================================================
    // // Etape 1
    // // Initialisation des paramètres
    // // ==================================================================================
    cout << "\n{...} Generating crypto context" << endl;

    int batch_size = 8192; 
    double normalization_factor = 1.0/256.0;
    // Dont work : 549764251649 35184371138561 243504973489 318083817907 435748987787 175650481151 592821132889
    // Work : 536903681 10000039937 1000000552961
    int mult_depth;
    cout << "Enter the multiplicative depth you want : ";
    cin >> mult_depth;
    
    TIC(t);
    CryptoContext<DCRTPoly> cc = initSystem(batch_size, mult_depth);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> Crypto context generation done [" << processing_time << " ms]" << endl;

    // // ==================================================================================      
    // // Etape 1.1
    // // Initialisation des répertoires de sauvegarde des résultats
    // // ==================================================================================
    string dir_path = initDirectory("../CKKS_MNIST_1/results", batch_size);

    // ==================================================================================
    // Etape 2
    // Récupération des données
    // ==================================================================================
    cout << "\n{...} Getting data" << endl;
    vector<int> labels = getLabels(batch_size);
    vector<vector<double>> input = transposeData(getInputs(batch_size, normalization_factor, input_scale));

    // ==================================================================================
    // Etape 3
    // Chiffrement des données
    // ==================================================================================
    cout << "\n{...} Encrypting data" << endl;
    TIC(t);
    vector<Ciphertext<DCRTPoly>> encrypted_input = encryptBatch(input, cc, keys.publicKey);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> Data encryption done [" << processing_time << " ms]" << endl;

    // Sauvegarder les inputs chiffrés
    // for (int j = 0; j < int(encrypted_input.size()); j++) {
    //     Serial::SerializeToFile(dir_path + "/input_ct_" + to_string(j) + ".txt", encrypted_input[j], SerType::BINARY);
    //     cout << j << endl;
    // }

    // ==================================================================================
    // Etape 4
    // Récupération des poids et des biais
    // ==================================================================================
    cout << "\n{...} Getting weights and biases" << endl;
    vector<vector<vector<double>>> weights_0;
    vector<double> bias_0, bias_1, bias_2, weights_1, weights_2;
    getWeights(weights_0, bias_0, weights_1, bias_1, weights_2, bias_2);

    // ==================================================================================
    // Etape 5
    // Encoder les biais
    // ==================================================================================
    cout << "\n{...} Encoding biases" << endl;
    vector<Plaintext> pt_bias_0, pt_bias_1, pt_bias_2;
    encodeParams(bias_0, pt_bias_0, cc, conv_scale*input_scale, batch_size);
    encodeParams(bias_1, pt_bias_1, cc, fc_scale*conv_scale*input_scale, batch_size);
    encodeParams(bias_2, pt_bias_2, cc, output_scale*fc_scale*conv_scale*input_scale, batch_size);
    
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
    vector<Ciphertext<DCRTPoly>> conv_layer_1 = ConvLayer(encrypted_input, weights_0, pt_bias_0, kernel_dim, stride, dim, cc, conv_scale, keys.publicKey, batch_size, padding); // 13*13*5 = 845
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> ConvLayer 1 done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 6
    // ActivationLayer Square 1
    // ==================================================================================
    cout << "\n{...} ActivationLayer Square 1" << endl;

    TIC(t);
    vector<Ciphertext<DCRTPoly>> activation_results_1 = SquareActLayer(conv_layer_1, cc);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> ActivationLayer Square 1 done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 7
    // ConvLayer 2
    // ==================================================================================
    cout << "\n{...} FCLayer 2" << endl;

    TIC(t);
    vector<Ciphertext<DCRTPoly>> fc_results_2 = FCLayer(activation_results_1, weights_1, pt_bias_1, cc, fc_scale, keys.publicKey, batch_size);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> FCLayer 2 done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 8
    // ActivationLayer Square 2
    // ==================================================================================
    cout << "\n{...} ActivationLayer Square 2" << endl;

    TIC(t);
    vector<Ciphertext<DCRTPoly>> activation_results_2 = SquareActLayer(fc_results_2, cc);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> ActivationLayer Square 2 done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 9
    // ConvLayer 3
    // ==================================================================================
    cout << "\n{...} FCLayer 3" << endl;

    TIC(t);
    vector<Ciphertext<DCRTPoly>> fc_results_3 = FCLayer(activation_results_2, weights_2, pt_bias_2, cc, output_scale, keys.publicKey, batch_size);
    processing_time = TOC(t);
    time_results.push_back(processing_time);
    cout << "=> FCLayer 3 done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 10
    // Calcul Precision
    // ==================================================================================
    cout << "\n{...} Calculating precision" << endl;
    int nb_total = 0;
    int nb_true = 0;
    int nb_false = 0;

    vector<Plaintext> decrypted_results = decryptVector(fc_results_3, cc, keys.secretKey, batch_size);

    for (int i = 0; i < batch_size; i++) {
        double max = decrypted_results[0]->GetRealPackedValue()[i];
        int index_max = 0;
        for (int j = 1; j < 10; j++) {
            if (decrypted_results[j]->GetRealPackedValue()[i] > max) {
                max = decrypted_results[j]->GetRealPackedValue()[i];
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

    // saveResults(dir_path, decrypted_results, labels, batch_size);
    saveResultsClear(dir_path, decrypted_results, labels, batch_size, nb_true, nb_false, nb_total, double(nb_true)/double(nb_total), time_results);


    cout << "Done ?" << endl;


    return 0;

}