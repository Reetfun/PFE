/*
    Spécifités :
        - Pas de bashing
        - Chacune des valeurs est représentée sous la forme d'un vecteur de taille 1x1.
        - Ce vecteur 1x1 est ensuite chiffré.
        - Chaque composant est donc un vecteur de chiffré.
        - Les opérations (multiplication et addition) se font donc entre vecteur de taille (1x1)
        - Les polynômes sont approximés avec Chevychev (la fonction d'OpenFHE)
    Tests réalisés :
        - 100 inférences (50 vraies et 50 fausses) sur les données d'entrée
        - Calcul de la moyenne de la précision de chaque couche
        - Calcul de la moyenne de la précision globale
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
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace std;
using namespace lbcrypto;

string initDirectory(string path, int polynomial_degree, int nb_test) {

    auto currentTimePoint = chrono::system_clock::now();
    time_t currentTime = chrono::system_clock::to_time_t(currentTimePoint);
    tm* localTime = localtime(&currentTime);
    stringstream timestamp;
    timestamp << put_time(localTime, "%Y%m%d_%H%M%S");

    string folderName = "Results_" + timestamp.str() + "_deg" + to_string(polynomial_degree) + "_nb" + to_string(nb_test);
    string full_path = path + "/" + folderName;
    filesystem::create_directory(full_path);

    ofstream file_accuracy;
    ofstream file_time;
    ofstream file_final_result;

    file_accuracy.open(full_path + "/" + "avg_prec_1_each_layer.csv");
    file_time.open(full_path + "/" + "avg_prec_1_each_layer_time.csv");
    file_final_result.open(full_path + "/" + "avg_prec_1_final_result.csv");

    if (file_accuracy.is_open()) {
        file_accuracy << "index,prec_layer1,gap_layer1,prec_activation1,gap_activation1,prec_layer2,gap_layer2,prec_activation2,gap_activation2,prec_layer3,gap_layer3,prec_activation3,gap_activation3" << endl;
        file_accuracy.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    if (file_time.is_open()) {
        file_time << "index,layer1,activation1,layer2,activation2,layer3,activation3" << endl;
        file_time.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    if (file_final_result.is_open()) {
        file_final_result << "index,gap_precision_pt_calcul,gap_precision_label,is_prediction_correct,verif_result,final_result,label" << endl;
        file_final_result.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    return full_path; 
}

CryptoContext<DCRTPoly> initSystem(int batch_size, int mult_depth, int scaling_modulus_bits) {
    CCParams<CryptoContextCKKSRNS> parametersCKKS;
    parametersCKKS.SetBatchSize(batch_size);
    parametersCKKS.SetScalingModSize(scaling_modulus_bits);
    parametersCKKS.SetMultiplicativeDepth(mult_depth);
    // parametersCKKS.SetExecutionMode(EXEC_NOISE_ESTIMATION);
    // parametersCKKS.SetDecryptionNoiseMode(NOISE_FLOODING_DECRYPT);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parametersCKKS);  
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

vector<vector<Plaintext>> encodeWeightsOneByOne(vector<vector<double>> data, CryptoContext<DCRTPoly> cc) {
    vector<vector<Plaintext>> encoded_data;
    for (int i = 0; i < int(data.size()); i++) {
        vector<Plaintext> row;
        for (int j = 0; j < int(data[0].size()); j++) {
            vector<double> val = {data[i][j]};
            Plaintext pt = cc->MakeCKKSPackedPlaintext(val);
            row.push_back(pt);
        }
        encoded_data.push_back(row);
    }
    return encoded_data;
}

vector<Plaintext> encodeBiasesOneByOne(vector<double> data, CryptoContext<DCRTPoly> cc) {
    vector<Plaintext> encoded_data;
    for (int i = 0; i < int(data.size()); i++) {
        vector<double> val = {data[i]};
        Plaintext pt = cc->MakeCKKSPackedPlaintext(val);
        encoded_data.push_back(pt);
    }
    return encoded_data;
}

vector<double> getBiases(string path, int row) {
    vector<double> biases;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return biases;
    }
    string line;
    getline(file, line);
    stringstream ss(line);
    string val;
    for (int i = 0; i < row; i++) {
        getline(ss, val, ',');
        biases.push_back(stod(val));
    }
    file.close();
    return biases;
}

vector<vector<double>> getWeights(string path, int row, int col) {
    vector<vector<double>> weights;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return weights;
    }
    string line;
    getline(file, line); 
    stringstream iss(line);
    string val;
    for (int i = 0; i < row; i++) {
        vector<double> row;
        for (int j = 0; j < col; j++) {
            getline(iss, val, ',');
            row.push_back(stod(val));
        }
        weights.push_back(row);
    }
    file.close();
    return weights;
}

vector<int> getLabels() {
    vector<int> labels;
    ifstream file("../model_data/label_bin1.csv");
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return labels;
    }
    string line;
    getline(file, line);
    stringstream ss(line);
    string val;
    while (getline(ss, val, ',')) {
        labels.push_back(stoi(val));
    }
    file.close();
    return labels;
}

vector<int> getLabelsTest(vector<int> index_test, vector<int> labels) {
    vector<int> labels_test;
    for (int i = 0; i < int(index_test.size()); i++) {
        labels_test.push_back(labels[index_test[i]]);
    }
    return labels_test;
}

vector<int> getRandomIndexTest(int nb_labels_1, int nb_labels_0, vector<int> labels) {
    int nb_1 = nb_labels_1;
    int nb_0 = nb_labels_0;
    vector<int> index_test;
    
    while (nb_1 > 0 || nb_0 > 0) {
        int index = rand() % labels.size();
        if (labels[index] == 1 && nb_1 > 0) {
            index_test.push_back(index);
            nb_1--;
        } else if (labels[index] == 0 && nb_0 > 0) {
            index_test.push_back(index);
            nb_0--;
        }
    }
    return index_test; 
}


vector<string> getCategories() {
    vector<string> categories;
    ifstream file("../model_data/input_normalized.csv");
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return categories;
    }
    string line;
    getline(file, line); // get the first line header
    stringstream ss(line);
    string val;
    getline(ss, val, ','); // skip the first column
    while (getline(ss, val, ',')) {
        categories.push_back(val);
    }
    file.close();
    return categories;
}

vector<vector<double>> getInputs() {
    vector<vector<double>> input;
    ifstream file("../model_data/input_normalized.csv");
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return input;
    }
    string line;
    getline(file, line); // skip the first line header
    while (getline(file, line)) {
        vector<double> row;
        stringstream iss(line);
        string val;
        getline(iss, val, ','); // skip the first column (index)
        while (getline(iss, val, ',')) {
            row.push_back(stod(val));
        }
        input.push_back(row);
    }
    file.close();
    return input;
}

vector<vector<double>> getInputsTest(vector<int> index_test, vector<vector<double>> input) {
    vector<vector<double>> input_test;
    for (int i = 0; i < int(index_test.size()); i++) {
        input_test.push_back(input[index_test[i]]);
    }
    return input_test;
}

vector<Ciphertext<DCRTPoly>> encryptDataOneByOne(vector<double> data, CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> pk) {
    vector<Ciphertext<DCRTPoly>> encrypted_data;
    for (int i = 0; i < int(data.size()); i++) {
        vector<double> row;
        row = {data[i]};
        Plaintext pt = cc->MakeCKKSPackedPlaintext(row);
        Ciphertext<DCRTPoly> ct = cc->Encrypt(pk, pt);
        encrypted_data.push_back(ct);
    }
    return encrypted_data;
}

vector<Plaintext> decryptVector(vector<Ciphertext<DCRTPoly>> ciphers, CryptoContext<DCRTPoly> cc, PrivateKey<DCRTPoly> sk) {
    vector<Plaintext> pt_vector;
    for (int i = 0; i < int(ciphers.size()); i++) {
        Plaintext pt;
        cc->Decrypt(sk, ciphers[i], &pt);
        pt_vector.push_back(pt);
    }
    return pt_vector;
}

vector<Ciphertext<DCRTPoly>> ctNeuron(CryptoContext<DCRTPoly> cc, vector<vector<Plaintext>> weights, vector<Plaintext> bias, vector<Ciphertext<DCRTPoly>> inputs) {
    vector<Ciphertext<DCRTPoly>> ct_layer;
    for (int i = 0; i < int(weights[0].size()); i++) {
        vector<Ciphertext<DCRTPoly>> row;
        for (int j = 0; j < int(weights.size()); j++) {
            Ciphertext<DCRTPoly> ct = cc->EvalMult(inputs[j], weights[j][i]);
            row.push_back(ct);
        }
        ct_layer.push_back(cc->EvalAddMany(row));
    }

    for (int i = 0; i < int(bias.size()); i++) {
        ct_layer[i] = cc->EvalAdd(ct_layer[i], bias[i]);
    }

    return ct_layer;
}

vector<double> ptNeuron(vector<vector<double>> weights, vector<double> bias, vector<double> inputs) {
    vector<double> layer;
    for (int i = 0; i < int(weights[0].size()); i++) {
        vector<double> row;
        for (int j = 0; j < int(weights.size()); j++) {
            double ct = inputs[j] * weights[j][i];
            row.push_back(ct);
        }
        layer.push_back(accumulate(row.begin(), row.end(), 0.0) + bias[i]);
    }
    return layer;
}

vector<Ciphertext<DCRTPoly>> PolynomialReluChebyshev(vector<Ciphertext<DCRTPoly>> ct, CryptoContext<DCRTPoly> cc, int degree) {
    vector<Ciphertext<DCRTPoly>> ct_relu;
    for (int i = 0; i < int(ct.size()); i++) {
        ct_relu.push_back(cc->EvalChebyshevFunction([](double x) -> double { return x > 0 ? x : 0; }, ct[i], -5, 5, degree));
    }
    return ct_relu;
}

vector<double> ptRelu(vector<double> verif) {
    vector<double> verif_relu;
    for (int i = 0; i < int(verif.size()); i++) {
        double ct = verif[i];
        if (ct < 0) {
            ct = 0;
        }
        verif_relu.push_back(ct);
    }
    return verif_relu;
}

vector<Ciphertext<DCRTPoly>> PolynomialSigmoidChebyshev(vector<Ciphertext<DCRTPoly>> ct, CryptoContext<DCRTPoly> cc, int degree) {
    vector<Ciphertext<DCRTPoly>> ct_sig;
    for (int i = 0; i < int(ct.size()); i++) {
        ct_sig.push_back(cc->EvalChebyshevFunction([](double x) -> double { return 1 / (1 + exp(-x)); }, ct[i], -5, 5, degree));
    }
    return ct_sig;
}

vector<double> ptSigmoid(vector<double> verif) {
    vector<double> verif_sigmoid;
    for (int i = 0; i < int(verif.size()); i++) {
        double ct = 1 / (1 + exp(-verif[i]));
        verif_sigmoid.push_back(ct);
    }
    return verif_sigmoid;
}

double calculAverageAccuracy(vector<double> verif, vector<Plaintext> ct) {
    vector<double> accuracy;
    for (int i = 0; i < int(verif.size()); i++) {
        // cout << i << endl;
        // cout << verif[i] << " | " << ct[i]->GetRealPackedValue()[0] << endl;
        // cout << 100 - (abs(verif[i] - ct[i]->GetRealPackedValue()[0])/verif[i]*100) << endl;
        accuracy.push_back(100 - (abs(verif[i]+0.00001 - ct[i]->GetRealPackedValue()[0])/(verif[i]+0.00001)*100));
    }
    return double(accumulate(accuracy.begin(), accuracy.end(), 0.0)/accuracy.size());
}

double calculAverageGap(vector<double> verif, vector<Plaintext> ct) {
    vector<double> gap;
    for (int i = 0; i < int(verif.size()); i++) {
        gap.push_back(abs(verif[i] - ct[i]->GetRealPackedValue()[0]));
    }
    return double(accumulate(gap.begin(), gap.end(), 0.0)/gap.size());
}

// void savingStats(vector<vector<double>> accuracy, string path, vector<int> index_input) {
//     ofstream file(path, ios::app);
//     if (!file.is_open()) {
//         cerr << "Error !" << endl;
//         return;
//     }
//     for (int j = 0; j < int(accuracy.size()); j++) {
//         for (int i = 0; i < int(accuracy[j].size()); i++) {
//             if (i == 0) {
//                 file << index_input[j] << ",";
//             }
//             file << accuracy[j][i] << ",";
//         }
//         file << endl;
//     }
//     file.close();
// }

void savingStats(vector<double> stats, string path, int index_input) {
    ofstream file(path, ios::app);
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return;
    }

    file << index_input << ",";

    for (int j = 0; j < int(stats.size()); j++) {
        file << stats[j] << ",";
    }
        
    file << endl;
    file.close();
}


void savingRecap(string path, int nb_test, int nb_true_test_1, int nb_false_test_1, int nb_true_test_0, int nb_false_test_0, CryptoContext<DCRTPoly> cc, int mult_depth, int scaling_modulus_bits, int polynomial_degree) {
    ofstream file(path, ios::app);
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return;
    }
    file << "Nombre de tests réalisés : " << nb_test << endl;
    file << "Nombre de tests censés être >50k : " << nb_test/2 << endl;
    file << "Nombre de tests censés être <50k : " << nb_test/2 << endl;
    file << "Nombre de vrais >50k : " << nb_true_test_1 << endl;
    file << "Nombre de faux >50k : " << nb_false_test_1 << endl;
    file << "Nombre de vrais <50k : " << nb_true_test_0 << endl;
    file << "Nombre de faux <50k : " << nb_false_test_0 << endl;
    file << "Précision totale : " << double(nb_true_test_1 + nb_true_test_0)/nb_test << "\n\n================" << endl;
    file << "Paramètres utilisés : \n" << "   - Module du plaintext : " << cc->GetCryptoParameters()->GetPlaintextModulus() << "\n   - Ring Dimension : " << cc->GetRingDimension() << "\n   - Profondeur multiplicative : " << mult_depth << endl;
    file << "   - Degré du polynome d'approximation : " << polynomial_degree << "\n   - Taille du scaling modulus : " << scaling_modulus_bits << endl;
    file.close();
}

double calculInferencePrecision(vector<double> verif, vector<Plaintext> ct) {
    vector<double> accuracy;
    for (int i = 0; i < int(verif.size()); i++) {
        accuracy.push_back(100 - (abs(verif[i]+0.00001 - ct[i]->GetRealPackedValue()[0])/(verif[i]+0.00001)*100));
    }
    return double(accumulate(accuracy.begin(), accuracy.end(), 0.0)/accuracy.size());
}

double isPredictionCorrect(int verif, double ct, int *nb_true_test_1, int *nb_false_test_1, int *nb_true_test_0, int *nb_false_test_0) {
    if (verif > 0.5 && ct > 0.5) {
        *nb_true_test_1 += 1;
        return 1;
    } else if (verif < 0.5 && ct < 0.5) {
        *nb_true_test_0 += 1;
        return 1;
    } else {
        if (verif > 0.5) {
            *nb_false_test_1 += 1;
        } else {
            *nb_false_test_0 += 1;
        }
        return 0;
    }
}

int main(int argc, char *argv[]) {
    TimeVar t;
    double processing_time(0.0);
    int nb_true_test_1 = 0, nb_false_test_1 = 0, nb_true_test_0 = 0, nb_false_test_0 = 0;
    vector<vector<double>> accuracy_results_each_layer; 
    vector<vector<double>> time_results_each_layer;
    vector<vector<double>> precision_final_results;
    vector<vector<double>> noise_size_each_layer;

    // ==================================================================================
    // Etape 1
    // Initialisation des paramètres
    // ==================================================================================
    cout << "\n{...} Generating crypto context" << endl;

    int scaling_modulus_bits = 50;
    int batch_size = 2;
    int polynomial_degree;
    cout << "Enter the polynomial degree : ";
    cin >> polynomial_degree;
    int mult_depth;
    cout << "Enter the multiplicative depth you want : ";
    cin >> mult_depth;
    int nb_test;
    cout << "Enter the number of tests you want to do : ";
    cin >> nb_test;
    
    TIC(t);
    CryptoContext<DCRTPoly> cc = initSystem(batch_size, mult_depth, scaling_modulus_bits);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    processing_time = TOC(t);
    cout << "=> Crypto context generation done [" << processing_time << " ms]" << endl;

    // ==================================================================================      
    // Etape 1.1
    // Initialisation des répertoires de sauvegarde des résultats
    // ==================================================================================
    string dir_path = initDirectory("../CKKS_1/results", polynomial_degree, nb_test);

    // ==================================================================================
    // Etape 2
    // Extraction et préparation des données
    // ==================================================================================
    cout << "\n{...} Reading inputs and labels from file" << endl;
    TIC(t);
    vector<string> const categories = getCategories();
    vector<int> const labels = getLabels();
    vector<vector<double>> const inputs = getInputs();
    // vector<int> const index_test = getRandomIndexTest(nb_test/2, nb_test/2, labels);
    vector<int> const index_test = { 11840, 11323, 2082, 22929, 4833, 29658, 2306, 22386, 26777 };
    processing_time = TOC(t);
    cout << "=> Inputs and labels reading done [" << processing_time << " ms]" << endl;

    cout << "\n{...} Fetching and preprocessing parameters and inputs" << endl;
    TIC(t);
    vector<vector<double>> const weights_layer_1 = getWeights("../model_data/weights1.csv", 88, 16);
    vector<double> const bias_layer_1 = getBiases("../model_data/bias1.csv", 16);
    vector<vector<double>> const weights_layer_2 = getWeights("../model_data/weights2.csv", 16, 16);
    vector<double> const bias_layer_2 = getBiases("../model_data/bias2.csv", 16);
    vector<vector<double>> const weights_layer_3 = getWeights("../model_data/weights3.csv", 16, 1);
    vector<double> const bias_layer_3 = getBiases("../model_data/bias3.csv", 1);    
    processing_time = TOC(t);
    cout << "=> Parameters fetching and preprocessing done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 3
    // Encoding des inputs et des paramètres du modèle
    // ==================================================================================
    cout << "\n{...} Encoding parameters" << endl;
    TIC(t);
    vector<vector<Plaintext>> const pt_weights_layer_1 = encodeWeightsOneByOne(weights_layer_1, cc);
    vector<vector<Plaintext>> const pt_weights_layer_2 = encodeWeightsOneByOne(weights_layer_2, cc);
    vector<vector<Plaintext>> const pt_weights_layer_3 = encodeWeightsOneByOne(weights_layer_3, cc);
    vector<Plaintext> const pt_bias_layer_1 = encodeBiasesOneByOne(bias_layer_1, cc);
    vector<Plaintext> const pt_bias_layer_2 = encodeBiasesOneByOne(bias_layer_2, cc);
    vector<Plaintext> const pt_bias_layer_3 = encodeBiasesOneByOne(bias_layer_3, cc);
    processing_time = TOC(t);
    cout << "=> Inputs and parameters encryption and encoding done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 4
    // Evaluation du modèle sur 100 inférences (50 vraies et 50 fausses)
    // ==================================================================================

    for (int i = 0; i < int(index_test.size()); i++) {
        vector<double> accuracy;
        vector<double> time;
        vector<double> precision_results;
        vector<double> noise_size;

        cout << "\n=================\n{...} Test " << i+1 << "\n" << "=================\n" << endl;
        cout << "{...} Encryption of the inputs" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_input = encryptDataOneByOne(inputs[index_test[i]], cc, keys.publicKey);
        processing_time = TOC(t);
        cout << "=> Inputs encryption done [" << processing_time << " ms]" << endl;

        // for (int j = 0; j < int(ct_input.size()); j++) {
        //     Serial::SerializeToFile(dir_path + "/input_ct_" + to_string(j) + ".txt", ct_input[j], SerType::BINARY);
        //     cout << j << endl;
        // }

        // ==================================================================================
        // Etape 4.1
        // Evaluation de la première couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the first layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_1 = ctNeuron(cc, pt_weights_layer_1, pt_bias_layer_1, ct_input);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 1 done [" << processing_time << " ms]" << endl;

        vector<double> layer_1_verif = ptNeuron(weights_layer_1, bias_layer_1, inputs[index_test[i]]);
        vector<Plaintext> pt_layer_1_dec = decryptVector(ct_layer_1, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_1_verif, pt_layer_1_dec));
        accuracy.push_back(calculAverageGap(layer_1_verif, pt_layer_1_dec));

        // ==================================================================================
        // Etape 4.2
        // Evaluation de la fonction d'activation de la première couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the first layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_1_act = PolynomialReluChebyshev(ct_layer_1, cc, polynomial_degree);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 1 done [" << processing_time << " ms]" << endl;

        vector<double> layer_1_act_verif = ptRelu(layer_1_verif);
        vector<Plaintext> pt_layer_1_act_dec = decryptVector(ct_layer_1_act, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_1_act_verif, pt_layer_1_act_dec));
        accuracy.push_back(calculAverageGap(layer_1_act_verif, pt_layer_1_act_dec));
        // cout << "Noise size 1 : " << pt_layer_1_act_dec[0]->GetLogError() << endl;
        // cout << "Noise size 2 : " << pt_layer_1_act_dec[0]->GetRealPackedValue()[1] << endl;

        // ==================================================================================
        // Etape 4.3
        // Evaluation de la deuxième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the second layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_2 = ctNeuron(cc, pt_weights_layer_2, pt_bias_layer_2, ct_layer_1_act);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 2 done [" << processing_time << " ms]" << endl;

        vector<double> layer_2_verif = ptNeuron(weights_layer_2, bias_layer_2, layer_1_act_verif);
        vector<Plaintext> pt_layer_2_dec = decryptVector(ct_layer_2, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_2_verif, pt_layer_2_dec));
        accuracy.push_back(calculAverageGap(layer_2_verif, pt_layer_2_dec));

        // ==================================================================================
        // Etape 4.4
        // Evaluation de la fonction d'activation de la deuxième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the second layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_2_act = PolynomialReluChebyshev(ct_layer_2, cc, polynomial_degree);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 2 done [" << processing_time << " ms]" << endl;

        vector<double> layer_2_act_verif = ptRelu(layer_2_verif);
        vector<Plaintext> pt_layer_2_act_dec = decryptVector(ct_layer_2_act, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_2_act_verif, pt_layer_2_act_dec));
        accuracy.push_back(calculAverageGap(layer_2_act_verif, pt_layer_2_act_dec));

        // ==================================================================================
        // Etape 4.5
        // Evaluation de la troisième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the third layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_3 = ctNeuron(cc, pt_weights_layer_3, pt_bias_layer_3, ct_layer_2_act);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 3 done [" << processing_time << " ms]" << endl;

        vector<double> layer_3_verif = ptNeuron(weights_layer_3, bias_layer_3, layer_2_act_verif);
        vector<Plaintext> pt_layer_3_dec = decryptVector(ct_layer_3, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_3_verif, pt_layer_3_dec));
        accuracy.push_back(calculAverageGap(layer_3_verif, pt_layer_3_dec));

        // ==================================================================================
        // Etape 4.6
        // Evaluation de la fonction d'activation de la troisième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the third layer" << endl;
        TIC(t);
        vector<Ciphertext<DCRTPoly>> ct_layer_3_act = PolynomialSigmoidChebyshev(ct_layer_3, cc, polynomial_degree);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 3 done [" << processing_time << " ms]" << endl;

        vector<double> layer_3_act_verif = ptSigmoid(layer_3_verif);
        vector<Plaintext> pt_layer_3_act_dec = decryptVector(ct_layer_3_act, cc, keys.secretKey);
        accuracy.push_back(calculAverageAccuracy(layer_3_act_verif, pt_layer_3_act_dec));
        accuracy.push_back(calculAverageGap(layer_3_act_verif, pt_layer_3_act_dec));

        // ==================================================================================
        // Etape 4.7
        // Evaluation de la précision globale
        // ==================================================================================
        cout << "{...} Result found :\n" << "   - CipherML result : " << pt_layer_3_act_dec[0]->GetRealPackedValue()[0] << "\n   - Plain result : " << layer_3_act_verif[0] << endl;
        double gap_precision_pt_calcul = abs(layer_3_act_verif[0] - pt_layer_3_act_dec[0]->GetRealPackedValue()[0]);
        double gap_precision_label = abs(layer_3_act_verif[0] - labels[index_test[i]]);
        double is_prediction_correct = isPredictionCorrect(labels[index_test[i]], pt_layer_3_act_dec[0]->GetRealPackedValue()[0], &nb_true_test_1, &nb_false_test_1, &nb_true_test_0, &nb_false_test_0);
        precision_results.push_back(gap_precision_pt_calcul);
        precision_results.push_back(gap_precision_label);
        precision_results.push_back(is_prediction_correct);
        precision_results.push_back(layer_3_act_verif[0]);
        precision_results.push_back(pt_layer_3_act_dec[0]->GetRealPackedValue()[0]);
        precision_results.push_back(labels[index_test[i]]);

        // ==================================================================================
        // Etape 4.8
        // Stockage des résultats
        // ==================================================================================
        cout << "\n{...} Saving results" << endl;

        savingStats(accuracy, dir_path + "/avg_prec_1_each_layer.csv", index_test[i]);
        savingStats(time, dir_path + "/avg_prec_1_each_layer_time.csv", index_test[i]);
        savingStats(precision_results, dir_path + "/avg_prec_1_final_result.csv", index_test[i]);
        // accuracy_results_each_layer.push_back(accuracy);
        // time_results_each_layer.push_back(time);
        // precision_final_results.push_back(precision_results);
    }

    // ==================================================================================
    // Etape 5
    // Stockage des résultats de l'inférence dans un fichier
    // ==================================================================================

    // savingStats(accuracy_results_each_layer, dir_path + "/avg_prec_1_each_layer.csv", index_test);
    // savingStats(time_results_each_layer, dir_path + "/avg_prec_1_each_layer_time.csv", index_test);
    // savingStats(precision_final_results, dir_path + "/avg_prec_1_final_result.csv", index_test);
    savingRecap(dir_path + "/avg_prec_1_recap.csv", nb_test, nb_true_test_1, nb_false_test_1, nb_true_test_0, nb_false_test_0, cc, mult_depth, scaling_modulus_bits, polynomial_degree);

    cout << "\n=================\n{...} Results\n" << "=================\n" << endl;
    cout << "Nombre de tests : " << nb_test << endl;
    cout << "Nombre de tests censés être >50k : " << nb_test/2 << endl;
    cout << "Nombre de tests censés être <50k : " << nb_test/2 << endl;
    cout << "Nombre de vrais >50k : " << nb_true_test_1 << endl;
    cout << "Nombre de faux >50k : " << nb_false_test_1 << endl;
    cout << "Nombre de vrais <50k : " << nb_true_test_0 << endl;
    cout << "Nombre de faux <50k : " << nb_false_test_0 << endl;
    cout << "Précision : " << double(nb_true_test_1 + nb_true_test_0)/double(nb_true_test_1 + nb_true_test_0 + nb_false_test_1 + nb_false_test_0) << endl;

    return 0;

}