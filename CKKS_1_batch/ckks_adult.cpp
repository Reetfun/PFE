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

    Remarque :
        - Batching à {8192, 4096, 2048, 1024} ne fonctionne pas avec multDepth = {18, 21} et poly = 13 et modSize = 50 => Erreur de bruit à la fin
        - Batching à 512 ne fonctionne pas bien (grosse erreur de bruit) avec multDepth = {18, 21} et poly = 13 et modSize = 50
        - Batching à {256, 128} fonctionne avec multDepth = {18, 21} et poly = 13 et modSize = 50, mais que du label 0 au label 511..
        - Batching à 8192 ne fonctionne pas avec multDepth = {18, 21} et poly = 13 et modSize = 60 et firstModSize = 59 => Terminate called recursively 
        - Batching à 8192 ne fonctionne pas avec multDepth = {18, 21} et poly = 13 et modSize = 50 et FLEXIBLEAUTOEXT => Erreur de bruit à la fin

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

string initDirectory(string path, int polynomial_degree, int nb_test, int i) {

    auto currentTimePoint = chrono::system_clock::now();
    time_t currentTime = chrono::system_clock::to_time_t(currentTimePoint);
    tm* localTime = localtime(&currentTime);
    stringstream timestamp;
    timestamp << put_time(localTime, "%Y%m%d_%H%M%S");

    string folderName = "Results_" + timestamp.str() + "_deg" + to_string(polynomial_degree) + "_nb" + to_string(nb_test) + "_" + to_string(i);
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
    // SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    // parametersCKKS.SetSecretKeyDist(secretKeyDist);

    parametersCKKS.SetBatchSize(batch_size);
    parametersCKKS.SetScalingModSize(scaling_modulus_bits);
    // parametersCKKS.SetScalingTechnique(FLEXIBLEAUTOEXT);
    // parametersCKKS.SetFirstModSize(59);
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

vector<vector<double>> getInputs(int batch_size) {
    vector<vector<double>> input;
    ifstream file("../model_data/input_normalized.csv");
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return input;
    }
    string line;
    getline(file, line); // skip the first line header
    for (int i = 0; i < batch_size; i++) {
        vector<double> row;
        getline(file, line);
        stringstream ss(line);
        string val;
        getline(ss, val, ','); // skip the first column (index)
        while (getline(ss, val, ',')) {
            row.push_back(stod(val));
        }
        input.push_back(row);
    }
    file.close();
    return input;
    // return transposeData(input);
}

vector<vector<double>> getInputsTest(vector<int> index_test, vector<vector<double>> input) {
    vector<vector<double>> input_test;
    for (int i = 0; i < int(index_test.size()); i++) {
        input_test.push_back(input[index_test[i]]);
    }
    return input_test;
}

vector<Ciphertext<DCRTPoly>> encryptData(vector<vector<double>> data, CryptoContext<DCRTPoly> cc, PublicKey<DCRTPoly> pk) {
    vector<Ciphertext<DCRTPoly>> encrypted_data;
    for (int i = 0; i < int(data.size()); i++) {
        Plaintext pt = cc->MakeCKKSPackedPlaintext(data[i]);
        Ciphertext<DCRTPoly> ct = cc->Encrypt(pk, pt);
        encrypted_data.push_back(ct);
    }
    return encrypted_data;
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
        Ciphertext<DCRTPoly> row = cc->EvalMult(inputs[0], weights[0][i]);
        for (int j = 1; j < int(weights.size()); j++) {
            Ciphertext<DCRTPoly> ct = cc->EvalMult(inputs[j], weights[j][i]); // j'ai changé le inputs[j] en inputs[i]
            row = cc->EvalAdd(row, ct);
        }
        ct_layer.push_back(cc->EvalAdd(row, bias[i]));
    }

    return ct_layer;
}

vector<double> vectorProductLinear(vector<double> v1, double v2) {
    vector<double> res;
    for (int i = 0; i < int(v1.size()); i++) {
        res.push_back(v1[i] * v2);
    }
    return res;
}

vector<double> vectorAddition(vector<double> v1, vector<double> v2) {
    vector<double> res;
    for (int i = 0; i < int(v1.size()); i++) {
        res.push_back(v1[i] + v2[i]);
    }
    return res;
}

vector<vector<double>> ptNeuron(vector<vector<double>> weights, vector<double> bias, vector<vector<double>> inputs, int batch_size) {
    // Fully Connected Neuron
    vector<vector<double>> layer;
    cout << weights[0].size() << endl;
    cout << weights.size() << endl;
    cout << inputs.size() << endl;
    cout << inputs[0].size() << endl;

    for (int i = 0; i < int(weights[0].size()); i++) {
        vector<double> row(batch_size, 0.0);
        for (int j = 0; j < int(weights.size()); j++) {
            vector<double> res = vectorProductLinear(inputs[j], weights[j][i]);
            row = vectorAddition(row, res);
        }
        row = vectorAddition(row, vector<double>(batch_size, bias[i]));
        layer.push_back(row);
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

    // vector<double> layer;
    // for (int i = 0; i < int(weights[0].size()); i++) {
    //     vector<double> row;
    //     for (int j = 0; j < int(weights.size()); j++) {
    //         double ct = inputs[j] * weights[j][i];
    //         row.push_back(ct);
    //     }
    //     layer.push_back(accumulate(row.begin(), row.end(), 0.0) + bias[i]);
    // }
    // return layer;


vector<vector<double>> ptRelu(vector<vector<double>> verif) {
    vector<vector<double>> verif_relu;
    for (int i = 0; i < int(verif.size()); i++) {
        vector<double> row;
        for (int j = 0; j < int(verif[0].size()); j++) {
            double ct = verif[i][j] > 0 ? verif[i][j] : 0;
            row.push_back(ct);
        }
        verif_relu.push_back(row);
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

vector<vector<double>> ptSigmoid(vector<vector<double>> verif) {
    vector<vector<double>> verif_sigmoid;
    for (int i = 0; i < int(verif.size()); i++) {
        vector<double> row;
        for (int j = 0; j < int(verif[0].size()); j++) {
            double ct = 1 / (1 + exp(-verif[i][j]));
            row.push_back(ct);
        }
        verif_sigmoid.push_back(row);
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

void savingRecap(string path, CryptoContext<DCRTPoly> cc, int mult_depth, int scaling_modulus_bits, int batch_size, int polynomial_degree) {
    ofstream file(path, ios::app);
    if (!file.is_open()) {
        cerr << "Error !" << endl;
        return;
    }
    file << "Paramètres utilisés : \n" << "   - Module du plaintext : " << cc->GetCryptoParameters()->GetPlaintextModulus() << "\n   - Ring Dimension : " << cc->GetRingDimension() << "\n   - Profondeur multiplicative : " << mult_depth << endl;
    file << "   - Degré du polynome d'approximation : " << polynomial_degree << "\n   - Taille du scaling modulus : " << scaling_modulus_bits << endl;
    file << "   - Batch size : " << batch_size << endl;
    file.close();
}

double calculInferencePrecision(vector<double> verif, vector<Plaintext> ct) {
    vector<double> accuracy;
    for (int i = 0; i < int(verif.size()); i++) {
        accuracy.push_back(100 - (abs(verif[i]+0.00001 - ct[i]->GetRealPackedValue()[0])/(verif[i]+0.00001)*100));
    }
    return double(accumulate(accuracy.begin(), accuracy.end(), 0.0)/accuracy.size());
}

double isPredictionCorrect(int verif, double ct) {
    if (verif > 0.5 && ct > 0.5) {
        return 1;
    } else if (verif < 0.5 && ct < 0.5) {
        return 1;
    } else {
        return 0;
    }
}

void encodeWeights(vector<vector<Plaintext>> &pt_weights, vector<vector<double>> weights, CryptoContext<DCRTPoly> cc, int batch_size) {
    for (int i = 0; i < int(weights.size()); i++) {
        vector<Plaintext> row;
        for (int j = 0; j < int(weights[0].size()); j++) {
            vector<double> val(batch_size, weights[i][j]);
            Plaintext pt = cc->MakeCKKSPackedPlaintext(val);
            row.push_back(pt);
        }
        pt_weights.push_back(row);
    }
}

void encodeBiases(vector<Plaintext> &pt_bias, vector<double> bias, CryptoContext<DCRTPoly> cc, int batch_size) {
    for (int i = 0; i < int(bias.size()); i++) {
        vector<double> val(batch_size, bias[i]);
        Plaintext pt = cc->MakeCKKSPackedPlaintext(val);
        pt_bias.push_back(pt);
    }
}

void evaluateStats(vector<vector<double>> results_verif, vector<Plaintext> pt_results, vector<int> labels, string dir_path, int batch_size, int index_loop, int mult_depth, int polynomial_degree, int scaling_modulus, CryptoContext<DCRTPoly> cc, vector<double> time) {

    for (int i = 0; i < batch_size; i++) {
        vector<double> stats;
        stats.push_back(abs(results_verif[0][i] - pt_results[0]->GetRealPackedValue()[i]));
        stats.push_back(abs(labels[i + index_loop * batch_size] - pt_results[0]->GetRealPackedValue()[i]));
        stats.push_back(isPredictionCorrect(labels[i + index_loop * batch_size], pt_results[0]->GetRealPackedValue()[i]));
        stats.push_back(results_verif[0][i]);
        stats.push_back(pt_results[0]->GetRealPackedValue()[i]);
        stats.push_back(labels[i + index_loop * batch_size]);
        savingStats(stats, dir_path + "/" + "avg_prec_1_final_result.csv", i + index_loop * batch_size);
    }

    ofstream file_time(dir_path + "/" + "avg_prec_1_each_layer_time.csv", ios::app);
    if (!file_time.is_open()) {
        cerr << "Error !" << endl;
        return;
    }
    file_time << index_loop << ",";
    for (int i = 0; i < int(time.size()); i++) {
        file_time << time[i] << ",";
    }
    file_time << endl;
    file_time.close();

    savingRecap(dir_path + "/" + "avg_prec_1_final_result.csv", cc, mult_depth, scaling_modulus, batch_size, polynomial_degree);

}

vector<vector<double>> inputsSplit(vector<vector<double>> inputs, int batch_size, int index_loop) {
    vector<vector<double>> inputs_split;
    for (int i = index_loop * batch_size; i < (index_loop + 1) * batch_size; i++) {
        inputs_split.push_back(inputs[i]);
    }
    return transposeData(inputs_split);
}

int main(int argc, char *argv[]) {
    TimeVar t;
    double processing_time(0.0);
    // int nb_true_test_1 = 0, nb_false_test_1 = 0, nb_true_test_0 = 0, nb_false_test_0 = 0;
    // vector<vector<double>> accuracy_results_each_layer; 
    // vector<vector<double>> time_results_each_layer;
    // vector<vector<double>> precision_final_results;
    // vector<vector<double>> noise_size_each_layer;
    vector<double> time;

    // ==================================================================================
    // Etape 1
    // Initialisation des paramètres
    // ==================================================================================
    cout << "\n{...} Generating crypto context" << endl;

    int scaling_modulus_bits;
    cout << "Enter the scaling modulus bits : ";
    cin >> scaling_modulus_bits;
    int batch_size;
    cout << "Enter the batch size : ";
    cin >> batch_size; 
    int i;
    cout << "Enter the number of the loop you want to do : ";
    cin >> i;
    int polynomial_degree;
    cout << "Enter the polynomial degree : ";
    cin >> polynomial_degree;
    int mult_depth;
    cout << "Enter the multiplicative depth you want : ";
    cin >> mult_depth;
    
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
    string dir_path = initDirectory("../CKKS_1_batch/results", polynomial_degree, batch_size, i);

    // ==================================================================================
    // Etape 2
    // Extraction et préparation des données
    // ==================================================================================
    cout << "\n{...} Reading inputs and labels from file" << endl;
    TIC(t);
    vector<string> const categories = getCategories();
    vector<int> const labels = getLabels();
    vector<vector<double>> const inputs = getInputs(batch_size * (i+1));
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
    vector<vector<Plaintext>> pt_weights_layer_1, pt_weights_layer_2, pt_weights_layer_3;
    vector<Plaintext> pt_bias_layer_1, pt_bias_layer_2, pt_bias_layer_3;
    encodeWeights(pt_weights_layer_1, weights_layer_1, cc, batch_size);
    encodeWeights(pt_weights_layer_2, weights_layer_2, cc, batch_size);
    encodeWeights(pt_weights_layer_3, weights_layer_3, cc, batch_size);
    encodeBiases(pt_bias_layer_1, bias_layer_1, cc, batch_size);
    encodeBiases(pt_bias_layer_2, bias_layer_2, cc, batch_size);
    encodeBiases(pt_bias_layer_3, bias_layer_3, cc, batch_size);
    processing_time = TOC(t);
    cout << "=> Inputs and parameters encryption and encoding done [" << processing_time << " ms]" << endl;

    // ==================================================================================
    // Etape 4
    // Evaluation du modèle
    // ==================================================================================

    vector<double> precision_results;

    cout << "{...} Encryption of the inputs" << endl;
    TIC(t);
    vector<Ciphertext<DCRTPoly>> ct_input = encryptData(inputsSplit(inputs, batch_size, i), cc, keys.publicKey);
    processing_time = TOC(t);
    cout << "=> Inputs encryption done [" << processing_time << " ms]" << endl;

    // for (int j = 0; j < int(ct_input.size()); j++) {
    //     Serial::SerializeToFile(dir_path + "/input_ct_" + to_string(j) + ".txt", ct_input[j], SerType::BINARY);
    //     cout << j << endl;
    // }


    // // vérification :
    // vector<vector<double>> inputs_verif = inputsSplit(inputs, batch_size, i);
    // for (int l = 0; l < 20; l++) {
    //     for (int j = 0; j < int(inputs_verif[0].size()); j++) {
    //         cout << inputs_verif[l][j] << " | ";
    //     }
    //     cout << "\n" << endl;
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

    // // Vérification :
    // vector<Plaintext> ct_layer_1_dec = decryptVector(ct_layer_1, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_1_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_1_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_1_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }


    vector<vector<double>> layer_1_verif = ptNeuron(weights_layer_1, bias_layer_1, inputsSplit(inputs, batch_size, i), batch_size);
    // vector<Plaintext> pt_layer_1_dec = decryptVector(ct_layer_1, cc, keys.secretKey);
    
    // Vérification des valeurs et comparaison
    // cout << "Vérification des valeurs et comparaison" << endl;
    // for (int i = 0; i < int(pt_layer_1_dec.size()); i++) {
    //     cout << "Valeur " << i << " : " << pt_layer_1_dec[i]->GetRealPackedValue()[0] << endl;
    // }
    // cout << "Vérification des valeurs et comparaison" << endl;
    // for (int i = 0; i < int(layer_1_verif.size()); i++) {
    //     cout << "Valeur " << i << " : " << layer_1_verif[i][0] << endl;
    // }

    // accuracy.push_back(calculAverageAccuracy(layer_1_verif, pt_layer_1_dec));
    // accuracy.push_back(calculAverageGap(layer_1_verif, pt_layer_1_dec));

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

    // // Vérification :
    // vector<Plaintext> ct_layer_1_act_dec = decryptVector(ct_layer_1_act, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_1_act_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_1_act_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_1_act_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }

    vector<vector<double>> layer_1_act_verif = ptRelu(layer_1_verif);
    // vector<Plaintext> pt_layer_1_act_dec = decryptVector(ct_layer_1_act, cc, keys.secretKey);
    // accuracy.push_back(calculAverageAccuracy(layer_1_act_verif, pt_layer_1_act_dec));
    // accuracy.push_back(calculAverageGap(layer_1_act_verif, pt_layer_1_act_dec));
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

    // // Vérification :
    // vector<Plaintext> ct_layer_2_dec = decryptVector(ct_layer_2, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_2_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_2_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_2_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }


    vector<vector<double>> layer_2_verif = ptNeuron(weights_layer_2, bias_layer_2, layer_1_act_verif, batch_size);
    // vector<Plaintext> pt_layer_2_dec = decryptVector(ct_layer_2, cc, keys.secretKey);
    // accuracy.push_back(calculAverageAccuracy(layer_2_verif, pt_layer_2_dec));
    // accuracy.push_back(calculAverageGap(layer_2_verif, pt_layer_2_dec));

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

    // // Vérification :
    // vector<Plaintext> ct_layer_2_act_dec = decryptVector(ct_layer_2_act, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_2_act_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_2_act_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_2_act_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }

    vector<vector<double>> layer_2_act_verif = ptRelu(layer_2_verif);
    // vector<Plaintext> pt_layer_2_act_dec = decryptVector(ct_layer_2_act, cc, keys.secretKey);
    // accuracy.push_back(calculAverageAccuracy(layer_2_act_verif, pt_layer_2_act_dec));
    // accuracy.push_back(calculAverageGap(layer_2_act_verif, pt_layer_2_act_dec));

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

    // // Vérification :
    // vector<Plaintext> ct_layer_3_dec = decryptVector(ct_layer_3, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_3_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_3_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_3_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }

    vector<vector<double>> layer_3_verif = ptNeuron(weights_layer_3, bias_layer_3, layer_2_act_verif, batch_size);
    // vector<Plaintext> pt_layer_3_dec = decryptVector(ct_layer_3, cc, keys.secretKey);
    // accuracy.push_back(calculAverageAccuracy(layer_3_verif, pt_layer_3_dec));
    // accuracy.push_back(calculAverageGap(layer_3_verif, pt_layer_3_dec));

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

    // // Vérification :
    // vector<Plaintext> ct_layer_3_act_dec = decryptVector(ct_layer_3_act, cc, keys.secretKey);
    // for (int l = 0; l < int(ct_layer_3_act_dec.size()); l++) {
    //     for (int j = 0; j < int(ct_layer_3_act_dec[0]->GetRealPackedValue().size()); j++) {
    //         cout << ct_layer_3_act_dec[l]->GetRealPackedValue()[j] << " | ";
    //     }
    //     cout << "\n" << endl;
    // }

    vector<vector<double>> layer_3_act_verif = ptSigmoid(layer_3_verif);
    vector<Plaintext> pt_layer_3_act_dec = decryptVector(ct_layer_3_act, cc, keys.secretKey);
    // accuracy.push_back(calculAverageAccuracy(layer_3_act_verif, pt_layer_3_act_dec));
    // accuracy.push_back(calculAverageGap(layer_3_act_verif, pt_layer_3_act_dec));

    // ==================================================================================
    // Etape 4.7
    // Evaluation de la précision globale
    // ==================================================================================
    cout << "\n{...} Evaluation of the global accuracy" << endl;
    evaluateStats(layer_3_act_verif, pt_layer_3_act_dec, labels, dir_path, batch_size, i, mult_depth, polynomial_degree, scaling_modulus_bits, cc, time);

    // double gap_precision_pt_calcul = abs(layer_3_act_verif[0] - pt_layer_3_act_dec[0]->GetRealPackedValue()[0]);
    // double gap_precision_label = abs(layer_3_act_verif[0] - labels[index_test[i]]);
    // double is_prediction_correct = isPredictionCorrect(labels[index_test[i]], pt_layer_3_act_dec[0]->GetRealPackedValue()[0], &nb_true_test_1, &nb_false_test_1, &nb_true_test_0, &nb_false_test_0);
    // precision_results.push_back(gap_precision_pt_calcul);
    // precision_results.push_back(gap_precision_label);
    // precision_results.push_back(is_prediction_correct);
    // precision_results.push_back(layer_3_act_verif[0]);
    // precision_results.push_back(pt_layer_3_act_dec[0]->GetRealPackedValue()[0]);
    // precision_results.push_back(labels[index_test[i]]);

    // ==================================================================================
    // Etape 4.8
    // Stockage des résultats
    // ==================================================================================
    // cout << "\n{...} Saving results" << endl;

    // savingStats(accuracy, dir_path + "/avg_prec_1_each_layer.csv", index_test[i]);
    // savingStats(time, dir_path + "/avg_prec_1_each_layer_time.csv", index_test[i]);
    // savingStats(precision_results, dir_path + "/avg_prec_1_final_result.csv", index_test[i]);
    // accuracy_results_each_layer.push_back(accuracy);
    // time_results_each_layer.push_back(time);
    // precision_final_results.push_back(precision_results);
    

    // ==================================================================================
    // Etape 5
    // Stockage des résultats de l'inférence dans un fichier
    // ==================================================================================

    // savingStats(accuracy_results_each_layer, dir_path + "/avg_prec_1_each_layer.csv", index_test);
    // savingStats(time_results_each_layer, dir_path + "/avg_prec_1_each_layer_time.csv", index_test);
    // savingStats(precision_final_results, dir_path + "/avg_prec_1_final_result.csv", index_test);
    // savingRecap(dir_path + "/avg_prec_1_recap.csv", nb_test, nb_true_test_1, nb_false_test_1, nb_true_test_0, nb_false_test_0, cc, mult_depth, scaling_modulus_bits, polynomial_degree);

    // cout << "\n=================\n{...} Results\n" << "=================\n" << endl;
    // cout << "Nombre de tests : " << nb_test << endl;
    // cout << "Nombre de tests censés être >50k : " << nb_test/2 << endl;
    // cout << "Nombre de tests censés être <50k : " << nb_test/2 << endl;
    // cout << "Nombre de vrais >50k : " << nb_true_test_1 << endl;
    // cout << "Nombre de faux >50k : " << nb_false_test_1 << endl;
    // cout << "Nombre de vrais <50k : " << nb_true_test_0 << endl;
    // cout << "Nombre de faux <50k : " << nb_false_test_0 << endl;
    // cout << "Précision : " << double(nb_true_test_1 + nb_true_test_0)/double(nb_true_test_1 + nb_true_test_0 + nb_false_test_1 + nb_false_test_0) << endl;

    return 0;

}