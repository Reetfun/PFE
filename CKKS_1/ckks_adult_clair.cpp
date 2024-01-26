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

using namespace std;
using namespace lbcrypto;

string initDirectory(string path, int nb_test) {

    auto currentTimePoint = chrono::system_clock::now();
    time_t currentTime = chrono::system_clock::to_time_t(currentTimePoint);
    tm* localTime = localtime(&currentTime);
    stringstream timestamp;
    timestamp << put_time(localTime, "%Y%m%d_%H%M%S");

    string folderName = "Results_" + timestamp.str() + "_nb" + to_string(nb_test);
    string full_path = path + "/" + folderName;
    filesystem::create_directory(full_path);

    ofstream file_time;

    file_time.open(full_path + "/" + "avg_prec_1_each_layer_time.csv");

    if (file_time.is_open()) {
        file_time << "index,layer1,activation1,layer2,activation2,layer3,activation3" << endl;
        file_time.close();
    } else {
        cerr << "Error opening the file." << endl;
    }

    return full_path; 
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

vector<double> ptSigmoid(vector<double> verif) {
    vector<double> verif_sigmoid;
    for (int i = 0; i < int(verif.size()); i++) {
        double ct = 1 / (1 + exp(-verif[i]));
        verif_sigmoid.push_back(ct);
    }
    return verif_sigmoid;
}

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

void isPredictionCorrect(int label, double prediction, int *nb_true_test_1, int *nb_false_test_1, int *nb_true_test_0, int *nb_false_test_0) {
    if (label == 1) {
        if (prediction >= 0.5) {
            *nb_true_test_1 += 1;
        } else {
            *nb_false_test_1 += 1;
        }
    } else {
        if (prediction < 0.5) {
            *nb_true_test_0 += 1;
        } else {
            *nb_false_test_0 += 1;
        }
    }
}

int main(int argc, char *argv[]) {
    TimeVar t;
    TimeVar t1;
    double proc_total_time(0.0);
    double processing_time(0.0);
    vector<vector<double>> time_results_each_layer;
    int nb_false_test_1 = 0;
    int nb_true_test_1 = 0;
    int nb_false_test_0 = 0;
    int nb_true_test_0 = 0;

    // ==================================================================================      
    // Etape 1.1
    // Initialisation des répertoires de sauvegarde des résultats
    // ==================================================================================
    string dir_path = initDirectory("../CKKS_1/results_pt", 8192);

    // ==================================================================================
    // Etape 2
    // Extraction et préparation des données
    // ==================================================================================
    cout << "\n{...} Reading inputs and labels from file" << endl;
    TIC(t);
    vector<string> const categories = getCategories();
    vector<int> const labels = getLabels();
    vector<vector<double>> const inputs = getInputs();
    vector<int> const index_test = getRandomIndexTest(4096, 4096, labels);
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
    // Etape 4
    // ==================================================================================

    TIC(t1);
    for (int i = 0; i < int(index_test.size()); i++) {
        vector<double> time;

        cout << "\n=================\n{...} Test " << i+1 << "\n" << "=================\n" << endl;

        // ==================================================================================
        // Etape 4.1
        // Evaluation de la première couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the first layer" << endl;
        TIC(t);
        vector<double> layer_1_verif = ptNeuron(weights_layer_1, bias_layer_1, inputs[index_test[i]]);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 1 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.2
        // Evaluation de la fonction d'activation de la première couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the first layer" << endl;
        TIC(t);
        vector<double> layer_1_act_verif = ptRelu(layer_1_verif);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 1 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.3
        // Evaluation de la deuxième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the second layer" << endl;
        TIC(t);
        vector<double> layer_2_verif = ptNeuron(weights_layer_2, bias_layer_2, layer_1_act_verif);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 2 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.4
        // Evaluation de la fonction d'activation de la deuxième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the second layer" << endl;
        TIC(t);
        vector<double> layer_2_act_verif = ptRelu(layer_2_verif);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 2 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.5
        // Evaluation de la troisième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the third layer" << endl;
        TIC(t);
        vector<double> layer_3_verif = ptNeuron(weights_layer_3, bias_layer_3, layer_2_act_verif);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Multiplication & Addition of layer 3 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.6
        // Evaluation de la fonction d'activation de la troisième couche
        // ==================================================================================
        cout << "\n{...} Evaluation of the activation function of the third layer" << endl;
        TIC(t);
        vector<double> layer_3_act_verif = ptSigmoid(layer_3_verif);
        processing_time = TOC(t);
        time.push_back(processing_time);
        cout << "=> Activation of layer 3 done [" << processing_time << " ms]" << endl;

        // ==================================================================================
        // Etape 4.7
        // Evaluation de la précision globale
        // ==================================================================================
        isPredictionCorrect(labels[index_test[i]], layer_3_act_verif[0], &nb_true_test_1, &nb_false_test_1, &nb_true_test_0, &nb_false_test_0);

        // ==================================================================================
        // Etape 4.8
        // Stockage des résultats
        // ==================================================================================
        cout << "\n{...} Saving results" << endl;

        savingStats(time, dir_path + "/avg_prec_1_each_layer_time.csv", index_test[i]);
        // accuracy_results_each_layer.push_back(accuracy);
        // time_results_each_layer.push_back(time);
        // precision_final_results.push_back(precision_results);
    }
    proc_total_time = TOC(t1);
    cout << proc_total_time << endl;


    // ==================================================================================
    // Etape 5
    // Stockage des résultats de l'inférence dans un fichier
    // ==================================================================================

    cout << "\n=================\n{...} Results\n" << "=================\n" << endl;
    cout << "Nombre de tests : " << 8192 << endl;
    cout << "Nombre de tests censés être >50k : " << 8192/2 << endl;
    cout << "Nombre de tests censés être <50k : " << 8192/2 << endl;
    cout << "Nombre de vrais >50k : " << nb_true_test_1 << endl;
    cout << "Nombre de faux >50k : " << nb_false_test_1 << endl;
    cout << "Nombre de vrais <50k : " << nb_true_test_0 << endl;
    cout << "Nombre de faux <50k : " << nb_false_test_0 << endl;
    cout << "Précision : " << double(nb_true_test_1 + nb_true_test_0)/double(nb_true_test_1 + nb_true_test_0 + nb_false_test_1 + nb_false_test_0) << endl;

    return 0;

}