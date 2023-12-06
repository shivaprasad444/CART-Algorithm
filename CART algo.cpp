#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>

using namespace std;

// Structure to represent a node in the decision tree
struct Node {
    int featureIndex;
    double splitValue;
    double classLabel;
    Node* left;
    Node* right;

    // Constructor for internal nodes
    Node(int featureIndex, double splitValue, Node* left, Node* right)
        : featureIndex(featureIndex), splitValue(splitValue), classLabel(-1.0), left(left), right(right) {}

    // Constructor for leaf nodes
    Node(double classLabel)
        : featureIndex(-1), splitValue(-1.0), classLabel(classLabel), left(nullptr), right(nullptr) {}
};

// Function to calculate Gini impurity
double calculateGini(const vector<double>& labels) {
    int size = labels.size();
    if (size == 0) return 0.0;

    // Count the occurrences of each class
    vector<int> classCounts(2, 0);
    for (double label : labels) {
        int classIndex = static_cast<int>(label);
        classCounts[classIndex]++;
    }

    // Calculate Gini impurity
    double gini = 1.0;
    for (int count : classCounts) {
        double probability = static_cast<double>(count) / size;
        gini -= probability * probability;
    }

    return gini;
}

// Function to split the dataset based on a given feature and split value
pair<vector<vector<double>>, vector<vector<double>>> splitDataset(const vector<vector<double>>& dataset, int featureIndex, double splitValue) {
    vector<vector<double>> leftSubset, rightSubset;
    for (const vector<double>& dataPoint : dataset) {
        if (dataPoint[featureIndex] < splitValue) {
            leftSubset.push_back(dataPoint);  // Entire data point goes to left subset
        } else {
            rightSubset.push_back(dataPoint); // Entire data point goes to right subset
        }
    }
    return make_pair(leftSubset, rightSubset);
}

// Function to find the best split for a given dataset and features
pair<int, double> findBestSplit(const vector<vector<double>>& dataset, const vector<int>& features) {
    int numFeatures = features.size();
    int numDataPoints = dataset.size();
    double bestGini = numeric_limits<double>::infinity();
    int bestFeatureIndex = -1;
    double bestSplitValue = 0.0;

    // Iterate over all features
    for (int featureIndex : features) {
        // Sort the dataset based on the current feature
        vector<vector<double>> sortedDataset = dataset;
        sort(sortedDataset.begin(), sortedDataset.end(),
            [featureIndex](const vector<double>& a, const vector<double>& b) {
                return a[featureIndex] < b[featureIndex];
            });

        // Iterate over possible split values
        for (int i = 1; i < numDataPoints; ++i) {
            double splitValue = (sortedDataset[i - 1][featureIndex] + sortedDataset[i][featureIndex]) / 2.0;

            // Split the dataset
            auto subsets = splitDataset(sortedDataset, featureIndex, splitValue);
            vector<vector<double>>& leftSubset = subsets.first;
            vector<vector<double>>& rightSubset = subsets.second;

            // Extract class labels from subsets
            vector<double> leftLabels, rightLabels;
            for (const auto& dataPoint : leftSubset) {
                leftLabels.push_back(dataPoint.back());
            }
            for (const auto& dataPoint : rightSubset) {
                rightLabels.push_back(dataPoint.back());
            }

            // Calculate Gini impurity for the split
            double gini = (static_cast<double>(leftSubset.size()) / numDataPoints) * calculateGini(leftLabels)
                        + (static_cast<double>(rightSubset.size()) / numDataPoints) * calculateGini(rightLabels);

            // Update best split if current split is better
            if (gini < bestGini) {
                bestGini = gini;
                bestFeatureIndex = featureIndex;
                bestSplitValue = splitValue;
            }
        }
    }

    return make_pair(bestFeatureIndex, bestSplitValue);
}


// Function to classify a data point using the decision tree
double classify(Node* node, const vector<double>& dataPoint) {
    if (node->featureIndex == -1) {
        // Leaf node, return the class label
        return node->classLabel;
    } else {
        // Internal node, traverse left or right based on the split
        if (dataPoint[node->featureIndex] < node->splitValue) {
            return classify(node->left, dataPoint);
        } else {
            return classify(node->right, dataPoint);
        }
    }
}


// Function to build the decision tree recursively
Node* buildTree(const vector<vector<double>>& dataset, const vector<int>& features) {
    // If all data points have the same class label, create a leaf node
    double firstLabel = dataset[0].back();
    if (all_of(dataset.begin(), dataset.end(), [firstLabel](const vector<double>& dataPoint) {
        return dataPoint.back() == firstLabel;
    })) {
        return new Node(firstLabel);
    }

    // If no features are left, create a leaf node with the majority class label
    if (features.empty()) {
        std::map<double, int> labelCounts; // Use std::map if not using namespace std
        for (const auto& dataPoint : dataset) {
            labelCounts[dataPoint.back()]++;
        }
        int majorityClass = max_element(labelCounts.begin(), labelCounts.end(), 
                                        [](const pair<double, int>& a, const pair<double, int>& b) {
                                            return a.second < b.second; 
                                        })->first;
        return new Node(majorityClass);
    }

    // Find the best split
    pair<int, double> bestSplit = findBestSplit(dataset, features);
    int bestFeatureIndex = bestSplit.first;
    double bestSplitValue = bestSplit.second;

    // Split the dataset
    pair<vector<vector<double>>, vector<vector<double>>> subsets = splitDataset(dataset, bestFeatureIndex, bestSplitValue);
    vector<vector<double>> leftSubset = subsets.first;   // Corrected type
    vector<vector<double>> rightSubset = subsets.second; // Corrected type

    // Recursively build the left and right subtrees
    Node* leftChild = buildTree(leftSubset, features);   // Pass leftSubset
    Node* rightChild = buildTree(rightSubset, features); // Pass rightSubset

    // Create and return the current node
    return new Node(bestFeatureIndex, bestSplitValue, leftChild, rightChild);
}

int main() {
    // Prompt the user for the number of data points and features
    int numDataPoints, numFeatures;
    cout << "Enter the number of data points: ";
    cin >> numDataPoints;
    cout << "Enter the number of features: ";
    cin >> numFeatures;

    // Prompt the user for the dataset
    vector<vector<double>> dataset(numDataPoints, vector<double>(numFeatures + 1, 0.0)); // +1 for the class label
    cout << "Enter the dataset (each row should contain features followed by the class label):" << endl;
    for (int i = 0; i < numDataPoints; ++i) {
        cout << "Data point " << i + 1 << ": ";
        for (int j = 0; j < numFeatures + 1; ++j) {
            cin >> dataset[i][j];
        }
    }

    // Prompt the user for the features available for splitting
    vector<int> features(numFeatures);
    cout << "Enter the features available for splitting (0-based indices): ";
    for (int i = 0; i < numFeatures; ++i) {
        cin >> features[i];
    }
    
    // Build the decision tree
    Node* root = buildTree(dataset, features);
    
    vector<double> newDataPoint(numFeatures);
    cout << "Enter the features of a new data point for classification:" << endl;
    for (int i = 0; i < numFeatures; ++i) {
        cout << "Feature " << i + 1 << ": ";
        cin >> newDataPoint[i];
    }

    double predictedClass = classify(root, newDataPoint);

    // Output the predicted class
    cout << "Predicted class for the new data point: " << predictedClass << endl;

    // You can now use the tree for classification or regression tasks.
  return 0;
}