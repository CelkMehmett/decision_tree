import 'dart:math';

/// A simple Decision Tree Classifier using the ID3 algorithm.
/// Supports categorical features only.
class DecisionTreeClassifier {
  late _Node _root;

  /// Trains the classifier on the given features [X] and target labels [y].
  /// [X] must be a list of samples where each sample is a list of categorical feature values.
  void fit(List<List<dynamic>> X, List<dynamic> y) {
    if (X.isEmpty || X[0].isEmpty) {
      throw ArgumentError('Input data X must be non-empty.');
    }
    if (X.length != y.length) {
      throw ArgumentError('Feature and label lengths must match.');
    }

    final featureIndices = List.generate(X[0].length, (i) => i);
    _root = _buildTree(X, y, featureIndices);
  }

  /// Predicts the labels for given feature set [X].
  List<dynamic> predict(List<List<dynamic>> X) {
    if (_root == null) {
      throw StateError('Model has not been trained. Call fit() first.');
    }

    return X.map((row) => _predictSingle(row, _root)).toList();
  }

  dynamic _predictSingle(List<dynamic> row, _Node node) {
    if (node.isLeaf) return node.label;

    final featureValue = row[node.featureIndex!];
    final nextNode = node.children?[featureValue];

    if (nextNode != null) {
      return _predictSingle(row, nextNode);
    } else {
      // Unknown feature value → return most likely class at this node
      return node.majorityLabel;
    }
  }

  _Node _buildTree(List<List<dynamic>> X, List<dynamic> y, List<int> features) {
    // Case 1: All labels are same → Leaf node
    if (y.toSet().length == 1) return _Node.leaf(y[0]);

    // Case 2: No features left → Return majority class
    if (features.isEmpty) return _Node.leaf(_majorityClass(y));

    // Choose best feature based on information gain
    final bestFeature = _selectBestFeature(X, y, features);
    final node = _Node.branch(bestFeature, _majorityClass(y));
    node.children = {};

    final uniqueValues = X.map((row) => row[bestFeature]).toSet();
    for (final value in uniqueValues) {
      final indices = <int>[];
      for (int i = 0; i < X.length; i++) {
        if (X[i][bestFeature] == value) {
          indices.add(i);
        }
      }

      final subX = indices.map((i) => X[i]).toList();
      final subY = indices.map((i) => y[i]).toList();
      final remainingFeatures = List.of(features)..remove(bestFeature);

      node.children![value] = _buildTree(subX, subY, remainingFeatures);
    }

    return node;
  }

  int _selectBestFeature(List<List<dynamic>> X, List<dynamic> y, List<int> features) {
    final baseEntropy = _calculateEntropy(y);
    double bestGain = -1;
    int bestFeature = features.first;

    for (final feature in features) {
      final subsets = <dynamic, List<dynamic>>{};

      for (int i = 0; i < X.length; i++) {
        final value = X[i][feature];
        subsets.putIfAbsent(value, () => []).add(y[i]);
      }

      double newEntropy = 0.0;
      for (final group in subsets.values) {
        final p = group.length / y.length;
        newEntropy += p * _calculateEntropy(group);
      }

      final gain = baseEntropy - newEntropy;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = feature;
      }
    }

    return bestFeature;
  }

  double _calculateEntropy(List<dynamic> labels) {
    final labelCounts = <dynamic, int>{};
    for (var label in labels) {
      labelCounts[label] = (labelCounts[label] ?? 0) + 1;
    }

    double entropy = 0.0;
    for (final count in labelCounts.values) {
      final p = count / labels.length;
      entropy -= p * log(p) / ln2;
    }

    return entropy;
  }

  dynamic _majorityClass(List<dynamic> labels) {
    final counts = <dynamic, int>{};
    for (final label in labels) {
      counts[label] = (counts[label] ?? 0) + 1;
    }

    return counts.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }
}

/// Internal representation of a node in the decision tree.
class _Node {
  final bool isLeaf;
  final dynamic label; // for leaf
  final int? featureIndex; // for branch
  final dynamic majorityLabel;
  Map<dynamic, _Node>? children;

  _Node.leaf(this.label)
      : isLeaf = true,
        featureIndex = null,
        majorityLabel = label;

  _Node.branch(this.featureIndex, this.majorityLabel)
      : isLeaf = false,
        label = null;
}
