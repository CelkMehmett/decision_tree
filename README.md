# ml_decision_tree

A simple and native implementation of Decision Tree Classifier (ID3) for Dart.

## Features

- Train and predict using decision trees
- Handles categorical data
- Lightweight and dependency-free

## Usage

```dart
import 'package:ml_decision_tree/ml_decision_tree.dart';

void main() {
  final X = [
    ['Sunny', 'Hot', 'High', false],
    ['Rain', 'Cool', 'Normal', true],
  ];

  final y = ['No', 'Yes'];

  final clf = DecisionTreeClassifier();
  clf.fit(X, y);

  final prediction = clf.predict([
    ['Sunny', 'Cool', 'High', true],
  ]);

  print(prediction); // ['No']
}
