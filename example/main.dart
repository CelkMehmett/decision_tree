import '../lib/decision_tree.dart';

void main() {
  final data = [
    ['Sunny', 'Hot', 'High', false],
    ['Sunny', 'Hot', 'High', true],
    ['Overcast', 'Hot', 'High', false],
    ['Rain', 'Mild', 'High', false],
    ['Rain', 'Cool', 'Normal', false],
    ['Rain', 'Cool', 'Normal', true],
    ['Overcast', 'Cool', 'Normal', true],
    ['Sunny', 'Mild', 'High', false],
    ['Sunny', 'Cool', 'Normal', false],
    ['Rain', 'Mild', 'Normal', false],
    ['Sunny', 'Mild', 'Normal', true],
    ['Overcast', 'Mild', 'High', true],
    ['Overcast', 'Hot', 'Normal', false],
    ['Rain', 'Mild', 'High', true],
  ];

  final labels = [
    'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
    'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'
  ];

  final clf = DecisionTreeClassifier();
  clf.fit(data, labels);

  final testSamples = [
    ['Sunny', 'Cool', 'High', true],
    ['Rain', 'Mild', 'Normal', false],
  ];

  final predictions = clf.predict(testSamples);
  print('Predictions: $predictions');
}
