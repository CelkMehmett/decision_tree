import 'package:test/test.dart';
import '../lib/decision_tree.dart';

void main() {
  group('DecisionTreeClassifier', () {
    test('basic training and prediction', () {
      final X = [
        ['Sunny', 'Hot', 'High', false],
        ['Rain', 'Cool', 'Normal', true],
        ['Overcast', 'Hot', 'High', false],
      ];
      final y = ['No', 'Yes', 'Yes'];

      final clf = DecisionTreeClassifier();
      clf.fit(X, y);

      final prediction = clf.predict([
        ['Sunny', 'Hot', 'High', false],
        ['Rain', 'Cool', 'Normal', true],
        ['Overcast', 'Hot', 'High', false],
      ]);

      expect(prediction, equals(['No', 'Yes', 'Yes']));
    });
  });
}
