import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np
import matplotlib.pyplot as plt


def data_prep(cv_df, feature_cols, label_col):
    cv_df.columns
    X = cv_df[feature_cols]
    y = cv_df[label_col]
    return train_test_split(X, y, test_size=0.00000001, random_state=1, shuffle=True)


def build_tree(X_train, X_test, y_train, y_test):
    print('Decision Tree')
    clf = DecisionTreeClassifier(max_depth=10, max_features=1, min_samples_split=int(X_train.__len__()*.2))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

    return clf


def visualize_dtree(clf, feature_names):
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
    #                 feature_names=feature_names, class_names=['0', '1'])
    # dtree_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # dtree_graph.write_png('cv_dtree_graph.png')
    # Image(dtree_graph.create_png())

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf,
                       feature_names=feature_names,
                       class_names=['0', '1'],
                       filled=True)
    fig.savefig("decistion_tree.png")


def get_lineage(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'

        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    for child in idx:
        for node in recurse(left, right, child):
            print(node)


def get_rules(clf, feature_names):
    tree_rules = export_text(clf, feature_names=feature_names)
    print(tree_rules)