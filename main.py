import os
from cv_data_etl import transform_data
from cv_dtree import data_prep, build_tree, visualize_dtree, get_lineage, get_rules


feature_names = ['BMI Category-Binary', 'BP_Category-Binary', 'Age', 'Gender-1', 'Cholestrol', 'Glucose', 'Smoke', 'Alcohol',
                 'Physical_Activity']
label_col = 'Cardio_Disease'


def main():
    print('Current Working Directory: ', os.curdir)
    cv_df = transform_data('data/Output_13Nov_v3.csv')
    X_train, X_test, y_train, y_test = data_prep(cv_df, feature_names, label_col)
    clf = build_tree(X_train, X_test, y_train, y_test)
    visualize_dtree(clf, feature_names)
    # get_lineage(clf, feature_names)
    get_rules(clf, feature_names)


if __name__ == '__main__':
    main()
