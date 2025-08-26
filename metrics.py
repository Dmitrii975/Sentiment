from sklearn.metrics import f1_score, accuracy_score, adjusted_rand_score

def show_metrics(y_true, y_pred):
    print('Macro F1 Score =', f1_score(y_true, y_pred, average='macro'))
    print('Accuracy =', accuracy_score(y_true, y_pred))
    print('ARS =', adjusted_rand_score(y_true, y_pred))