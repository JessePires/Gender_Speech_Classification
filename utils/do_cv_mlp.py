import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from tqdm.notebook import tqdm
from sklearn.neural_network import MLPClassifier


def do_cv_mlp(X, y, cv_splits, params_cv_folds, params={'random_state':[1], 'solver' : ['adam'], 'max_iter': [2000], 'alpha': [0.0001, 0.001, 0.005], 'hidden_layer_sizes': [(10), (10, 10), (10, 50), (100, 100, 100), (71, 71), (200, 200, 200, 200), (500, 500, 500, 500)]}, n_jobs=1):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    fold = 1
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)
        
        score = 'f1' if len(set(y_treino)) < 3 else 'f1_weighted'
        
        grid = GridSearchCV(MLPClassifier(), param_grid=params, cv=StratifiedKFold(n_splits=params_cv_folds), n_jobs=n_jobs, scoring=score)   
        grid.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
        
        pred = grid.predict(X_teste)

        acuracias.append(f1_score(y_teste, pred))

        fold+=1
        pgb.update(1)
        
    pgb.close()
    
    return acuracias