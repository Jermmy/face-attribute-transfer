import pickle

with open('data/aus_openface.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin')
    print(data)