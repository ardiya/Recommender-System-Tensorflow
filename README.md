# Recommender-System-Tensorflow
Simple Recommender System implemented using TensorFlow and TF-Slim

## Collaborative Filtering using AutoEncoder
This model was created using step-by-step tutorial from https://github.com/fstrub95/torch.github.io/blob/master/blog/_posts/2016-02-21-cfn.md
In this method we tried to predict the rating 

Result
 - On the MovieLens-1M dataset, we achieve Mean Average Error of 0.560396 and Root Mean Squared Error of 0.748613
 - ... TODO show more result

```bash
python trainAE.py
python evaluateAE.py
```

## Hybrid collaborative filtering using Denoising AutoEncoders and Embedding
This method uses Embedding on userdata to improve the model that uses Denoising AutoEncoder on rating matrix.

```bash
python trainDAE.py
python evaluateDAE.py
```

... TODO introduce embedding
