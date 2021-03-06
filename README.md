# video_action_recognition
Purpose: Modified versions of gluoncv's inference.py and feat_extract.py files for local run and similarity matrix code generation for .npy vectors.

The inference_v1.py file and feat_extract_v1.py file both are versions of the gluoncv files of similar names. All command line functionality has been replaced by global variables to allow for local runs and more flexible functionality. Some default models and settings are provided. Both files generate .npy files of features, and the inference_v1.py file also generates predictions from the model's classes.

The similarity_matrix.py file loads .npy feature matrices as generated by the other files, and calculates column-average vectors before running correlation, cosine, and euclidean pairwise distance calculations to generate similarity matrices to identify distances between videos.

Here is an example of the similarity matrices generated by similarity_matrix.py for a snippet of the Office Episode 1 divided into 5 second segments, with the SomethingSomething_Resnet50 model:

![img](https://lh6.googleusercontent.com/DzqqfFkIRtiufPTlz7O9y6FJ-G8zjaoSXSAgo8gjseYBa_at3zF7Rvm9A5_oAEjZsO79w3Q2MWdx0fnbG1uUPPYzCIQ2Fs8xLpzDhUVLkAQlVk4G3iSM91Mv_KO4N6WwMT9Uc88H)

![img](https://lh5.googleusercontent.com/bEpxhI9vUG805nmvU7HrsK1oaDPq960z2nK1M6qHV9vXG8EcIQaG0_YtU5d7iPyZgQ1YcdH63pT2Nm2mhmkei1BGJDudcY88VH3P-c-nASLbr07PBzPrcaxL2zqCW6Q7UXsJQvMd)

![img](https://lh4.googleusercontent.com/uFJlu70qE40YxFB3RQnkrtnukamVGjCv0YxfInju-AQiDFDxZeLFx4fB3Ti9vHjYynZZhuPAFiB84TAuMxkdZWBrkRznWpjinP0vx-io8Cy7arl6910AILo6AoNJq4RLIxvQyoLV)

This code was made by Snigdha Saha (Caltech UG'2024) and Umit Keles (Caltech PhD) for the Adolphs Lab.

