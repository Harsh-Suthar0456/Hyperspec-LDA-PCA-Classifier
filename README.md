## analysis.ipynb
Contains all the analysis conducted on the outputs of Fischer's Linear Discriminant classification and PCA based Logistic Regressor.
Includes Exploratory Data Analysis done to further improve the accuracies and achieve 90%+ accuracy.

The results have been compiled up in results.md for quick access

Dataset used: Indian Pines dataset <href>https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes</href>

### Further additions
## app.py
Web Application based GUI to analyze the predictions from PCA and LDA for various datasets, with ability to regenerate any previous plots

This generates a server on the port <href>http://127.0.0.1:5000</href> after it is run, where you can upload your files to get predictions and accuracy from both LDA and PCA.

You can also get plots from any previous submissions by downloading the pickle file from the results page and reuploading it.


