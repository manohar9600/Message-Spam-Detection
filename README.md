# Message Spam Detection

Created an app for detecting spam message using sequence modelling in deep learning.

* App useful to detect messages from different languages ( English, Hindi, Telugu ) as spam or not.
* By using Bag of words and applying normal Machine Learning techniques obtained maximum of 95.2% accuracy for test data. 
* But we observed that it ignored context of message for message classification.
* As to classify messages based on context of message we used Recurrent Neural Networks.
* By using word embeddings and LSTM we obtained 98.2% accuracy.
* Python is used for training above model. Created sample electron app using HTML, CSS and NodeJs.



commands for project:  ( in project folder)

```bash
cd Electron/Src/Python
python rpc_server.py
```

wait for some time -- &&

open another terminal in project folder:

```bash
cd Electron

npm install
npm start
```




( optional ) - For Training the model :

```bash
cd src
python lstm.py
```