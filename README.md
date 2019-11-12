Kim_CNN paper description: 

"NLP@ UIOWA at SemEval-2019 Task 6: Classifying the Crass using Multi-windowed CNNs."
Rusert, Jonathan, and Padmini Srinivasan.
Proceedings of the 13th International Workshop on Semantic Evaluation.
2019.
http://homepage.divms.uiowa.edu/~jrusert/S19-2125.pdf



How to run: 
python3 Kim_CNN_random.py trainingData testData outputFile

NOTE: Uses tensorflow 1.7.0, will error on the newest versions due to change in documentation



How to evaluate output: 
python3 evaluate.py outputFile goldFile