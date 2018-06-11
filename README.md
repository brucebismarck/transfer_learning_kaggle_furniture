# transfer_learning_kaggle_furniture
kaggle furniture transfer learning task


7% in final leaderboard. Learned a lot from https://github.com/skrypka/imaterialist-furniture-2018 and also from official pytorch tutorial.


Some Reflections after read the solution from first place team final solution.
1. As long as we get some information about how to train the model, we dont need to try fixed epoch numbers which wastes time.
2. In my working, augumentation on color, hue, contrast etc. does not work. But I did not realize by using some corners and center(5-crop) it works better.
3. Model understanding. Didn't try , dpn92, xception, inceptionResNetv2, senet154, renext101 etc.
4. Tried many summarization methods mean, median, weighted mean. According to them, gmean is better.
5. Calibration.
