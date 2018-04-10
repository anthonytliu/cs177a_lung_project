- installed tensorflow
- installed tensorpack
- cloned densenet for tensorpack, tried to run, failed
- cloned FC-densenet: https://github.com/SimJeg/FC-DenseNet
- tried running FC-densenet, first problem was with the dataset_loaders, found in:
----
RuntimeError: The config.ini is missing. Make sure to create one in "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/\site-packages/dataset_loaders-1.0.0-py2.7.egg/
dataset_loaders/config.ini" according to the config.ini.example in the same path.
----
- I copied the existing config.ini.example from the dataset_loaders file downloaded and pasted my version (with the proper directories)
- Next error that came up was with:
----
File "/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/dataset_loaders-1.0.0-py2.7.egg/
dataset_loaders/images/camvid.py", line 83, in filenames
    with open(os.path.join(self.path, self.which_set + '.txt')) as f:

IOError: [Errno 2] No such file or directory: '/Users/anthonyliu/Documents/College/Junior Year/COSI177a/FC-DenseNet/local/camvid/train.txt'
----
- I expect that it's looking for
