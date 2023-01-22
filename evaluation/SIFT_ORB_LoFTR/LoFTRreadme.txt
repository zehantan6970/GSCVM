This file is used "LoFTR: Detector-Free Local Feature Matching with Transformers" to test the ablation experiment.
The test.py for LoFTR test.
The train.py for LoFTR train.
The main code is loftr.py.
You should get clone LoFTR from https://github.com/zju3dv/LoFTR.
A brief explanation: Considering the " Matches with high confidence are selected from these dense matches and later refined to a subpixel level with a correlation-based approach"  mentioned in LoFTR's paper, that means "The key points of loFTR in the top 100 pairs of points with high confidence are almost concentrated in some small areas", the experimental results of LoFTR+Votenet are not as good as the non-dense feature point pairs found by SuperGlue+Votenet.
