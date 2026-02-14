# CROCO-Hook: Crochet stitch counter

It would be so great to watch a movie while crocheting and not having to keep up with the stitch count in the back of your mind. 

Currently it is specialized for simple chain stitch which is a foundational stitch. (The only stitch that didn't take me ages to learn)

DEMO: https://huggingface.co/spaces/samriddhisr/CROCO

###Feature extraction:
**DINO Embeddings:** We pass every video frame through the DINOv2 model. 
**Timestamps:** For every frame processed, we record a timestamp. The stitch boundary labelling was done through recording timestamps and not frames due to different FPS of the training videos. 


###Training:
We used a Bi-GRU as the core of our model. Gaussian smoothing was used to create a probability mountain around each stitch since the goal was to count the number of stitches.

We utilized a lower sigma (0.25) in our Gaussian smoothing function to define sharper, more distinct probability peaks around the stitch boundaries.

To feed the data into the Bi-GRU, we broke the videos into windows of 64 frames. We moved this window across the footage with a stride of 16.

Because a stride of 16 rarely lands perfectly at the end of a video, we implemented an extra tail coverage by manually forcing the system to reach back from the very last frame to grab the final windows.


**To simulate a live webcam feed during inference, we implemented a Sliding Window Buffer system. This ensures that the model always has enough temporal context to make a decision without letting new data corrupt the counts we’ve already finalized.**

I really want to learn how to crochet amigurumi dachshunds. 
