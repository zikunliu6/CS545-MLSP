# CS545-PS Final Project

This is the repo for cs543 computer vision final project. 

We are interested in using generative models to help music improvisation and we believe this will give in-sights into music creation. With a small piece of music from the user as input, we manage to generate another piece of music that naturally follows the melody and the style of the input by using conditional GAN. Using generative models for music is an emerging area of study and we envision providing an useful tool for the music industry

**Dependency:**

```
pytorch
numpy
scipy
matplotlib
```

**File Explanation**

1. main.py contains the GAN structure and training function. It will automatically save the trained model into a newly built folder.
2. generate_music is used to generate music using the trained model and save the format as .midi file in the current folder

- To run the training, do following:

```
python main.py
```

- The dataset contains over 3000 piece of music that was randomly chosen from the MAESTRO music dataset
- The dataset can be found here:

```
https://drive.google.com/file/d/1FP-6PisCP8or4409tynz3R3yET12Afss/view?usp=sharing
```

Each .npy file in the folder means a music piece, which has the dimension of 100 by 1, representing 100 notes.
