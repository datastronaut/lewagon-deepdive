# :whale: DEEP DIVE :dolphin:

## Description
Final project of [Le Wagon](https://www.lewagon.com/) Data Science Bootcamp - batch :hash::eight::zero::two:
It has been developed in two weeks by three beginners in Python and Deep Learning
- [Victoria Metzger](https://github.com/VictoriaMetzger)
- [Timothée Filhol](https://github.com/timfilhol96)
- [Christian Lajouanie](https://github.com/ChristianDesCodes)

## Objective
Goal of the project: build a classification deep learning model.
**Input:** sound recording of an unknown marine mammal song (.wav format).
**Output:** ranking of corresponding marine mammal species from most to least probable.

## Source Data
All the data used to train our model has been downloaded from the [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds).
We have used the sounds available in the “Best of cuts” section of the site (around 1700 recordings).
We intended to work on the “all cuts” section (around 16000 recordings) but suffered RAM issues that we couldn’t solve in the allowed two weeks.

## Workflow

### Data selection
From the raw dataset, we selected:
1. The number of families and species to observe (we chose to work with all 8 families and 31 species)
2. The minimum duration (we chose none i.e. we selected all samples)
3. The ‘quality’ of audio recording (we chose to work only with cleaned samples (i.e. no external noises (boat, rain, icebergs…etc.) nor multi-species recordings (i.e. 2 or more species heard on the recording
The generated dataset amounted to approximately 900 audio samples.

### Preprocessing Workflow
Beforehand, we observed 2 main ‘issues’:
1. The species in our selected dataset were extremely unbalanced
2. The audio samples were of different durations and would thus result to spectrograms of different lengths, which would be impossible for our model to process

To address both problems, we agreed on the following preprocessing workflow:

**Target duration**: duration of all audio signals that will be processed by our model, we deemed that **5 seconds** was a sufficient duration to observe and hear significant patterns.

Step 1: **Train** - **Validation** - **Test** split on the dataset, as we will perform some Data Augmentation solely on the Train set.

Step 2: **Train** set:
- Check class balance in terms of duration (through a barplot)
- For over-represented classes :
  - **samples >= 5s:** slice them into 5s consecutive slices (+ pad the last one randomly if >= 3s)
  - **samples < 5s:** pad randomly
- For under-represented classes :
  - **samples >= 5s:** slice them into 5s consecutive slices 3 times at different intervals, for each new signal generated, apply White Noise and Random Gain, thus generating 2 additional signals
  - **samples < 5s:** pad randomly per sample
- Check class balance again and adapt preprocessing if needed

Step 3: **Validation** and **Test** set:
  - **samples >= 5s:** slice them into 5s consecutive slices (+ pad the last one randomly if >= 3s)
  - **samples < 5s:** pad randomly

Step 4: Convert all audio signals to mel spectrograms (i.e. numpy arrays)

*Note: the audio signals must be converted to mel spectrograms as CNN cannot process audio signals as such.*

### Model training

CNN architecture and training history can be found in /03_model_training/notebooks and /models

## Repositories
- **front-end repo:** [ChristianDesCodes/lewagon-deepdive-front](https://github.com/ChristianDesCodes/lewagon-deepdive-front)
- **back-end repo:** [ChristianDesCodes/lewagon-deepdive](https://github.com/ChristianDesCodes/lewagon-deepdive)
