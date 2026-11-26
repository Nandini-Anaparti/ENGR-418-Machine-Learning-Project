
# Rock, Paper, Scissors: Image Classification

**ENGR 418 – Applied Machine Learning | UBC Okanagan**

**Project Team (#24): Nandini Anaparti, Saksham Bhattrai, Reid Pratley**

## About the project

This project looks at how machine learning models classify images of hands showing rock, paper, or scissors. We wanted to use a problem that's easy to recognize but still lets us explore some of the methods covered in ENGR 418.

We're comparing models and input features to see which changes actually make a difference. Instead of focusing only on getting the highest accuracy, we're also looking at what happens when we reduce the number of image features, change training settings, and test the models on images they haven't seen before.

## What we're investigating

- How well does logistic regression classify rock, paper, and scissors from image pixels?
- Can principal component analysis (PCA) reduce the number of features without losing too much accuracy?
- How do the number of training iterations and neural network settings affect performance?
- Do results from the validation images carry over to a separate test set?

## Dataset

The repository contains two image collections, each organized into `rock`, `paper`, and `scissors` folders.

| Collection | Images per class | Total images | How we use it |
| --- | ---: | ---: | --- |
| `Dataset/rps/` | 840 | 2,520 | Training and validation |
| `Dataset/rps-test-set/` | 124 | 372 | Final evaluation |

We use a stratified 80/20 split of the first collection: 2,016 training images and 504 validation images. The separate 372-image collection is reserved for testing.

Images are converted to grayscale, resized to fit a 48 × 48 pixel canvas, and represented as 2,304 pixel features. We fit feature scaling and PCA on the training data, then apply those transformations to the validation and test data.

**Dataset source and license:** We still need to add and verify the original download links and usage terms before submission.

## Approach

Our notebook follows the same overall process for each model:

1. Load and inspect the images, then create the training and validation split.
2. Explore PCA and how much of the variation in the images is retained by different component counts.
3. Use linear regression as a short illustration of why continuous predictions are not a natural fit for class labels.
4. Train multinomial logistic regression as the classification baseline, first on pixels and then on PCA features.
5. Compare training iteration limits and a small feedforward neural network with different settings.
6. Evaluate the selected configurations on the separate test set using accuracy, balanced accuracy, macro F1-score, cross-entropy loss, and a confusion matrix.

The settings are selected using validation results, not the test set.

## Preliminary results

Our first run shows a noticeable gap between validation and test performance. Several configurations reached 100% validation accuracy, but their performance on the separate test images was much lower:

| Model | Test accuracy |
| --- | ---: |
| Logistic regression using pixels | 55.9% |
| Logistic regression using 100 PCA components | 58.6% |
| Selected neural network (learning rate 0.01) | 53.8% |

These results are preliminary. They suggest that doing well on the validation split doesn't necessarily mean the model will work as well on a different image collection. We still need to look more closely at differences between the datasets, potential near-duplicates, and the kinds of images the models misclassify.

## Repository contents

```text
Dataset/
├── rps/                                             # Images used for training and validation
└── rps-test-set/                                    # Separate test images
notebooks/
└── rock_paper_scissors_classification.ipynb         # Shared notebook for our experiments
results/                                             # Saved tables and figures
principal-component-analysis-visualization.ipynb     # External PCA reference notebook
README.md
```

The shared classification notebook contains our experiments and their results. The separate PCA visualization notebook was kept as a reference from another author, not as work created by our team.

## Tools

Python, Jupyter Notebook, NumPy, pandas, Pillow, Matplotlib, and scikit-learn.

## References and acknowledgments

- **Course:** ENGR 418, Applied Machine Learning, UBC Okanagan.
- **PCA visualization reference:** [DrGFreeman's notebook on Kaggle](https://www.kaggle.com/code/drgfreeman/principal-component-analysis-visualization). We retained a copy in the repository for reference and will credit the original author when discussing it.
- **Dataset:** Original source and license details to be added after verification.

## Reproducing the experiments

We'll add the final environment setup, dependencies, and step-by-step instructions after the notebook and experiments have been reviewed and finalized.
