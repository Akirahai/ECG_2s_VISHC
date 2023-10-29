# Light weight deep-learning model for ECG classification


Cardiovascular diseases (CVDs), the main cause of mortality worldwide, are being
detected in an increasing number of people nowadays. Electrocardiograms (ECGs) are
the gold standard for detecting certain cardiac issues. The majority of recent research
and clinical practice both make extensive use of the typical 12-lead ECG. In this study,
we build a lightweight model for six common heart diseases based on 12-lead ECGs
using a ResNet1D-18 modified model.

## Table of Contents

- [Installation](#installation)
- [Data source](#datasource)
- [Contributing](#contributing)
## Installation

To get started with the project, you can follow these steps:

To use this project, you'll need to clone the repository. You can do this using Git:

```bash
git clone https://github.com/Akirahai/ECG_VISHC_Project.git
```

Next, make sure to install the necessary dependencies. You can create a Python virtual environment and install the requirements:

```bash
cd ECG_VISHC_Project
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Data Source

The data for this project was obtained from [The China Physiological Signal Challenge 2018](http://2018.icbeb.org/Challenge.html). Please refer to their website for information about the dataset and any usage guidelines.

## Contributing

In this study, we have designed and constructed a classification model utilizing a modified ResNet-18 architecture, which has demonstrated superior accuracy while maintaining a lightweight profile. This model has the capability to effectively differentiate between six distinct heart-related diseases. Moreover, we have developed a mobile application that facilitates the classification of these diseases directly on smartphones. These forward-looking products hold significant promise in supporting medical professionals throughout the diagnostic process.

Additionally, we conducted a comprehensive review of alternative models, including AlexNet, SqueezeNet, and Se_ResNet, to assess their performance on the dataset.

