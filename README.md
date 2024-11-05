# Fraud Detection Project

This repository contains a ML pipeline designed for fraud detection, 
that includes principal steps of model lifecycle management. It begins with training
an anomaly detection model using Isolation Forest, deploys the model through 
a dockerized Flask API to handle real-time predictions, and implements model 
performance monitoring. The project also includes CI/CD automation with 
with github actions, integrating with Google Cloud Platform that is used to store the data.


## Functional features in this project
- **Model Training**: Train a model for fraud detection.
- **Deployment**: Deploy the model via dockerized Flask API.
- **Monitoring**: Monitor model performance and trigger retraining if the preformance degrades.

## Repository Structure
- `.github/workflows/`: GitHub actions workflows for CI/CD and MLOps
- `experimental/`: Contains script to test the prediction flask app
- `src/`: Contains Python scripts for data processing, training, and monitoring.
- `tests/`: Contains scripts for unit testing.
- `Dockerfile`: Docker file for deploying the model API.
- `configuration.ini`: Configuration for model parameters and file paths.
- `requirements.txt`: Requirements file.

## Setup and Installation

### Install Dependencies
Clone the repository and install dependencies:
```bash
git clone https://github.com/mahmoudalbardan/test.git
cd test
pip install -r requirements.txt
```
## Launching and deployment
###  Launch and test the app on your Local Machine
To train the model locally, use the data already stored in Google Cloud Storage
by following these steps:
1. Create a service account in your GCP project and download its key (in JSON format).
2. Copy the key to the `test/` directory.
3. Open a terminal and run:
```bash
 export GOOGLE_APPLICATION_CREDENTIALS=<your-key-name>
 ```
4. Run the training script:
```bash
python src/scripts/train_model.py --configuration configuration.ini --retrain false
```
The model will be automaticaly saved in: `src/model/fraud_detection_model.pkl`

5. Build docker image:
```bash
docker build -t fraudapp .
```
6.  Run docker container:
```bash
docker run -p 5000:5000 fraudapp
```
7. Test the app by running the following script:
```bash
python experimental/testing_the_app.py
```

###  MLOps with GitHub actions
The `.github/workflows` directory in the project repository contains 4 workflow `.yml` files:
1. `build_and_test.yml`: it automates the process of checking out the code, setting up the python
environment, installing dependencies, analyzing code quality with Pylint, and running unit tests whenever changes 
are pushed to the main branch or when triggered manually.
2. `train.yml`: it automates the process of training a machine learning model whenever 
changes are pushed to the main branch or when manually triggered. It consists of several important steps: 
decoding a Google Cloud Platform service account key from a secret and writing it to a JSON file that is used by 
`src/scripts/train_model.py` in order to access google cloud storage and read the raw data from it.
It executes the model training script with the following arguments (`--configuration configuration.ini`: configuration file and  `--retrain false`: to specify if it
is a first time training. Finally it uploads the trained model as an artifact. 
3. `deploy.yml`: it automates the deployment of the machine learning that is built in `train.yml`
model **after the completion** of the `train.yml` workflow or when manually triggered.
it consists of several important steps: 
building and tagging a docker image for the application using `Dockerfile`, and pushing the built image to my docker hub personal account. 
Finally, the workflow deploys the application to a google cloud compute engine by pulling the 
docker image from Docker Hub and running it.
4. `monitor.yml`: this workflow is designed to monitor the performance of the machine learning model on a weekly basis (each sunday at midnight)
or when manually triggered. The workflow includes several important steps:
decoding a Google Cloud Platform service account key from a secret and writing it to a JSON file that is used by
`src/scripts/monitor.py` in order to access google cloud storage and read the **updated** data from it.
It executes the model monitoring script with the following arguments (`--configuration configuration.ini`: configuration file 
and  `--retrain true`: to specify if is a retrain due to new data updates and model preformance decrease)
the output of the monitoring script is read from a file `src/monitoring_output/monitor.txt`,
and the status is stored in GitHub env variable. 
Based on the monitoring results, the workflow will trigger the model retraining process if the variable is **retrain new model** otherwise,
it confirms that the model's performance is still good and no retraining is required.
