# Image Retrieval for Visual Geolocalization: Extensions and Experiments

Official repository for the Machine learning and Deep learning project "Image Retrieval for Visual Geolocalization: Extensions and Experiments".

* The project is also available on [[Kaggle](https://www.kaggle.com/code/giovannimonco22/image-retrieval-for-visual-geolocalization)] with all the datasets loaded.
* Training, Validation and Test can be run from `main.py`.
* Relative report is [here](report.pdf).

## Repository Structure

### Datasets
The `datasets` folder contains the following scripts:
- `Train.py`: Class definition for the Train 
- `Test.py`: Class definition for the Validation and Test. 

### Models
The `models` folder includes the model architecture for the VPR pipeline.
- `Aggregators.py`: Script that contains the aggregation layers.
- `Backbone.py`: Script that contains a modified backbone of ResNet18.
- `Evaluation_loop.py`: Script that defines the Recall@K and the Evaluation loop.
- `Training_loop.py`: Script that defines the Training loop.

### Visualizations
The `visualization` folder contains the file `Visualization.py` that defines functions to plot images and predictions.

## Authors
The authors of the project are:
- Francesco Gaza s315489@studenti.polito.it 
- Giovanni Monco s315001@studenti.polito.it 
- Erika Spada s318375@studenti.polito.it 
