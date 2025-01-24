# Short Term Temperature Predictor Using k-NN (k-Nearest Neighbor) Machine Learning Model

## Purpose

The `ML_Temp_Predict.ipynb` notebook creates a short-term temperature predictor using the k-Nearest Neighbor (k-NN) machine learning algorithm.

The script in the [Jupyter](https://jupyter.org/install) notebook will:

- Fetch historical weather data for a given location from a [WeatherAPI](https://www.weatherapi.com/)
- Preprocess the data
- Save weather data to a .CSV file
- Create and train a [k-NN ML model](https://www.geeksforgeeks.org/k-nearest-neighbours/) using your downloaded data
- Predict temperature for a future date entered in the terminal by user
 	- In practice, model accuracy will be between 2-6 degrees Fahrenheit ([RMSE](https://help.sap.com/docs/SAP_PREDICTIVE_ANALYTICS/41d1a6d4e7574e32b815f1cc87c00f42/5e5198fd4afe4ae5b48fefe0d3161810.html) = 2.00 - 5.79) based on the quantity and quality of weather data you provide for training the model.
  		- Predicted accuracy is directly proportional to the amount of historical data you provide.
  		- Providing only a weeks worth of data will net terrible performance if you ask the model to predict temperatures a more than a day or two out from the last data point in the model.
  		- At least 6 months of valid historical weather data from your chosen weather API are recommended

## Platform Prerequisites

- [Python](https://www.python.org/downloads/) 3.x or later (Developed with Python [3.13.1](https://www.python.org/downloads/release/python-3131/))
- [Jupyter](https://jupyter.org/install) notebook
- [pip](https://pypi.org/project/pip/#description)
- Required Python libraries: [pandas](https://pandas.pydata.org/docs/getting_started/install.html), [numpy](https://numpy.org/install/), [sklearn](https://scikit-learn.org/stable/install.html), [requests](https://pypi.org/project/requests/), [csv](https://docs.python.org/3/library/csv.html), [datetime](https://docs.python.org/3/library/datetime.html)

## Weather Dataset Prerequisites

- If you just want to try out the code, you can use my preexisting `weather_data.csv` file included in the repository.
- If you want to get your own weather data, you will need to have your own personal API key to access either the [WeatherAPI.com](https://www.weatherapi.com/signup.aspx) or [NOAA](https://www.ncdc.noaa.gov/cdo-web/token) datasets. Do *NOT* share your API key with anyone else or post it publicly.
- You must edit the code blocks to insert your own API key (Do not share this with others!) in order to retrieve your own set of weather data.

## Installation

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to save the script.
3. Download the script `ML_Temp_Predict.ipynb` either manually or by using the [git](https://github.com/git-guides/install-git) command:
 - ```git clone https://github.com/DWNewton/KNN_Temperature_Predictor```
 - Navigate to the cloned directory: `cd KNN_Temperature_Predictor`
 - (Optional)
   If you want to run the script in a virtual environment, create a new one and activate it:

  `python -m venv .venv && source`

  `source venv/bin/activate` [macOS/Linux]

  `venv\Scripts\activate` [Windows]

  NOTE: To uninstall the script, deactivate the virtual environment and remove it:

  `deactivate && rm -rf venv` [macOS/Linux]
   
  `venv\Scripts\deactivate` [Windows]

 - Install the required libraries:
  ```pip install pandas numpy sklearn requests```
  If you encounter any issues while installing libraries, try updating your [pip](https://pypi.org/project/pip/#description) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) versions.
  
  `csv`, and `datetime` are built-in Python libraries and should not need to be installed separately.

  (Optional)
   To run the script in a [Docker](https://www.docker.com/) container:

 - build the image: ```docker build -t knn_temperature_predictor.```
    
 - run the container: ```docker run -it knn_temperature_predictor```
