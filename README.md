# ML_Cropinator
A Machine Learning Approach of Farming, to find the right crop for any Climatic conditions.

This project aims in designing a system which is capable of giving an optimized crop recommendation based on
various physical parameters like temperature, soilmoisture, humidity etc. The input parameters are real-time
data which would enable us to give a better prediction and more feasibility. The machine learning algorithm gives the
prediction based on the small-scale real-time dataset.

Electronic Components used : 
⬡ Capacitive Soil Moisture Sensor
⬡ DHT22
⬡ Nodemcu
⬡ Raspberry pi
⬡ Lcd 16*2

The "ML Cropinator" uses 2 sensors , one capacitive soil moisture sensor and a
temperature sensor which is fed into the ML model. The model which is pre-trained into predicting what crop can be
grown in a given soil (with corresponding temperature and moisture values) takes real-time readings through the
sensors and shows the most favourable crop's name on a LCD display. The model uses K-NN classifier (K-nearest neighbour
classifier). It is wirelessly loaded into Raspberry-Pi through NodeMCU microcontroller. 

Machine Learning Model
⬡ We chose and implemented a machine learning classifier (in python) which decides which crop will show optimum growth on
any given type of soil.
⬡ Our model decides using 2 parameters: Optimum rainfall and favourable average temperature required

K-Nearest Neighbor Algorithm
➢Our ML model uses the K-NN classifier algorithm based on Supervised Learning technique.
➢After plotting the datapoints from the training dataset on a 2-D feature plane , the new datapoint is assigned a category , depending
on “K” nearest neighbours of that point.

MAIN FEATURE
• By taking the real time values from the sensors, and running them in our K-NN
model , we made actual predictions of which crop would be best suitable for optimum
growth !

ACCURACY
The model was trained and tested using KNN with value of k=3. The algorithm checked the 3 nearest Euclidian distance between the
3 pairs of 2 points.
• The accuracy on training set was 88%.
• The accuracy on test set was 92%.
