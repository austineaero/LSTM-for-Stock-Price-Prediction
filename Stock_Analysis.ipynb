{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import tensorflow as tf\n",
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<module 'tensorflow._api.v2.version' from 'C:\\\\Users\\\\Austin\\\\anaconda3\\\\envs\\\\deep_learning\\\\lib\\\\site-packages\\\\tensorflow\\\\_api\\\\v2\\\\version\\\\__init__.py'>"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tf.version"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date_time</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2/22/2016 10:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.000780</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/22/2016 11:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.000497</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/22/2016 12:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.000290</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/22/2016 14:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001711</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.005131</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/22/2016 15:00</th>\n",
       "      <td>0.001763</td>\n",
       "      <td>0.001711</td>\n",
       "      <td>0.001745</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001335</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 12:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.002884</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 14:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.001814</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 15:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001719</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.002310</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 16:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001693</td>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.007861</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 17:00</th>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.001685</td>\n",
       "      <td>0.001693</td>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.004217</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6503 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                      Open      High       Low     Close    Volume\n",
       "date_time                                                         \n",
       "2/22/2016 10:00   0.001738  0.001685  0.001719  0.001712  0.000780\n",
       "2/22/2016 11:00   0.001738  0.001685  0.001719  0.001738  0.000497\n",
       "2/22/2016 12:00   0.001738  0.001685  0.001719  0.001738  0.000290\n",
       "2/22/2016 14:00   0.001738  0.001711  0.001719  0.001738  0.005131\n",
       "2/22/2016 15:00   0.001763  0.001711  0.001745  0.001738  0.001335\n",
       "...                    ...       ...       ...       ...       ...\n",
       "11/29/2019 12:00  0.001738  0.001685  0.001719  0.001738  0.002884\n",
       "11/29/2019 14:00  0.001738  0.001685  0.001719  0.001712  0.001814\n",
       "11/29/2019 15:00  0.001738  0.001685  0.001719  0.001738  0.002310\n",
       "11/29/2019 16:00  0.001738  0.001685  0.001693  0.001712  0.007861\n",
       "11/29/2019 17:00  0.001712  0.001685  0.001693  0.001738  0.004217\n",
       "\n",
       "[6503 rows x 5 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the stock quote\n",
    "df = pd.read_csv('C:/Users/Austin/Documents/Mustafa_Aydemir/Stock.csv') # Change this directory to yours.\n",
    "# Set date_time as index\n",
    "df = df.set_index('date_time')\n",
    "# Show the data\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6503, 5)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the number of rows and columns in the data set\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x648 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (18,9))\n",
    "plt.plot(range(df.shape[0]),(df['Close']))\n",
    "#plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)\n",
    "plt.xlabel('Date_Time',fontsize=18)\n",
    "plt.ylabel('Close Price',fontsize=18)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5202"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Create a new dataframe with only the 'Close' column\n",
    "data = df.filter(['Close'])\n",
    "# Convert the dataframe to a numpy array\n",
    "dataset = data.values\n",
    "# Get the number of rows to train the model on\n",
    "training_data_len = round(len(dataset) * .8)\n",
    "training_data_len"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.49397603],\n",
       "       [0.50602397],\n",
       "       [0.50602397],\n",
       "       ...,\n",
       "       [0.50602397],\n",
       "       [0.49397603],\n",
       "       [0.50602397]])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Scale the data\n",
    "scaler = MinMaxScaler(feature_range=(0,1))\n",
    "scaled_data = scaler.fit_transform(dataset)\n",
    "\n",
    "scaled_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create the training data set\n",
    "# Create the scaled training data set\n",
    "train_data = scaled_data[0:training_data_len, :]\n",
    "# Split the data into x_train and y_train data sets\n",
    "x_train = []\n",
    "y_train = []\n",
    "\n",
    "for i in range(60, len(train_data)):\n",
    "    x_train.append(train_data[i-60:i, 0])\n",
    "    y_train.append(train_data[i, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5142, 60)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Convert the x_train and y_train to numpy arrays\n",
    "x_train, y_train = np.array(x_train), np.array(y_train)\n",
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(5142, 60, 1)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Reshape the data since LSTM expects 3D\n",
    "x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))\n",
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build the LSTM model\n",
    "import keras\n",
    "from keras.models import Sequential \n",
    "from keras.layers import Dense, LSTM\n",
    "model = Sequential()\n",
    "model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))\n",
    "model.add(LSTM(50, return_sequences=False))\n",
    "model.add(Dense(25))\n",
    "model.add(Dense(1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "lstm (LSTM)                  (None, 60, 50)            10400     \n",
      "_________________________________________________________________\n",
      "lstm_1 (LSTM)                (None, 50)                20200     \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 25)                1275      \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 1)                 26        \n",
      "=================================================================\n",
      "Total params: 31,901\n",
      "Trainable params: 31,901\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5142/5142 [==============================] - 189s 36ms/step - loss: 0.0014 - accuracy: 3.8323e-04\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.keras.callbacks.History at 0x20ef6f165f8>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the model\n",
    "model.fit(x_train, y_train, batch_size=1, epochs=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create the testing data set\n",
    "# Create a new array containing scaled values from index 5142 to 6503\n",
    "test_data = scaled_data[training_data_len-60:, :]\n",
    "# Create the data sets x_test and y_test\n",
    "x_test = []\n",
    "y_test = dataset[training_data_len:, :]\n",
    "for i in range(60, len(test_data)):\n",
    "    x_test.append(test_data[i-60:i, 0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert the data to a numpy array\n",
    "x_test = np.array(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reshape the data\n",
    "x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the model predicted price values\n",
    "predictions = model.predict(x_test)\n",
    "predictions = scaler.inverse_transform(predictions) # unscaling the values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.865268141790341e-06"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Get the root mean squared error (RMSE)\n",
    "rmse = np.sqrt(np.mean(predictions - y_test)**2)\n",
    "rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Austin\\anaconda3\\envs\\deep_learning\\lib\\site-packages\\ipykernel_launcher.py:4: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  after removing the cwd from sys.path.\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA88AAAH4CAYAAAB0a9//AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAACuCElEQVR4nOzdd5gb1dUG8PeqbS9e7667ve7GuOMCppluY0roEDoBQkklEEzoEEoIfLRQAqEHAoQSigEDptiADe4Vd6+9625vrypzvz+kkUbSqEur9v6ex4+labrSrmbnzLn3XCGlBBEREREREREFZkh2A4iIiIiIiIhSHYNnIiIiIiIiohAYPBMRERERERGFwOCZiIiIiIiIKAQGz0REREREREQhMHgmIiIiIiIiCoHBMxEREekSQlQJIaQQwhTGtpcJIb7rinYRERElA4NnIiKiDCGEqBZCWIUQ5T7Ll7uC4KokNY2IiCjtMXgmIiLKLFsBXKA+EUKMBpCXvOYQERFlBgbPREREmeU1AJdonl8K4FX1iRCiRAjxqhBinxBimxDiNiGEwbXOKIR4WAixXwixBcBM7YFd+74ghNglhNghhPirEMLYFW+KiIgo2Rg8ExERZZaFAIqFEAe5AtvzAPxbs/5JACUABgE4Gs5A+3LXuqsAnAJgPICJAM72OfYrAOwAhri2ORHAlYl5G0RERKmFwTMREVHmUbPPJwBYB2CHa7kaTN8ipWyWUlYDeATAxa715wJ4TEpZI6WsA/CAekAhRA8AMwD8QUrZKqXcC+BRAOd3wfshIiJKupDVM4mIiCjtvAZgHoCB0HTZBlAOwAJgm2bZNgB9XI97A6jxWacaAMAMYJcQQl1m8NmeiIgoYzF4JiIiyjBSym1CiK0ATgbwK82q/QBscAbCa13L+sOTmd4FoJ9m+/6axzUAOgGUSyntiWg3ERFRKmO3bSIiosz0KwDHSilbNcscAN4GcJ8QokgIMQDADfCMiX4bwO+EEH2FEN0AzFJ3lFLuAvA5gEeEEMVCCIMQYrAQ4ugueTdERERJxuCZiIgoA0kpN0spF+us+i2AVgBbAHwH4A0AL7rWPQ9gDoAVAJYCeM9n30vg7Pa9FkA9gHcA9Ip744mIiFKQkFImuw1EREREREREKY2ZZyIiIiIiIqIQGDwTERERERERhcDgmYiIiIiIiCgEBs9EREREREREITB4JiIiIiIiIgrBlOwGpLLy8nJZVVWV7GYQERERERFRAixZsmS/lLIinG0ZPAdRVVWFxYv1psgkIiIiIiKidCeE2Bbutuy2TURERERERBQCg2ciIiIiIiKiEBg8ExEREREREYXA4JmIiIiIiIgoBAbPRERERERERCEweCYiIiIiIiIKgcEzERERERERUQgMnomIiIiIiIhCYPBMREREREREFAKDZyIiIiIiIqIQGDwTERERERERhcDgmYiIiIiIiCgEBs9EREREREREITB4JiIiIiIiIgqBwTMRERERERFRCAyeiYiIiIiIiEJg8EwJ1WFzYMHmA5BSJrspREREREREUWPwTAn1r/lbcMHzC/Hj1rpkN4WIiIiIiChqDJ4poX7e3QwA2NfcmeSWEBERERERRY/BMyWU2SAAAHZFSXJLiIiIiIiIosfgmRLKaHD+itkcHPNMRERERETpi8EzJZTJlXl+Yu5G7GpsT3JriIiIiIiIosPgmRKqe6EFAFBb346L/vVjkltDREREREQUHQbPlFBFuWb3471NLBpGRERERETpicEzdRmOeiYiIiIionTF4JkSSpEMmYmIiIiIKP0xeKaEkprguaXTnsSWEBERERERRY/BMyWUokk8D6ooSF5DiIiIiIiIYsDgmRJK2227R1FuEltCREREREQUPQbPlFCKJvXs4PhnIiIiIiJKUwyeKaG0AbNDYfBMRERERETpicEzxWRvUweqZs3Gp6t26a7/cMVO92MGz0RERERElK4YPFNM1u9pBgD8+8dtuuvLCnIAAD2Kcxg8ExERERFR2jIluwGU3sxG5/0Xm10/MJZSYtrwCpgMAjsbOrqyaURERERERHHDzDPFRA2erQ5Fd70iJYxCwGgQXpW3iYiIiIiI0gmDZ4qJxRU817dZ/da1Wx2orW+HcAXPdnbbJiIiIiKiNMXgmWJiMTl/hbYdaPNbd9Wri9HQZoNBAAYhvKatIiIiIiIiSicMnikmBuH8P99i9Fv33ab9AAAhAJNBcJ5nIiIiIiJKWwyeKSbhhMMORcJgELA7GDwTEREREVF6YvBMcREsqWxXnEXDWDCMiIiIiIjSFYNniokaD8sgOeja+nZIALsaOzjumYiIiIiI0hKDZ4pJsKC5f1k+AGDT3ha8s6QWAPDu0touaRcREREREVE8MXimmATriT11cHe/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\n",
      "text/plain": [
       "<Figure size 1152x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot the data\n",
    "train = data[:training_data_len]\n",
    "valid = data[training_data_len:]\n",
    "valid['Predictions'] = predictions\n",
    "\n",
    "# Visualise the data\n",
    "plt.figure(figsize=(16,8))\n",
    "plt.title('Model')\n",
    "plt.xlabel('Date_Time', fontsize=18)\n",
    "plt.ylabel('Close Price', fontsize=18)\n",
    "plt.plot(train['Close'])\n",
    "plt.plot(valid[['Close', 'Predictions']])\n",
    "plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Close</th>\n",
       "      <th>Predictions</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date_time</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2/15/2019 11:00</th>\n",
       "      <td>0.001150</td>\n",
       "      <td>0.001139</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/15/2019 12:00</th>\n",
       "      <td>0.001150</td>\n",
       "      <td>0.001137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/15/2019 14:00</th>\n",
       "      <td>0.001150</td>\n",
       "      <td>0.001139</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/15/2019 15:00</th>\n",
       "      <td>0.001150</td>\n",
       "      <td>0.001141</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2/15/2019 16:00</th>\n",
       "      <td>0.001124</td>\n",
       "      <td>0.001143</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 12:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001756</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 14:00</th>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.001747</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 15:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001731</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 16:00</th>\n",
       "      <td>0.001712</td>\n",
       "      <td>0.001730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11/29/2019 17:00</th>\n",
       "      <td>0.001738</td>\n",
       "      <td>0.001723</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1301 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                     Close  Predictions\n",
       "date_time                              \n",
       "2/15/2019 11:00   0.001150     0.001139\n",
       "2/15/2019 12:00   0.001150     0.001137\n",
       "2/15/2019 14:00   0.001150     0.001139\n",
       "2/15/2019 15:00   0.001150     0.001141\n",
       "2/15/2019 16:00   0.001124     0.001143\n",
       "...                    ...          ...\n",
       "11/29/2019 12:00  0.001738     0.001756\n",
       "11/29/2019 14:00  0.001712     0.001747\n",
       "11/29/2019 15:00  0.001738     0.001731\n",
       "11/29/2019 16:00  0.001712     0.001730\n",
       "11/29/2019 17:00  0.001738     0.001723\n",
       "\n",
       "[1301 rows x 2 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Show the valid and predicted prices\n",
    "valid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
