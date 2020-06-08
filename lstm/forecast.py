import numpy as np

def forecast(training_set, model, time_steps, future):
    forecasts = training_set[-time_steps:]
    for _ in range(future):
        x = forecasts[-time_steps:]
        x = x.reshape((1, time_steps, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    forecasts = forecasts[time_steps-1:]
    return forecasts