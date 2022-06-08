# loading the library
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st

# setting the basic configuration of the web app. This is shown in the Tab
st.set_page_config(page_title = "Price and Sales relationship" 
                    ,page_icon = ":bar_chart:" 
                    ,layout = "wide")

# page title 
st.title( "Price and Sales relationship :bar_chart:")

st.markdown('-----')
st.write(
"""
In an organisation, units sold is often inversely linked to price. As the price increases, the units sold will drop. If we wish to predict units sold using price, we have to come up with a mathematical relationship.

If we use a **linear regression** to find out this relationship, the mathematical formulae will mimick the line equation of the form $y= mx+c$, where $m$ is the **slope** and $c$ is a **constant** value.

So the relationship between price and units sold will look like :    
                       $Units = coefficient * price + constant$

We will explore this relationship in a simple app. You can manipulate the value of weights and constants and see how the values in the equation change along with the change in the chart
"""
)

st.markdown('-----')


# Taken from Andrew NG ML course
# function to compute the line
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w : weight:  weights of the model 
      b: constant: starting point of the model 
      
      Returns
      y (ndarray (m,)): target values
    """
    x = np.array(x)
    
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

# data 
price = [8.75,8.99,7.5,7.25,7.4,8.5,8.4,7.9,7.25,8.7,8.4,8.1,8.4,7.4,8,8.3,8.1,8.2,8.99,7.99,8.5,7.9,7.99,8.25]
units_sold = [73959,71544,78587,80364,78771,71986,74885,73345,76659,71880,73598,74893,69003,78542,72543,74247,76253,72582,69022,76200,69701,77005,70987,75643]


# sliders
st.sidebar.header("Please use the slider to change the values")

# variables
weights = st.sidebar.slider('Set the  coefficient of the price', min_value = -15000, max_value = 0, step = 1000, value = -3000)
constant = st.sidebar.slider('Set the constant value', min_value = 60000, max_value = 100000, step = 1500, value = 100000)

# predictions
predicted_units = compute_model_output(price, weights, constant)

# equation
st.write('The equation is: $Units$ = ', weights, '$ * \,price\, + $', constant)


## Visualizations

scatter_plot = px.scatter(x = price, y = units_sold)

line_plot = px.line(x = price, y = predicted_units)

combined_plot = go.Figure(data = scatter_plot.data + line_plot.data)

combined_plot.update_layout(title_text = "Prices vs. Units Sold")
combined_plot.update_xaxes(title_text ="Price")
combined_plot.update_yaxes(title_text="Units Sold")

# color of the background and grid
# combined_plot.update_layout(
#      plot_bgcolor = "rgba(0,0,0,0)"
#     ,xaxis = (dict(showgrid = False))
# )

# show the plot
st.plotly_chart(combined_plot, use_container_width = True)


## THINGS TO UPDATE

# Change the color of the scatter plot
# Change the color of the line plot
# Add more info
# Maybe make a 3-d chart with advertisment and promo values
# add labels and legend to the chart

