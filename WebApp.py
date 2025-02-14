import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Medicinal Herb Recognition System"
app.config["suppress_callback_exceptions"] = True

def load_model():
    model_path = "Add_Your_Model_Path"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    return tf.keras.models.load_model(model_path)

def model_prediction(model, test_image):
    image = Image.open(test_image) 
    image = image.resize((224, 224))  
    input_arr = np.array(image) / 255.0 
    input_arr = np.expand_dims(input_arr, axis=0) 
    predictions = model.predict(input_arr) 
    return np.argmax(predictions)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1(
                            "Medicinal Herb Recognition System 🌿",
                            className="text-center text-white mt-4",
                        ),
                        html.P(
                            "Identify and Learn About Medicinal Herbs",
                            className="lead text-center text-white mb-5",
                        ),
                    ],
                    className="p-5 text-center",
                    style={
                        "backgroundImage": "url('/assets/1141106.jpg')",
                          "backgroundSize": "cover",
                          "backgroundPosition": "center", 
                          "borderRadius": "20px",
                    },
                ),
                width=12,
            )
        ),
        
        dbc.Tabs(
            [
                dbc.Tab(label="Home", tab_id="home"),
                dbc.Tab(label="About", tab_id="about"),
                dbc.Tab(label="Medicinal Herb Recognition", tab_id="recognition"),
            ],
            id="tabs",
            active_tab="home",
        ),
        
        html.Div(id="tab-content", className="mt-4"),
    ],
    fluid=True,
)
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    if active_tab == "home":
        return html.Div(
            [
                html.Img(
                    src="/assets/home-image.jpg",
                    style={"width": "100%", "borderRadius": "10px"},
                ),
                html.Br(),
                dcc.Markdown(
                    """
                    Welcome to the Medicinal Herbs Recognition System! 🌿🔍  
                    Our mission is to help identify medicinal herbs efficiently. Upload an image of a herb and explore natural remedies.

                    ### How It Works
                    - **Upload Image:** Navigate to the 'Medicinal Herb Recognition' page and upload a plant image.  
                    - **Analysis:** The system will analyze the image and identify the plant using advanced algorithms.  
                    - **Results:** View results and recommendations for further action.  

                    ### Why Choose Us?
                    - **Accuracy:** Cutting-edge machine learning for precise herb identification.  
                    - **User-Friendly:** Seamless and intuitive interface.  
                    - **Fast Results:** Get predictions in seconds.
                    """
                ),
            ]
        )
    elif active_tab == "about":
        return html.Div(
            [
                html.H2("About Us", className="text-center text-primary"),
                dcc.Markdown(
                    """
                    ### Faculty of Engineering and Technology (FEAT)
                    #### AIML 3rd Year  
                    **Team Members:**  
                    - Nikhil Warkad (DMET1322015) AIML  
                    - **Mentor:** Dr. Swapnil Gundewar
                    """
                ),
            ]
        )
    elif active_tab == "recognition":
        return html.Div(
            [
                dcc.Upload(
                    id="upload-image",
                    children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
                    style={
                        "width": "100%",
                        "height": "80px",
                        "lineHeight": "30px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "backgroundColor": "#f8f9fa",
                    },
                    multiple=False,
                ),
                html.Div(id="output-image", className="mt-3"),
                html.Button(
                    "Predict", id="predict-button", className="btn btn-primary mt-3"
                ),
                html.Div(id="prediction-result", className="mt-3"),
            ]
        )
    return html.Div("Tab not found.")

@app.callback(
    [Output("output-image", "children"), Output("prediction-result", "children")],
    [Input("predict-button", "n_clicks")],
    [State("upload-image", "contents"), State("upload-image", "filename")],
)
def update_output(n_clicks, contents, filename):
    if contents is None:
        return None, ""

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        test_image = io.BytesIO(decoded)
        display_image = html.Img(
            src=contents, style={"width": "20%", "margin": "10px 0", "borderRadius": "10px"}
        )
        if n_clicks:
            model = load_model()
            result_index = model_prediction(model, test_image)
            class_names = [ "Arive_Dantu", "Besale", "betel", "Crape_Jasmine", "Curry"]
            if result_index < len(class_names):
                prediction_result = html.H5(
                    f"The model predicts this is a {class_names[result_index]}.",
                    className="text-success",
                )
            else:
                prediction_result = html.H5(
                    " Prediction index is out of range. Please check the class names.",
                    className="text-danger",
                )

            return display_image, prediction_result

        return display_image, ""

    except Exception as e:
        return None, html.H5(f"Error: {str(e)}", className="text-danger")

if __name__ == "__main__":
    app.run_server(debug=True)
