import dash
import base64
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from PIL import Image
# from app import app
# from app import server
from io import BytesIO as _BytesIO

app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                assets_folder='assets/')
server = app.server

app.layout = html.Div([html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-image-upload')
], style={
    'width': '50%'
}),
    html.Div(id='results',
             style={
                 'width': '50%'
             })
], style={'display': 'flex'})


def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)
    rgb = Image.new('RGB', im.size)
    rgb.paste(im)
    image = rgb
    test_image = image.resize((72, 72))

    return test_image


def b64_to_numpy(string, to_scalar=True):
    im = b64_to_pil(string)
    np_array = np.asarray(im)

    if to_scalar:
        np_array = np_array / 255.

    return np_array


def parse_contents(contents, filename):
    return html.Div([
        # html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,
                 style={'width': '100%'})
    ], style={'width': '100%',
              'height': 'fit-content',
              'padding': '28px',
              'box-sizing': 'border-box'})


def parse_image(contents):
    classes_ = ['apple, apple scab',
                'apple, black rot',
                'apple, cedar apple rust',
                'apple, healthy',
                'banana, healthy',
                'banana, segatoka',
                'banana, xamthomonas',
                'cherry, (including sour) healthy',
                'cherry, (including sour) powdery mildew',
                'corn, (maize) cercospora leaf spot gray leaf spot',
                'corn, (maize) common rust',
                'corn, (maize) healthy',
                'corn, (maize) northern leaf blight',
                'grape, black rot',
                'grape, esca (black measles)',
                'grape, healthy',
                'grape, leaf blight (isariopsis leaf spot)',
                'orange, haunglongbing (citrus greening)',
                'bell pepper, bacterial spot',
                'bell pepper, healthy',
                'potato, early blight',
                'potato, healthy',
                'potato, late blight',
                'rice, brownspot',
                'rice, healthy',
                'rice, hispa',
                'rice, leafblast',
                'strawberry, healthy',
                'strawberry, leaf scorch',
                'tea, leaf blight',
                'tea, red leaf spot',
                'tea, red scab',
                'tomato, bacterial spot',
                'tomato, early blight',
                'tomato, healthy',
                'tomato, late blight',
                'tomato, leaf mold',
                'tomato, septoria leaf spot',
                'tomato, spider mites two-spotted spider mite',
                'tomato, target spot',
                'tomato, tomato mosaic virus',
                'tomato, tomato yellow leaf curl']
    # model = tf.keras.models.load_model('assets/best_net_82.hdf5')

    # change to use tensorflow lite model instead of full tensorflow model
    tflite_model_path = 'assets/DeepWideNet_BE_WND.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    content_type, content_string = contents.split(",")
    image = b64_to_numpy(content_string)
    image = image.reshape((1, 72, 72, 3))
    image = image.astype('float32')

    # make predictions
    input_dtl = interpreter.get_input_details()
    output_dtl = interpreter.get_output_details()

    interpreter.set_tensor(input_dtl[0]['index'], image)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_dtl[0]['index'])

    p = zip(list(classes_), list(pred[0]))
    p = sorted(list(p), key=lambda z: z[1], reverse=True)[:20]
    temp = pd.DataFrame(data=p, columns=['label', 'prob'])
    temp['text'] = [f'{round(t * 100, 2)}%' for t in temp.prob]

    bar = go.Figure(data=[go.Bar(x=temp.prob,
                                 y=temp.label,
                                 text=temp.text,
                                 orientation='h')])
    bar.update_layout(hovermode=False,
                      paper_bgcolor='#fff',
                      plot_bgcolor='#fff',
                      height=800,
                      margin_pad=10,
                      xaxis=dict(showline=False,
                                 showgrid=False,
                                 showticklabels=False),
                      yaxis=dict(showline=False,
                                 showgrid=False)
                      )

    return html.Div([html.H2('ILDD app'),
                     html.P('for detections upload a image to agent which is plant type'
                            'among Apple, Cassava, Cherry, Corn, Grape, Orange, Pepper,'
                            'Potato, Rice, Strawberry, Tomato.'),
                     dcc.Graph(id='bar',
                               figure=bar,
                               config={
                                   'displayModeBar': False
                               },
                               style={'position': 'static',
                                      'padding': '0px 30px'}
                               )
                     ])


@app.callback([Output('output-image-upload', 'children'), Output('results', 'children')],
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        return [[parse_contents(list_of_contents, list_of_names)], [parse_image(list_of_contents)]]


if __name__ == '__main__':
    app.run_server(debug=False)
