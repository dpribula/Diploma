import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

import evaluation_helper
import nn_model_tensorflow
import data_helper
import output_writer


def create_data_for_nn(input_answers, input_questions, length):
    batch_size = 10
    num_steps = 499
    answers = np.zeros((batch_size, num_steps), dtype=int)
    questions = np.zeros((batch_size, num_steps), dtype=int)
    seq_len = np.zeros(batch_size, dtype=int)

    seq_len[0] = length

    answers[0] = data_helper.add_padding_array(input_answers, num_steps)
    questions[0] = data_helper.add_padding_array(input_questions, num_steps)


    return questions, answers, seq_len


def run_data_on_nn(model, answer, question, seq_len, number):
    prediction_labels = []
    correct_labels = []


    test_predictions = model.run_model_test(question, answer, seq_len)
    # print(question[0])
    # print(answer[0])
    questions = (evaluation_helper.get_questions(question))
    prediction_labels += (evaluation_helper.get_predictions(test_predictions, questions))
    correct_labels += (evaluation_helper.get_labels(answer, questions))
    # OUTPUT
    output_writer.output_for_map('maps/nn_output/', question, answer, seq_len,
                                 test_predictions, number)
    #return questions, prediction_labels, correct_labels, test_predictions


def get_map_from_data():
    datafile = os.path.expanduser('/home/dave/projects/diploma/data/place_code2.csv')

    datafile2 = os.path.expanduser(
        '/home/dave/projects/diploma/maps/data/API_IT.NET.USER.ZS_DS2_en_csv_v2_10082442.csv')
    nn_file = os.path.expanduser('/home/dave/projects/diploma/maps/nn_output/something2.csv')
    shapefile = os.path.expanduser('/home/dave/projects/diploma/maps/data/ne_10m_admin_0_countries_lakes.shp')

    cols = ['id', 'name', 'country_code']

    colors = 9
    cmap = 'Blues'
    figsize = (16, 10)
    year = '2016'
    cols2 = ['Country Name', 'Country Code']


    gdf = gpd.read_file(shapefile)[['ADM0_A3', 'geometry']].to_crs('+proj=robin')
    gdf.sample(5)

    df = pd.read_csv(datafile, skiprows=0, usecols=cols, delimiter=',')
    df.sample(5)

    df2 = pd.read_csv(datafile2, skiprows=4, usecols=cols2)
    df2.sample(5)

    nn = pd.read_csv(nn_file, skiprows=0, delimiter=';')
    nn.sample(5)

    merged = nn.merge(df, left_on='id', right_on='id')
    merged.sample(5)

    merged2 = merged.merge(df2, left_on='country_code', right_on='Country Code')
    merged2.sample(5)

    merged3 = gdf.merge(merged2, left_on='ADM0_A3', right_on='Country Code')
    merged3.describe()

    prediction = "prediction"
    ax = merged3.dropna().plot(column=prediction, cmap='OrRd', figsize=figsize, scheme='equal_interval', k=colors,
                               legend=True)
    plt.rcParams['axes.facecolor'] = 'blue'
    ax.set_axis_off()
    ax.set_xlim([-1.5e7, 1.7e7])
    ax.get_legend().set_bbox_to_anchor((.12, .4))
    ax.get_figure()


