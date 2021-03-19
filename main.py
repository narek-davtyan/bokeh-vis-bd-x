import pandas as pd
import numpy as np


# Import multiprocessing libraries
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

# Load data
df_orig = pd.read_excel(r'bokeh-vis-bd-x/result_data_x.xlsx', names=['index', 'filter_one', 'filter_two', 'filter_three', 'filter_four', 'recommendation', 'easiness', 'question_one', 'question_two', 'question_three', 'question_four', 'rec_sc', 'eas_sc', \
    'sentiment', 'question_one_filtered_lemmas', 'question_two_filtered_lemmas', 'question_three_filtered_lemmas', 'question_four_filtered_lemmas'])

# df_orig = pd.read_excel(r'result_data_cdd_2.xlsx', names=['company', 'client_name', 'sector', 'country', 'service', 'commercial', 'ca_total', 'cp', 'town', 'recommendation', 'easiness', 'question_one', 'question_one_filtered_lemmas', 'rec_sc', 'eas_sc'])#.astype({'country':'category', 'company':'int16', 'service':'category', 'recommendation':'int8', 'question_one':'string', 'easiness':'int8', 'question_two':'string'})

# Transform filtered lemmas string into list of strings
df_orig['question_one_filtered_lemmas'] = df_orig.loc[~df_orig['question_one_filtered_lemmas'].isna()]['question_one_filtered_lemmas'].parallel_apply(lambda x: x[2:-2].split("', '"))
df_orig['question_two_filtered_lemmas'] = df_orig.loc[~df_orig['question_two_filtered_lemmas'].isna()]['question_two_filtered_lemmas'].parallel_apply(lambda x: x[2:-2].split("', '"))
df_orig['question_three_filtered_lemmas'] = df_orig.loc[~df_orig['question_three_filtered_lemmas'].isna()]['question_three_filtered_lemmas'].parallel_apply(lambda x: x[2:-2].split("', '"))
df_orig['question_four_filtered_lemmas'] = df_orig.loc[~df_orig['question_four_filtered_lemmas'].isna()]['question_four_filtered_lemmas'].parallel_apply(lambda x: x[2:-2].split("', '"))

# Create dictionary of all plots, filter lock, filters
general_dict = {}
# # Calculate all filters
# selected_country_list = np.concatenate((df_orig.country.unique(),df_orig.service.unique()))

def calculate_barycenter(df_temp, country_list):   
    # Create visual data points
    df_tempo = df_temp[['recommendation', 'easiness', 'filter_three']].groupby(['recommendation', 'easiness'],as_index=False).count().rename(columns={'filter_three' : 'sum'}).astype({'sum': 'float32'})
    
    # df_tempy = df_tempo
    df_tempy = pd.merge(df_temp, df_tempo, how='left', on=['recommendation', 'easiness'])
    # df_tempy = pd.merge(df_orig, df_tempo, how='left', on=['recommendation', 'easiness'])

    # Calculate size of circles
    df_tempy.loc[df_tempy['sum'] > 25.0, 'sum'] = 25.0
    df_tempy['visual_sum'] = df_tempy['sum']# * 2.95

    # Create visual barycenter with edges
    if len(df_temp) == 0 or len(country_list) == 0:
        barycenter = np.array([0.0, 0.0])
    else:
        barycenter = df_temp[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).mean().to_numpy()
    
    # Create barycenter dataframe
    # bary_numpy = df_temp.to_numpy()
    bary_numpy = df_temp[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).to_numpy()

    # row_bary = [np.nan,np.nan,np.nan,barycenter[0],np.nan,barycenter[1],np.nan,np.nan,np.nan]
    row_bary = [barycenter[0], barycenter[1]]
    row_empty = np.empty((1,bary_numpy.shape[1]))
    row_empty.fill(np.nan)

    bary_numpy = np.insert(bary_numpy, range(1, len(bary_numpy)+1, 1), row_bary, axis=0)
    bary_numpy = np.insert(bary_numpy, range(2, len(bary_numpy), 2), row_empty, axis=0)
    bary_data = pd.DataFrame(bary_numpy, columns=['recommendation', 'easiness'])

    return df_tempy, barycenter, bary_data

# Unset initial filter lock
general_dict['filter_called'] = False
# Set initial filters to all
general_dict['filter_list'] = np.concatenate((df_orig.filter_one.unique(),df_orig.filter_two.unique(),df_orig.filter_three.unique(),df_orig.filter_four.unique()))
general_dict['full_filter_list'] = np.concatenate((df_orig.filter_one.unique(),df_orig.filter_two.unique(),df_orig.filter_three.unique(),df_orig.filter_four.unique()))

# Calculating filtered dataframe
filtered_df = df_orig.loc[(df_orig['filter_one'].isin(general_dict['filter_list']) & df_orig['filter_two'].isin(general_dict['filter_list']) & df_orig['filter_three'].isin(general_dict['filter_list']) & df_orig['filter_four'].isin(general_dict['filter_list']))]
# Calculating new data points, barycenter and its edges
df_points, barycenter, df_bary = calculate_barycenter(filtered_df[['index', 'recommendation', 'easiness', 'filter_one', 'filter_two', 'filter_three', 'filter_four', 'rec_sc', 'eas_sc']], general_dict['filter_list'])

###################################################################################
###################################################################################

from bokeh.models import ColumnDataSource, Callback, Toggle, BoxAnnotation, LabelSet, Label, HoverTool, DataTable, TableColumn, Image, TapTool, Tap, HBar, Plot, Div
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, Spacer

###################################################################################
############################## Visual 4 - Data Table ##############################
# Create data table structure
data_columns = [
        TableColumn(field="filter_one", title="Company"),
        TableColumn(field="filter_two", title="Service"),
        TableColumn(field="filter_three", title="Country"),
    ]
data_source = ColumnDataSource(pd.DataFrame(columns=['filter_one', 'filter_two', 'filter_three']))
data_table = DataTable(source=data_source, columns=data_columns, width=400, height=600)

###################################################################################
###################################################################################


###################################################################################
############################## Visual 1 - Points Plot #############################


#---------------------------------------------------------------------------------#
#------------------------------- Static Background -------------------------------#
# Create points plot
general_dict['points_plot'] = figure(x_range=(0, 10), y_range=(0, 10), plot_width=600, plot_height=600, match_aspect=True, tools=['tap'])

# Hide real axis
general_dict['points_plot'].axis.visible = False

# Hide real grid
general_dict['points_plot'].xgrid.grid_line_color = None
general_dict['points_plot'].ygrid.grid_line_color = None

# Define grid lines
general_dict['points_plot'].xaxis.ticker = list(range(11))
general_dict['points_plot'].yaxis.ticker = list(range(11))

# Create color zones
general_dict['points_plot'].circle(x=7.0, y=7.0, radius=1, fill_alpha=1, fill_color='#fbe5d6', radius_units='data', line_color=None, level='underlay')
ba1 = BoxAnnotation(bottom=7, top=10, left=0, right=7, fill_alpha=1, fill_color='#fbe5d6', level='underlay')
ba2 = BoxAnnotation(bottom=0, top=7, left=7, right=10, fill_alpha=1, fill_color='#fbe5d6', level='underlay')
ba3 = BoxAnnotation(bottom=0, top=7, left=0, right=7, fill_alpha=0.3, fill_color='#bf0603', level='underlay')
ba4 = BoxAnnotation(bottom=7, top=10, left=7, right=10, fill_alpha=0.3, fill_color='#538d22', level='underlay')
general_dict['points_plot'].add_layout(ba1)
general_dict['points_plot'].add_layout(ba2)
general_dict['points_plot'].add_layout(ba3)
general_dict['points_plot'].add_layout(ba4)

# Create fake axis lines with ticks and labels
general_dict['points_plot'].line(x=[0, 10], y=[7, 7], line_color='skyblue', level='underlay')
general_dict['points_plot'].line(x=[7, 7], y=[0, 10], line_color='forestgreen', level='underlay')
general_dict['points_plot'].segment(x0=list(range(11)), y0=list(np.array(range(7,8))-0.1)*11,
             x1=list(range(11)), y1=list(np.array(range(7,8))+0.1)*11,
             color='skyblue', line_width=2, level='underlay')
general_dict['points_plot'].segment(x0=list(np.array(range(7,8))-0.1)*11, y0=list(range(11)),
             x1=list(np.array(range(7,8))+0.1)*11, y1=list(range(11)),
             color='forestgreen', line_width=1, level='underlay')
source = ColumnDataSource(data=dict(height=list(range(11)),
                                    weight=list(np.array(range(7,8)))*11,
                                    names=list(range(11))))
labels = LabelSet(x='weight', y='height', text='names', level='glyph',
              x_offset=8, y_offset=2, source=source, render_mode='canvas')
general_dict['points_plot'].add_layout(labels)
labels = LabelSet(x='height', y='weight', text='names', level='glyph',
              x_offset=5, y_offset=-20, source=source, render_mode='canvas')
general_dict['points_plot'].add_layout(labels)

# Create quadrant labels
citation = Label(x=8, y=8, text='Love', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=3, y=8, text='Frustration', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=3, y=3, text='Repulsion', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=8, y=3, text='Frustration', render_mode='css')
general_dict['points_plot'].add_layout(citation)
#----------------------------- ^ Static Background ^ -----------------------------#
#---------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------#
#------------------------------ Ineractive Triggers ------------------------------#

# Filter countries on button click
def callback_h(selected_state):
    
    if general_dict['filter_called']:
        return None

    # Get selected filters from toggle buttons
    selected_country_list = []
    if filter_button1.active:
        general_dict['filter_called'] = True
        for button in logical_buttons1:
            button.active = False
        general_dict['filter_called'] = False
        filter_button1.active = False
    elif filter_button3.active:
        general_dict['filter_called'] = True
        for button in logical_buttons1:
            button.active = True
            selected_country_list.append(button.name)
        general_dict['filter_called'] = False
        filter_button3.active = False
        if len(selected_country_list) == len(general_dict['full_filter_list']):
            return None
    else:
        for button in logical_buttons1:
            if button.active:
                selected_country_list.append(button.name)
    if filter_button2.active:
        general_dict['filter_called'] = True
        for button in logical_buttons2:
            button.active = False
        general_dict['filter_called'] = False
        filter_button2.active = False
    elif filter_button4.active:
        general_dict['filter_called'] = True
        for button in logical_buttons2:
            button.active = True
            selected_country_list.append(button.name)
        general_dict['filter_called'] = False
        filter_button4.active = False
        if len(selected_country_list) == len(general_dict['full_filter_list']):
            return None
    else:
        for button in logical_buttons2:
            if button.active:
                selected_country_list.append(button.name)
    if filter_button5.active:
        general_dict['filter_called'] = True
        for button in logical_buttons3:
            button.active = False
        general_dict['filter_called'] = False
        filter_button5.active = False
    elif filter_button7.active:
        general_dict['filter_called'] = True
        for button in logical_buttons3:
            button.active = True
            selected_country_list.append(button.name)
        general_dict['filter_called'] = False
        filter_button7.active = False
        if len(selected_country_list) == len(general_dict['full_filter_list']):
            return None
    else:
        for button in logical_buttons3:
            if button.active:
                selected_country_list.append(button.name)
    if filter_button6.active:
        general_dict['filter_called'] = True
        for button in logical_buttons4:
            button.active = False
        general_dict['filter_called'] = False
        filter_button6.active = False
    elif filter_button8.active:
        general_dict['filter_called'] = True
        for button in logical_buttons4:
            button.active = True
            selected_country_list.append(button.name)
        general_dict['filter_called'] = False
        filter_button8.active = False
        if len(selected_country_list) == len(general_dict['full_filter_list']):
            return None
    else:
        for button in logical_buttons4:
            if button.active:
                selected_country_list.append(button.name)

    # Setting new filters
    general_dict['filter_list'] = selected_country_list
    # Calculating new filtered dataframe
    filtered_df = df_orig.loc[(df_orig['filter_one'].isin(general_dict['filter_list']) & df_orig['filter_two'].isin(general_dict['filter_list']) & df_orig['filter_three'].isin(general_dict['filter_list']) & df_orig['filter_four'].isin(general_dict['filter_list']))]
    # Calculating new data points, barycenter and its edges
    df_points, barycenter, df_bary = calculate_barycenter(filtered_df[['index', 'recommendation', 'easiness', 'filter_one', 'filter_two', 'filter_three', 'filter_four', 'rec_sc', 'eas_sc']], general_dict['filter_list'])
    
    # Create data source for points plot
    general_dict['data_points'] = ColumnDataSource(df_points)
    
    # Attach circle tap callback to new circles
    general_dict['data_points'].selected.on_change('indices', callback)
    
    # Remove old data points
    general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='data_points')[0])
    # Plot new data points
    general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', source=general_dict['data_points'], selection_fill_alpha=0.2, selection_color="firebrick", line_width=1, nonselection_line_color="firebrick")    
    
    # Remove old barycenter and connecting edges
    if len(general_dict['points_plot'].select(name='bary')) > 0 and len(general_dict['points_plot'].select(name='barypoint')) > 0:
        general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='bary')[0])
        general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='barypoint')[0])

    # Plot new barycenter and connecting edges
    general_dict['points_plot'].line(x='recommendation', y='easiness', source=ColumnDataSource(df_bary), name='bary', line_width=1, level='overlay', color='#2a679d')
    general_dict['points_plot'].circle(x=barycenter[0], y=barycenter[1], color='firebrick', size=barycenter[0]+barycenter[1]+1, name='barypoint', level='overlay')

    # Calculate new scores
    df_emotions = filtered_df[['rec_sc','eas_sc']]

    if len(df_emotions) > 0:
        rec_score = df_emotions['rec_sc'].mean() * 100
        easy_score = df_emotions['eas_sc'].mean() * 100
    else:
        rec_score = 0.0
        easy_score = 0.0
    
    # Update scores
    general_dict['emotions_rec_score'].patch({ 'right' : [(0,rec_score)], 'left' : [(0,rec_score)] })
    general_dict['emotions_easy_score'].patch({ 'right' : [(0,easy_score)], 'left' : [(0,easy_score)] })

    # Calculate new word frequencies
    d_freq_pv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] else 0 for item in v[:]]) for k, v in d_pv.items()}
    dict_freq_pv = d_freq_pv
    d_freq_uv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] else 0 for item in v[:]]) for k, v in d_uv.items()}
    d_freq_uv = pd.DataFrame.from_dict({k: d_freq_uv[k] for k in d_freq_pv.keys() if k in d_freq_uv}, orient='index', columns=['freq_uv']).reset_index()
    d_freq_nv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] else 0 for item in v[:]]) for k, v in d_nv.items()}
    d_freq_nv = pd.DataFrame.from_dict({k: d_freq_nv[k] for k in d_freq_pv.keys() if k in d_freq_nv}, orient='index', columns=['freq_nv']).reset_index()
    d_freq_pv = pd.DataFrame.from_dict(d_freq_pv, orient='index', columns=['freq_pv']).reset_index()
    frequency_df = d_freq_pv.merge(d_freq_uv, how='left').merge(d_freq_nv, how='left').fillna(0).astype({'index':'category', 'freq_pv':'int16', 'freq_uv':'int16', 'freq_nv':'int16'}).rename(columns = {'index':'freq_w'})
    
    # Update word frequencies
    general_dict['freq_source'].patch({ 'freq_pv' : [(general_dict['freq_words_slice'],frequency_df['freq_pv'])], 'freq_uv' : [(general_dict['freq_words_slice'],frequency_df['freq_uv'])], 'freq_nv' : [(general_dict['freq_words_slice'],frequency_df['freq_nv'])] })
    
    # Remove old word cloud
    if len(general_dict['words_plot'].select(name='words')) > 0:
        general_dict['words_plot'].renderers.remove(general_dict['words_plot'].select(name='words')[0])

    # Plot new word cloud
    if d_freq_pv['freq_pv'].sum() != 0:
        # Calculate frequency-based word cloud
        general_dict['wordcloud'].generate_from_frequencies(frequencies=dict_freq_pv)

        general_dict['words_plot'].image(image=[np.flipud(np.mean(general_dict['wordcloud'].to_array(), axis=2))], x=0, y=0, dw=200, dh=400, level="image", name='words')

# Update data table on circle tap
def callback(attr, old, new):
    recommendations, easinesses = ([],[])

    inds = general_dict['data_points'].selected.indices
    if (len(inds) == 0):
        pass

    for i in range(0, len(inds)):
        recommendations.append(general_dict['data_points'].data['recommendation'][inds[i]])
        easinesses.append(general_dict['data_points'].data['easiness'][inds[i]])
    
    current = df_points.loc[(df_points['recommendation'].isin(recommendations)) & (df_points['easiness'].isin(easinesses)) & (df_points['filter_one'].isin(general_dict['filter_list'])) & (df_points['filter_two'].isin(general_dict['filter_list'])) & (df_points['filter_three'].isin(general_dict['filter_list'])) & (df_points['filter_four'].isin(general_dict['filter_list']))]
    
    data_source.data = {
            'filter_one' : current.filter_one,
            'filter_two' : current.filter_two,
            'filter_three' : current.filter_three,
        }
    
#---------------------------- ^ Ineractive Triggers ^ ----------------------------#
#---------------------------------------------------------------------------------#

# Create data source for points plot
general_dict['data_points'] = ColumnDataSource(df_points)

# Attach circle tap callback to circles
general_dict['data_points'].selected.on_change('indices', callback)

# Plot data circles
general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', source=general_dict['data_points'], selection_fill_alpha=0.2, selection_color="firebrick", line_width=1, nonselection_line_color="firebrick")
# general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', alpha=0.4, source=general_dict['data_points'], selection_color="firebrick", selection_alpha=0.4, tags=['country','service'], line_width=1, nonselection_fill_alpha=0.2, nonselection_fill_color="blue", nonselection_line_color="firebrick", nonselection_line_alpha=1.0)

# Plot barycenter and connecting edges
general_dict['bary_points'] = ColumnDataSource(df_bary)
general_dict['points_plot'].line(x='recommendation', y='easiness', source=general_dict['bary_points'], name='bary', line_width=1, level='overlay', color='#2a679d')
general_dict['points_plot'].circle(x=barycenter[0], y=barycenter[1], color='firebrick', size=barycenter[0]+barycenter[1], name='barypoint', level='overlay')

###################################################################################
###################################################################################

###################################################################################
############################ Visual 2 - Buttons Columns ###########################
# buttons1, buttons2 = ([],[])
logical_buttons1, logical_buttons2, logical_buttons3, logical_buttons4 = ([],[],[],[])
for country in df_orig.filter_one.unique():
    # Plot buttons
    button = Toggle(label=str(country), button_type="warning", name=str(country), width_policy='fixed', width=105)
    button.active = True
    button.on_click(callback_h)
    logical_buttons1.append(button)
for country in df_orig.filter_two.unique():
    # Plot buttons
    button = Toggle(label=str(country), button_type="primary", name=str(country), width_policy='fixed', width=105)
    button.active = True
    button.on_click(callback_h)
    logical_buttons2.append(button)
for country in df_orig.filter_three.unique():
    # Plot buttons
    button = Toggle(label=str(country), button_type="warning", name=str(country), width_policy='fixed', width=105)
    button.active = True
    button.on_click(callback_h)
    logical_buttons3.append(button)
for country in df_orig.filter_four.unique():
    # Plot buttons
    button = Toggle(label=str(country), button_type="primary", name=str(country), width_policy='fixed', width=105)
    button.active = True
    button.on_click(callback_h)
    logical_buttons4.append(button)

filter_button1 = Toggle(label='Select None', button_type="warning", name='filter1', width_policy='fixed', width=105)
filter_button2 = Toggle(label='Select None', button_type="primary", name='filter2', width_policy='fixed', width=105)
filter_button1.active = False
filter_button2.active = False
filter_button1.on_click(callback_h)
filter_button2.on_click(callback_h)

filter_button3 = Toggle(label='Select All', button_type="warning", name='filter3', width_policy='fixed', width=105)
filter_button4 = Toggle(label='Select All', button_type="primary", name='filter4', width_policy='fixed', width=105)
filter_button3.active = False
filter_button4.active = False
filter_button3.on_click(callback_h)
filter_button4.on_click(callback_h)

filter_button5 = Toggle(label='Select None', button_type="warning", name='filter5', width_policy='fixed', width=105)
filter_button6 = Toggle(label='Select None', button_type="primary", name='filter6', width_policy='fixed', width=105)
filter_button5.active = False
filter_button6.active = False
filter_button5.on_click(callback_h)
filter_button6.on_click(callback_h)

filter_button7 = Toggle(label='Select All', button_type="warning", name='filter7', width_policy='fixed', width=105)
filter_button8 = Toggle(label='Select All', button_type="primary", name='filter8', width_policy='fixed', width=105)
filter_button7.active = False
filter_button8.active = False
filter_button7.on_click(callback_h)
filter_button8.on_click(callback_h)

buttons1 = [filter_button1, filter_button3].extend(logical_buttons1)
buttons2 = [filter_button2, filter_button4].extend(logical_buttons2)
buttons3 = [filter_button5, filter_button7].extend(logical_buttons3)
buttons4 = [filter_button6, filter_button8].extend(logical_buttons4)
###################################################################################
###################################################################################


###################################################################################
############################# Visual 6 - Emotions Plot ############################  

df_emotions = filtered_df[['rec_sc','eas_sc']]

rec_score = df_emotions['rec_sc'].mean() * 100
easy_score = df_emotions['eas_sc'].mean() * 100

general_dict['emotions_rec_score'] = ColumnDataSource(dict(right=[rec_score], left=[rec_score],))
general_dict['emotions_easy_score'] = ColumnDataSource(dict(right=[easy_score], left=[easy_score],))

general_dict['emotions_plot'] = Plot(
    title=None, plot_width=600, plot_height=180,
    min_border=0, toolbar_location=None, outline_line_color=None, output_backend="webgl")

general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))

general_dict['emotions_plot'].add_glyph(general_dict['emotions_rec_score'], HBar(y=0.4, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='rec_s')
general_dict['emotions_plot'].add_glyph(general_dict['emotions_easy_score'], HBar(y=0.0, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='easy_s')

# Create labels
citation = Label(x=-24, y=0.55, text='Recommendation', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-12, y=0.16, text='Easiness', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-82, y=-0.2, text='NEEDS IMPROVEMENT', render_mode='css', text_color="#931a25")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=7, y=-0.2, text='GOOD', render_mode='css', text_color="#ffc93c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=40, y=-0.2, text='GREAT', render_mode='css', text_color="#b3de69")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=68, y=-0.2, text='EXCELLENT', render_mode='css', text_color="#158467")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-103, y=0.16, text='-100', render_mode='css', text_color="#4c4c4c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=93, y=0.16, text='100', render_mode='css', text_color="#4c4c4c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=1.5, y=0.35, text='0', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=31.5, y=0.35, text='30', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=71.5, y=0.35, text='70', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=1.5, y=-0.05, text='0', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=31.5, y=-0.05, text='30', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=71.5, y=-0.05, text='70', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)

###################################################################################
###################################################################################

###################################################################################
############################# Visual 4 - Frequency Plot ###########################

# Split data into recommendation and easiness frames
df_one = filtered_df.loc[~filtered_df['question_one_filtered_lemmas'].isna(), ['sentiment', 'question_one_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
df_two = filtered_df.loc[~filtered_df['question_two_filtered_lemmas'].isna(), ['sentiment', 'question_two_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
df_three = filtered_df.loc[~filtered_df['question_three_filtered_lemmas'].isna(), ['sentiment', 'question_three_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
df_four = filtered_df.loc[~filtered_df['question_four_filtered_lemmas'].isna(), ['sentiment', 'question_four_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]


# Fill sentiment column
# df_one['sentiment'] = (np.select(condlist=[df_one['recommendation'] < 7, df_one['recommendation'] < 9], choicelist=[-1, 0], default=+1))
# df_one = df_one.astype({'sentiment':'int8'})

d_pv = dict()

selected_lemmas_pv = df_one.loc[(df_one['sentiment'] > 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][['question_one_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_two.loc[(df_two['sentiment'] > 0.0) & (df_two['question_two_filtered_lemmas'].str.len() > 1)][['question_two_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_two_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_three.loc[(df_three['sentiment'] > 0.0) & (df_three['question_three_filtered_lemmas'].str.len() > 1)][['question_three_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_three_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_four.loc[(df_four['sentiment'] > 0.0) & (df_four['question_four_filtered_lemmas'].str.len() > 1)][['question_four_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_four_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
for lemmas_row in selected_lemmas_pv.itertuples():
    for word in lemmas_row[1]:
        if not word in d_pv:
            d_pv[word] = np.array([[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]])
        d_pv[word] = np.vstack((d_pv[word],np.array(  [[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]]  ) ))

d_uv = dict()

selected_lemmas_pv = df_one.loc[(df_one['sentiment'] == 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][['question_one_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_two.loc[(df_two['sentiment'] == 0.0) & (df_two['question_two_filtered_lemmas'].str.len() > 1)][['question_two_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_two_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_three.loc[(df_three['sentiment'] == 0.0) & (df_three['question_three_filtered_lemmas'].str.len() > 1)][['question_three_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_three_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_four.loc[(df_four['sentiment'] == 0.0) & (df_four['question_four_filtered_lemmas'].str.len() > 1)][['question_four_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_four_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
for lemmas_row in selected_lemmas_pv.itertuples():
    for word in lemmas_row[1]:
        if not word in d_uv:
            d_uv[word] = np.array([[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]])
        d_uv[word] = np.vstack((d_uv[word],np.array(  [[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]]  ) ))

d_nv = dict()

selected_lemmas_pv = df_one.loc[(df_one['sentiment'] < 0.0) & (df_one['question_one_filtered_lemmas'].str.len() > 1)][['question_one_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']]
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_two.loc[(df_two['sentiment'] < 0.0) & (df_two['question_two_filtered_lemmas'].str.len() > 1)][['question_two_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_two_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_three.loc[(df_three['sentiment'] < 0.0) & (df_three['question_three_filtered_lemmas'].str.len() > 1)][['question_three_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_three_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
selected_lemmas_pv = pd.concat([selected_lemmas_pv, df_four.loc[(df_four['sentiment'] < 0.0) & (df_four['question_four_filtered_lemmas'].str.len() > 1)][['question_four_filtered_lemmas', 'filter_one', 'filter_two', 'filter_three', 'filter_four']].rename(columns={'question_four_filtered_lemmas':'question_one_filtered_lemmas'})  ], ignore_index=True)
for lemmas_row in selected_lemmas_pv.itertuples():
    for word in lemmas_row[1]:
        if not word in d_nv:
            d_nv[word] = np.array([[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]])
        d_nv[word] = np.vstack((d_nv[word],np.array(  [[lemmas_row[2],lemmas_row[3],lemmas_row[4],lemmas_row[5]]]  ) ))



d_freq_pv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] in general_dict['filter_list'] else 0 for item in v[:]]) for k, v in d_pv.items()}
d_freq_uv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] in general_dict['filter_list'] else 0 for item in v[:]]) for k, v in d_uv.items()}
d_freq_uv = pd.DataFrame.from_dict({k: d_freq_uv[k] for k in d_freq_pv.keys() if k in d_freq_uv}, orient='index', columns=['freq_uv']).reset_index()
d_freq_nv = {k: sum([1 if item[0] in general_dict['filter_list'] and item[1] in general_dict['filter_list'] and item[2] in general_dict['filter_list'] and item[3] in general_dict['filter_list'] else 0 for item in v[:]]) for k, v in d_nv.items()}
d_freq_nv = pd.DataFrame.from_dict({k: d_freq_nv[k] for k in d_freq_pv.keys() if k in d_freq_nv}, orient='index', columns=['freq_nv']).reset_index()
d_freq_pv = pd.DataFrame.from_dict(d_freq_pv, orient='index', columns=['freq_pv']).reset_index()
frequency_df = d_freq_pv.merge(d_freq_uv, how='left').merge(d_freq_nv, how='left').fillna(0).astype({'index':'category', 'freq_pv':'int16', 'freq_uv':'int16', 'freq_nv':'int16'}).rename(columns = {'index':'freq_w'})

general_dict['freq_words_slice'] = slice(len(frequency_df))
general_dict['freq_source'] = ColumnDataSource(frequency_df)

general_dict['freq_plot'] = figure(y_range=frequency_df['freq_w'].to_list(), plot_height=450, plot_width=500, match_aspect=True, toolbar_location=None, tools="", outline_line_color=None, output_backend="webgl", name='freq_f')

general_dict['freq_plot'].title.text = "Frequencies"
general_dict['freq_plot'].title.align = "center"
general_dict['freq_plot'].title.text_color = "#4c4c4c"
general_dict['freq_plot'].title.render_mode = 'css'
general_dict['freq_plot'].title.text_font_size = "16px"
general_dict['freq_plot'].title.text_font_style='bold'

general_dict['freq_plot'].hbar_stack(['freq_pv', 'freq_uv', 'freq_nv'], y='freq_w', width=0.9, color=['#29ce42', '#ffc12f', '#ff473d'], source=general_dict['freq_source'], name='freq_w')

general_dict['freq_plot'].xgrid.grid_line_color = None
general_dict['freq_plot'].ygrid.grid_line_color = None
general_dict['freq_plot'].x_range.start = 0.0

###################################################################################
###################################################################################


###################################################################################
############################## Visual 5 - Words Cloud ############################# 

from wordcloud import WordCloud

# Prepare dataframe
df_viso = frequency_df[['freq_w','freq_pv']].set_index('freq_w')

# Calculate frequency-based word cloud
general_dict['wordcloud'] = WordCloud(background_color ='white')
general_dict['wordcloud'].generate_from_frequencies(frequencies=df_viso.to_dict()['freq_pv'])

# Convert 3d RGB uint ndarray to 2d Luminescence float ndarray
# l_array = np.rot90(np.transpose(np.apply_along_axis(np.mean, 2, general_dict['wordcloud'].to_array())/256))

general_dict['words_plot'] = figure(width = 400, height=200, toolbar_location=None, tools="", output_backend="webgl")
# general_dict['words_plot'].image(image=[l_array], x=0, y=0, dw=200, dh=400, palette="Greys256", level="image", name='words')
general_dict['words_plot'].image(image=[np.flipud(np.mean(general_dict['wordcloud'].to_array(), axis=2))], x=0, y=0, dw=200, dh=400, level="image", name='words')

general_dict['words_plot'].title.text = "Word Cloud"
general_dict['words_plot'].title.align = "center"
general_dict['words_plot'].title.text_color = "#4c4c4c"
general_dict['words_plot'].title.render_mode = 'css'
general_dict['words_plot'].title.text_font_size = "16px"
general_dict['words_plot'].title.text_font_style='bold'

# Hide axis, grid, padding
general_dict['words_plot'].axis.visible = False
general_dict['words_plot'].xgrid.grid_line_color = None
general_dict['words_plot'].ygrid.grid_line_color = None
general_dict['words_plot'].x_range.range_padding = 0
general_dict['words_plot'].y_range.range_padding = 0

###################################################################################
###################################################################################

# Connect all plots into one object and set layout
buttons_1_col = column(filter_button1, filter_button3, column(logical_buttons1, height=520, width=130, css_classes=['scrollable'], max_height=520, min_width=130))
buttons_2_col = column(filter_button2, filter_button4, column(logical_buttons2, height=520, width=130, css_classes=['scrollable'], max_height=520, min_width=130))
buttons_3_col = column(filter_button5, filter_button7, column(logical_buttons3, height=520, width=130, css_classes=['scrollable'], max_height=520, min_width=130))
buttons_4_col = column(filter_button6, filter_button8, column(logical_buttons4, height=520, width=130, css_classes=['scrollable'], max_height=520, min_width=130))
buttons_row = row(children=[buttons_1_col, buttons_2_col, buttons_3_col, buttons_4_col], height=600, max_height=600)


# header = Div(text="<link rel='stylesheet' type='text/css' href='./Templates/styles.css'>")
# layout = column(header, other_content)
# curdoc().add_root(layout)


curdoc().add_root(column(row(general_dict['points_plot'], buttons_row, data_table), Spacer(height=50), row(general_dict['freq_plot'], general_dict['words_plot'], general_dict['emotions_plot'])))



# curdoc().add_root(column(row(general_dict['points_plot'], buttons_row, data_table), Spacer(height=50), row(general_dict['freq_plot'], general_dict['emotions_plot'])))