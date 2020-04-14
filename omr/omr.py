import numpy as np
from PIL import Image, ImageDraw
import os
from omr.TemplateDetection import rescale_template, template_match, temp_position
from omr.NoteDetection import specify_notes
from omr.HoughSpace import EdgeDetection, get_staff_and_spacing_parameter
import pandas as pd
from flask_socketio import SocketIO, emit
from config import socketio

def main(request_image):


    DATA_DIR  = 'omr/test-images/'
    # music_file = sys.argv[1]
    music_file = 'music1.png'
    template1_file = 'template1.png'
    template2_file = 'template2.png'
    template3_file = 'template3.png'

    # getting space and starting position parameter
    socketio.emit('omr_event', {'data': 'Calculating Edge Matrix'})
    EdgeMatrix = EdgeDetection(request_image, self = False)

    socketio.emit('omr_event', {'data': 'Calculating Hough Space'})
    space, starting_positions = get_staff_and_spacing_parameter(EdgeMatrix)


    # reading images and templates
    # im = Image.open(os.path.join(DATA_DIR, music_file),mode="r").convert('L')
    socketio.emit('omr_event', {'data': 'Reading Templates'})
    im = request_image
    temp1 = Image.open(os.path.join(DATA_DIR,template1_file),mode="r").convert('L')
    temp2 = Image.open(os.path.join(DATA_DIR,template2_file),mode="r").convert('L')
    temp3 = Image.open(os.path.join(DATA_DIR,template3_file),mode="r").convert('L')

    # convert into array and scale
    im_array = np.array(im)
    temp_array1 = np.array(temp1)
    temp_array2 = np.array(temp2)
    temp_array3 = np.array(temp3)

    socketio.emit('omr_event', {'data': 'Rescaling Templates'})

    # scaling template from spacing parameter
    # for template1
    width1 = temp_array1.shape[1]
    height1 = temp_array1.shape[0]
    new_height1 = space
    new_width1  = int(new_height1 * (width1 / height1))
    size1 = new_height1, new_width1
    temp_array_1 = rescale_template(temp_array1,size1)


    # scaling template from spacing parameter
    # for template2
    width2 = temp_array2.shape[1]
    height2 = temp_array2.shape[0]
    new_height2 = 3*space
    new_width2  = int(new_height2 * (width2 / height2))
    size2 = new_height2, new_width2
    temp_array_2 = rescale_template(temp_array2,size2)

    # scaling template from spacing parameter
    # for template3
    width3 = temp_array3.shape[1]
    height3 = temp_array3.shape[0]

    new_height3 = int(2.5*space)
    new_width3  = int(new_height3 * (width3 / height3))
    size3 = new_height3, new_width3

    temp_array_3 = rescale_template(temp_array3,size3)


    # scaling values of image and templates
    im_scale = (im_array - im_array.min())/(im_array.max() - im_array.min())
    temp_scale1 = (temp_array_1 - temp_array_1.min())/(temp_array_1.max() - temp_array_1.min())
    temp_scale2 = (temp_array_2 - temp_array_2.min())/(temp_array_2.max() - temp_array_2.min())
    temp_scale3 = (temp_array_3 - temp_array_3.min())/(temp_array_3.max() - temp_array_3.min())

    socketio.emit('omr_event', {'data': 'Matching templates'})
    # matching template with image using hemming distance
    im_match1 = template_match(im_scale,temp_scale1)
    im_match2 = template_match(im_scale,temp_scale2)
    im_match3 = template_match(im_scale,temp_scale3)

    # setting cutoff for template 1 = 220
    # setting cutoff for template 2 = 250
    # setting cutoff for template 3 = 245
    im_match1[im_match1 > 220] = 255
    im_match1[im_match1 <= 220] = 0

    im_match2[im_match2 > 250] = 255
    im_match2[im_match2 <= 250] = 0


    im_match3[im_match3 > 245] = 255
    im_match3[im_match3 <= 245] = 0

    socketio.emit('omr_event', {'data': 'Normalizing Values'})

    # trimming last rows and columns for noise removal in resizing
    im_match_1 = np.delete(im_match1,np.s_[-5:], axis=1)
    image_match1 = Image.fromarray(im_match_1)

    im_match_2 = np.delete(im_match2,np.s_[-5:], axis=1)
    image_match2 = Image.fromarray(im_match_2)

    im_match_3 = np.delete(im_match3,np.s_[-5:], axis=1)
    image_match3 = Image.fromarray(im_match_3)

    # count number of 255 values
    match1_count = np.count_nonzero(im_match_1 == 255)
    match2_count = np.count_nonzero(im_match_2 == 255)
    match3_count = np.count_nonzero(im_match_3 == 255)


    # getting x, y coordinates of templates
    x1, y1 = temp_position(im_match_1,temp_array_1.shape)
    x2, y2 = temp_position(im_match_2,temp_array_2.shape)
    x3, y3 = temp_position(im_match_3,temp_array_3.shape)

    socketio.emit('omr_event', {'data': 'Calculating Confidence Scores'})
    # getting confidence score for each templates
    conf_temp1 = match1_count / (len(x1) * temp_array_1.shape[0] * temp_array_1.shape[1])
    conf_temp2 = match2_count / (len(x2) * temp_array_2.shape[0] * temp_array_2.shape[1])
    conf_temp3 = round(match3_count / (len(x3) * temp_array_3.shape[0] * temp_array_3.shape[1]), 5)

    # making array of confidence score for each template and then normalising it
    conf_array = np.array([conf_temp1, conf_temp2, conf_temp3])
    normalized_conf_array = conf_array / np.sqrt(np.sum(conf_array ** 2))


    # specify notes space, starting position and coordinates
    image_new = np.zeros(shape = im_match1.shape)
    for i in range(len(x1)):
        if(int(x1[i] + space/2) < image_new.shape[0]):
            image_new[int(x1[i]+space/2),y1[i]] = 255

        else:
            image_new[x1[i], y1[i]] = 255

    socketio.emit('omr_event', {'data': 'Detecting Notes'})
    notes = specify_notes(space, starting_positions, image_new)


    # creating dataframe for detected.txt output from notes dictionary
    # for template1
    detect_temp1 = pd.DataFrame(data=notes, columns=['row','col', 'symbol_type'])
    detect_temp1['height'] = temp_array_1.shape[0]
    detect_temp1['width'] = temp_array_1.shape[1]
    detect_temp1['confidence'] = normalized_conf_array[0]
    # adjusting for space/2 parameter in row position to get top left coordinates of template
    detect_temp1['row'] = detect_temp1['row'] - space/2
    detect_temp1['row'] = detect_temp1['row'].astype('int')

    # for template2
    detect_temp2 = pd.DataFrame({'row': x2,'col': y2})
    detect_temp2['symbol_type'] = 'quarter_rest'
    detect_temp2['height'] = temp_array_2.shape[0]
    detect_temp2['width'] = temp_array_2.shape[1]
    detect_temp2['confidence'] = normalized_conf_array[1]

    # for template3
    detect_temp3 = pd.DataFrame({'row': x3,'col': y3})
    detect_temp3['symbol_type'] = 'eighth_rest'
    detect_temp3['height'] = temp_array_3.shape[0]
    detect_temp3['width'] = temp_array_3.shape[1]
    detect_temp3['confidence'] = normalized_conf_array[2]

    # concatenate three dataframes of respective template into one
    detect_temp = pd.concat([detect_temp1, detect_temp2, detect_temp3], axis=0)
    detect_temp = detect_temp[['row','col','height','width','symbol_type','confidence']]


    # convert original image into rgb format
    # im1 = im.convert('RGB')
    convert_image = Image.fromarray(im)
    im1 = convert_image.convert('RGB')

    socketio.emit('omr_event', {'data': 'Drawing on sheet'})
    # drawing rectangle and text over matched templates
    draw = ImageDraw.Draw(im1)
    for i in range(len(notes)):
        x1_1 = int(notes[i][0] - space/2)
        y1_1 = notes[i][1]
        x1_2 = int(notes[i][0] - space/2) + temp_array_1.shape[0]
        y1_2 = notes[i][1] + temp_array_1.shape[1]
        draw.rectangle(((y1_1,x1_1),(y1_2,x1_2)), outline='red', width=2)
        draw.text((y1_1-10,x1_1-10), notes[i][2], fill='red')


    for i in range(len(x2)):
        x2_1 = x2[i]
        y2_1 = y2[i]
        x2_2 = x2[i] + temp_array_2.shape[0]
        y2_2 = y2[i] + temp_array_2.shape[1]
        draw.rectangle(((y2_1,x2_1),(y2_2,x2_2)), outline='green',width=2)
        draw.text((y2_1-10,x2_1-10), "quarter",fill='green')


    for i in range(len(x3)):
        x3_1 = x3[i]
        y3_1 = y3[i]
        x3_2 = x3[i] + temp_array_3.shape[0]
        y3_2 = y3[i] + temp_array_3.shape[1]
        draw.rectangle(((y3_1,x3_1),(y3_2,x3_2)), outline='blue',width=2)
        draw.text((y3_1-10,x3_1-10), "eighth",fill='blue')

    # return detect_temp.to_dict()
    # saving detected notes, quarter and eighth image file
    # im1.save(os.path.join(DATA_DIR, 'detected.png'))
    socketio.emit('omr_event', {'data': 'Done'})

    return (im1, detect_temp.to_dict())

    # saving detected .txt file having row, col, height, width, symbol and confidence
    # detect_temp.to_csv(os.path.join(DATA_DIR, 'detected.txt'), header=False, index=False, sep='\t')

